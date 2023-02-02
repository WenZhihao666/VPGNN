import argparse
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import Data
import random
from torch_geometric.nn import DeepGraphInfomax, SAGEConv
from torch_geometric.utils import degree

from final_model_gnn import Final_m
from prompt import Prompt
from NeighborSampler import NeighborSampler

criterion = nn.NLLLoss()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            if i == 0:
                x_target = x[:size[1]]
                x = self.convs[0]((x, x_target), edge_index)
                x = torch.relu(x)
            if i == 1:
                x_target = x[:size[1]]
                x = self.convs[1]((x, x_target), edge_index)
        return x

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def main_finetune(args, all_labels):

    train_normal_ids = random.sample(pos_ids, args.shot_num*9)
    train_abnormal_ids = random.sample(neg_ids, args.shot_num)
    sup_train_ids = train_normal_ids + train_abnormal_ids
    val_test_normal_set = set(pos_ids) - set(train_normal_ids)
    val_test_abnormal_set = set(neg_ids) - set(train_abnormal_ids)
    val_normal_ids = random.sample(val_test_normal_set, args.shot_num*9)
    val_abnormal_ids = random.sample(val_test_abnormal_set, args.shot_num)
    sup_val_ids = val_normal_ids + val_abnormal_ids
    test_normal_set = val_test_normal_set - set(val_normal_ids)
    test_abnormal_set = val_test_abnormal_set -set(val_abnormal_ids)
    sup_test_ids = list(test_normal_set | test_abnormal_set)

    print('supvised len(train_ids): ', len(sup_train_ids), '\n', 'supervised len(test_ids): ', len(sup_test_ids))
    train_ids, test_ids = torch.tensor(sup_train_ids, dtype=torch.long).to(device), torch.tensor(sup_test_ids, dtype=torch.long).to(device)
    val_ids = torch.tensor(sup_val_ids, dtype=torch.long).to(device)

    all_labels = torch.tensor(all_labels, dtype=torch.long).to(device)
    prompt_normal_ids = torch.tensor(train_normal_ids, dtype=torch.long)
    prompt_abnormal_ids = torch.tensor(train_abnormal_ids, dtype=torch.long)

    data = Data(x=g_ft, edge_index=adj)

    best_model = DeepGraphInfomax(
        hidden_channels=args.gnn_hid, encoder=SAGE(data.num_features, args.gnn_hid, args.gnn_output),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption)#.to(device)
    best_model.load_state_dict(torch.load('./res/care_{}.pkl'.format(args.data), map_location=device))
    gnn = best_model.encoder.to(device)
    dgi_w = best_model.weight.to(device)

    train_loader = NeighborSampler(data.edge_index, sizes=args.sample_size, node_idx=train_ids, batch_size=args.batch_size,
                                   shuffle=True, num_nodes=data.num_nodes)

    val_loader = NeighborSampler(data.edge_index, sizes=args.sample_size, node_idx=val_ids, batch_size=args.batch_size,
                                   shuffle=False, num_nodes=data.num_nodes)

    test_loader = NeighborSampler(data.edge_index, sizes=args.sample_size, node_idx=test_ids, batch_size=args.batch_size,
                                   shuffle=False, num_nodes=data.num_nodes)

    prompt_normal_loader = NeighborSampler(data.edge_index, sizes=[5], node_idx=prompt_normal_ids, batch_size=args.batch_size,
                                   shuffle=False, num_nodes=data.num_nodes)
    prompt_abnormal_loader = NeighborSampler(data.edge_index, sizes=[5], node_idx=prompt_abnormal_ids, batch_size=args.batch_size,
                                   shuffle=False, num_nodes=data.num_nodes)


    def get_prompt():
        gnn.eval()
        with torch.no_grad():
            for _, nor_n_id, _, _ in prompt_normal_loader:
                a = nor_n_id
            for _, abnor_n_id, _, _ in prompt_abnormal_loader:
                b = abnor_n_id
            normal_loader = NeighborSampler(data.edge_index, sizes=args.sample_size, node_idx=a, batch_size=args.batch_size,
                                           shuffle=False, num_nodes=data.num_nodes)
            abnormal_loader = NeighborSampler(data.edge_index, sizes=args.sample_size, node_idx=b, batch_size=args.batch_size,
                                           shuffle=False, num_nodes=data.num_nodes)
            for batch_size, n_id, adjs, raw_batch in normal_loader:
                adjs = [adj.to(device) for adj in adjs]
                pos_p = gnn(data.x[n_id].to(device), adjs)
                pos_p = torch.sigmoid(pos_p.mean(dim=0, keepdim=True)).t()
                pos_p = torch.matmul(dgi_w, pos_p)
            for batch_size, n_id, adjs, raw_batch in abnormal_loader:
                adjs = [adj.to(device) for adj in adjs]
                neg_p = gnn(data.x[n_id].to(device), adjs)
                neg_p = torch.sigmoid(neg_p.mean(dim=0, keepdim=True)).t()
                neg_p = torch.matmul(dgi_w, neg_p)
            prompt_vectors = torch.cat((pos_p, neg_p), dim=1)
        return prompt_vectors

    prompt_vectors = get_prompt()
    f_prompt = Prompt(prompt_vectors)

    model = Final_m(gnn, f_prompt).to(device)
    optimizer = torch.optim.Adam([{'params': model.parameters()}, ],
                                  lr=args.lr)

    def train():
        model.train()
        total_loss = 0
        for batch_size, n_id, adjs, raw_batch in train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]  # two different adjs
            # print(raw_batch[:100])
            res = model(data.x[n_id].to(device), adjs)
            y_label = all_labels[raw_batch]
            task_loss = criterion(torch.log_softmax(res.squeeze(), dim=-1), y_label.to(device))
            prompt = model.parameters()[-1].T
            prompt = prompt / prompt.norm(dim=-1, keepdim=True)
            l2_loss = torch.norm(torch.mm(prompt, prompt.T) - torch.eye(prompt.shape[0]).to(device))
            loss = task_loss + args.lr_c * l2_loss
            # loss = criterion(torch.log_softmax(res.squeeze(), dim=-1), y_label.to(device))
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().clone().cpu()) * res.size(0)
            return total_loss /train_ids.size(0)

    cnt_wait = 0
    best = 0
    best_t = 0
    # patience = 50
    for epoch in range(1, args.epoch_n):
        loss = train()
        print(f'train loss= {loss:.4f}')
        model.eval()
        with torch.no_grad():
            res_list = []
            y_label_list = []
            for batch_size, n_id, adjs, raw_batch in val_loader:
                adjs = [adj.to(device) for adj in adjs]
                res = model(data.x[n_id].to(device), adjs)
                y_label = all_labels[raw_batch]
                res_list.append(res)
                y_label_list.append(y_label)
            res_list = torch.cat(res_list, dim=0)
            y_pred = torch.argmax(res_list, dim=1)
            y_label_list = torch.cat(y_label_list, dim=0)
            val_acc = f1_score(y_label_list.detach().clone().cpu(), y_pred.detach().clone().cpu())

        if val_acc > best:
            print(f'Epoch: {epoch:03d}, best_f1_score: {best:.4f} ----->  val_f1_score: {val_acc:.4f}')
            best = val_acc
            best_t = epoch
            cnt_wait = 0
            best_model = model
        else:
            cnt_wait += 1
        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    print('Loading {}th epoch'.format(best_t))

    if val_acc == 0:
        best_model = model
    best_model.eval()
    with torch.no_grad():
        res_list = []
        y_label_list = []
        for batch_size, n_id, adjs, raw_batch in test_loader:
            adjs = [adj.to(device) for adj in adjs]
            res = best_model(data.x[n_id].to(device), adjs)
            y_label = all_labels[raw_batch]
            res_list.append(res)
            y_label_list.append(y_label)
        res_list = torch.cat(res_list, dim=0)
        y_pred = torch.argmax(res_list, dim=1)
        y_label_list = torch.cat(y_label_list, dim=0)
        y_test, predictions = y_label_list.detach().clone().cpu(), y_pred.detach().clone().cpu()
        f1 = f1_score(y_test, predictions, pos_label=1, average='binary')
        # precision = precision_score(y_test, predictions, pos_label=1, average='binary')
        # recall = recall_score(y_test, predictions, pos_label=1, average='binary')
        y_prob = torch.softmax(res_list.squeeze(), dim=-1)
        y_prob = y_prob[:, 1]
        auc = roc_auc_score(y_label_list.detach().clone().cpu(), y_prob.clone().cpu())

    print('-' * 20, 'the test f1_score here', '-' * 20)
    print('auc', round(auc, 4))
    print('f1', round(f1, 4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch table IO')

    parser.add_argument('--tables',
                        default="./data/amazon_feat_label.npy,./data/amazon_edge_index.npy",
                        type=str, help='ODPS input table names')
    parser.add_argument('--data', default="amazon", type=str)

    parser.add_argument('--ftsize', default=32, type=int, help='feature size')
    parser.add_argument('--lr', default=1e-3, type=float, help='lr')
    parser.add_argument('--lr_c', default=1e-1, type=float, help='lr_c')

    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--gnn_input', type=int, default=128)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)
    parser.add_argument('--epoch_n', type=int, default=501, help='epoch number')
    parser.add_argument('--patience', type=int, default=50, help='patience number')

    parser.add_argument('--heads', type=int, default=2)

    parser.add_argument('--sample_size', default=[25, 10], type=list)

    parser.add_argument('--table_name', default='amazon', type=str)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using CPU.")
    parser.add_argument("--shot_num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=77)



    args = parser.parse_args()

    start = time.perf_counter()
    t = args.tables.split(',')
    feat_label = np.load(t[0])
    node_ft2 = feat_label[:, :-1]
    all_labels = feat_label[:, -1].astype(int)
    pos_ids = []
    neg_ids = []
    for i in range(all_labels.shape[0]):
        if all_labels[i] == 0:
            pos_ids.append(i)
        else:
            neg_ids.append(i)

    edge_index = np.load(t[1])

    adj = torch.tensor(edge_index, dtype=torch.int64)

    row = adj[0]
    deg = degree(row, dtype=torch.float32)
    deg = deg.numpy().reshape(-1, 1)
    print('deg', deg[:10])
    print('node_ft2.shape', node_ft2.shape)
    node_ft2 = np.concatenate((node_ft2, deg), axis=1)
    print('node_ft2.shape', node_ft2.shape)

    node_ft2 = preprocessing.StandardScaler().fit_transform(node_ft2)
    g_ft = torch.tensor(node_ft2, dtype=torch.float32)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    setup_seed(args.seed)
    main_finetune(args, all_labels)

    end = time.perf_counter()

    print("time consuming {:.2f}".format(end - start))

