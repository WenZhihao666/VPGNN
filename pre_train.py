
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from NeighborSampler import NeighborSampler
from torch_geometric.data import Data
from torch_geometric.utils import degree
import random
from torch_geometric.nn import DeepGraphInfomax, SAGEConv

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch table IO')

    parser.add_argument('--tables',
                        default="./data/amazon_feat_label.npy,./data/amazon_edge_index.npy",
                        type=str)
    parser.add_argument('--data', default="amazon", type=str)

    parser.add_argument('--outputs', default="", type=str, help='ODPS output table names')
    parser.add_argument('--ftsize', default=32, type=int, help='feature size')
    parser.add_argument('--model', default="SAGE", type=str, help='model of choice')
    parser.add_argument('--lr', default=1e-4, type=float, help='lr')

    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--gnn_input', type=int, default=128)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)
    parser.add_argument('--class_in', type=int, default=128)
    parser.add_argument('--epoch_n', type=int, default=101, help='epoch number')
    parser.add_argument('--patience', type=int, default=20, help='patience number')

    parser.add_argument('--sample_size', default=[25, 10], type=list)

    parser.add_argument('--table_name', default='amazon', type=str)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using CPU.")

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

    setup_seed(42)
    node_ft2 = preprocessing.StandardScaler().fit_transform(node_ft2)
    g_ft = torch.tensor(node_ft2, dtype=torch.float32)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    data = Data(x=g_ft, edge_index=adj)
    model = DeepGraphInfomax(
        hidden_channels=args.gnn_hid, encoder=SAGE(data.num_features, args.gnn_hid, args.gnn_output),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters()}, ],
                                 lr=args.lr)


    pretrain_loader = NeighborSampler(data.edge_index, sizes=args.sample_size, batch_size=args.batch_size,
                                      shuffle=True, num_nodes=data.num_nodes)

    test_loader = NeighborSampler(data.edge_index, sizes=args.sample_size, batch_size=args.batch_size,
                                  shuffle=False, num_nodes=data.num_nodes)

    x = data.x.to(device)


    def pretrain():
        model.train()
        total_loss = total_examples = 0
        for batch_size, n_id, adjs, raw_batch in pretrain_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]  # two different adjs
            # print(raw_batch[:100])

            pos_z, neg_z, summary = model(x[n_id], adjs)
            loss = model.loss(pos_z, neg_z, summary)
            optimizer.zero_grad()

            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pos_z.size(0)
            total_examples += pos_z.size(0)

        return total_loss / total_examples


    cnt_wait = 0
    best = 10000000
    best_t = 0
    # patience = 50
    for epoch in range(1, args.epoch_n):
        loss = pretrain()
        print(f'pretrain loss= {loss:.4f}')

        if loss < best:
            print(f'Epoch: {epoch:03d}, best_loss: {best:.4f} ----->  cur_loss: {loss:.4f}')
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), './res/care_{}.pkl'.format(args.data))
        else:
            cnt_wait += 1
        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    print('Loading {}th epoch'.format(best_t))

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))

