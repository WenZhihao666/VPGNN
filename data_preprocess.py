from utils_caregnn import sparse_to_adjlist
from scipy.io import loadmat

prefix = 'data/'

amz = loadmat('data/Amazon.mat')
net_upu = amz['net_upu']
net_usu = amz['net_usu']
net_uvu = amz['net_uvu']
amz_homo = amz['homo']

sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')


import pickle
import random as rd
import numpy as np
"""
	Utility functions to handle data and evaluate model.
"""


def load_data(data):
    """
    Load graph, feature, and label given dataset name
    :returns: home and single-relation graphs, feature, label
    """
    prefix = 'data/'
    if data == 'yelp':
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        # load the preprocessed adj_lists
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)
        file.close()
    elif data == 'amazon':
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        # load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)
    return [homo, relation1, relation2, relation3], feat_data, labels


for data in ['amazon']:
    print('data', data)
    [homo, relation1, relation2, relation3], feat_data, labels = load_data("{}".format(data))
    edge_index = []
    for k, vs in homo.items():
        for v in vs:
            edge = [k, v]
            edge_index.append(edge)

    edge_index.sort()
    print("edge_index after sort", edge_index[:10])
    edge_arr = np.array(edge_index)
    print("edge arr before transpose", edge_arr)
    edge_arr = edge_arr.T
    print("edge arr before transpose", edge_arr)
    labels = labels.reshape(-1, 1)
    feat_label = np.concatenate((feat_data, labels), axis=1)
    print('feat_label', feat_label[:3])
    print('edge_arr', edge_arr[:, 3])

    np.save("./data/{}_feat_label.npy".format(data), feat_label)
    np.save("./data/{}_edge_index.npy".format(data), edge_arr)

