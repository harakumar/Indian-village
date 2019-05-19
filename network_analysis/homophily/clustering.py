import pandas as pd
import sklearn.cluster as cluster

def spectral_clustering():
    hh_cluster_dict = dict()
    for village in [1, 2, 3, 4, 6, 9, 10, 12, 15, 19, 20, 21, 23, 24, 25, 28, 29, 31, 32, 33, 36, 37, 39, 41, 42, 43, 45,
                46, 47, 48, 50, 51, 52, 55, 57, 59, 60, 62, 64, 65, 66, 67, 68, 70, 71, 72, 73, 75, 77]:
        A = pd.read_csv('../data/network_data/adjacency_matrices/adj_allVillageRelationships_HH_vilno_' + str(village) + '.csv',
                        header=None).values
        # print('A', A)
        # print('A type: ', type(A))
        #print('A shape', A.shape)
        #print('village {}'.format(village))
        hh_cluster_dict[village] = dict()
        clusterid = cluster.spectral_clustering(A)
        clusterset = set(clusterid)
        #print('clusterset: ', clusterset)
        for id in clusterset:
            #print('cluster set id: ', id)
            hh_cluster_dict[village][id] = list()
        #print('hh_cluster_dict', hh_cluster_dict)
        for i in range(A.shape[0]):
            # print('cluster id: ', clusterid[i])
            hh_cluster_dict[village][clusterid[i]].append(i + 1)
            # print('hh_cluster_dict[village][clusterid[{}]]: '.format(i), hh_cluster_dict[village][clusterid[i]])
        #print('hh_cluster_dict: ', hh_cluster_dict)
        # print('len clusterid: ', len(clusterid))

    return hh_cluster_dict




