from collections import Counter

import matplotlib.pyplot as plt
# load networkx graph objects G1 and G2
import networkx as nx
import numpy as np
import pandas as pd
from homophily.clustering import spectral_clustering

data_filepath = '../data/'
hh_cluster_dict = spectral_clustering()


def create_df():
    fields = ['hhid', 'religion', 'caste', 'resp_status']
    fields_hh = ['hhid', 'adjmatrix_key', 'village']
    df = pd.read_csv(data_filepath + "demographics/individual_characteristics.csv", usecols=fields)
    df = df.loc[df['resp_status'] == 'Head of Household']
    #df_caste = df.groupby(['hhid'])['caste'].agg(pd.Series.mode).to_frame()
    #df_caste['hhid'] = df_caste.index
    #df_caste.index.names = ['index']
    #df_religion = df.groupby(['hhid'])['religion'].agg(pd.Series.mode).to_frame()
    #df_religion['hhid'] = df_religion.index
    #df_religion.index.names = ['index']
    df_hh = pd.read_csv(data_filepath + "demographics/household_characteristics.csv", usecols=fields_hh)
    #df_hh = pd.merge(df_hh, df_caste, on="hhid", how='outer')
    #df_hh = pd.merge(df_hh, df_religion, on="hhid", how='outer')
    #df_step = [df, df_hh]
    #df_final = reduce(lambda left, right: pd.merge(left, right, on='hhid'), df_step)
    df_hh = pd.merge(df_hh, df, on="hhid",  how='outer')
    df_hh = df_hh.drop_duplicates()
    dfs = list()

    for i in range(1, 78):
        try:
            df_mf = pd.read_csv(data_filepath + "mf_participation/MF" + str(i) + ".csv", header=None)
            df_hh_village = df_hh[df_hh.village == i]
            if i == 4 or i == 9 or i == 12:
                print("len df: " + str(len(df_hh_village.index)))
            df_hh_village = df_hh_village.assign(mf_participation=df_mf.values)
            dfs.append(df_hh_village)
        except Exception as e:
            #print("creating dfs: i: " + str(i) + " " + str(e))
            pass

    return dfs

# the observed homophily in our network
# measure of homophily is the proportion of edges whose nodes share a characteristic.
def homophily(G, chars, IDs, isCluster):
    """
    Given a network G, a dict of characteristics chars for node IDs,
    and dict of node IDs for each node in the network,
    find the homophily of the network.
    """
    num_same_ties, num_ties = 0, 0

    #try:
    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 > n2:  # do not double-count edges!
                if isCluster:
                    if n1 in chars.keys() and n2 in chars.keys():
                        if G.has_edge(n1, n2):
                            num_ties += 1  # Should `num_ties` be incremented?  What about `num_same_ties`?
                            if chars[n1] == chars[n2]:
                                num_same_ties += 1  # Should `num_ties` be incremented?  What about `num_same_ties`?
                else:
                    if IDs[n1] in chars.keys() and IDs[n2] in chars.keys():
                        if G.has_edge(n1, n2):
                            num_ties += 1  # Should `num_ties` be incremented?  What about `num_same_ties`?
                            if chars[IDs[n1]] == chars[IDs[n2]]:
                                num_same_ties += 1  # Should `num_ties` be incremented?  What about `num_same_ties`?

    if num_ties == 0:
        print("G: ", G.__repr__())
    # except Exception as e:
    #     print("exception: " + str(e) + " IDs[n1]: " + str(IDs[n1]) + " IDs[n2]" + str(IDs[n2]))
    return float(num_same_ties / num_ties)


# measure of homophily will be the proportion of edges in the network whose constituent nodes share that characteristic.
# If characteristics are distributed completely randomly, the probability that
# two nodes x and y share characteristic a is the probability both nodes have characteristic a,
# which is the frequency of a squared.
# total probability that nodes x and y share their characteristic is
# therefore the sum of the frequency of each characteristic in the network
def chance_homophily(chars):
    """
    Computes the chance homophily of a characteristic,
    specified

    """
    chars_counts_dict = Counter(chars.values())
    chars_counts = np.array(list(chars_counts_dict.values()))
    chars_props = chars_counts / sum(chars_counts)
    return sum(chars_props ** 2)

def hh_chars():
    df = pd.read_stata(data_filepath + "demographics/household_characteristics.dta")

    return dict(zip(df.hhid, df.adjmatrix_key))


def map_features(dfs):
    cast_maps = dict()
    religion_maps = dict()
    for df in dfs:
        caste = dict()
        religion = dict()
        village = df.village.iloc[0]
        for i, v in df.adjmatrix_key.items():
            if v not in caste:
                caste[v] = df.caste[i]
            if v not in religion:
                religion[v] = df.religion[i]
        cast_maps[village] = caste
        religion_maps[village] = religion

    return cast_maps, religion_maps

def hh_pids():
    pids = list()
    for i in range(1, 78):
        try:
            pid = pd.read_csv(data_filepath + "network_data/adjacency_matrix_keys/key_HH_vilno_" + str(i) + ".csv", dtype=int, header=None)
            pids.append(pid)
        except:
            pass
    return pids

def find_mf_participation_percentage(dfs):
    participation = dict()

    for df in dfs:
        df = df.dropna()
        count = 0
        village = df.village.iloc[0]
        participation[village] = 0
        for p in df.mf_participation:
            count += 1
            if p == 1:
                participation[village] += 1
        participation[village] = participation[village] * 100 / count
    return participation

def find_homophilies():
    dfs = create_df()
    cast_maps, religion_maps = map_features(dfs)
    #pids = hh_pids()
    village_caste_homophily = dict()
    village_religion_homophily = dict()
    village_cluster_caste_homophily = dict()
    village_cluster_religion_homophily = dict()
    for df in dfs:
        #try:
        village = df.village.iloc[0]
        village_cluster_caste_homophily[village] = dict()
        village_cluster_religion_homophily[village] = dict()
        adj_csv = pd.read_csv(data_filepath + "network_data/adjacency_matrices/adj_allVillageRelationships_HH_vilno_" + str(village)+".csv",
                          dtype=int, header=None)
        village_cluster_caste_homophily[village],  village_cluster_religion_homophily[village] = \
            cluster_homophily_dict(adj_csv, cast_maps, religion_maps, village)

        adj = np.loadtxt(data_filepath + "network_data/adjacency_matrices/adj_allVillageRelationships_HH_vilno_" + str(village)+".csv",
                        delimiter=",")
        graph = nx.to_networkx_graph(adj)
        array_pid = df.adjmatrix_key.values
        village_caste_homophily[village] = homophily(graph, cast_maps[village], array_pid, False)
        village_religion_homophily[village] = homophily(graph, religion_maps[village], array_pid, False)
        # except Exception as e:
        #     print("final exception: " + str(e))
    participation = find_mf_participation_percentage(dfs)
    df1 = pd.Series(participation)
    df2 = pd.Series(village_caste_homophily)
    df3 = pd.Series(village_religion_homophily)
    return df1, df2, df3, village_cluster_caste_homophily, village_cluster_religion_homophily


def cluster_homophily_dict(adj_csv, cast_maps, religion_maps, village):
    village_cluster_caste_homophily = dict()
    village_cluster_religion_homophily = dict()
    # find homophilies in clusters in the same village
    # it is a dict
    village_clusters = find_homophily_in_clusters(village)
    # iterate over the dictionary and each element is a dict
    # cluster is a key, list of hh ids is a value
    # create adj ndarray matrix of the hh ids in the list
    for c in village_clusters.keys():
        cluster_g = cluster_graph(adj_csv, village_clusters[c])
        village_cluster_caste_homophily[c] = homophily(cluster_g, cast_maps[village], village_clusters[c], True)
        village_cluster_religion_homophily[c] = homophily(cluster_g, religion_maps[village], village_clusters[c], True)

    return village_cluster_caste_homophily, village_cluster_religion_homophily


def cluster_graph(adj_csv, cluster):
    G = nx.Graph()
    size_of_cluster = len(cluster)
    for i in range(size_of_cluster):
        G.add_node(cluster[i])
    for i in range(size_of_cluster):
        for j in range(size_of_cluster):
            if i != j and adj_csv[cluster[i]-1][cluster[j]-1] == 1:
                G.add_edge(cluster[i], cluster[j])
    return G


def find_homophily_in_clusters(village):
    return hh_cluster_dict[village]


def correlation(df):
    corr = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(df.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)
    plt.show()
    #plt.savefig('corr.png')


df1, df2, df3, cluster_caste, cluster_rel = find_homophilies()
df = pd.DataFrame()
df = df.assign(mf_participation=df1)
df = df.assign(village_caste_homophily=df2)
df = df.assign(village_religion_homophily=df3)

correlation(df)
