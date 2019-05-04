from collections import Counter
import pandas as pd
import numpy as np
# load networkx graph objects G1 and G2
import networkx as nx


# the observed homophily in our network
# measure of homophily is the proportion of edges whose nodes share a characteristic.
def homophily(G, chars, IDs):
    """
    Given a network G, a dict of characteristics chars for node IDs,
    and dict of node IDs for each node in the network,
    find the homophily of the network.
    """
    num_same_ties, num_ties = 0, 0
    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 > n2:  # do not double-count edges!
                if IDs[n1] in chars.keys() and IDs[n2] in chars.keys():
                    if G.has_edge(n1, n2):
                        num_ties += 1  # Should `num_ties` be incremented?  What about `num_same_ties`?
                        if chars[IDs[n1]] == chars[IDs[n2]]:
                            num_same_ties += 1  # Should `num_ties` be incremented?  What about `num_same_ties`?
    return (num_same_ties / num_ties)


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


data_filepath = '../data/'
df = pd.read_stata(data_filepath + "demographics/individual_characteristics.dta")

# Store separate datasets for individuals belonging to Villages 1 and 2 as df1 and df2, respectively.
df1 = df[df.village == 1]
df2 = df[df.village == 2]

# display the first few entries of df1.
print(df1.head())

# n this dataset, each individual has a personal ID, or PID,
# stored in key_vilno_1.csv and key_vilno_2.csv for villages 1 and 2, respectively.
pid1 = pd.read_csv(data_filepath + "network_data/adjacency_matrix_keys/key_vilno_1.csv", dtype=int, header=None)
pid2 = pd.read_csv(data_filepath + "network_data/adjacency_matrix_keys/key_vilno_2.csv", dtype=int, header=None)

# Define Python dictionaries with personal IDs as keys
# and sex, caste, and religion covariates, for Villages 1 and 2 as values
sex1 = dict(zip(df1.pid, df1.resp_gend))
caste1 = dict(zip(df1.pid, df1.caste))
religion1 = dict(zip(df1.pid, df1.religion))

# Continue for df2 as well.
sex2 = dict(zip(df2.pid, df2.resp_gend))
caste2 = dict(zip(df2.pid, df2.caste))
religion2 = dict(zip(df2.pid, df2.religion))

favorite_colors = {
    "ankit": "red",
    "xiaoyu": "blue",
    "mary": "blue"
}

color_homophily = chance_homophily(favorite_colors)
print(color_homophily)

print("Village 1 chance of same sex:", chance_homophily(sex1))
print("Village 1 chance of same caste:", chance_homophily(caste1))
print("Village 1 chance of same religion:", chance_homophily(religion1))
print()
print("Village 2 chance of same sex:", chance_homophily(sex2))
print("Village 2 chance of same caste:", chance_homophily(caste2))
print("Village 2 chance of same religion:", chance_homophily(religion2))

A1 = np.loadtxt(data_filepath + "network_data/adjacency_matrices/adj_allVillageRelationships_vilno_1.csv", delimiter=",")
A2 = np.loadtxt(data_filepath + "network_data/adjacency_matrices/adj_allVillageRelationships_vilno_2.csv", delimiter=",")

G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

# convert dataframe to numpy array
array_pid1 = np.array(pid1[0])
array_pid2 = np.array(pid2[0])

print("Village 1 observed proportion of same sex:", homophily(G1, sex1, array_pid1))
print("Village 1 observed proportion of same caste:", homophily(G1, caste1, array_pid1))
print("Village 1 observed proportion of same religion:", homophily(G1, religion1, array_pid1))
print()
print("Village 2 observed proportion of same sex:", homophily(G2, sex2, array_pid2))
print("Village 2 observed proportion of same caste:", homophily(G2, caste2, array_pid2))
print("Village 2 observed proportion of same religion:", homophily(G2, religion2, array_pid2))
