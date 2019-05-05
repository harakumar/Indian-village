import numpy as np
import pandas as pd
import random
import networkx as nx
from networkx.algorithms import approximation
import matplotlib.pyplot as plt
import os




def show_graph_with_labels(adjacency_matrix,vilnumber):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    #labels = {k:k for k in power_iteration(adjacency_matrix,1000)}
    gr = nx.Graph()
    gr.clear()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=5, with_labels=True)
    #plt.show()
    plt.savefig('./graphs/vil_'+str(vilnumber)+'.png')
    plt.close()

def show_correlation(corrmat,mat,vilnumber):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corrmat,cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,len(mat.columns),1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(mat.columns)
        ax.set_yticklabels(mat.columns)
        #plt.show()
        plt.savefig('./graphs/vil_'+str(vilnumber)+'_corr.png')
        plt.close()



hhcharac = pd.read_csv("household_characteristics.csv")
vilchar = pd.DataFrame(columns=['village','aveClustCoeff','density','MFpartrate'])

directory = os.path.join("./adjmat") #adjacency matrices directory
for root,dirs,files in os.walk(directory):
    for filename in files:
        if filename.endswith(('.csv')) and filename.startswith(('adj_allVillageRelationships_HH_vilno_')):
                curVillage = int(filename[37:-4])
                mat = pd.read_csv(os.path.join(directory,filename), header=None)
                show_graph_with_labels(mat.values,curVillage)

                FG = nx.from_numpy_matrix(mat.values)
                try: 
                        eigveccentNWX = pd.DataFrame.from_dict(nx.algorithms.eigenvector_centrality(FG), orient='index').squeeze() #convert dictionary to pandas dataframe, maybe .squeeze() into a Series?
                except:
                        print("Could not converge with power iteration for the Eigenvector Centrality for village: " + str(curVillage))
                        eigveccentNWX = pd.DataFrame.from_dict(nx.eigenvector_centrality_numpy(FG), orient='index').squeeze() #https://stackoverflow.com/questions/43208737/using-networkx-to-calculate-eigenvector-centrality
                degcentNWX = pd.DataFrame.from_dict(nx.algorithms.degree_centrality(FG), orient='index').squeeze() #convert dictionary to pandas dataframe, maybe .squeeze() into a Series?
                closecentNWX = pd.DataFrame.from_dict(nx.algorithms.closeness_centrality(FG), orient='index').squeeze() #convert dictionary to pandas dataframe, maybe .squeeze() into a Series?
                betcentNWX = pd.DataFrame.from_dict(nx.algorithms.betweenness_centrality(FG), orient='index').squeeze() #convert dictionary to pandas dataframe, maybe .squeeze() into a Series?

                indexVil = hhcharac[ hhcharac['village'] == curVillage ].index

                tmphhcharac = hhcharac.iloc[indexVil]
                tmphhcharac = tmphhcharac.assign(degcent=degcentNWX.values)
                tmphhcharac = tmphhcharac.assign(eigcent=eigveccentNWX.values)
                tmphhcharac = tmphhcharac.assign(closecent=closecentNWX.values)
                tmphhcharac = tmphhcharac.assign(betcent=betcentNWX.values)
                try:
                        microfinvil = pd.read_csv(os.path.join("./microfinance/","MF"+str(curVillage)+".csv"), header=None)
                        tmphhcharac = tmphhcharac.assign(microfinup=microfinvil.values)
                        partrate = float(microfinvil.sum(axis=0))/len(microfinvil.index) #microfinance participation rate
                except:
                        print("Microfinancing uptake information not found for village: "+str(curVillage))
                        partrate = 0

                tmphhcharac.to_csv("./aggregation/aggvil"+str(curVillage)+".csv")

                vilcorrel = tmphhcharac.corr(method='pearson')
                vilcorrel.to_csv("./correlations/corvil"+str(curVillage)+".csv")

                show_correlation(vilcorrel,vilcorrel,curVillage)

                aveClustCoeff = nx.algorithms.approximation.clustering_coefficient.average_clustering(FG)
                density = nx.density(FG)

                vilchar = vilchar.append({'village':curVillage,'aveClustCoeff':aveClustCoeff,'density':density,'MFpartrate':partrate}, ignore_index=True)

vilchar.to_csv("./villagesagg.csv")
vilcorrelgen = vilchar.corr(method='pearson')
vilcorrelgen.to_csv("./corvilgen.csv")



directory = os.path.join("./correlations") #correlation matrices directory
for root,dirs,files in os.walk(directory):
    i = 0
    for filename in files:
        if filename.endswith(('.csv')) and filename.startswith(('corvil')):
                mat = pd.read_csv(os.path.join(directory,filename))
                if i==0:
                        mat2 = mat.copy()
                mat2.add(mat)
                i += 1

vilcorrelgensum = mat2 #/(i+1)
vilcorrelgensum.to_csv("./corvilgengen.csv")

