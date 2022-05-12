# Facebok_network_analysis
Data Visualization Project Work
Martina Crisafulli 
Enrico Grandi 
Lorenzo Antolini 

Ego Facebook Network Analysis


1 Data Exploration and Network Basic Analysis

The dataset was taken from SNAP (Stanford Network Analysis Project) and regards Facebook network.
The data used in this analysis are two: facebook combined.txt.gz describing all the edges of the 10
nodes in the and facebook.tar.gz which provides additional features for each of the 10 nodes selected.
Therefore, for each of the ten nodes, labeled as node 0, 107, 348, 414, 686, 698, 1684, 1912, 3437 and
3980 we have 5 additional features:

1. Circles: describes the relationships between features of the vertex v’s friends (circles). Each line
contains one circle, consisting of a series of node ids
2. Edges: edges for the network of vertex v
3. Featnames: all the anonymized features’ names used in the following data set ”egofeat” and
”feat”
4. Egofeat: features belonging to the ego node (0 stands for ”feature does not belong to the node”,
1 instead ”feature belongs to the node”
5. Feat: same format as the egofeat data set but in this case it describes the features of the nodes
friends of vertex v.
