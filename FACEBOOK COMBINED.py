#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and dataset

# In[8]:


# Importing dataset and libraries
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import community
import matplotlib.cm as cm
import seaborn as sns
from package import utilities as ut
from itertools import chain
import os


# In[4]:


# Importing the dataset
G = nx.read_edgelist('combined/facebook_combined.txt', create_using=nx.Graph(),nodetype= int)


# In[5]:


# Plotting the entire graph
pos = nx.spring_layout(G)
plt.figure(figsize=(20,10))
nx.draw_networkx(G, pos= pos,with_labels= False,node_size=10)


# In[6]:


#Label dictionary
labeldict = {}
labeldict[348] = "Node 348"
labeldict[414] = "Node 414"
labeldict[0] = "Node 0"
labeldict[107] = "Node 107"
labeldict[3980] = "Node 3980"
labeldict[3437] = "Node 3437"
labeldict[686] = "Node 686"
labeldict[1684] = "Node 1684"
labeldict[1912] = "Node 1912"
labeldict[698] = "Node 698"


# In[13]:


# Plot facebook network using betweenness as metric for the node size and communities indicated by the diverse colors
# Compute betweenness centrality
centrality = nx.betweenness_centrality(G, k = 10, endpoints = True)

# Compute communities
parts = community.best_partition(G)
values = [parts.get(node) for node in G.nodes()]

# Draw the graph
fig, ax = plt.subplots(figsize = (20, 15))
node_color = [values[n] for n in G]
node_size = [v * 20000 for v in centrality.values()]
nx.draw_networkx(
    G,
    pos=pos,
    with_labels=True,
    labels=labeldict,
    node_color=node_color,
    node_size=node_size,
    edge_color="gainsboro",
    alpha=0.4,
)

# Title/legend
font = {"color": "k", "fontweight": "bold", "fontsize": 20}
ax.set_title("Facebook Network", font)
# Change font color for legend
font["color"] = "black"

ax.text(
    0.80,
    0.10,
    "node color = community structure",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.text(
    0.80,
    0.06,
    "node size = betweeness centrality",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)

# Network communities
ego_nodes = set([int(name.split('.')[0]) for name in os.listdir("data/")])

for node in ego_nodes:
    print(node, "is in community number", parts.get(node))
n_sizes = [5]*len(G.nodes())
for node in ego_nodes:
    n_sizes[node] = 250


# Resize figure for label readibility
ax.margins(0.1, 0.05)
fig.tight_layout()
plt.axis("off")
#plt.show()
plt.savefig("Facebook network.png")


# # Network Statistics

# Statistics regarding structure of the network (is it connected?, is it bipartite?, etc).

# In[ ]:


# Summary of the graph
info = nx.info(G)
print(info)
# True if the graph is connected, false otherwise.
connceted = nx.is_connected(G)
print("Network is connected?",connceted)
# Returns True if graph G is bipartite, False if not.
bipartite = nx.is_bipartite(G)
print("Network is bipartite?",bipartite)
# Returns the value of the density
density = nx.density(G)
print("Network's density:",density)


# In[15]:


# Computation of average clustering
avg_clustering = nx.average_clustering(G)
print("Network Average Clustering: ", avg_clustering)
# Calculates average distance
avg_distance = nx.average_shortest_path_length(G)
print("Network Average distance: ", avg_distance)
# Calculates diameter
diameter = nx.diameter(G, e=None, usebounds=False)
print("Network Diameter: ", diameter)


# In[16]:


# Computes degree centrality and most influential nodes
centrality = nx.degree_centrality(G)
print(" ")
print("----------Degree centrality----------")
print(" ")
for w in sorted(centrality, key=centrality.get, reverse=True)[0:10]:
    print("Most influent node: ",w," Degree of: ", centrality[w])
# PageRank centrality and most influential nodes
centrality_page_rank = nx.pagerank(G)
print(" ")
print("----------PageRank centrality----------")
print(" ")
for w in sorted(centrality_page_rank, key=centrality_page_rank.get, reverse=True)[0:10]:
    print("Most influent node: ",w," Degree of: ", centrality_page_rank[w])


# # Random graph and Real Network

# Random graphs usually follow a normal distribution instead real networks have a power-law structure (connected to scale-free property). We use it to extract useful features from the real networks and study properties in controlled environment.

# In[17]:


# Generate a random graph with 4039 nodes and a probability od 0.01 per edges
G_random = nx.gnp_random_graph(4039, 0.01, seed=42)
# summary of random graph
info_random = nx.info(G_random)
print(info_random)


# ### Random Graph

# In[19]:


# Create a gridspec for adding subplots of different sizes
fig = plt.figure(figsize=(8, 8))
axgrid = fig.add_gridspec(5, 4)
# Plot the random graph with spring layout
ax0 = fig.add_subplot(axgrid[0:3, :])
pos = nx.spring_layout(G)
nx.draw_networkx(G_random, pos= pos,with_labels= False, node_size=10)
# Set title
ax0.set_title("Random Network Graph")
# Do not plot axis on graph
ax0.set_axis_off()
# Second plot for degree distribution
ax1 = fig.add_subplot(axgrid[3:, :])
# Create a list with the degree of random graph and sort 
degree_sequence = sorted((d for n, d in G_random.degree()), reverse=True)
# Returns the sorted unique elements of an array with its number of times each unique value comes up in the input array
frequency_degree = np.unique(degree_sequence, return_counts=True)
ax1.bar(*frequency_degree)
ax1.set_title("Random Network Degree histogram")
ax1.set_xlabel("# of Nodes")
ax1.set_ylabel("Number of neighbors")
fig.tight_layout()
#plt.show()
plt.savefig("Random Graph.png")


# ### Real Network

# In[20]:


# Create a gridspec for adding subplots of different sizes
fig = plt.figure(figsize=(8, 8))
axgrid = fig.add_gridspec(5, 4)
# Plot
ax0 = fig.add_subplot(axgrid[0:3, :])
nx.draw_networkx(G, pos= pos,with_labels= False, node_size=10)
# Set title
ax0.set_title("Real Network Graph")
# Do not plot axis on graph
ax0.set_axis_off()
# Second plot for degree distribution
ax1 = fig.add_subplot(axgrid[3:, :])
# Create a list with the degree of random graph and sort
degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
# Returns the sorted unique elements of an array with its number of times each unique value comes up in the input array
frequency_degree = np.unique(degree_sequence, return_counts=True)
ax1.bar(*frequency_degree)
ax1.set_title("Real Network Degree histogram")
ax1.set_xlabel("# of Nodes")
ax1.set_ylabel("Number of neighbors")
fig.tight_layout()
plt.show()
plt.savefig("Real FB Network.png")


# # Communities

# Compute the partition of the graph nodes which maximises the modularity using the Louvain heuristices (best_partition)

# In[22]:


# Compute the partition, the total number of communities is equal to 16
partition = community.best_partition(G)
values = [partition.get(node) for node in G.nodes()]
df_communities = pd.DataFrame(partition, index=[0]).T.reset_index()
df_communities.columns = ["nodes", "communities"]
print("How Many communities are there? ", len(set(values)))


# In[25]:


# Plot the communities by maximizing modularity
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
plt.figure(figsize=(20,10))
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.title("Communities computed with modularity maximization")
#plt.show()
plt.savefig("Communities.png")


# In[26]:


# Summary of nodes in each community created
df_communities_statistics = df_communities.groupby("communities").count().sort_values("nodes", ascending=False).reset_index()
sorted_communities = list(df_communities_statistics.communities)
df_communities_statistics["communities"] = ["Community " + str(df_communities_statistics["communities"][x]) for x in range(len(df_communities_statistics))]
df_communities_statistics


# In[28]:


# Define the plot
plt.figure(figsize = (15,8))
ax = sns.barplot(x = "communities", y = "nodes", data = df_communities_statistics, palette= "Blues_r")
ax.bar_label(ax.containers[0])
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
ax.set_title("Number of nodes per communities", fontsize = 20)


# # Analysis

# In[30]:


# Brief overview over all of the communities statistics
comm_info = []
comm_avg_distance = []
comm_diameter = []
comm_density = []
comm_influencer = []

for x in sorted_communities:
    nodes = list(df_communities[df_communities["communities"]==x]["nodes"])
    community = G.subgraph(nodes)
    info = nx.info(community)
    comm_info += [info]
    avg_distance = nx.average_shortest_path_length(community)
    comm_avg_distance += [avg_distance]
    diameter = nx.diameter(community, e=None, usebounds=False)
    comm_diameter += [diameter]
    influencer = sorted(nx.degree_centrality(community), key=centrality.get, reverse=True)[0]
    comm_influencer += [influencer]
    density = nx.density(community)
    comm_density += [density]
    plt.figure(figsize=(20,10))
    plt.title("Community: "+str(x)+", "+str(info) + " Diameter: " + str(diameter) + ", influencer node: " + str(influencer))
    nx.draw_networkx_nodes(community, pos, node_size=10)
    nx.draw_networkx_edges(community, pos, alpha=0.5)
    plt.show()

df_communities_statistics["comm_info"] = comm_info
df_communities_statistics["comm_avg_distance"] = comm_avg_distance
df_communities_statistics["comm_diameter"] = comm_diameter
df_communities_statistics["comm_density"] = comm_density
df_communities_statistics["comm_influencer"] = comm_influencer


# In[31]:


# Print all of the statistics
df_communities_statistics


# # Information virality in networks

# How does information spread in networks? Nowadays misinformation is one of the most important negative externalities of social networks. In order to clearly explain the reasoning method behind this research some assumptions must be made: 
# 1. since the network was unweighted it was assigned a weight to each edge obtained by the average between source node's one centrality and node's two one.
# 2. the information spread in this case will be defined as "secret".
# 3. it is assumed that at the beginning of the infection process or information flow, only two people know the secret.
# 4. the spread of information has three main characters in this analysis: two people that know the secret (red nodes in the graphs and one which will be the first person "infected" by the secret (yellow). All the remaining blue nodes will never know the secret

# # Information spreading in dense community

# In[41]:


# Take community 12 as example of information spreading process because it is the community with the highest density
nodes = list(df_communities[df_communities["communities"]==12]["nodes"])


# In[42]:


# Assign a weight to each edge obtained by the average between source node's one centrality and node's two one.
df = pd.read_csv('combined/facebook_combined.txt', sep=" ", header=None)
df.columns = ["node1", "node2"]
df = pd.read_csv('combined/facebook_combined.txt', sep=" ", header=None)
df.columns = ["node1", "node2"]
centrality = nx.degree_centrality(G)
centrality = pd.DataFrame(centrality, index=[0]).T.reset_index()
centrality.columns = ["node1","centrality"]
df_centrality = pd.merge(df, centrality, on="node1")
df_centrality.columns = ["node1","node2","node1_centrality"]
centrality.columns = ["node2","centrality"]
df_centrality = pd.merge(df_centrality, centrality, on="node2")
df_centrality.columns = ["node1","node2","node1_centrality","node2_centrality"]
df_centrality["weight"] = ((df_centrality["node1_centrality"]+df_centrality["node2_centrality"])/2)*10000
df_centrality = df_centrality.drop(columns=["node1_centrality","node2_centrality"])
df_centrality = df_centrality[df_centrality["node1"].isin(nodes)].reset_index(drop=True)
df_centrality.sort_values("weight", ascending=False)


# In[43]:


# Create a graph from the new dataset containing weight
G_inf = nx.from_pandas_edgelist(
    df_centrality,
    source="node1",
    target="node2",
    edge_attr="weight",
    create_using=nx.Graph())


# In[44]:


nx.info(G_inf)


# In[46]:


# High centrality nodes in 
infection_times = {583:-1,658:-1,578:0}
label = {583:"583",658:"658",578:"578"}

for t in range(5):
    ut.plot_G(G_inf,pos,infection_times,t, label)
    ut.infection_times = ut.independent_cascade(G_inf,t,infection_times)


# In[47]:


# High centrality nodes in 
infection_times = {628:-1,647:-1,599:0}
label = {628:"628",647:"647",599:"599"}

for t in range(5):
    ut.plot_G(G_inf,pos,infection_times,t, label)
    ut.infection_times = ut.independent_cascade(G_inf,t,infection_times)


# # Information spreading in low-density community

# In[48]:


# Take community 12 as example of information spreading process because it is the community with the highest density
nodes_ = list(df_communities[df_communities["communities"]==9]["nodes"])


# In[49]:


# Assign a weight to each edge obtained by the average between source node's one centrality and node's two one.
df = pd.read_csv('combined/facebook_combined.txt', sep=" ", header=None)
df.columns = ["node1", "node2"]
df = pd.read_csv('combined/facebook_combined.txt', sep=" ", header=None)
df.columns = ["node1", "node2"]
centrality = nx.degree_centrality(G)
centrality = pd.DataFrame(centrality, index=[0]).T.reset_index()
centrality.columns = ["node1","centrality"]
df_centrality = pd.merge(df, centrality, on="node1")
df_centrality.columns = ["node1","node2","node1_centrality"]
centrality.columns = ["node2","centrality"]
df_centrality = pd.merge(df_centrality, centrality, on="node2")
df_centrality.columns = ["node1","node2","node1_centrality","node2_centrality"]
df_centrality["weight"] = ((df_centrality["node1_centrality"]+df_centrality["node2_centrality"])/2)*10000
df_centrality = df_centrality.drop(columns=["node1_centrality","node2_centrality"])
df_centrality = df_centrality[df_centrality["node1"].isin(nodes_)].reset_index(drop=True)
df_centrality.sort_values("weight", ascending=False)


# In[50]:


# Create a graph from the new dataset containing weight
G_new = nx.from_pandas_edgelist(
    df_centrality,
    source="node1",
    target="node2",
    edge_attr="weight",
    create_using=nx.Graph())


# In[51]:


nx.info(G_new)


# In[ ]:


# High centrality nodes in 
infection_times = {3437:-1,3596:-1,3604:0}
label = {3437:"3437",3596:"3596",3604:"3604"}

for t in range(5):
    ut.plot_G(G_new,pos,infection_times,t, label)
    ut.infection_times = ut.independent_cascade(G_new,t,infection_times)


# In[ ]:




