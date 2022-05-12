import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import community
import matplotlib.cm as cm
import seaborn as sns
from itertools import chain
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo

edge_style_def = dict(color = 'gray', width = 0.5)  # define the default style for edges
node_style_def = dict(symbol = 'circle', size = 5, opacity = 0.9, 
                      color = 'blue', line = dict(color = 'rgb(0,0,0)', width = 0.5))   # define the default style for nodes

def set_layout(G, ndim = 2, kamada = True):    
    print('Calculating coordinates')
    layout = nx.kamada_kawai_layout(G, dim = ndim) if kamada else nx.spring_layout(G, dim = ndim) # define the node layout with NetworkX
    print('End')
    edgelist = list(G.edges())
    
    # Extract coordinates
    ax_nodes = pd.DataFrame(layout).transpose() 
    ax_nodes.columns = ['X', 'Y'] if ndim == 2 else ['X', 'Y', 'Z'] # 2D or 3D?
    
    
    # define layout for edges
    if ndim == 2:
        layout_e = dict(map(lambda e: 
                (e, # edge key
                ([layout[e[0]][0], layout[e[1]][0], None],  # X
                [layout[e[0]][1], layout[e[1]][1], None])  # Y
                ), edgelist))
    else:
        layout_e = dict(map(lambda e: 
            (e, # edge key
            ([layout[e[0]][0], layout[e[1]][0], None], # X
            [layout[e[0]][1], layout[e[1]][1], None],  # Y
            [layout[e[0]][2], layout[e[1]][2], None])  # Z
            ), edgelist))
    
    
    # extract coordinates
    ax_edges = pd.DataFrame(layout_e).transpose()
    ax_edges.columns = ['X', 'Y'] if ndim == 2 else ['X', 'Y', 'Z']

    return ax_nodes, ax_edges

def set_trace(X, Y, kind, style, name, Z = None, text = None): #draw nodes or edges
    mode = 'markers' if kind == 'node' else 'lines'
    hoverinfo = 'text' if kind == 'node' else 'none'
    if Z is None:
        if kind == 'node':
            return go.Scatter(x = X, y = Y, mode = mode, name = name, marker = style, text = text, hoverinfo = hoverinfo)  # draw nodes (i.e., markers) in 2D
        else:
            return go.Scatter(x = X, y = Y, mode = mode, name = name, line = style, text = text, hoverinfo = hoverinfo)  # draw edges (i.e., lines) in 2D
    else: 
        if kind == 'node':
            return go.Scatter3d(x = X, y = Y, z = Z, mode = mode, marker = style,  text = text, name = name, hoverinfo = hoverinfo)  # draw nodes (i.e., markers) in 3D
        else:
            return go.Scatter3d(x = X, y = Y, z = Z, mode = mode, line = style,  text = text, name = name, hoverinfo = hoverinfo)  # draw edges (i.e., lines) in 3D

    
def set_traces(nodes_ax: pd.DataFrame, edges_ax: pd.DataFrame, communities = None):  # Function to set up the main figure with nodes, edges, and communities
    traces = []
    Xe = list(chain(*edges_ax.X.tolist())) 
    Ye = list(chain(*edges_ax.Y.tolist()))
    Ze = list(chain(*edges_ax.Z.tolist())) if 'Z' in edges_ax.columns else None

    traces.append(set_trace(Xe, Ye, Z = Ze, kind = 'edge', style = edge_style_def, name = 'Link'))
    
    if communities is None:
        traces.append(set_trace(nodes_ax.X.tolist(), nodes_ax.Y.tolist(), 
                                Z = nodes_ax.Z.tolist() if 'Z' in nodes_ax.columns else None, 
                                kind = 'node', style = node_style_def, 
                                name = 'Node', text = nodes_ax.index))
    else:
        colors = list(sns.color_palette(n_colors =len(communities)).as_hex())
        for idcom, community in enumerate(communities):
            node_style = node_style_def.copy()
            node_style['color'] = colors[idcom]
            temp = nodes_ax[nodes_ax.index.isin(community)]
            traces.append(set_trace(temp.X.tolist(), temp.Y.tolist(), 
                                Z = temp.Z.tolist() if 'Z' in temp.columns else None, 
                                kind = 'node', style = node_style, 
                                name = 'Class '+str(idcom), text = temp.index))
            
                                       
    return traces

def draw_plotly_network(net, ndim = 2, communities = None, kamada = True):  # Main function to draw a network
    nodes_ax, edges_ax = set_layout(net, ndim = ndim, kamada = kamada)  # the kamada parameter sets the Networkx layout to use. The ndim parameter sets the dimensions the layout (i.e., 2D or 3D)
    traces = set_traces(nodes_ax, edges_ax, communities)  # the kamada parameter set the Networkx layout to use.
    pyo.iplot(traces, filename = 'basic-line')
    
def plot_features(df, node):
    #define the plot
    plt.figure(figsize=(20,10))
    df = df.sort_values("sum_values",ascending=False)
    ax = sns.barplot(x="feature",y="sum_values", data=df, ci=None)
    #attach the value of each bar
    ax.bar_label(ax.containers[0])
    #rotate the label on X axis in order to allow to read each value
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    #set the title of the plot
    ax.set_title("Sum of the top 10 feature per community: "+str(node), fontsize = 20)
    
def communities_statistics(df_communities,sorted_communities,feat,G):
    comm_info = []
    comm_avg_distance = []
    comm_diameter = []
    comm_influencer = []
    pos = nx.spring_layout(G)
    for x in sorted_communities:
        nodes = list(df_communities[df_communities["communities"]==0]["nodes"])
        nodes = list(df_communities[df_communities["communities"]==x]["nodes"])
        fet_com = feat[feat["node"].isin(nodes)].sum().to_frame().sort_values(0, ascending=False).reset_index()[1:10]
        fet_com.columns = ["feature","sum_values"]
        plot_features(fet_com, x)
        community = G.subgraph(nodes)
        info = nx.info(community)
        comm_info += [info]
        avg_distance = nx.average_shortest_path_length(community)
        comm_avg_distance += [avg_distance]
        diameter = nx.diameter(community, e=None, usebounds=False)
        comm_diameter += [diameter]
        centrality = nx.degree_centrality(community)
        influencer = sorted(nx.degree_centrality(community), key=centrality.get, reverse=True)[0]
        comm_influencer += [influencer]
        plt.figure(figsize=(20,10))
        plt.title("Community: "+str(x)+", "+str(info) + " Diameter: " + str(diameter) + ", influencer node: " + str(influencer))
        nx.draw_networkx_nodes(community, pos, node_size=10)
        nx.draw_networkx_edges(community, pos, alpha=0.5)
        plt.show()
        
    df_communities_statistics = df_communities.groupby("communities").count().sort_values("nodes", ascending=False).reset_index()
    df_communities_statistics["comm_info"] = comm_info
    df_communities_statistics["comm_avg_distance"] = comm_avg_distance
    df_communities_statistics["comm_diameter"] = comm_diameter
    df_communities_statistics["comm_influencer"] = comm_influencer
    return df_communities_statistics


def independent_cascade(G,t,infection_times):
    #doing a t->t+1 step of independent_cascade simulation
    #each infectious node infects neigbors with probabilty proportional to the weight
    max_weight = max([e[2]['weight'] for e in G.edges(data=True)])
    current_infectious = [n for n in infection_times if infection_times[n]==t]
    for n in current_infectious:
        for v in G.neighbors(n):
            if v not in infection_times:
                if  G.get_edge_data(n,v)['weight'] >= np.random.random()*max_weight:
                    infection_times[v] = t+1
    return infection_times
    
def plot_G(G,pos,infection_times,t,label):
    current_infectious = [n for n in infection_times if infection_times[n]==t]
    plt.figure()
    plt.axis('off')
    plt.title('Spread of information in community, t={}'.format(t),fontsize = 24)
    plt.figure(figsize=(20,10))
    weighted_degrees = dict(nx.degree(G,weight='weight'))
    for node in G.nodes():
        size = 100*weighted_degrees[node]**0.5
        if node in current_infectious:
            ns = nx.draw_networkx_nodes(G,pos,nodelist=[node], node_size=size, node_color='yellow')#don't know, yet
        elif infection_times.get(node,9999999)<t:
            ns = nx.draw_networkx_nodes(G,pos,nodelist=[node], node_size=size, node_color='red')#you know
        else:
            ns = nx.draw_networkx_nodes(G,pos,nodelist=[node], node_size=size, node_color='#009fe3')#never know
        ns.set_edgecolor('#f2f6fa')
    
    nx.draw_networkx_labels(G,pos,label,font_size=10);

    for e in G.edges(data=True):
        if e[2]['weight']>10:
            nx.draw_networkx_edges(G,pos,[e],width=e[2]['weight']/100,edge_color='#707070')
    