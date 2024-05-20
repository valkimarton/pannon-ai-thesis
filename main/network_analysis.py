import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import networkx as nx
from networkx.algorithms import community

from random_forest import read_inputs
import feature_extractor

NETWORK_ANALYSIS_DATA_PATH = './network_analysis_data/'
PLOT_SEED = 1

# list of color codes for plotting with 17 colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#000000']


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    df_std = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_std

# function to define distance metric: Eucledian distance
def distance(x, y):
    return np.linalg.norm(x - y)

def KNN_graph(df: pd.DataFrame, k: int):
    # create an empty graph
    graph = {}
    for i in range(len(df)):
        graph[i] = []
    
    # calculate distance between all points
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            dist = distance(df.iloc[i], df.iloc[j])
            graph[i].append((j, dist))
            graph[j].append((i, dist))
    
    # sort the distances
    for i in range(len(df)):
        graph[i].sort(key=lambda x: x[1])
    
    # create the KNN graph
    knn_graph = {}
    for i in range(len(df)):
        knn_graph[i] = []
        for j in range(k):
            knn_graph[i].append(graph[i][j][0])
    
    return knn_graph

def get_number_of_clusters(graph: dict):
    # create a dictionary to store the clusters
    clusters = {}
    for i in graph:
        clusters[i] = i
    
    # merge the clusters
    for i in graph:
        for j in graph[i]:
            if clusters[i] != clusters[j]:
                for k in clusters:
                    if clusters[k] == clusters[j]:
                        clusters[k] = clusters[i]
    
    # count the number of clusters
    num_clusters = len(set(clusters.values()))
    return num_clusters

def get_clusters(graph: dict):
    # create a dictionary to store the clusters
    clusters = {}
    for i in graph:
        clusters[i] = i
    
    # merge the clusters
    for i in graph:
        for j in graph[i]:
            if clusters[i] != clusters[j]:
                for k in clusters:
                    if clusters[k] == clusters[j]:
                        clusters[k] = clusters[i]
    
    return clusters

def plot_graph(graph: nx.Graph):
    pos = nx.fruchterman_reingold_layout(graph, seed=PLOT_SEED)
    plt.figure(figsize=(16,16))
    plt.title('KNN graph')
    plt.axis("off")
    nx.draw_networkx_nodes(graph, pos, node_size=100, node_color="black")
    nx.draw_networkx_edges(graph, pos, alpha=0.500)
    #nx.draw_networkx_labels(graph, pos, font_color="white", font_size=8)
    plt.show()

def save_graph(graph: dict, filename: str):
    folder = NETWORK_ANALYSIS_DATA_PATH
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = folder + filename
    with open(file, 'w') as f:
        for i in graph:
            f.write(str(i) + ':')
            for j in graph[i]:
                f.write(str(j) + ',')
            f.write('\n')

def create_KNN_graph_data_for_biax_tension(k: int = 5):
    print('Network analysis')
    # read extracted features
    df_biax_tension = read_inputs('./extracted_features_data/biax_tension_features.csv')
    df_biax_tension_X = df_biax_tension[[
        feature_extractor.TOTAL_CURVATURE, 
        feature_extractor.CURVATURE_RATIO,
        feature_extractor.FINAL_STRAIN,
        feature_extractor.FINAL_STRESS
        ]]
    df_biax_tension_X_std = standardize(df_biax_tension_X)
    KNN_graph = KNN_graph(df_biax_tension_X_std, 5)
    save_graph(KNN_graph, 'biax_tension_graph.csv')

def read_KNN_graph(file_path) -> nx.Graph:
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split(':')
            node = int(line[0])
            neighbors = line[1].split(',')
            neighbors = [int(i) for i in neighbors if i != '']
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
    return G

def get_classes():
    class1 = list(range(0, 100))
    class2 = list(range(100, 200))
    class3 = list(range(200, 400))
    return {
        0: class1,
        1: class2,
        2: class3
    }

def plot_graph_by_communities(graph: nx.Graph, communities: list):
    pos = nx.fruchterman_reingold_layout(graph, seed=PLOT_SEED)
    plt.figure(figsize=(16,16))
    plt.title('Detected communities')
    plt.axis("off")
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(graph, pos, nodelist=community, node_size=100, node_color=colors[i])
    nx.draw_networkx_edges(graph, pos, alpha=0.500)
    #nx.draw_networkx_labels(graph, pos, font_color="white", font_size=8)
    plt.show()

def plot_graph_by_classes(graph: nx.Graph, classes: dict[int, list[int]]):
    pos = nx.fruchterman_reingold_layout(graph, seed=PLOT_SEED)
    plt.figure(figsize=(16,16))
    plt.title('Actual classes')
    plt.axis("off")
    for i, class_ in classes.items():
        nx.draw_networkx_nodes(graph, pos, nodelist=class_, node_size=100, node_color=colors[i])
    nx.draw_networkx_edges(graph, pos, alpha=0.500)
    #nx.draw_networkx_labels(graph, pos, font_color="white", font_size=8)
    plt.show()

def get_community_class_nodes(community_classes: dict[int, list[int]], communities: list):
    community_class_nodes = {
        0: [],
        1: [],
        2: []
    }
    for class_, community_indices in community_classes.items():
        for community_index in community_indices:
            community = communities[community_index]
            for node in community:
                community_class_nodes[class_].append(node)
    return community_class_nodes

def get_community_classes(communities: list, classes: dict[int, list[int]]):
    community_classes = {
        0: [],
        1: [],
        2: []
    }
    for i, community in enumerate(communities):
        class_counts = {
            0: 0,
            1: 0,
            2: 0
        }
        for node in community:
            for class_, class_nodes in classes.items():
                if node in class_nodes:
                    class_counts[class_] += 1
        max_class = max(class_counts, key=class_counts.get)
        community_classes[max_class].append(i)

    print(community_classes)
    return community_classes

def plot_graph_by_community_classes(graph: nx.Graph, community_class_nodes: dict[int, list[int]]):
    pos = nx.fruchterman_reingold_layout(graph, seed=PLOT_SEED)
    plt.figure(figsize=(16,16))
    plt.title('Classes based on communities')
    plt.axis("off")
    for i, class_nodes in community_class_nodes.items():
        nx.draw_networkx_nodes(graph, pos, nodelist=class_nodes, node_size=100, node_color=colors[i])
    nx.draw_networkx_edges(graph, pos, alpha=0.500)
    #nx.draw_networkx_labels(graph, pos, font_color="white", font_size=8)
    plt.show()

def get_class_for_node(node: int, classes: dict[int, list[int]]):
    for class_, class_nodes in classes.items():
        if node in class_nodes:
            return class_
    raise Exception('Node not found in any class')

def evaluate_detected_classes(community_class_nodes: dict[int, list[int]], classes: dict[int, list[int]]):
    y = []
    y_pred = []
    for node in range(400):
        y.append(get_class_for_node(node, classes))
        y_pred.append(get_class_for_node(node, community_class_nodes))
    
    # confusion matrix for all nodes (actual from classes, predicted from communities)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='g'
                
                )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for classes by community detection')
    # display numerical values in real number format


    plt.show()

    # print accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy: {accuracy}')

# TASK:
# Read extracted features
# define distance metric
# KNN gráffal gráf a mintákból -> megnézni hogy kialakul-e a három klaszter
if __name__ == '__main__':
    #create_KNN_graph_data_for_biax_tension(5)

    KNN_graph_biax_tension: nx.Graph = read_KNN_graph(NETWORK_ANALYSIS_DATA_PATH + 'biax_tension_graph.csv')
    plot_graph(KNN_graph_biax_tension)

    # community detection
    communities = community.louvain_communities(KNN_graph_biax_tension, seed=3)
    plot_graph_by_communities(KNN_graph_biax_tension, communities)

    classes = get_classes()
    plot_graph_by_classes(KNN_graph_biax_tension, classes)

    # assign each community to a class: assign community to the class from wich it has the most nodes
    community_classes: dict[int, list[int]] = get_community_classes(communities, classes)
    community_class_nodes: dict[int, list[int]] = get_community_class_nodes(community_classes, communities)
    plot_graph_by_community_classes(KNN_graph_biax_tension, community_class_nodes)

    evaluate_detected_classes(community_class_nodes, classes)
    
    print('done')

