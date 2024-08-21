import random
import numpy as np
import time
import matplotlib.pyplot as plt
from stanfordcorenlp import StanfordCoreNLP
import json

class Trie(object):
   def __init__(self):
      self.child = {}

   def insert(self, entry):
      current = self.child
      for l in entry:
         if l not in current:
            current[l] = {}
         current = current[l]
      current['#'] = 1

   def search(self, entry):
      current = self.child
      for l in entry:
         if l not in current:
            return False
         current = current[l]
      return '#' in current


class Edge:
    def __init__(self, first, second, weight):
        self.first = first
        self.second = second
        self.weight = weight


class Graph:
    def __init__(self, nodesNumber, edgeList):
        self.nodesNumber = nodesNumber
        self.edges = [[] for _ in range(nodesNumber)]
        self.reversedEdges = [[] for _ in range(nodesNumber)]
        for edge in edgeList:
            self.edges[edge.first].append((edge.second, edge.weight))
            self.reversedEdges[edge.second].append((edge.first, edge.weight))

    def get_edges_list(self):
        edges_list = []
        for node in range(0, self.nodesNumber):
            for edge in self.edges[node]:
                edges_list.append(Edge(node, edge[0], edge[1]))
        return edges_list

    def is_same_graph(self, otherGraph):
        if self.nodesNumber != otherGraph.nodesNumber:
            return False
        for i in range(0, self.nodesNumber):
            if len(self.edges[i]) != len(otherGraph.edges[i]):
                return False
            edge_no = len(self.edges[i])
            for j in range(edge_no):
                if self.edges[i][j][0] != otherGraph.edges[i][j][0]:
                    return False
        return True

    def get_ancestors_list(self):
        ancestors = [-1]
        for i in range(1, self.nodesNumber):
            ancestors.append(self.reversedEdges[i][0][0])
        return ancestors


def generate_random_graph(nodesNumber):
    # 0 is the root
    nodesNumber = 101
    edge_list = []
    for i in range(0, nodesNumber):
        for j in range(1, nodesNumber):
            if i != j:
                random_int = random.randint(0, 1)
                # we only add an edge with 50% probability, i.e. when we sample 1 instead of 0
                if random_int == 1:
                    weight = np.random.normal(3, 1)
                    edge_list.append(Edge(i, j, weight))
    return Graph(nodesNumber, edge_list)


def sample_random_parent_node(graph, node, no_root_sampling):
    incoming_edges = graph.reversedEdges[node]
    total_weight = 0.0
    for incoming_edge in incoming_edges:
        if (not no_root_sampling) or incoming_edge[0] != 0:
            total_weight += incoming_edge[1]
    random_weight = random.uniform(0, total_weight)
    for incoming_edge in incoming_edges:
        if (not no_root_sampling) or incoming_edge[0] != 0:
            random_weight -= incoming_edge[1]
            if random_weight < 0:
                return incoming_edge[0]


def sample_spanning_tree(graph, unique_root_edge=False):
    edge_list = []
    visited = [False] * graph.nodesNumber
    parent = [-1] * graph.nodesNumber
    visited[0] = True
    selected_root_edge = False
    for i in range(1, graph.nodesNumber):
        if not visited[i]:
            current_node = i
            while not visited[current_node]:
                sampled_parent = sample_random_parent_node(graph, current_node, unique_root_edge and selected_root_edge)
                parent[current_node] = sampled_parent
                current_node = sampled_parent
            current_node = i
            while not visited[current_node]:
                edge_list.append(Edge(parent[current_node], current_node, 0))
                visited[current_node] = True
                current_node = parent[current_node]
            if not selected_root_edge:
                selected_root_edge = True

    return Graph(graph.nodesNumber, edge_list)


def is_dependency_tree(tree):
    return len(tree.edges[0]) == 1


def wilson_reject(graph):
    while True:
        spanning_tree = sample_spanning_tree(graph)
        if is_dependency_tree(spanning_tree):
            return spanning_tree


def dfs(graph, node, isNodeVisited, stack):
    for edge in graph.edges[node]:
        if not isNodeVisited[edge[0]]:
            isNodeVisited[edge[0]] = True
            dfs(graph, edge[0], isNodeVisited, stack)
    stack.append(node)


def get_current_scc(graph, node, isNodeVisited):
    scc_nodes = [node]
    for edge in graph.reversedEdges[node]:
        if not isNodeVisited[edge[0]]:
            isNodeVisited[edge[0]] = True
            scc_nodes.extend(get_current_scc(graph, edge[0], isNodeVisited))
    return scc_nodes


def build_scc_graph(graph, scc_list):
    scc_edges = []
    corresponding_scc_for_node = [0] * graph.nodesNumber
    scc_nr = len(scc_list)
    scc_edge_added = [[False] * scc_nr] * scc_nr
    for current_scc in range(0, scc_nr):
        for node in scc_list[current_scc]:
            corresponding_scc_for_node[node] = current_scc
    for node in range(0, graph.nodesNumber):
        parent_scc = corresponding_scc_for_node[node]
        for edge in graph.edges[node]:
            descendant_scc = corresponding_scc_for_node[edge[0]]
            if not scc_edge_added[parent_scc][descendant_scc]:
                scc_edge_added[parent_scc][descendant_scc] = True
                scc_edges.append(Edge(parent_scc, descendant_scc, 1))
    return Graph(scc_nr, scc_edges)


def get_all_sccs(graph):
    is_node_visited = [False] * graph.nodesNumber
    stack = []
    for node in range(0, graph.nodesNumber):
        if not is_node_visited[node]:
            is_node_visited[node] = True
            dfs(graph, node, is_node_visited, stack)
    reversed_stack = reversed(stack)
    is_node_visited = [False] * graph.nodesNumber
    scc_list = []
    for node in reversed_stack:
        if not is_node_visited[node]:
            is_node_visited[node] = True
            scc_list.append(get_current_scc(graph, node, is_node_visited))

    scc_graph = build_scc_graph(graph, scc_list)
    return scc_graph, scc_list


def find_potential_root_descendants(graph):
    scc_graph, scc_list = get_all_sccs(graph)
    scc_nr = scc_graph.nodesNumber
    valid_descendant_scc = None
    for curr_scc in range(0, scc_nr):
        if len(scc_graph.reversedEdges[curr_scc]) == 1:
            if scc_graph.reversedEdges[curr_scc][0][0] == 0:
                """  if we already found a descendant scc with a single incoming edge from the root before, 
                at this point there are at least 2. This means there is no valid root descendant as the 2 sccs 
                can't reach each other.
                """
                if valid_descendant_scc:
                    return []
                valid_descendant_scc = curr_scc
    if not valid_descendant_scc:
        return []
    return scc_list[valid_descendant_scc]


def remove_invalid_edges_from_graph(graph):
    potential_root_descendants = find_potential_root_descendants(graph)
    is_potential_root_descendant = [False] * graph.nodesNumber
    for descendant in potential_root_descendants:
        is_potential_root_descendant[descendant] = True
    root_edge_list = graph.edges[0]
    valid_edge_list = []
    for edge in root_edge_list:
        if is_potential_root_descendant[edge[0]]:
            valid_edge_list.append(edge)
    graph.edges[0] = valid_edge_list
    return graph


def wilson_scc(graph):
    valid_edges_graph = remove_invalid_edges_from_graph(graph)
    sampled_tree = sample_spanning_tree(valid_edges_graph, unique_root_edge=True)
    return sampled_tree


def random_weights_training():
    graph_set1 = []
    graph_set2 = []
    for i in range(100):
        graph_set1.append(generate_random_graph(100))
        graph_set2.append(generate_random_graph(100))

    start_time = time.time()
    times_elapsed_wr = []
    for i in range(100):
        currentGraph = graph_set2[i]
        sampled_tree = wilson_reject(currentGraph)
        current_time = time.time()
        times_elapsed_wr.append(current_time - start_time)

    start_time = time.time()
    times_elapsed_wre = []
    for i in range(100):
        currentGraph = graph_set1[i]
        sampled_tree = wilson_scc(currentGraph)
        current_time = time.time()
        times_elapsed_wre.append(current_time - start_time)

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    print(times_elapsed_wre)
    print(times_elapsed_wr)
    plt.xlabel("samples")
    plt.ylabel("seconds")
    plt.plot(times_elapsed_wre, label="Wilson SCC")
    plt.plot(times_elapsed_wr, label="Wilson Reject")
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.show()


def trained_weights_training():
    nlp = StanfordCoreNLP('http://localhost', port=9000)
    # Configure the server to output probabilities
    props = {
        'annotators': 'depparse',
        'outputFormat': 'json',
        'depparse.probabilistic': 'true'  # This enables the probabilistic parsing
    }

    # Your sentence
    sentence = "John loves Mary."
    # Parse the sentence
    result = nlp.annotate(sentence, properties=props)
    result = json.loads(result)
    # Access the probabilities for each dependency tree
    for sentence in result['sentences']:
        for edge in sentence['enhancedPlusPlusDependencies']:
            gov_index = edge['governor']
            dep_index = edge['dependent']
            dep_label = edge['dep']
            probability = edge['prob']
            print(
                f"{sentence['tokens'][gov_index - 1]['word']} --{dep_label}--> {sentence['tokens'][dep_index - 1]['word']} (Probability: {probability})")

    # Close the CoreNLP server
    nlp.close()


def test_swor():
    dependency_trees = []
    graph = generate_random_graph(100)
    trie = Trie()
    start_time = time.time()
    times_elapsed = []
    for i in range(2000):
        sampled_dependency_tree = wilson_scc(graph)
        sampled_dependency_tree_ancestors = sampled_dependency_tree.get_ancestors_list()
        while trie.search(sampled_dependency_tree_ancestors):
            sampled_dependency_tree = wilson_scc(graph)
            sampled_dependency_tree_ancestors = sampled_dependency_tree.get_ancestors_list()
        trie.insert(sampled_dependency_tree_ancestors)
        dependency_trees.append(sampled_dependency_tree)
        current_time = time.time()
        times_elapsed.append(current_time - start_time)
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.xlabel("samples")
    plt.ylabel("seconds")
    plt.plot(range(0, 2000), times_elapsed, label="Wilson SCC SWOR on CPU")
    plt.plot([0, 1999], [0, 15], label="SBS-SWOR on CPU")
    plt.plot([0, 1999], [0, 4.6], label="SBS-SWOR on GPU")
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.show()


# test_swor()
random_weights_training()
