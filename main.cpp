#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <fstream>
#include <iomanip>

// Use a namespace to encapsulate the logic and avoid polluting the global scope.
namespace GraphSampling {

// A simple structure for a graph edge.
struct Edge {
    // In-class initializers ensure members are zero-initialized by default.
    int first = 0;
    int second = 0;
    double weight = 0.0;
};

class Graph {
public:
    int nodesNumber;
    std::vector<std::vector<std::pair<int, double>>> edges;
    std::vector<std::vector<std::pair<int, double>>> reversedEdges;

    // ADDED: Default constructor for an empty graph.
    Graph() : nodesNumber(0) {}

    // Existing constructor to build a graph from an edge list.
    explicit Graph(int numNodes, const std::vector<Edge>& edgeList = {})
        : nodesNumber(numNodes),
          edges(numNodes),
          reversedEdges(numNodes) {
        for (const auto& edge : edgeList) {
            if (edge.first < numNodes && edge.second < numNodes) {
                edges[edge.first].emplace_back(edge.second, edge.weight);
                reversedEdges[edge.second].emplace_back(edge.first, edge.weight);
            }
        }
    }

    // Creates a flat list of all edges in the graph.
    std::vector<Edge> get_edges_list() const {
        std::vector<Edge> edge_list;
        for (int node = 0; node < nodesNumber; ++node) {
            for (const auto& edge : edges[node]) {
                edge_list.push_back({node, edge.first, edge.second});
            }
        }
        return edge_list;
    }

    // Returns a list of parent nodes for each node (assuming it's a tree structure).
    std::vector<int> get_ancestors_list() const {
        if (nodesNumber == 0) return {}; // Handle empty graph case
        std::vector<int> ancestors(nodesNumber);
        ancestors[0] = -1; // Root has no ancestor.
        for (int i = 1; i < nodesNumber; ++i) {
            if (!reversedEdges[i].empty()) {
                ancestors[i] = reversedEdges[i][0].first;
            } else {
                ancestors[i] = -1; // Node i has no parent.
            }
        }
        return ancestors;
    }
};

// A Trie (Prefix Tree) to store sequences of integers.
// Useful for checking if a sampled dependency tree has been seen before.
class Trie {
private:
    struct TrieNode {
        std::unordered_map<int, std::unique_ptr<TrieNode>> children;
        bool isEndOfSequence = false;
    };
    std::unique_ptr<TrieNode> root;

public:
    // This class already has a default constructor.
    Trie() : root(std::make_unique<TrieNode>()) {}

    void insert(const std::vector<int>& entry) {
        TrieNode* current = root.get();
        for (int val : entry) {
            if (current->children.find(val) == current->children.end()) {
                current->children[val] = std::make_unique<TrieNode>();
            }
            current = current->children[val].get();
        }
        current->isEndOfSequence = true;
    }

    bool search(const std::vector<int>& entry) const {
        const TrieNode* current = root.get();
        for (int val : entry) {
            auto it = current->children.find(val);
            if (it == current->children.end()) {
                return false;
            }
            current = it->second.get();
        }
        return current && current->isEndOfSequence;
    }
};


// Utility class for thread-safe random number generation.
class RandomGenerator {
private:
    // Use static to ensure the generator is seeded only once.
    static std::mt19937& get_engine() {
        static std::random_device rd;
        static std::mt19937 engine(rd());
        return engine;
    }

public:
    static int get_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(get_engine());
    }

    static double get_uniform(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(get_engine());
    }

    static double get_normal(double mean, double stddev) {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(get_engine());
    }
};

Graph generate_random_graph(int nodesNumber) {
    // The original Python code hardcodes this value inside the function.
    nodesNumber = 101;
    std::vector<Edge> edge_list;
    for (int i = 0; i < nodesNumber; ++i) {
        for (int j = 1; j < nodesNumber; ++j) {
            if (i != j) {
                // Add an edge with 50% probability.
                if (RandomGenerator::get_int(0, 1) == 1) {
                    double weight = RandomGenerator::get_normal(3.0, 1.0);
                    edge_list.push_back({i, j, weight});
                }
            }
        }
    }
    return Graph(nodesNumber, edge_list);
}

int sample_random_parent_node(const Graph& graph, int node, bool no_root_sampling) {
    const auto& incoming_edges = graph.reversedEdges[node];
    double total_weight = 0.0;

    for (const auto& edge : incoming_edges) {
        if (!no_root_sampling || edge.first != 0) {
            total_weight += edge.second;
        }
    }

    if (total_weight <= 0) return -1; // No valid parent to sample

    double random_weight = RandomGenerator::get_uniform(0.0, total_weight);

    for (const auto& edge : incoming_edges) {
        if (!no_root_sampling || edge.first != 0) {
            random_weight -= edge.second;
            if (random_weight < 0) {
                return edge.first;
            }
        }
    }
    // Should not be reached if there are valid edges, but as a fallback.
    return incoming_edges.back().first;
}

Graph sample_spanning_tree(const Graph& graph, bool unique_root_edge = false) {
    std::vector<Edge> edge_list;
    std::vector<bool> visited(graph.nodesNumber, false);
    std::vector<int> parent(graph.nodesNumber, -1);

    if (graph.nodesNumber == 0) return Graph{}; // Handle empty graph

    visited[0] = true;
    bool selected_root_edge = false;

    for (int i = 1; i < graph.nodesNumber; ++i) {
        if (!visited[i]) {
            int current_node = i;
            // 1. Find a path back to an already visited node (random walk).
            while (!visited[current_node]) {
                int sampled_parent = sample_random_parent_node(graph, current_node, unique_root_edge && selected_root_edge);
                if (sampled_parent == -1) break; // Path broken
                parent[current_node] = sampled_parent;
                current_node = sampled_parent;
            }

            // 2. Add the path to the tree, erasing any cycles found along the way.
            current_node = i;
            while (!visited[current_node]) {
                if (parent[current_node] == -1) break; // Path broken
                edge_list.push_back({parent[current_node], current_node, 0.0});
                visited[current_node] = true;
                current_node = parent[current_node];
            }
            if (!selected_root_edge) {
                selected_root_edge = true;
            }
        }
    }
    return Graph(graph.nodesNumber, edge_list);
}

bool is_dependency_tree(const Graph& tree) {
    if (tree.nodesNumber == 0) return false;
    return tree.edges[0].size() == 1;
}

Graph wilson_reject(const Graph& graph) {
    while (true) {
        Graph spanning_tree = sample_spanning_tree(graph);
        if (is_dependency_tree(spanning_tree)) {
            return spanning_tree;
        }
    }
}

// --- Kosaraju's Algorithm for Strongly Connected Components (SCCs) ---

void dfs_first_pass(const Graph& graph, int node, std::vector<bool>& visited, std::vector<int>& stack) {
    visited[node] = true;
    for (const auto& edge : graph.edges[node]) {
        if (!visited[edge.first]) {
            dfs_first_pass(graph, edge.first, visited, stack);
        }
    }
    stack.push_back(node);
}

void dfs_second_pass(const Graph& graph, int node, std::vector<bool>& visited, std::vector<int>& component) {
    visited[node] = true;
    component.push_back(node);
    for (const auto& edge : graph.reversedEdges[node]) {
        if (!visited[edge.first]) {
            dfs_second_pass(graph, edge.first, visited, component);
        }
    }
}

std::pair<Graph, std::vector<std::vector<int>>> get_all_sccs(const Graph& graph) {
    if (graph.nodesNumber == 0) {
        return {Graph{}, {}};
    }

    std::vector<int> stack;
    std::vector<bool> visited(graph.nodesNumber, false);
    for (int i = 0; i < graph.nodesNumber; ++i) {
        if (!visited[i]) {
            dfs_first_pass(graph, i, visited, stack);
        }
    }

    std::fill(visited.begin(), visited.end(), false);
    std::vector<std::vector<int>> scc_list;
    while (!stack.empty()) {
        int node = stack.back();
        stack.pop_back();
        if (!visited[node]) {
            std::vector<int> component;
            dfs_second_pass(graph, node, visited, component);
            scc_list.push_back(component);
        }
    }

    // Build the SCC graph
    int scc_nr = scc_list.size();
    std::vector<int> node_to_scc_map(graph.nodesNumber);
    for (int i = 0; i < scc_nr; ++i) {
        for (int node : scc_list[i]) {
            node_to_scc_map[node] = i;
        }
    }

    std::vector<Edge> scc_edges;
    // Note: Correctly create a 2D boolean vector, unlike the Python version's bug.
    std::vector<std::vector<bool>> scc_edge_added(scc_nr, std::vector<bool>(scc_nr, false));

    for (int node = 0; node < graph.nodesNumber; ++node) {
        int parent_scc = node_to_scc_map[node];
        for (const auto& edge : graph.edges[node]) {
            int descendant_scc = node_to_scc_map[edge.first];
            if (parent_scc != descendant_scc && !scc_edge_added[parent_scc][descendant_scc]) {
                scc_edge_added[parent_scc][descendant_scc] = true;
                scc_edges.push_back({parent_scc, descendant_scc, 1.0});
            }
        }
    }

    return {Graph(scc_nr, scc_edges), scc_list};
}

// --- End of SCC ---

std::vector<int> find_potential_root_descendants(const Graph& graph) {
    auto [scc_graph, scc_list] = get_all_sccs(graph);
    if (scc_graph.nodesNumber == 0) return {};

    int scc_nr = scc_graph.nodesNumber;
    int valid_descendant_scc = -1;

    for (int curr_scc = 0; curr_scc < scc_nr; ++curr_scc) {
        if (scc_graph.reversedEdges[curr_scc].size() == 1) {
            if (scc_graph.reversedEdges[curr_scc][0].first == 0) {
                if (valid_descendant_scc != -1) {
                    return {}; // Found more than one, so no valid path.
                }
                valid_descendant_scc = curr_scc;
            }
        }
    }

    if (valid_descendant_scc == -1) {
        return {};
    }
    return scc_list[valid_descendant_scc];
}

Graph remove_invalid_edges_from_graph(const Graph& graph) {
    Graph newGraph = graph; // Make a mutable copy
    if (newGraph.nodesNumber == 0) return newGraph;

    std::vector<int> potential_descendants = find_potential_root_descendants(newGraph);
    if (potential_descendants.empty()) {
        newGraph.edges[0].clear(); // No valid descendants, so remove all root edges.
        return newGraph;
    }

    std::vector<bool> is_potential_descendant(graph.nodesNumber, false);
    for (int descendant : potential_descendants) {
        is_potential_descendant[descendant] = true;
    }

    std::vector<std::pair<int, double>> valid_edges;
    for (const auto& edge : newGraph.edges[0]) {
        if (is_potential_descendant[edge.first]) {
            valid_edges.push_back(edge);
        }
    }
    newGraph.edges[0] = valid_edges;
    return newGraph;
}

Graph wilson_scc(const Graph& graph) {
    Graph valid_edges_graph = remove_invalid_edges_from_graph(graph);
    return sample_spanning_tree(valid_edges_graph, true);
}


// --- Test Harnesses ---

void random_weights_training() {
    std::cout << "Running comparison: Wilson with SCC vs. Wilson with Rejection Sampling..." << std::endl;
    constexpr int num_samples = 100;
    std::vector<Graph> graph_set1, graph_set2;
    graph_set1.reserve(num_samples);
    graph_set2.reserve(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        graph_set1.push_back(generate_random_graph(100));
        graph_set2.push_back(generate_random_graph(100));
    }

    std::vector<double> times_elapsed_wr;
    times_elapsed_wr.reserve(num_samples);
    auto start_time_wr = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_samples; ++i) {
        wilson_reject(graph_set2[i]);
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time_wr;
        times_elapsed_wr.push_back(elapsed.count());
    }

    std::vector<double> times_elapsed_wre;
    times_elapsed_wre.reserve(num_samples);
    auto start_time_wre = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_samples; ++i) {
        wilson_scc(graph_set1[i]);
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time_wre;
        times_elapsed_wre.push_back(elapsed.count());
    }

    // Write results to CSV file for plotting
    std::ofstream outfile("random_weights_training.csv");
    outfile << "Sample,Wilson_SCC_Time,Wilson_Reject_Time\n";
    for (int i = 0; i < num_samples; ++i) {
        outfile << i + 1 << "," << times_elapsed_wre[i] << "," << times_elapsed_wr[i] << "\n";
    }
    outfile.close();
    std::cout << "Results saved to 'random_weights_training.csv'." << std::endl;
}

void test_swor() {
    std::cout << "Running SWOR (Sampling Without Replacement) test..." << std::endl;
    constexpr int num_samples = 2000;
    Graph graph = generate_random_graph(100);
    Trie trie;

    std::vector<double> times_elapsed;
    times_elapsed.reserve(num_samples);
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_samples; ++i) {
        Graph sampled_tree;
        std::vector<int> ancestors;
        do {
            sampled_tree = wilson_scc(graph);
            ancestors = sampled_tree.get_ancestors_list();
        } while (trie.search(ancestors));

        trie.insert(ancestors);

        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time;
        times_elapsed.push_back(elapsed.count());
    }

    // Write results to CSV file for plotting
    std::ofstream outfile("test_swor.csv");
    outfile << "Sample,Time_Elapsed\n";
    for (int i = 0; i < num_samples; ++i) {
        outfile << i + 1 << "," << times_elapsed[i] << "\n";
    }
    outfile.close();
    std::cout << "Results saved to 'test_swor.csv'." << std::endl;
}

} // namespace GraphSampling

int main() {
    // Set floating point precision for output
    std::cout << std::fixed << std::setprecision(6);

    // To switch between tests, comment/uncomment the desired function.
    // GraphSampling::test_swor();
    GraphSampling::random_weights_training();

    return 0;
}