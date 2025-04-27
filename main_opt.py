import numpy as np
from scipy.sparse import csr_matrix
import psutil
import time

def read_data(file_path):
    sources = []
    targets = []
    node_set = set()
    with open(file_path, 'r') as file:
        for line in file:
            source, target = map(int, line.strip().split())
            sources.append(source)
            targets.append(target)
            node_set.add(source)
            node_set.add(target)
    num_nodes = max(node_set) + 1
    data = np.ones(len(sources))
    adj_matrix = csr_matrix((data, (sources, targets)), shape=(num_nodes, num_nodes))
    return adj_matrix, num_nodes

def pagerank(adj_matrix, num_nodes, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    out_degree = np.array(adj_matrix.sum(axis=1)).flatten()
    out_degree[out_degree == 0] = 1
    transition = adj_matrix.multiply(1 / out_degree[:, None])
    pr = np.ones(num_nodes) / num_nodes
    teleport = np.ones(num_nodes) / num_nodes

    for _ in range(max_iterations):
        new_pr = damping_factor * transition.T.dot(pr) + (1 - damping_factor) * teleport
        if np.linalg.norm(new_pr - pr, 1) < tolerance:
            break
        pr = new_pr
    return pr

def write_result(pr, num_nodes, output_file):
    node_scores = [(i, pr[i]) for i in range(num_nodes)]
    node_scores.sort(key=lambda x: x[1], reverse=True)
    top_100 = node_scores[:100]
    with open(output_file, 'w') as file:
        for node, score in top_100:
            file.write(f"{node} {score:.8f}\n")

if __name__ == "__main__":
    process = psutil.Process()
    start_time = time.time()
    input_file = "Data.txt"
    output_file = "Res.txt"

    adj_matrix, num_nodes = read_data(input_file)
    pr = pagerank(adj_matrix, num_nodes)
    write_result(pr, num_nodes, output_file)

    end_time = time.time()
    print(f"PageRank computation completed in {end_time - start_time:.4f} seconds.")

    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Final memory usage: {final_memory:.2f} MB")
