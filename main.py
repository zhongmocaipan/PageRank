import numpy as np
from scipy.sparse import csr_matrix


def read_data(file_path):
    """
    从文件中读取数据并构建稀疏矩阵
    :param file_path: 数据文件路径
    :return: 稀疏矩阵和节点数量
    """
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
    """
    计算 PageRank 分数
    :param adj_matrix: 邻接矩阵
    :param num_nodes: 节点数量
    :param damping_factor: 阻尼因子
    :param max_iterations: 最大迭代次数
    :param tolerance: 收敛阈值
    :return: PageRank 分数
    """
    # 计算每个节点的出度
    out_degree = adj_matrix.sum(axis=1).A1
    # 处理死端节点
    dead_end_mask = (out_degree == 0)
    # 初始化 PageRank 向量
    pr = np.ones(num_nodes) / num_nodes
    for _ in range(max_iterations):
        new_pr = np.zeros(num_nodes)
        # 计算 PageRank 分数
        for i in range(num_nodes):
            if not dead_end_mask[i]:
                new_pr += damping_factor * pr[i] * adj_matrix[i].toarray()[0] / out_degree[i]
        # 处理死端节点
        new_pr += damping_factor * np.sum(pr[dead_end_mask]) / num_nodes
        # 加上随机跳转的部分
        new_pr += (1 - damping_factor) / num_nodes
        # 检查收敛
        if np.linalg.norm(new_pr - pr) < tolerance:
            break
        pr = new_pr
    return pr


def write_result(pr, num_nodes, output_file):
    """
    将得分最高的前 100 个 NodeID 及其 PageRank 分数写入文件
    :param pr: PageRank 分数
    :param num_nodes: 节点数量
    :param output_file: 输出文件路径
    """
    node_scores = [(i, pr[i]) for i in range(num_nodes)]
    node_scores.sort(key=lambda x: x[1], reverse=True)
    top_100 = node_scores[:100]
    with open(output_file, 'w') as file:
        for node, score in top_100:
            file.write(f"{node} {score:.8f}\n")


if __name__ == "__main__":
    input_file = "Data.txt"
    output_file = "Res.txt"
    adj_matrix, num_nodes = read_data(input_file)
    pr = pagerank(adj_matrix, num_nodes)
    write_result(pr, num_nodes, output_file)
    