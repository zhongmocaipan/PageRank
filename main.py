import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags
import psutil 
import time

def read_data(file_path):
    """
    从文件中读取数据并构建稀疏矩阵
    :param file_path: 数据文件路径
    :return: 转置稀疏矩阵和节点数量
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
    matrix = csr_matrix((data, (targets, sources)), shape=(num_nodes, num_nodes)) 

    return matrix, num_nodes


def pagerank(matrix, num_nodes, damping_factor=0.85, max_iterations=100, tolerance=1e-6, block_size=350):
    """
    计算 PageRank 分数
    :param matrix: 转置邻接矩阵
    :param num_nodes: 节点数量
    :param damping_factor: 阻尼因子
    :param max_iterations: 最大迭代次数
    :param tolerance: 收敛阈值
    :param block_size: 分块大小
    :return: PageRank 分数
    """

    # 计算每个节点的出度
    out_degree = np.array(matrix.sum(axis=0)).flatten()
    # 死端节点处理：将出度为零的节点的出度设为 1，避免除零
    out_degree[out_degree == 0] = 1  
    inv_out = diags(1 / out_degree)
    # 初始化 PageRank 向量
    pr = np.ones(num_nodes) / num_nodes
    # M 矩阵为稀疏标准化转置矩阵
    transition = matrix @ inv_out

    # 按块进行迭代更新
    for iteration in range(max_iterations):
        new_pr = np.zeros(num_nodes)

        # 减少阻尼因子，增加随机跳跃
        if iteration > max_iterations // 2:
            damping_factor = 0.75  

        # 分块计算，每个块大小为 block_size
        for block_start in range(0, num_nodes, block_size):
            block_end = min(block_start + block_size, num_nodes)
            block_transition = transition[block_start:block_end]  # 获取当前块的子矩阵
            new_pr[block_start:block_end] = damping_factor * block_transition @ pr + (1 - damping_factor) / num_nodes

        # 加上 dead-end 的 PageRank 补偿（total leaked mass）
        leak = damping_factor * pr[out_degree == 0].sum() / num_nodes
        new_pr += leak       
        # 检查收敛
        if np.linalg.norm(new_pr - pr, 1) < tolerance:
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

    # 获取当前进程的内存使用情况
    process = psutil.Process()
    # 记录开始时间
    start_time = time.time()
    input_file = "Data.txt"
    output_file = "Res.txt"

    matrix, num_nodes = read_data(input_file)

    pr = pagerank(matrix, num_nodes)
    
    write_result(pr, num_nodes, output_file)

    # 记录结束时间并打印运行时间
    end_time = time.time()
    print(f"PageRank computation completed in {end_time - start_time:.4f} seconds.")

        
    # 输出结束时的内存使用情况
    final_memory = process.memory_info().rss / 1024 / 1024  # 转换为MB
    print(f"Final memory usage: {final_memory:.2f} MB")
    