# test.py
import psutil 
import time
import networkx as nx

def load_graph(file_path):
    """
    使用 networkx 加载图
    """
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        for line in f:
            source, target = map(int, line.strip().split())
            G.add_edge(source, target)
    return G

def run_pagerank(file_path, output_path):
    """
    只跑 NetworkX 的 PageRank，并保存前100个节点结果
    """
    G = load_graph(file_path)
    pr_nx_dict = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)

    # 排序，取前100个
    top_100 = sorted(pr_nx_dict.items(), key=lambda x: x[1], reverse=True)[:100]

    # 保存到 test.txt
    with open(output_path, 'w') as f:
        for node, score in top_100:
            f.write(f"{node}\t{score:.8f}\n")

if __name__ == "__main__":

    # 获取当前进程的内存使用情况
    process = psutil.Process()
    # 记录开始时间
    start_time = time.time()

    file_path = "Data.txt"    # 你的输入数据文件
    output_path = "test.txt"  # 保存结果的文件
    run_pagerank(file_path, output_path)
    
    # 记录结束时间并打印运行时间
    end_time = time.time()
    print(f"PageRank computation completed in {end_time - start_time:.4f} seconds.")

        
    # 输出结束时的内存使用情况
    final_memory = process.memory_info().rss / 1024 / 1024  # 转换为MB
    print(f"Final memory usage: {final_memory:.2f} MB")