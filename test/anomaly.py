import pickle
from datetime import datetime

import pandas as pd

# 统计每个图快照中正常和异常边的数量
def count_edges_by_anomaly(graphs):
    stats = []

    for i, G in enumerate(graphs):
        # 获取所有边的属性
        normal_edges = sum(1 for u, v, data in G.edges(data=True) if not data['anom'])
        anomaly_edges = sum(1 for u, v, data in G.edges(data=True) if data['anom'])
        total_edges = normal_edges + anomaly_edges

        stats.append({
            'snapshot': i,
            'normal_edges': normal_edges,
            'anomaly_edges': anomaly_edges,
            'total_edges': total_edges,
            'anomaly_ratio': round(anomaly_edges / total_edges * 100, 2) if total_edges > 0 else 0
        })

    # 转换为DataFrame以便更好地展示
    stats_df = pd.DataFrame(stats)
    return stats_df

def load_saved_data(graphs_file):
    # print(f"成功加载节点映射，包含 {len(node_map)} 个节点")
    # 加载图数据
    print(f"正在加载图数据从: {graphs_file}")
    with open(graphs_file, 'rb') as f:
        graphs = pickle.load(f)
    return  graphs

# 使用示例：
def analyze_graphs(graphs_file,output_file):
    # 加载数据
    graphs = load_saved_data(graphs_file)

    # 统计边的信息
    stats_df = count_edges_by_anomaly(graphs)

    # 准备输出内容
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入文件头部信息
        f.write(f"图快照边统计分析报告\n")
        f.write(f"生成时间: {current_time} UTC\n")
        f.write("=" * 80 + "\n\n")

        # 写入详细统计信息
        f.write("每个快照的统计信息：\n")
        f.write(stats_df.to_string())
        f.write("\n\n")

        # 写入总体统计
        f.write("总体统计：\n")
        f.write("-" * 40 + "\n")
        f.write(f"总快照数：{len(graphs)}\n")
        f.write(f"总正常边数：{stats_df['normal_edges'].sum()}\n")
        f.write(f"总异常边数：{stats_df['anomaly_edges'].sum()}\n")
        f.write(f"平均异常比例：{stats_df['anomaly_ratio'].mean():.2f}%\n")

    print(f"统计结果已保存到: {output_file}")
    return stats_df

if __name__ == "__main__":
    # 调用分析函数
    stats = analyze_graphs(graphs_file='D:/Python_File/Janus/dataset/output/graphs_20250117_051246.pkl',
                           output_file="graph_statistics.txt")