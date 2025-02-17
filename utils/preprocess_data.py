import json
import os
import pickle
import sys
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

data_name = 'D:/Python_File/Janus/dataset/lanl/auth.csv'

data_folder = 'D:/Python_File/Janus/dataset/lanl/'

output_path = 'D:/Python_File/Janus/dataset/output/'

# 节点映射：将源计算机和目标计算机映射到唯一的整数 ID
def get_node_map(data_df):
    print("... Generating Node Map ... \n")
    node_map = {}
    node_id = 0
    for index, row in tqdm(data_df.iterrows()):
        scomp = row.src
        dcomp = row.dst
        if scomp not in node_map:
            node_map[scomp] = node_id
            node_id += 1
        if dcomp not in node_map:
            node_map[dcomp] = node_id
            node_id += 1
    return node_map


# 生成数据帧 data_df，并调用 get_node_map 方法生成节点映射 node_map
def get_map(data_name):
    data_df = pd.read_csv(data_name, header=0)
    node_df = data_df[['src', 'dst']]
    node_df = node_df.drop_duplicates()
    node_map = get_node_map(node_df)
    return data_df, node_map


# 预处理LANL数据集，得到数据帧auth_df,并从redteam.txt中标记异常记录，并将结果保存到新的csv文件中
def preprocess_lanl_data(data_name):
    auth_df = pd.DataFrame()
    # 参数 chunksize=10000 指定每次读取 10000 行数据，usecols 参数指定要读取的列，dtype 参数指定每列的数据类型
    for chunk_df in tqdm(pd.read_csv(data_name,
                                     usecols=['timestamp', 'src_user', 'dst_user', 'src_computer', 'dst_computer',
                                              'auth_type', 'logon_type', 'auth_orient', 'result'],
                                     dtype={'timestamp': np.int32, 'src_user': str, 'dst_user': str,
                                            'src_computer': str, 'dst_computer': str, 'auth_type': 'category',
                                            'logon_type': 'category', 'auth_orient': 'category', 'result': 'category'},
                                     chunksize=10000)):
        # 过滤掉 auth_type 或 logon_type 列中包含 ? 的行
        chunk_df = chunk_df[~((chunk_df['auth_type'] == '?') | (chunk_df['logon_type'] == '?'))]
        # 过滤掉 src_user 列中包含特定字符串（如 ANONYMOUS, LOCAL, NETWORK, ADMIN）的行
        chunk_df = chunk_df[~((chunk_df['src_user'].str.contains(r'ANONYMOUS(?!$)')) | (
            chunk_df['src_user'].str.contains(r'LOCAL(?!$)')) | (chunk_df['src_user'].str.contains(r'NETWORK(?!$)')) | (
                                  chunk_df['src_user'].str.contains(r'ADMIN(?!$)')))]
        # 过滤掉 src_computer 和 dst_computer 列值相同的行
        # chunk_df = chunk_df[chunk_df['src_computer'] != chunk_df['dst_computer']]

        # 将处理后的数据块 chunk_df 追加到 auth_df DataFrame 中，并重置索引
        auth_df = pd.concat([auth_df, chunk_df], ignore_index=True)

    # 初始化 anom 列为 0
    auth_df['anom'] = 0

    # 加载恶意文件，红队攻击记录
    rt_df = pd.read_csv(data_folder + 'redteam.txt', header=0)
    # 数据帧列名重命名
    rt_df.columns = ['timestamp', 'src_user', 'src_computer', 'dst_computer']

    # 标记异常记录
    for index, row in rt_df.iterrows():
        mask = ((auth_df['timestamp'] == row['timestamp']) &
                (auth_df['src_user'] == row['src_user']) &
                (auth_df['src_computer'] == row['src_computer']) &
                (auth_df['dst_computer'] == row['dst_computer']))
        auth_df.loc[mask, 'anom'] = 1

    # 获取最小的时间
    initial_time = min(auth_df['timestamp'])
    # 计算与初始时间戳差值
    auth_df['delta'] = auth_df['timestamp'] - initial_time
    # 转为小时存储
    auth_df['snapshot'] = auth_df['delta'] // 3600

    # 在完成异常标记和时间处理后，创建合并的 src 和 dst 列
    auth_df['src'] = auth_df['src_user'] + ':' + auth_df['src_computer']
    auth_df['dst'] = auth_df['dst_user'] + ':' + auth_df['dst_computer']

    # 过滤掉 src 和 dst 相同的行
    # auth_df = auth_df[auth_df['src'] != auth_df['dst']]

    # 删除原始列和delta列
    auth_df = auth_df.drop(['delta', 'src_user', 'src_computer', 'dst_user', 'dst_computer'], axis=1).reset_index(
        drop=True)

    auth_df.to_csv(data_folder + "auth_all_anom_1hr.csv")
    print("... Finished Preprocessing ... \n")


# 创建单个快照图
def create_graph(snapshot_df, node_map):
    G = nx.MultiGraph()
    anom_node = []
    # 获取边
    edge_feature_cols = ['auth_type', 'logon_type', 'auth_orient', 'result']
    # 使用 LabelEncoder 将分类特征转换为数值
    le_dict = {col: LabelEncoder() for col in edge_feature_cols}

    # 保持索引一致性，使用 reset_index 确保索引重置
    snapshot_df = snapshot_df.reset_index(drop=True)
    edge_features_encoded = pd.DataFrame()

    # 逐列进行编码，保持索引一致性
    for col in edge_feature_cols:
        edge_features_encoded[col] = le_dict[col].fit_transform(snapshot_df[col])

    # 边索引计数器
    edge_idx = 0

    for index, row in snapshot_df.iterrows():
        scomp = row.src
        dcomp = row.dst
        time = row.timestamp
        gid = row.snapshot

        src_id = node_map[scomp]
        dst_id = node_map[dcomp]
        is_anomaly = False
        # 标记异常
        if row.anom == 1:
            is_anomaly = True
            if scomp not in anom_node:
                anom_node.append(scomp)
            if dcomp not in anom_node:
                anom_node.append(dcomp)


        # 添加节点，使用更丰富的节点特征
        if not G.has_node(src_id):
            G.add_node(src_id,anom = (scomp in anom_node),
                       feature=np.array([src_id,
                                         len(snapshot_df[snapshot_df['src'] == scomp])],
                                        dtype=np.float32))
        if not G.has_node(dst_id):
            G.add_node(dst_id,anom = (dcomp in anom_node),
                       feature=np.array([dst_id,
                                         len(snapshot_df[snapshot_df['dst'] == dcomp])],
                                        dtype=np.float32))

            # 安全地获取边特征
            try:
                edge_feature = edge_features_encoded.iloc[index].values
                G.add_edge(src_id, dst_id,
                           time=time,
                           anom=is_anomaly,
                           snapshot=gid,
                           idx=edge_idx,
                           feature=edge_feature)
                edge_idx += 1
            except IndexError as e:
                print(f"警告：索引 {index} 访问失败。DataFrame 长度：{len(edge_features_encoded)}")
                continue
    return G


def save_snapshot_features(G, snapshot_idx, output_path):
    """保存快照的节点特征和边特征"""
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 提取并保存节点特征
    node_list = sorted(G.nodes())
    node_features = np.array([G.nodes[node]['feature'] for node in node_list])

    # 提取并保存边特征
    edges = list(G.edges(data=True))
    edge_features = np.array([e[2]['feature'] for e in edges])

    # 如果没有边，创建空的特征矩阵
    if len(edge_features) == 0:
        edge_features = np.zeros((1, 4))  # 4是边特征的维度

    # 保存特征
    np.save(f'{output_path}/snapshot_{snapshot_idx}_node_features.npy', node_features)
    np.save(f'{output_path}/snapshot_{snapshot_idx}_edge_features.npy', edge_features)


def build_and_save_graphs(auth_df, node_map, output_path):
    """构建并保存图快照"""
    graphs = []
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    features_path = f'{output_path}/features_{timestamp}'

    print("构建图快照...")
    for t in tqdm(sorted(auth_df.snapshot.unique())):
        # 构建图
        snapshot_df = auth_df[auth_df['snapshot'] == t]
        G = create_graph(snapshot_df, node_map)
        graphs.append(G)

        # 保存特征
        save_snapshot_features(G, t, features_path)

    # 保存图快照列表
    graphs_file = f'{output_path}/graphs_{timestamp}.pkl'
    with open(graphs_file, 'wb') as f:
        pickle.dump(graphs, f)

    print(f"图快照已保存到: {graphs_file}")
    print(f"特征数据已保存到: {features_path}")

    # 图快照已保存到: D:\Python_File\Pikachu\dataset\output/graphs_20250115_051649.pkl
    # 特征数据已保存到: D:\Python_File\Pikachu\dataset\output/features_20250115_051649

    return graphs, graphs_file, features_path


if __name__ == '__main__':
    # 主程序中调用
    preprocess_lanl_data(data_name)
    data_df, node_map = get_map(data_folder + "auth_all_anom_1hr.csv")
    # 序列化并保存节点映射
    with open(f'{data_folder}/node_map_lanl.pickle', 'wb') as f:
        pickle.dump(node_map, f)
    graphs, graphs_file, features_path = build_and_save_graphs(data_df, node_map, output_path)

    # 验证保存的数据
    # for snapshot_idx in range(len(graphs)):
    #     node_features = np.load(f'{features_path}/snapshot_{snapshot_idx}_node_features.npy')
    #     edge_features = np.load(f'{features_path}/snapshot_{snapshot_idx}_edge_features.npy')
    #
    #     print(f"快照 {snapshot_idx}:")
    #     print(f"节点特征: {node_features}")
    #     print(f"边特征: {edge_features}")
    #     sys.exit(0)