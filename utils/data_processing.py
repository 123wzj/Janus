import pickle

import numpy as np

features_path = f'D:/Python_File/Janus/dataset/output/features_20250117_051246'

# 加载图快照数据并按比例划分训练集、验证集和测试集
def get_data_snapshots_split(graphs_file, use_validation=False):
    """
    Args:
        graphs_file: 图快照文件路径
        use_validation: 是否使用验证集，默认False

    Returns:
        train_range: 训练集的图索引范围 (start_idx, end_idx)
        val_range: 验证集的图索引范围 (start_idx, end_idx)
        test_range: 测试集的图索引范围 (start_idx, end_idx)
    """
    # 加载图快照数据
    with open(graphs_file, 'rb') as f:
        graphs = pickle.load(f)
    # 获取图快照总数
    num_snapshots = len(graphs)
    # 计算分割点
    if use_validation:
        # 70% 训练, 15% 验证, 15% 测试
        val_idx = int(num_snapshots * 0.70)
        test_idx = int(num_snapshots * 0.85)

        train_range = (0, val_idx)
        val_range = (val_idx, test_idx)
        test_range = (test_idx, num_snapshots)
    else:
        # 85% 训练, 15% 测试
        test_idx = int(num_snapshots * 0.85)

        train_range = (0, test_idx)
        val_range = (test_idx, num_snapshots)  # 验证集与测试集相同
        test_range = (test_idx, num_snapshots)

    # print(f"Total number of snapshots: {num_snapshots}")
    # print(f"Training snapshots: {train_range[0]} to {train_range[1] - 1}")
    # if use_validation:
    #     print(f"Validation snapshots: {val_range[0]} to {val_range[1] - 1}")
    # print(f"Test snapshots: {test_range[0]} to {test_range[1] - 1}")
    return train_range, val_range, test_range

# 定义 Data 类，用于存储图数据的基本信息
class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)

# 从已构建的图快照中获取数据
def get_data(snapshot, snapshot_idx):
    """
    snapshot: NetworkX图对象，包含预处理好的节点和边特征
    snapshot_idx: 快照索引
    logger: 日志记录器
    返回:
    node_features: 已预处理的节点特征
    edge_features: 已预处理的边特征
    full_data: 包含图结构信息的Data对象
    """
    try:
        print(f"加载快照 {snapshot_idx} 的数据")
        # 1. 获取已保存的节点特征和边特征
        node_features = np.load(f'{features_path}/snapshot_{snapshot_idx}_node_features.npy')
        edge_features = np.load(f'{features_path}/snapshot_{snapshot_idx}_edge_features.npy')

        # 2. 收集边的信息
        sources = []
        destinations = []
        timestamps = []
        edge_idxs = []
        labels = []
        # edge_features_list = []

        # 从边属性中获取已有信息
        for u, v, data in snapshot.edges(data=True):
            sources.append(u)
            destinations.append(v)
            timestamps.append(data['time'])
            edge_idxs.append(data['idx'])
            # labels.append(int(data['anom']))
            labels.append(data['feature'])
            # edge_features_list.append(data['feature'])

        # 4. 创建Data对象
        full_data = Data(
            sources=np.array(sources),
            destinations=np.array(destinations),
            timestamps=np.array(timestamps),
            edge_idxs=np.array(edge_idxs),
            labels=np.array(labels)
        )
        return node_features, edge_features, full_data

    except Exception as e:
        logger.error(f"处理快照 {snapshot_idx} 时出错: {str(e)}")
        return None, None, None
# 加载数据并将其分割为训练、验证和测试集，同时处理新节点的情况，以测试模型的归纳能力
# def get_data(snapshot_idx, induct, n, different_new_nodes_between_val_and_test=False, randomize_features=False, logger=None):
#     logger.info("归纳能力 {}".format(induct))
#     ### 加载数据并进行训练、验证和测试集的划分
#     graph_df = pd.read_csv('./data/ml_snapshot_{}.csv'.format(snapshot_idx))
#     graph_df = graph_df.head(n)
#     edge_features = np.load('./data/ml_snapshot_{}.npy'.format(snapshot_idx))
#     node_features = np.load('./data/ml_snapshot_{}_node.npy'.format(snapshot_idx))
#
#     if randomize_features:
#         node_features = np.random.rand(node_features.shape[0], node_features.shape[1])
#
#     val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
#
#     sources = graph_df.u.values
#     destinations = graph_df.i.values
#     edge_idxs = graph_df.idx.values
#     labels = graph_df.label.values
#     timestamps = graph_df.ts.values
#
#     full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
#
#     random.seed(2020)
#
#     node_set = set(sources) | set(destinations)
#     n_total_unique_nodes = len(node_set)
#
#     # 计算在测试阶段出现的节点
#     test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
#
#     # 抽取要作为新节点的节点（用于测试归纳能力），并移除训练集中的所有涉及这些节点的边
#     new_test_node_set = set(random.sample(sorted(test_node_set), int(induct * n_total_unique_nodes)))
#
#     # 标记每个源节点和目标节点是否为新测试节点
#     new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
#     new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
#
#     # 标记那些源节点和目标节点都不是新测试节点的边（因为我们要移除涉及任何新测试节点的边）
#     observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
#
#     # 对于训练集，我们保留验证时间之前且不涉及任何新节点的边
#     train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
#
#     train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
#                       edge_idxs[train_mask], labels[train_mask])
#
#     # 定义用于测试模型归纳能力的新节点集
#     train_node_set = set(train_data.sources).union(train_data.destinations)
#     assert len(train_node_set & new_test_node_set) == 0
#     new_node_set = node_set - train_node_set
#
#     val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
#     test_mask = timestamps > test_time
#
#     if different_new_nodes_between_val_and_test:
#         n_new_nodes = len(new_test_node_set) // 2
#         val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
#         test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])
#
#         edge_contains_new_val_node_mask = np.array(
#             [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
#         edge_contains_new_test_node_mask = np.array(
#             [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
#         new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
#         new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
#     else:
#         edge_contains_new_node_mask = np.array(
#             [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
#         new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
#         new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)
#
#     # 包含所有边的验证和测试集
#     val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
#                     edge_idxs[val_mask], labels[val_mask])
#
#     test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
#                      edge_idxs[test_mask], labels[test_mask])
#
#     # 包含至少一个新节点（不在训练集中）的验证和测试集
#     new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
#                              timestamps[new_node_val_mask],
#                              edge_idxs[new_node_val_mask], labels[new_node_val_mask])
#
#     new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
#                               timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
#                               labels[new_node_test_mask])
#
#     logger.info("数据集包含 {} 次交互，涉及 {} 个不同的节点".format(full_data.n_interactions, full_data.n_unique_nodes))
#     logger.info("训练数据集包含 {} 次交互，涉及 {} 个不同的节点".format(train_data.n_interactions, train_data.n_unique_nodes))
#     logger.info("验证数据集包含 {} 次交互，涉及 {} 个不同的节点".format(val_data.n_interactions, val_data.n_unique_nodes))
#     logger.info("测试数据集包含 {} 次交互，涉及 {} 个不同的节点".format(test_data.n_interactions, test_data.n_unique_nodes))
#     logger.info("新节点验证数据集包含 {} 次交互，涉及 {} 个不同的节点".format(new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
#     logger.info("新节点测试数据集包含 {} 次交互，涉及 {} 个不同的节点".format(new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
#     logger.info("用于归纳测试的节点数为 {}，即在训练期间从未见过".format(len(new_test_node_set)))
#
#     return node_features, edge_features, full_data, train_data, val_data, test_data, \
#            new_node_val_data, new_node_test_data

# 计算时间统计信息
def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst