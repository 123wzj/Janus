# ******************************************************************************
# processing.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 3/4/21   Paudel     Initial version,
# ******************************************************************************
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx
from timeit import default_timer as timer
import random

class GraphUtils:
    def __init__(self, node_map):
        self.node_map = node_map
        pass

    # 计算两个向量 u 和 v 的 Hadamard 积（逐元素相乘）
    def embedding_hadamard(self, u, v):
        return u * v

    # 计算两个向量 u 和 v 的 L1 距离（逐元素绝对差）
    def embedding_l1(self, u, v):
        return np.abs(u - v)
    # 计算两个向量 u 和 v 的 L2 距离（逐元素平方差）
    def embedding_l2(self, u, v):
        return (u - v) ** 2
    # 计算两个向量 u 和 v 的平均值
    def embedding_avg(self, u, v):
        return (u + v) / 2.0

    def create_graph(self, snapshot_df):
        G = nx.MultiGraph()
        anom_node = []

        for index, row in snapshot_df.iterrows():
            scomp = row.src_computer
            dcomp = row.dst_computer
            # host_name = row.src_user #row.host_name
            time = index #row.timestamp
            gid = row.snapshot
            is_anomaly = False
            # 标记为异常图
            if row.label == 1:
                # print(row)
                is_anomaly = True
                if scomp not in anom_node:
                    anom_node.append(scomp)
                if dcomp not in anom_node:
                    anom_node.append(dcomp)
            # 节点的属性 anom 表示该节点是否是异常节点，值为 True 或 False，取决于 scomp 是否在 anom_node 列表中
            G.add_node(self.node_map[scomp], anom= (scomp in anom_node))
            G.add_node(self.node_map[dcomp], anom= (dcomp in anom_node))

            G.add_edge(self.node_map[scomp], self.node_map[dcomp], time=time, anom=is_anomaly, snapshot = gid, weight=1)
        # print("Auth N: %d E: %d \n" % (G.number_of_nodes(), G.number_of_edges()))
        return G

# 传入数据集
class DataUtils:
    def __init__(self, data_folder):
        self.data_folder = data_folder
    # 节点映射：将源计算机和目标计算机映射到唯一的整数 ID
    def get_node_map(self, data_df):
        print("... Generating Node Map ... \n")
        node_map = {}
        node_id = 0
        for index, row in tqdm(data_df.iterrows()):
            scomp = row.src_computer
            dcomp = row.dst_computer
            if scomp not in node_map:
                node_map[scomp] = node_id
                node_id += 1
            if dcomp not in node_map:
                node_map[dcomp] = node_id
                node_id += 1
        return node_map

    # 生成数据帧 data_df，并调用 get_node_map 方法生成节点映射 node_map
    def get_data(self):
        data_df = pd.read_csv(self.data_folder, header=0)
        node_df = data_df[['src_computer', 'dst_computer']]
        node_df = node_df.drop_duplicates()
        node_map = self.get_node_map(node_df)
        return data_df, node_map

    # 预处理LANL数据集，得到数据帧auth_df,并从redteam.txt中标记异常记录，并将结果保存到新的csv文件中
    def preprocess_lanl_data(self):
        auth_df = pd.DataFrame()
        # 参数 chunksize=10000 指定每次读取 10000 行数据，usecols 参数指定要读取的列，dtype 参数指定每列的数据类型
        for chunk_df in tqdm(pd.read_csv(self.data_folder, usecols=['timestamp', 'src_user', 'dst_user', 'src_computer', 'dst_computer', 'auth_type', 'logon_type'], dtype={'timestamp': np.int32, 'src_user': str, 'dst_user': str,
                                     'src_computer': str, 'dst_computer': str, 'auth_type': 'category','logon_type': 'category'}, chunksize=10000)):
            # 过滤掉 auth_type 或 logon_type 列中包含 ? 的行
            chunk_df = chunk_df[~((chunk_df['auth_type'] == '?') | (chunk_df['logon_type'] == '?'))]
            # 过滤掉 src_user 列中包含特定字符串（如 ANONYMOUS, LOCAL, NETWORK, ADMIN）的行
            chunk_df = chunk_df[~((chunk_df['src_user'].str.contains(r'ANONYMOUS(?!$)')) | (
                chunk_df['src_user'].str.contains(r'LOCAL(?!$)')) | (chunk_df['src_user'].str.contains(r'NETWORK(?!$)')) | (
                chunk_df['src_user'].str.contains(r'ADMIN(?!$)')))]
            # 过滤掉 src_computer 和 dst_computer 列值相同的行
            chunk_df = chunk_df[chunk_df['src_computer'] != chunk_df['dst_computer']]
            # 删除 dst_user, auth_type, 和 logon_type 列，并重置索引
            chunk_df = chunk_df.drop(['dst_user', 'auth_type', 'logon_type'], axis=1).reset_index(
                drop=True)
            # 将处理后的数据块 chunk_df 追加到 auth_df DataFrame 中，并重置索引
            auth_df = auth_df.append(chunk_df, ignore_index=True)

        # 加载恶意文件，红队攻击记录
        rt_df = pd.read_csv('../dataset/lanl/redteam.txt', header=0)
        # 数据帧列名重命名
        rt_df.columns = ['timestamp', 'src_user', 'src_computer', 'dst_computer']
        filter_col_name = ['timestamp', 'src_user', 'src_computer', 'dst_computer']  # rt_df.columns.tolist()
        # 将 auth_df 和 rt_df 数据帧进行合并:使用内连接（inner join），只保留在两个数据帧中都存在的记录,指定用于合并的列名，即 timestamp, src_user, src_computer, 和 dst_computer
        comm_df = pd.merge(auth_df.reset_index(), rt_df.reset_index(), how='inner', on=filter_col_name)
        # print("Anomalous rows: \n", comm_df)

        # 将 comm_df 数据帧中的 index_x 列转换为一个列表，并将其存储在变量 anom_row_index 中。这个列表包含了所有在 auth_df 中与 rt_df 匹配的记录的索引
        anom_row_index = comm_df.index_x.to_list()
        # print("Anom rows index: ", anom_row_index)

        # label row as anom or norm
        auth_df['label'] = 0
        auth_df.loc[anom_row_index, 'label'] = 1
        # 获取最小的时间
        initial_time = min(auth_df['timestamp'])
        # 计算与初始时间戳差值
        auth_df['delta'] = auth_df['timestamp'] - initial_time
        # 转为小时存储
        auth_df['snapshot'] = auth_df['delta'].sec // 3600
        # 删除列，并重置索引
        auth_df = auth_df.drop(['delta'], axis=1).reset_index(
            drop=True)
        auth_df.to_csv("dataset/lanl/auth_all_anom_1hr.csv")
        self.get_data()

    # 生成一个包含异常用户和正常用户子集的数据帧
    def lanl_user_subset(self):

        lanl_df = pd.read_csv(self.data_folder, header=0, index_col=0, dtype={'timestamp': np.int32, 'src_user': str, 'src_computer': str, 'dst_computer': str, 'label': np.bool,
                                     'snapshot': int})
        anom_user_df = lanl_df[lanl_df['label'] == 1]
        anom_row_index = anom_user_df.index.to_list()
        print("Total Anom Edges: ", len(anom_row_index))
        print("Anom rows index: ", anom_row_index)

        # 从数据帧中提取异常记录的源用户并用set去重，并将这些用户存储在一个列表中
        anom_user = list(set(lanl_df.loc[anom_row_index, 'src_user'].tolist()))
        print("Anomalous Users: ", len(anom_user), anom_user)
        all_user = lanl_df.src_user.unique()
        print("total users: ", len(all_user))

        # anom_user = ['U748@DOM1', 'U1723@DOM1', 'U636@DOM1', 'U6115@DOM1', 'U620@DOM1']#, 'U737@DOM1', 'U825@DOM1', 'U1653@DOM1', 'U293@DOM1',

        # 提取正常用户
        norm_users = np.setdiff1d(all_user, anom_user).tolist()
        print("Norm users: ", len(norm_users))
        # 随机抽取与异常用户数量两倍相等的正常用户
        norm_users = random.sample(norm_users, len(anom_user) * 2)
        all_users = norm_users + anom_user
        print("all users: ", len(all_users), all_users)
        # 生成包含所有用户所在行的子集数据帧
        all_user_df = lanl_df[lanl_df['src_user'].isin(all_users)]
        all_user_df.to_csv("dataset/lanl/anom_full_2xuser_1hr.csv")

    # 生成节点标签矩阵
    def get_node_label(graphs, node_list):
        node_labels = []
        for G in graphs:
            # 初始化标签数组
            label = np.zeros((len(node_list), 1), dtype=np.float32)
            for n, data in G.nodes(data=True):
                # 将节点 n 的标签设置为节点属性 data['anom'] 的值
                label[node_list.index(str(n))] = data['anom']
            node_labels.append(label)
        node_labels = np.array(node_labels)
        # print("Node Label: ", node_labels.shape)
        return node_labels

    # 根据节点 ID 返回节点名称
    def get_node(node_map, n):
        for k, v in node_map.items():
            if v == n:
                return k
        return None

    # 生成具有回溯序列的训练数据
    def generate_seq_lookback(static_emb, lookback):
        X_train = []
        for sample in range(static_emb.shape[0]):
            for i in range(static_emb.shape[1] - lookback + 1):
                X_train.append(static_emb[sample, i:i + lookback, :])
        return np.array(X_train, dtype=np.float32)


    # 生成训练数据序列
    def generate_seq(static_emb):
        X_train, Y_train, = [], []
        for sample in range(static_emb.shape[0]):
            for i in range(static_emb.shape[1]):
                X_train.append(static_emb[sample, i, :])
        X_train = np.array(X_train, dtype=np.float32)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        return np.array(X_train, dtype=np.float32)

    # 将动态嵌入矩阵 dynamic_emb 回滚为图嵌入矩阵
    def rollback_seq(dynamic_emb, total_node, batch_size):
        graph_emb = []
        for node_idx in range(total_node):
            offset = (node_idx + 1) * batch_size
            node_emb = dynamic_emb[node_idx*batch_size:offset,:]
            graph_emb.append(node_emb)
        return np.array(graph_emb, dtype=np.float32)
