import csv
import time
import networkx as nx

# 读取CSV文件并处理时间戳
def read_csv_and_create_multigraph(csv_file_path):
    multigraph = nx.MultiGraph()

    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            # 获取source、target和timestamp
            source, target, timestamp_str = row[0], row[1], row[-1]

            # 将timestamp转换为时间结构
            timestamp = int(timestamp_str)
            time_struct = time.gmtime(timestamp)

            # 添加边到multigraph，使用时间戳作为边属性
            multigraph.add_edge(source, target, timestamp=timestamp, time_struct=time_struct)

    return multigraph

# 示例用法
csv_file_path = 'E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/bitcoin-alpha/soc-sign-bitcoinalpha.csv'  # 替换为你的CSV文件路径
result_multigraph = read_csv_and_create_multigraph(csv_file_path)

# 打印结果
for edge in result_multigraph.edges(data=True):
    print(f"Edge: {edge[0]} - {edge[1]}, Timestamp: {edge[2]['timestamp']}, TimeStruct: {edge[2]['time_struct']}")
