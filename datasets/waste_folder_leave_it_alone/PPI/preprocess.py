import csv
import time
import networkx as nx
from datetime import datetime
import ast
import numpy as np
# 读取CSV文件并处理时间戳
def read_csv_and_create_multigraph(csv_file_path):
    multigraph = nx.MultiGraph()

    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        for row in csv_reader:
            # 获取source、target和timestamp
            source, target, timestamp_str = row[-1], row[-2], row[-5]
            if timestamp_str:
                timestamp_str=ast.literal_eval(timestamp_str[8:])
                #print(type(timestamp_str))
                timestamp = datetime(*timestamp_str)
                # 将timestamp转换为时间结构
                # 添加边到multigraph，使用时间戳作为边属性
                multigraph.add_edge(source, target, **{'date':timestamp})

    return multigraph

# 示例用法
csv_file_path = 'E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/PPI/interactions_with_dates.csv'  # 替换为你的CSV文件路径
result_multigraph = read_csv_and_create_multigraph(csv_file_path)

# 打印结果
for edge in result_multigraph.edges(data=True):
    print(edge)

def divide_multigraph(original_multigraph, num_divisions):
    edges_sorted = sorted(original_multigraph.edges(data=True), key=lambda x: x[2]['date'])

    # Calculate the number of edges in each sub-multigraph
    total_edges = len(edges_sorted)
    edges_per_subgraph = total_edges // 10

    # Divide the edges into 10 sub-multigraphs
    subgraphs = []
    for i in range(num_divisions):
        start_idx = i * edges_per_subgraph
        end_idx = (i + 1) * edges_per_subgraph if i < 9 else total_edges
        subgraph_edges = edges_sorted[start_idx:end_idx]
        #print(subgraph_edges)
        # Create a sub-multigraph
        subgraph = nx.MultiDiGraph(subgraph_edges)

        # Append the sub-multigraph to the list
        subgraphs.append(subgraph)
    #print(subgraphs)
    return subgraphs


num_divisions = 10
divided_multigraphs = divide_multigraph(result_multigraph, num_divisions)

# Save each multigraph to the NPZ file
#final_dict={}
#final_dict['graph']=divided_multigraphs
final_dict={'graph': divided_multigraphs}
np.savez('E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/PPI/PPI.npz', **final_dict)
