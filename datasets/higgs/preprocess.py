import networkx as nx
from datetime import datetime
import time
import numpy as np
# Function to convert timestamp to datetime object
def parse_timestamp(timestamp_str):
    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

# Read the edges from the text file
file_path = "E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/higgs/higgs-activity_time.txt"  # Update with your file path
edges = []
multigraph = nx.MultiGraph()

with open(file_path, 'r') as file:
    for line in file:
        # Assuming the format is "node1 node2 timestamp"
        node1, node2, timestamp_str, _ = line.strip().split()
        timestamp = int(timestamp_str)
        struct_time_obj = time.gmtime(timestamp)
        time_struct = time.strftime('%Y-%m-%d %H:%M:%S', struct_time_obj)
        timestamp = datetime.strptime(time_struct, '%Y-%m-%d %H:%M:%S')  # Adjust the format based on your actual timestamp format
        # 添加边到multigraph，使用时间戳作为边属性
        #print(timestamp)
        #edge=(source, target, **{'date':timestamp})
        multigraph.add_edge(node1, node2, **{'date':timestamp})

for edge in multigraph.edges(data=True):
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
        subgraph = nx.MultiGraph(subgraph_edges)

        # Append the sub-multigraph to the list
        subgraphs.append(subgraph)
    #print(subgraphs)
    return subgraphs


num_divisions = 10
divided_multigraphs = divide_multigraph(multigraph, num_divisions)

# Save each multigraph to the NPZ file
#final_dict={}
#final_dict['graph']=divided_multigraphs
final_dict={'graph': divided_multigraphs}
np.savez('E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/higgs/graphs.npz', **final_dict)
