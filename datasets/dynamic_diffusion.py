import random
import networkx as nx
#from readnpz import load_graph_features
import matplotlib.pyplot as plt
import numpy as np
import EoN
import pickle
#Note that epidemics in networkx is only available after 2.6.0
# 使用 NetworkX 内置的 IC 模型模拟传播
npz_file_path = 'E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/bitcoin-alpha/graphs.npz'

# For the dynamic diffusion, we use a different propagation method, IC.

data=np.load(npz_file_path, allow_pickle=True, encoding='latin1')

tau=0.2

def simulate_diffusion(graph, seed_nodes, current_time):
    active_nodes = set(seed_nodes)
    new_nodes_activated = True

    while new_nodes_activated:
        new_nodes_activated = False
        for node in list(active_nodes):
            for neighbor in graph.neighbors(node):
                # Check if the edge is active and if the neighbor is not already active
                if graph[node][neighbor][0]['date'] <= current_time and neighbor not in active_nodes:
                    # Check if the propagation probability is successful
                    if random.random() <= tau:
                        active_nodes.add(neighbor)
                        new_nodes_activated = True
    return active_nodes

for key,graph_list in data.items():
    for j, graph in enumerate(graph_list):
        #Make id integer
        mapping = {node: idx for idx, node in enumerate(graph.nodes())}
        # Create a new graph with nodes relabeled to integers
        graph = nx.relabel.relabel_nodes(graph, mapping)

        #print(graph.edges())
        min_start_time = min([attr['date'] for _, _, attr in graph.edges(data=True)])
        max_end_time = max([attr['date'] for _, _, attr in graph.edges(data=True)])

        # Calculate the time step
        dataset_length=200
        num_pieces = 100
        time_step = (max_end_time - min_start_time) / num_pieces

        # Iterate through the time pieces
        time_list=[]
        for i in range(num_pieces):
            current_time = min_start_time + i * time_step
            time_list.append(current_time)

        #print(time_list)
        #(time_list)
        iterations = 5 #run 5 simulations

        thorough_dataset=[]
        for i in range(dataset_length):
            all_nodes = list(graph.nodes())
            initial_infected = random.sample(all_nodes, k=50)
            #print(initial_infected)
            source=[0] * len(graph.nodes())
            for index in initial_infected:
                source[index] = 1 
            value_dict=[]
            for counter in range(iterations):
                temp_arr=[]
                for idx, current_time in enumerate(time_list):
                    result=simulate_diffusion(graph, initial_infected, current_time)
                    if idx % 10 ==0:
                        source=[0] * len(graph.nodes())
                        for index in result:
                            source[index] = 1 
                        temp_arr.append(source)
                    initial_infected=result
                counts_per_list = [sum(1 for element in sublist if element >= 0.5) for sublist in temp_arr]
                numbers_list = list(range(0, 10))
                if counter == 0:
                    plt.plot(numbers_list, counts_per_list, color = 'k', alpha=0.3, label='Simulation')
                plt.plot(numbers_list, counts_per_list, color = 'k', alpha=0.3)
                #plt.show()
                #print(a)
                value_dict.append(temp_arr)
            val=np.array(value_dict)
            avg_arr=np.mean(val, axis=0)
            result_array = np.vstack([source, avg_arr])
            print(result_array.shape)
            thorough_dataset.append(result_array)
        with open(f'E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/bitcoin-alpha/dynamic/dynamic_{j}.pkl', 'wb') as f:
            pickle.dump({
            'multi_graph': graph,
            'tau': tau,
            'influ_list': thorough_dataset
            }, f)
            