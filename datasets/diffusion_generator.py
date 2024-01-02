import random
import networkx as nx
#from readnpz import load_graph_features
import matplotlib.pyplot as plt
import numpy as np
import EoN
import pickle
#Note that epidemics in networkx is only available after 2.6.0
# 使用 NetworkX 内置的 IC 模型模拟传播
npz_file_path = 'E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/Enron_new/graphs.npz'
data=np.load(npz_file_path, allow_pickle=True, encoding='latin1')
for key,graph_list in data.items():
    for i, graph in enumerate(graph_list):
        tmax = 100
        iterations = 5 #run 5 simulations
        tau = 0.1           #transmission rate per edge
        gamma = 1.0    #recovery rate per edge
        dataset_length=100
        thorough_dataset=[]
        for i in range(dataset_length):
            all_nodes = list(graph.nodes())
            initial_infected = random.sample(all_nodes, k=10)
            print(initial_infected)
            source=[0] * len(graph.nodes())
            for index in initial_infected:
                source[index] = 1 
            value_dict=[]
            for counter in range(iterations): #run simulations
                result = EoN.fast_SIS(graph, tau, gamma, initial_infecteds=initial_infected, tmax = tmax, return_full_data=True)
                prog_list=[]
                for time in range(int(tmax/10)):
                    result_tmp=result.get_statuses(time=10*(time+1)-1)
                    result_values=list(result_tmp.values())
                    modified_list = [0 if element == 'S' else 1 for element in result_values]
                    prog_list.append(modified_list)
                if counter == 0:
                    plt.plot(result.t(), result.I(), color = 'k', alpha=0.3, label='Simulation')
                plt.plot(result.t(), result.I(), color = 'k', alpha=0.3)
                value_dict.append(prog_list)
            val=np.array(value_dict)
            avg_arr=np.mean(val, axis=0)
            #t, S, I = EoN.SIS_compact_pairwise_from_graph(graph, tau, gamma, rho=rho, tmax=tmax)
            #plt.plot(t, I, '--', label = 'Compact pairwise', linewidth = 5)
            result_array = np.vstack([source, avg_arr])
            print(result_array.shape)
            thorough_dataset.append(result_array)
        with open(f'E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/Enron_new/saved/saved_movielens_{i}.pkl', 'wb') as f:
            pickle.dump(graph, f)  # Save the MultiGraph
            pickle.dump(tau, f)  # Save parameter 1
            pickle.dump(gamma, f)  # Save parameter 2
            pickle.dump(thorough_dataset, f)  # Save the multi-dimensional list
        
            