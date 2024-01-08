from baselines.LPSI import LPSI_ALG
from dataloader import CustomDataset
import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


pickle_file_path = "E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/bitcoin-alpha/saved/saved_8.pkl"
custom_dataset = CustomDataset(pickle_file_path)
adjacency_matrix = nx.to_numpy_matrix(custom_dataset['multi_graph'])

print(type(custom_dataset['multi_graph']))
print(custom_dataset['tau'])
print(np.array(custom_dataset['influ_list']).shape)
print(custom_dataset['influ_list'][0])
print(custom_dataset['influ_list'][-1])

#Now test on the whole dataset.

dataset=np.array(custom_dataset['influ_list'])

dataset_sereis=[]
for i in range(dataset.shape[0]):
    snapshot_series=dataset[i]
    #print(snapshot_series.shape)
    series_prediction=[]
    for j in range(snapshot_series.shape[0]-1):
        #print(len(snapshot_series[j+1]))
        #print(adjacency_matrix.shape)
        Pred=LPSI_ALG(Y=snapshot_series[j+1], adjM=adjacency_matrix)
        Pred=np.array(Pred).flatten()
        series_prediction.append(Pred)
        #print(Pred)
        #print(snapshot_series[0])
        #print(Pred.shape)
        #print(snapshot_series[0].shape)
        #print(f1_score(Pred,snapshot_series[0]))
        #print(a)
        #print(snapshot_series[0])
    dataset_sereis.append(series_prediction)

ground_truth=dataset[:, 0, :]
#print(ground_truth.shape)
dataset_sereis=np.array(dataset_sereis)
# Assign weights based on timestamps (higher weights for earlier timestamps)
weights = np.arange(1, dataset_sereis.shape[1] + 1)

for m in range(dataset_sereis.shape[1]):
    current_slice = dataset_sereis[:, m, :]
    #print(current_slice.shape)
    f1_scores=[]
    for true_labels, predicted_labels in zip(ground_truth, current_slice):
        predicted_labels=[1 if x >= 0.5 else 0 for x in predicted_labels]
        f1_scores.append(f1_score(true_labels,predicted_labels))
    #f1_scores = [f1_score(true_labels, predicted_labels) for true_labels, predicted_labels[1 if x >= 0.5 else 0 for x in predicted_labels] in zip(ground_truth, current_slice)]
    # Calculate the average F1 score
    average_f1_score = np.mean(f1_scores)
    print("m ,f1_score", m, average_f1_score)


decay_factor = 0.05  # Adjust this factor based on the decay rate you want
weights = [1 * np.exp(-decay_factor * i) for i in range(dataset_sereis.shape[1])]
#weights = [decay_factor ** i for i in range(dataset_sereis.shape[1])]
weights /= np.sum(weights)

def weighted_sum(predictions, weights):
    # Ensure predictions and weights have the same length
    assert len(predictions) == len(weights), "Number of predictions and weights must be the same"

    # Multiply each element of each prediction by its corresponding weight
    weighted_predictions = [prediction * weight for prediction, weight in zip(predictions, weights)]

    # Sum up the weighted predictions element-wise
    total_weighted_sum = np.sum(weighted_predictions, axis=0)

    return total_weighted_sum

Tim_pred=[]
for t in range(dataset_sereis.shape[0]):
    current_slice = dataset_sereis[t, :, :]
    pred_result=weighted_sum(current_slice, weights)
    predicted_labels=[1 if x >= 0.5 else 0 for x in pred_result]
    Tim_pred.append(predicted_labels)

f1_scores=[]
for true_labels, predicted_labels in zip(ground_truth, Tim_pred):
        predicted_labels=[1 if x >= 0.5 else 0 for x in predicted_labels]
        f1_scores.append(f1_score(true_labels,predicted_labels))
    #f1_scores = [f1_score(true_labels, predicted_labels) for true_labels, predicted_labels[1 if x >= 0.5 else 0 for x in predicted_labels] in zip(ground_truth, current_slice)]
    # Calculate the average F1 score
average_f1_score = np.mean(f1_scores)
print("weighted f1_score", average_f1_score)



"""
m ,f1_score 0 0.07586051213632711
m ,f1_score 1 0.09008604920615518
m ,f1_score 2 0.09244070718446276
m ,f1_score 3 0.09114514277559252
m ,f1_score 4 0.09507205349859658
m ,f1_score 5 0.0885964559507677
m ,f1_score 6 0.09126140944674663
m ,f1_score 7 0.09257775075402983
m ,f1_score 8 0.0935013221286129
m ,f1_score 9 0.09367233781053116

"""

#With weighted decay:
"""
m ,f1_score 0 0.0871820795969907
m ,f1_score 1 0.09171620763988228
m ,f1_score 2 0.09489630460004062
m ,f1_score 3 0.09548075727243457
m ,f1_score 4 0.09524771715469772
m ,f1_score 5 0.09392085502020407
m ,f1_score 6 0.09706497973716456
m ,f1_score 7 0.0977694023366023
m ,f1_score 8 0.0929128162161554
m ,f1_score 9 0.09801937961064779
weighted f1_score 0.1249931192910633
"""