import pickle
import os
from torch.utils.data import Dataset, DataLoader
import networkx as nx
class CustomDataset(Dataset):
    def __init__(self, pickle_file_path):
        self.data = self.load_data(pickle_file_path)

    def load_data(self, pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
        #print(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # You can process the sample if needed
        return sample

# Example usage:
"""
if __name__ == "__main__":
    pickle_file_path = "E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/bitcoin-alpha/saved/saved_1.pkl"
    custom_dataset = CustomDataset(pickle_file_path)
    
    # Create a DataLoader
    batch_size = 32
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    # Iterate through the DataLoader
""" 