import numpy as np

def load_graph_features(npz_file_path):
    try:
        # Load the .npz file
        with np.load(npz_file_path, allow_pickle=True, encoding='latin1') as data:
            # Check if 'features' and 'labels' arrays are present in the file
            print("Keys in the .npz file:")
            for key in data.keys():
                print(key)

            # Access and display the corresponding values
            print("\nContent of the .npz file:")
            for key, value in data.items():
                print(f"{key}:")
                print(value)
                for i, graph in enumerate(value):
                    print(f"Graph {i + 1}:")

                    # Print nodes
                    nodes = graph.nodes()
                    print(f"Nodes: {list(nodes)}")  # Convert nodes to a list for printing

                    # Print edges with attributes
                    edges = graph.edges(data=True)
                    print("Edges:")
                    for edge in edges:
                        print(edge)

                    print("-" * 30)

                print("\n" + "-" * 30)
    except IOError as e:
        print(f"Error loading the .npz file: {e}")
        return None, None

# Example usage:
#npz_file_path = 'E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/yelp_new/graphs.npz'
#npz_file_path = 'E:/summer_intern/Hua_zheng_Wang/source_localization/DySL/datasets/Movielens-10M/graphs.npz'
npz_file_path = 'E:\summer_intern\Hua_zheng_Wang\source_localization\DySL\datasets\Enron_new\graphs.npz'
graph_features, labels = load_graph_features(npz_file_path)

if graph_features is not None and labels is not None:
    print(f"Graph Features shape: {graph_features.shape}")
    print(f"Labels shape: {labels.shape}")
    # You can now use 'graph_features' and 'labels' in your further analysis or processing
