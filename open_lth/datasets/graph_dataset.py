import torch
from torch_geometric.data import Data, InMemoryDataset
import networkx as nx
from torch_geometric.utils import from_networkx
import torch_geometric.utils.to_dense_adj as to_dense_adj
from collections import deque
from torch_geometric.utils import from_networkx
from datasets import base_graph
from tqdm import tqdm
from platforms.platform import get_platform

def classify_graph_BFS(num_nodes, x, edge_index, dist):
  # Check whether there are two vertices with the same color at a <=dist distance
  adj_mat = to_dense_adj(edge_index=edge_index, max_num_nodes=num_nodes).squeeze(dim=0)
  ans = torch.zeros(num_nodes, dtype=torch.long)
  for start in range(num_nodes):
    if x[start]==0:
      continue
    visited = torch.zeros(num_nodes)
    queue = deque([start])
    distances = torch.zeros(num_nodes)
    visited[start] = 1
    while queue:
      vertex = queue.popleft()
      if distances[vertex] > dist:
        break
      if start!=vertex and x[start] == x[vertex]:
        ans[start] = 1
        break
      for neighbour in range(num_nodes):
        if vertex!=neighbour and adj_mat[vertex][neighbour]==1 and visited[neighbour]==0:
          distances[neighbour]=distances[vertex] + 1
          visited[neighbour] = 1
          queue.append(neighbour)
  return ans


class Dataset(InMemoryDataset):
    def __init__(self, root="/graph_dataset/.", num_graphs=10, num_nodes=10, k=2, num_class=2, num_colors=3, dist=3, transform=None, pre_transform=None):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.k = k
        self.num_class = num_class
        self.num_colors = num_colors
        self.dist = dist
        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
      
    def cuda(self):
        self.x = self.x.to(get_platform().torch_device)
        self.edge_index = self.edge_index.to(get_platform().torch_device)
        self.y = self.y.to(get_platform().torch_device)
        return self
        
                
    def process(self):
        data_list = []

        for ind in tqdm(range(self.num_graphs)):
            # Generate a k-regular graph
            G = nx.random_regular_graph(self.k, self.num_nodes)
            edge_index = from_networkx(G).edge_index
            x = torch.randint(0, self.num_colors+1, (self.num_nodes, 1))
            x = x.to(torch.float32)
            # Assign a random label to the graph
            y = classify_graph_BFS(self.num_nodes, x, edge_index, self.dist)

            # Apply training, validation and test masks; 60% graphs are for
            #   training, 20% for validation and 20% for testing
            """
            if ind < self.num_graphs * 0.6:
              train_mask = torch.ones(self.num_nodes, dtype=torch.bool)
              val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
              test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            elif ind < self.num_graphs * 0.8:
              train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
              val_mask = torch.ones(self.num_nodes, dtype=torch.bool)
              test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            else:
              train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
              val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
              test_mask = torch.ones(self.num_nodes, dtype=torch.bool)
            """

            # Create a data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def num_train_examples(self): return int(self.num_graphs*0.8)
    
    def num_test_examples(self): return int(self.num_graphs*0.2)

    def num_classes(self): return self.num_classes

    def get_train_set(root, num_graphs, num_nodes, k, num_class, num_colors, dist, use_augmentation=0):
        # No augmentation for GraphClassificationDataset.
        train_set=Dataset(root=root, num_graphs=num_graphs, num_nodes=num_nodes, k=k, num_class=num_class, num_colors=num_colors, dist=dist)
        train_set = train_set[:train_set.num_train_examples()];
        return train_set

    def get_test_set(root, num_graphs, num_nodes, k, num_class, num_colors, dist, use_augmentation=0):
        # No augmentation for GraphClassificationDataset.
        test_set=Dataset(root=root, num_graphs=num_graphs, num_nodes=num_nodes, k=k, num_class=num_class, num_colors=num_colors, dist=dist)
        test_set = test_set[test_set.num_train_examples():]
        return test_set

DataLoader = base_graph.DataLoader
