from transformers import GPT2Config, GPT2Model
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


"""
The primary model to which apply the secondary model
"""
# Configure GPT-2 to output attentions
config = GPT2Config(
    n_layer=4,  # Number of layers
    n_head=4,   # Number of attention heads
    n_embd=128, # Embedding size
    output_attentions=True  # Enable attention outputs
)

# Load the model with this configuration
model = GPT2Model(config)

def get_attention(model, input_ids):
    # Get the attentions
    outputs = model(input_ids)
    # Extract attention activations
    attentions = outputs.attentions
    return attentions

def mean_head_aggregation(attentions):
    return attentions.mean(dim=1).detach().numpy()


"""
Function that creates a graph from the attention weights
"""

# Create the graph structure from the attention weights
def attention_to_graph(attention):
    # Get the number of nodes
    n = attention.shape[-1] # number of tokens

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes from attention
    for i in range(n):
        # TODO: weight dependend on the number of the token
        G.add_node(f'token_{i}', weight=1.0)
    
    # Add edges from attention
    for i in range(n):
        for j in range(n):
            if (j <= i): # attention masking
                G.add_edge(f'token_{i}', f'token_{j}', weight=attention[i, j])

    # TODO: Check for further aggregation for transformer input

    return G

"""
The GNN aggragating the attentions
"""
"""
this is used for num node feature in the GCNConv
pyg_data = from_networkx(mean_attention_graph)
pyg_data.x = torch.ones(pyg_data.num_nodes, 1)
"""

# construct aggregation Network for the attention graph
# This network just compresses the data in the attnention mechanism for further processing
class AggregationNetwork(torch.nn.Module):
    def __init__(self, num_node_feature, hidden_dim, dropout=0.2, adj_dropout=0.2): # TODO: hyperparameter tuning for dropout
        super(AggregationNetwork, self).__init__()
        self.conv1 = GCNConv(num_node_feature, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.adj_dropout = adj_dropout
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        edge_index, edge_mask = pyg_utils.dropout_edge(edge_index, p=self.adj_dropout, training=self.training)

        if edge_weight is not None:
            edge_weight = edge_weight[edge_mask]

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight)
        return x
    
"""
Linear Model to compress the aggregated attention
"""
# This compresses the attention graph into a smaller representation
class CompressionNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, compressed_dim, dropout=0.1):
        super(CompressionNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)  # Expand
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_dim, compressed_dim)  # Compress

    def forward(self, x):
        x = self.fc1(x)  # (num_tokens, 1) → (num_tokens, compressed_dim*2)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (num_tokens, compressed_dim*2) → (num_tokens, compressed_dim)
        return x
    
"""
Transformer Model to predict a reward based on alignment from the compressed attention
"""

# Tries to predict the current token from the attention graph
# This should allow to predict deviations from network intentions, during runtime
# And allow for better interpretability
class AttentionToRewardEncoder(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(AttentionToRewardEncoder, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)  # Project attention features
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))  # Positional encoding

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.output_layer = nn.Linear(d_model, 1)  # Predict reward

    def forward(self, x):
        x = self.embedding(x)  
        x = x + self.pos_encoder[:, :x.size(1), :]  # Add positional info

        x = self.transformer_encoder(x)  # No causal mask needed
        x = x.mean(dim=1)  # Pool over token representations (global understanding)

        x = self.output_layer(x)  # Predict token logits  
        
        return x  # Output shape: (reward)
    
""""
Math Dataset for Alignment
"""

# Function to read problems and solutions
def load_math_data(problem_filename="math_problems.txt", solution_filename="math_solutions.txt"):
    problems = [line.strip() for line in open(problem_filename, "r")]
    solutions = [line.strip() for line in open(solution_filename, "r")]
    
    df = pd.DataFrame({"problem": problems, "solution": solutions})
    return df

# PyTorch Dataset class
class MathDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=20):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        problem = self.dataframe.iloc[idx]["problem"]
        solution = self.dataframe.iloc[idx]["solution"]
        
        problem_enc = self.tokenizer(problem, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        solution_enc = self.tokenizer(solution, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        
        return problem_enc.input_ids.squeeze(), solution_enc.input_ids.squeeze()

# Example usage
data = load_math_data("math_problems.txt", "math_solutions.txt")
print(data.head())

# Attention Reward Dataset
class AttentionRewardDataset(Dataset):
    def __init__(self, dataframe, max_length=20):
        self.dataframe = dataframe
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        problem = self.dataframe.iloc[idx]["problem"]
        solution = self.dataframe.iloc[idx]["solution"]
        
        problem_enc = self.tokenizer(problem, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        solution_enc = self.tokenizer(solution, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        
        return problem_enc.input_ids.squeeze(), solution_enc.input_ids.squeeze()
    
    def extract_attention(self, model):
        # Extract attention from model
        pass

    def extract_results(self, model):
        # Extract results from model
        pass

    def new_datapoint(self, solution):
        # Add new datapoint
        pass
        
