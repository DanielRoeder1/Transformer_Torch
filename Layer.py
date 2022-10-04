from torch import nn
import torch
from numpy import sqrt
import numpy as np


class MultiHeadAttentionFAST(nn.Module):
    """
    Calculate all attention heads using a single Dense layer (Original implementation)
    """
    def __init__(self, config, masked_attention = False):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.q_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False) 
        self.k_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False) 
        self.v_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.final_lin = nn.Linear(config.hidden_size, config.hidden_size, bias= False)
        self.drop = nn.Dropout(config.dropout_prob)

        self.num_heads = config.num_attention_heads 
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.masked_attention = masked_attention
    
    def forward(self, x, mask, kv = None):
        # In the case of the Encoder-Decoder attention layer k,v are set by the Encoder ouput
        Q = self.split_heads(self.q_layer(x))
        K = self.split_heads(self.k_layer(kv if torch.is_tensor(kv) else x)).transpose(2,3)
        V = self.split_heads(self.v_layer(kv if torch.is_tensor(kv) else x))

        scaled_attention_scores = torch.matmul(Q,K) / sqrt(Q.size(-1))  # [batch, num_heads,seq_len, seq_len]
        #print(f"Attention scores before masking {scaled_attention_scores}")
        scaled_attention_scores = scaled_attention_scores.masked_fill(mask == 0, float("-inf"))
        #print(f"Attention scores after masking {scaled_attention_scores}")
        
        attention_weights = nn.functional.softmax(scaled_attention_scores, dim= -1)
       # print(f"Mask {mask}")
        #print(f"Shape attention weigths {attention_weights.shape}")
        #print(f"attention weights {attention_weights}")
        #print("########")
        attention_results = self.combine_heads(torch.matmul(attention_weights, V)) # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, embed_dim]
        attention_results = self.final_lin(attention_results)
        attention_results = self.drop(attention_results)
        return attention_results

    def split_heads(self, x):
        seperate_heads = torch.reshape(x, x.shape[:-1]+(self.num_heads, self.head_dim))
        return seperate_heads.permute([0,2,1,3])
    
    def combine_heads(self,x):
        x = x.permute([0,2,1,3])
        a,b = x.shape[-2:]
        return torch.reshape(x, x.shape[:-2]+(a*b,))

class FeedForward(nn.Module):
    """
    The two feed forward layer found in each Encoder Block
    """
    def __init__(self, config) -> None:
        super().__init__()

        self.lin1 = nn.Linear(config.hidden_size, config.intermediate_dim)
        self.lin2= nn.Linear(config.intermediate_dim, config.hidden_size)
        self.drop = nn.Dropout(config.dropout_prob)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.drop(x)
        return x

class PositionalEmbedding(nn.Module):
    """
    Using the static sin and cosin embeddings proposed in the original Paper
    As these encodings are not learned we can calculate them once
    """
    def __init__(self, config) -> None:
        super().__init__()
        i = np.arange(config.hidden_size)[np.newaxis, :]
        pos = np.arange(config.seq_len)[:, np.newaxis]
        angles =  pos * (1  / np.power(10000., (2*i) / np.sqrt(config.hidden_size)))
        angles[:,0::2] = np.sin(angles[:,0::2])
        angles[:,1::2] = np.sin(angles[:,1::2])
        # Register as buffer to ensure its moved to gpu alongside model
        self.register_buffer("pos_enc", torch.from_numpy(angles).unsqueeze(0).float())

    def forward(self):
        return self.pos_enc  