from lib2to3.pgen2 import token
from turtle import forward
import torch
from numpy import sqrt
from torch import nn
from transformers import PretrainedConfig, AutoTokenizer

from utils import get_praram_count
from Layer import MultiHeadAttentionFAST, FeedForward, PositionalEmbedding


class Transformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embedding_layer = EmbeddingBlock(config)
        # Can create a nn.Sequential when using contents of list *[]
        self.encoder_stack = nn.Sequential(*[EncoderBlock(config) for _ in range(config.num_encoder_blocks)])
        self.decoder_stack = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_blocks)])

        # In the publication the final lin layer shares the weights with the embedding layer
        self.final_lin = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_lin.weight = self.embedding_layer.token_embeddings.weight
        self.softmax = nn.Softmax(dim = -1)
    
        self.pad_id = config.pad_idx
        self.register_buffer("attention_mask", torch.tril(torch.ones(config.seq_len, config.seq_len)).view(1,1,config.seq_len, config.seq_len).bool())

    def forward(self,src_ids, trgt_ids):
        src_mask = self.get_pad_mask(src_ids)
        trgt_mask = self.get_pad_mask(trgt_ids) & self.attention_mask

        src_embed = self.embedding_layer(src_ids)
        trgt_embed = self.embedding_layer(trgt_ids)

        enc_out = self.encoder_stack((src_embed,src_mask))

        for dec in self.decoder_stack:
            trgt_embed = dec(trgt_embed,trgt_mask, enc_out, src_mask)
        
        trgt_embed = self.final_lin(trgt_embed)
        #trgt_embed = self.softmax(trgt_embed)
        return trgt_embed 
    
    def get_pad_mask(self,input_ids):
        # This expands the dim to [batch, 1,1,seq_len]
        return (input_ids != self.pad_id)[:,None,None,:]

class EncoderBlock(nn.Module):
    """
    Enconder Block as outlined in Attention is all you need paper
    Using Post Layer normalization
    """
    def __init__(self, config) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.multi_attention = MultiHeadAttentionFAST(config)
        self.feed_forward = FeedForward(config)

    def forward(self, input):
        src, pad_mask = input
        src = src + self.multi_attention(src,pad_mask)
        src = self.norm1(src)
        src = src + self.feed_forward(src)
        src = self.norm2(src)
        return (src, pad_mask) 


class DecoderBlock(nn.Module):
    """
    Decoder Block as outlined in Attention is all you need paper
    Using Post Layer normalization
    dec_mask: Combines causal masking and padding mask on decoder input
    enc_mask: Padding mask on encoder input
    """
    def __init__(self, config) -> None:
        super().__init__()
        
        self.masked_attention = MultiHeadAttentionFAST(config, masked_attention=True)
        self.enc_dec_attention = MultiHeadAttentionFAST(config)
        self.feed_forward = FeedForward(config)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)

    def forward(self, trgt_embed,trgt_mask, enc_output, enc_mask):
        trgt_embed = trgt_embed + self.masked_attention(trgt_embed, trgt_mask)
        trgt_embed = self.norm1(trgt_embed)
        trgt_embed = trgt_embed + self.enc_dec_attention(trgt_embed,enc_mask,enc_output)
        trgt_embed = self.norm2(trgt_embed)
        trgt_embed = trgt_embed + self.feed_forward(trgt_embed)
        trgt_embed = self.norm3(trgt_embed)
        return trgt_embed

class EmbeddingBlock(nn.Module):
    """
    Receives input_ids from tokenizer and creates embedding based on id and position in sequence
    """
    def __init__(self, config) -> None:
        super().__init__()

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size) 
        self.positional_embeddings = PositionalEmbedding(config)
        self.drop = nn.Dropout(config.dropout_prob)

        self.scale_factor = sqrt(config.hidden_size)
    
    def forward(self, input_ids):
        token_embeddings = self.token_embeddings(input_ids) 
        # Scale the embeddings by the sqrt of the hidden size -> original publication
        token_embeddings *= self.scale_factor
        position_embeddings = self.positional_embeddings()

        embeddings = token_embeddings + position_embeddings
        embeddings = self.drop(embeddings)
        return embeddings
        

if __name__ == "__main__":
    # We are using this wmt tokenizer as it shares a vocab for english and german
    tokenizer = AutoTokenizer.from_pretrained("rossanez/t5-small-finetuned-de-en-lr2e-4")
    tokenizer.add_special_tokens({"bos_token": "<start>"})

    config = PretrainedConfig.from_json_file("config.json")
    config.update({"vocab_size": len(tokenizer), "pad_idx": tokenizer.pad_token_id})

    transformer = Transformer(config)

    out = tokenizer(["A test src sentence", "Ein Test source Satz"], add_special_tokens= False, max_length= config.seq_len, padding= "max_length", return_token_type_ids= False, return_attention_mask= False, return_tensors="pt")
    src = out.input_ids[0].unsqueeze(0)
    trgt = torch.cat((torch.Tensor([tokenizer.bos_token_id]).unsqueeze(0).long(), out.input_ids[1].unsqueeze(0)), dim = 1)
    trgt = trgt[:,:-1]

    output = transformer(src, trgt)

    print(f"Trainable parameters Transformer: {get_praram_count(transformer)}")