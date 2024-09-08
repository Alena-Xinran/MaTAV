# import torch
# import torch.nn as nn
# import torch.nn.functional as F




# '''
# Bidirectional cross-attention layers.
# '''
# class BidirectionalCrossAttention(nn.Module):

#     def __init__(self, model_dim, Q_dim, K_dim, V_dim):
#         super().__init__()

#         self.query_matrix = nn.Linear(model_dim, Q_dim)
#         self.key_matrix = nn.Linear(model_dim, K_dim)
#         self.value_matrix = nn.Linear(model_dim, V_dim)
    

#     def bidirectional_scaled_dot_product_attention(self, Q, K, V):
#         score = torch.bmm(Q, K.transpose(-1, -2))
#         scaled_score = score / (K.shape[-1]**0.5)
#         attention = torch.bmm(F.softmax(scaled_score, dim = -1), V)

#         return attention
    

#     def forward(self, query, key, value):
#         Q = self.query_matrix(query)
#         K = self.key_matrix(key)
#         V = self.value_matrix(value)
#         attention = self.bidirectional_scaled_dot_product_attention(Q, K, V)

#         return attention



# '''
# Multi-head bidirectional cross-attention layers.
# '''
# class MultiHeadAttention(nn.Module):

#     def __init__(self, num_heads, model_dim, Q_dim, K_dim, V_dim):
#         super().__init__()

#         self.num_heads = num_heads
#         self.attention_heads = nn.ModuleList(
#             [BidirectionalCrossAttention(model_dim, Q_dim, K_dim, V_dim) for _ in range(self.num_heads)]
#         )
#         self.projection_matrix = nn.Linear(num_heads * V_dim, model_dim)
    

#     def forward(self, query, key, value):
#         heads = [self.attention_heads[i](query, key, value) for i in range(self.num_heads)]
#         multihead_attention = self.projection_matrix(torch.cat(heads, dim = -1))

#         return multihead_attention



# '''
# A feed-forward network, which operates as a key-value memory.
# '''
# class Feedforward(nn.Module):

#     def __init__(self, model_dim, hidden_dim, dropout_rate):
#         super().__init__()

#         self.linear_W1 = nn.Linear(model_dim, hidden_dim)
#         self.linear_W2 = nn.Linear(hidden_dim, model_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_rate)
    

#     def forward(self, x):
#         return self.dropout(self.linear_W2(self.relu(self.linear_W1(x))))



# '''
# Residual connection to smooth the learning process.
# '''
# class AddNorm(nn.Module):

#     def __init__(self, model_dim, dropout_rate):
#         super().__init__()

#         self.layer_norm = nn.LayerNorm(model_dim)
#         self.dropout = nn.Dropout(dropout_rate)

    
#     def forward(self, x, sublayer):
#         output = self.layer_norm(x + self.dropout(sublayer(x)))

#         return output



# '''
# MultiAttn is a multimodal fusion model which aims to capture the complicated interactions and 
# dependencies across textual, audio and visual modalities through bidirectional cross-attention layers.
# MultiAttn is made up of three sub-components:
# 1. MultiAttn_text: integrate the textual modality with audio and visual information;
# 2. MultiAttn_audio: incorporate the audio modality with textual and visual information;
# 3. MultiAttn_visual: fuse the visual modality with textual and visual cues.
# '''
# class MultiAttnLayer(nn.Module):

#     def __init__(self, num_heads, model_dim, hidden_dim, dropout_rate):
#         super().__init__()

#         Q_dim = K_dim = V_dim = model_dim // num_heads
#         self.attn_1 = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim)
#         self.add_norm_1 = AddNorm(model_dim, dropout_rate)
#         self.attn_2 = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim)
#         self.add_norm_2 = AddNorm(model_dim, dropout_rate)
#         self.ff = Feedforward(model_dim, hidden_dim, dropout_rate)
#         self.add_norm_3 = AddNorm(model_dim, dropout_rate)


#     def forward(self, query_modality, modality_A, modality_B):
#         attn_output_1 = self.add_norm_1(query_modality, lambda query_modality: self.attn_1(query_modality, modality_A, modality_A))
#         attn_output_2 = self.add_norm_2(attn_output_1, lambda attn_output_1: self.attn_2(attn_output_1, modality_B, modality_B))
#         ff_output = self.add_norm_3(attn_output_2, self.ff)

#         return ff_output



# '''
# Stacks of MultiAttn layers.
# '''
# class MultiAttn(nn.Module):

#     def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
#         super().__init__()

#         self.multiattn_layers = nn.ModuleList([
#             MultiAttnLayer(num_heads, model_dim, hidden_dim, dropout_rate) for _ in range(num_layers)])


#     def forward(self, query_modality, modality_A, modality_B):
#         for multiattn_layer in self.multiattn_layers:
#             query_modality = multiattn_layer(query_modality, modality_A, modality_B)
        
#         return query_modality



# class MultiAttnModel(nn.Module):

#     def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
#         super().__init__()

#         self.multiattn_text = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
#         self.multiattn_audio = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
#         self.multiattn_visual = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)

    
#     def forward(self, text_features, audio_features, visual_features):
#         f_t = self.multiattn_text(text_features, audio_features, visual_features)
#         f_a = self.multiattn_audio(audio_features, text_features, visual_features)
#         f_v = self.multiattn_visual(visual_features, text_features, audio_features)

#         return f_t, f_a, f_v
import torch
import torch.nn as nn
from transformers import Blip2Model, Blip2Config
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim, head_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.model_dim = model_dim

        # Ensure the total dimension of keys/values matches the model dimension
        self.key_dim = self.value_dim = head_dim
        self.total_dim = self.key_dim * num_heads

        # Define layers
        self.query_layer = nn.Linear(model_dim, self.total_dim)
        self.key_layer = nn.Linear(model_dim, self.total_dim)
        self.value_layer = nn.Linear(model_dim, self.total_dim)
        self.output_layer = nn.Linear(self.total_dim, model_dim)

    def forward(self, queries, keys, values):
        batch_size = queries.shape[0]

        # Shape: (batch_size, num_heads, seq_length, head_dim)
        Q = self.query_layer(queries).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_layer(keys).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_layer(values).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention using scaled dot product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.key_dim)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.total_dim)
        output = self.output_layer(context)

        return output

class MultiAttnModel(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super(MultiAttnModel, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        config = Blip2Config()
        self.blip2 = Blip2Model.from_pretrained("/home/xrl/MultiEMO-ACL2023-main/Model/blip2-opt-2.7b", torch_dtype=torch.float32)
        
        # Define multi-head attention modules
        self.attention_modules = nn.ModuleList([
            MultiHeadAttention(num_heads, model_dim, model_dim // num_heads) for _ in range(num_layers)
        ])
        
        # Additional layers for text and audio processing
        self.text_processor = nn.Linear(768, model_dim)
        self.audio_processor = nn.Linear(768, model_dim)
        
        # Assume the qformer is suitable for visual modality
        self.qformer_visual = copy.deepcopy(self.blip2).to('cuda:0')
        self.qformer_text = copy.deepcopy(self.blip2).to('cuda:1')
        self.qformer_audio = copy.deepcopy(self.blip2).to('cuda:2')

    def forward(self, text_features, audio_features, visual_features):
        # Process each modality
        text_features = self.text_processor(text_features).to('cuda:1')
        audio_features = self.audio_processor(audio_features).to('cuda:2')
        visual_features = visual_features.to('cuda:0')

        # Apply multi-head attention to each modality
        for mod in (self.qformer_text, self.qformer_audio, self.qformer_visual):
            text_features = mod(text_features).last_hidden_state.mean(dim=1)
            audio_features = mod(audio_features).last_hidden_state.mean(dim=1)
            visual_features = mod(visual_features).last_hidden_state.mean(dim=1)

        return text_features, audio_features, visual_features
