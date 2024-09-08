import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from tqdm import tqdm
import math
import os
import urllib.request
from zipfile import ZipFile
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class S6(nn.Module):
    def __init__(self, d_model, state_size, device):
        super(S6, self).__init__()
        # Linear layers
        self.fc1 = nn.Linear(d_model, d_model).to(device)
        self.fc2 = nn.Linear(d_model, state_size).to(device)
        self.fc3 = nn.Linear(d_model, state_size).to(device)

        # d_model and state_size are now properties of the class
        self.d_model = d_model
        self.state_size = state_size
        self.device = device

        # Learnable matrix A normalized along the last dimension
        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

    def forward(self, x):
        # Calculate the batch size and sequence length dynamically
        batch_size, seq_len, _ = x.shape

        # Compute B, C and delta for the current input
        B = self.fc2(x)  # B has shape [batch_size, seq_len, state_size]
        C = self.fc3(x)  # C has shape [batch_size, seq_len, state_size]
        delta = F.softplus(self.fc1(x))  # delta has shape [batch_size, seq_len, d_model]

        # Calculating dB and dA using Einstein summation notation
        dB = torch.einsum("bld,bln->bldn", delta, B)  # dB has shape [batch_size, seq_len, d_model, state_size]
        dA = torch.exp(torch.einsum("bld,dn->bldn", delta, self.A))  # dA has shape [batch_size, seq_len, d_model, state_size]

        # Initialize h with zeros having the shape of dA
        h = torch.zeros(batch_size, seq_len, self.d_model, self.state_size, device=self.device)

        # Update h using the discretized dynamics
        h_new = torch.einsum('bldn,bldn->bldn', dA, h) + rearrange(x, "b l d -> b l d 1") * dB

        # Calculate the output y using matrix C and the new state h_new
        y = torch.einsum('bln,bldn->bld', C, h_new)  # y has shape [batch_size, seq_len, d_model]

        return y


import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model, state_size, device):
        super(MambaBlock, self).__init__()
        self.inp_proj = nn.Linear(d_model, 2 * d_model).to(device)
        self.conv = nn.Conv1d(2 * d_model, 2 * d_model, kernel_size=3, padding=1).to(device)
        self.conv_linear = nn.Linear(2 * d_model, d_model).to(device)
        self.out_proj = nn.Linear(d_model, d_model).to(device)
        self.norm = RMSNorm( d_model, device=device)  # Adjusting the normalization to the correct dimensionality

    def forward(self, x):
        x = self.norm(x)
        x_proj = self.inp_proj(x)

        # Ensure the tensor has the correct shape for Conv1d
        if x_proj.dim() == 2:  # Only batch and features, no length dimension
            x_proj = x_proj.unsqueeze(2)  # Add a single-length dimension
        elif x_proj.dim() == 3:
            x_proj = x_proj.transpose(1, 2)  # Correct shape for Conv1d

        x_conv = self.conv(x_proj)
        x_conv = x_conv.transpose(1, 2)  # Transpose back for further processing

        x_conv_out = self.conv_linear(x_conv)
        x_out = self.out_proj(x_conv_out)
        return x_out



import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class Mamba(nn.Module):
    def __init__(self, d_model, state_size, device):
        super(Mamba, self).__init__()
        self.mamba_block1 = MambaBlock(d_model, state_size, device)
        self.mamba_block2 = MambaBlock(d_model, state_size, device)
        self.mamba_block3 = MambaBlock(d_model, state_size, device)

    def mask_to_lengths(self, mask):
        if mask is None or mask.nelement() == 0:
            raise ValueError("Mask is empty or None")
        # Ensure that you sum along the sequence dimension to get lengths for each sequence in the batch
        lengths = mask.sum(dim=1)  # This should sum along the sequence dimension (axis 1)
        return lengths


    def forward(self, x, mask):
        # Convert mask to lengths
        lengths = self.mask_to_lengths(mask)

        # Ensure lengths is a 1D tensor of dtype torch.int64 on the CPU
        lengths = lengths.cpu().to(torch.int64)

        # Pack the padded sequence with the new shape
        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Process the packed input through blocks that can handle PackedSequence
        x_packed = self.mamba_block1(packed_input.data)
        x_packed = self.mamba_block2(x_packed)
        x_packed = self.mamba_block3(x_packed)

        # Need to repack the data since we've modified it
        packed_output = rnn_utils.PackedSequence(x_packed, packed_input.batch_sizes)

        # Unpack the processed sequence
        x, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        return x



import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device='cuda'):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        if isinstance(x, rnn_utils.PackedSequence):
            # Unpack sequence
            padded_output, lengths = rnn_utils.pad_packed_sequence(x, batch_first=True)

            # Normalize the unpacked data
            mean_sq = padded_output.pow(2).mean(dim=-1, keepdim=True)
            normalized_output = padded_output * torch.rsqrt(mean_sq + self.eps) * self.weight

            # Repack normalized data if needed, otherwise return it directly
            repacked_output = rnn_utils.pack_padded_sequence(normalized_output, lengths, batch_first=True, enforce_sorted=False)
            return repacked_output
        else:
            # Apply normalization directly to regular tensor
            mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
            output = x * torch.rsqrt(mean_sq + self.eps) * self.weight
            return output


class MambaModel(nn.Module):
    def __init__(self, d_model, state_size, num_classes, dropout_rate, device):
        super(MambaModel, self).__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.device = device

        # 初始化 Mamba 模型
        self.mamba = Mamba(d_model, state_size, device)
        # 分类层
        self.classifier = nn.Linear(d_model, num_classes)  # 假设输出是双向或有concat
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        if mask is not None:
            """
            x: Tensor, [seq_len, batch_size, d_model]，输入特征
            mask: Tensor, [batch_size, seq_len]，有效长度掩码
            """

            x = self.mamba(x, mask)
    

            # 应用 Dropout 和分类层
            x = self.dropout(x)
            x = torch.squeeze(x, -2)  # Squeeze out the third dimension if it's size is 1
            print("Shape after squeezing:", x.shape)  # Should now be [83, 32, 256]
            class_logits = self.classifier(x)
            return class_logits