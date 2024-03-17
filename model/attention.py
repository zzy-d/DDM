import torch
import torch.nn as nn

from typing import Tuple, Optional

import math
import torch.nn.functional as F

from utils.config import hparams

from transformers import DebertaTokenizer, DebertaModel

class PositionalEncoding(nn.Module):
    def __init__(self, in_dim=100, v_size=200, max_seq_len=100):
        super().__init__()

        self.in_dim = in_dim
        self.v_size = v_size
        self.max_seq_len = max_seq_len

        self.position_embedding = nn.Embedding(max_seq_len, in_dim)  # 位置编码

        self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
        self.proj_v = nn.Linear(self.v_size, 1)

    def forward(self, context):
        """Additive Attention with Positional Encoding

        Args:
            context (tensor): [B, seq_len, in_dim]

        Returns:
            outputs, weights: [B, seq_len, out_dim], [B, seq_len]
        """
        position_indices = torch.arange(context.size(1), device=context.device).unsqueeze(0)  # 生成位置索引
        position_encoded = self.position_embedding(position_indices)
        position_encoded = position_encoded.expand(context.size(0), -1, -1)

        context_with_position = context + position_encoded

        weights = self.proj_v(self.proj(context_with_position)).squeeze(-1)
        weights = torch.softmax(weights, dim=-1)

        return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights





# class PositionalEncoding(nn.Module):
#     def __init__(self, max_seq_len, d_model):
#         super(PositionalEncoding, self).__init__()
#         self.position_encodings = self.build_position_encodings(max_seq_len, d_model)
#
#     def build_position_encodings(self, max_seq_len, d_model):
#         position_encodings = torch.zeros(max_seq_len, d_model)
#         position = torch.arange(0, max_seq_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         position_encodings[:, 0::2] = torch.sin(position * div_term)
#         position_encodings[:, 1::2] = torch.cos(position * div_term)
#         return position_encodings
#
#     def forward(self, context):
#         position_encodings = self.position_encodings[:context.size(1), :].unsqueeze(0)
#         context_with_position = context + position_encodings
#         return context_with_position
#
#
#
#
#
# class PositionAttention(nn.Module):
#     def __init__(self, in_dim=100, v_size=200, max_seq_len=100):
#         super().__init__()
#
#         self.in_dim = in_dim
#         self.v_size = v_size
#         self.max_seq_len = max_seq_len
#         self.position_embedding = nn.Embedding(max_seq_len, in_dim)  # 位置编码
#
#         self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
#         self.proj_v = nn.Linear(self.v_size, 1)
#
#         self.proj_k = nn.Linear(self.v_size, self.in_dim // 30)  # 修改输出维度
#
#         self.proj_l = nn.Linear(self.v_size, self.in_dim // 30)  # 修改输出维度
#
#     def forward(self, context):
#         """Position Attention
#
#         Args:
#             context (tensor): [B, seq_len, in_dim]
#
#         Returns:
#             outputs, weights: [B, seq_len, in_dim], [B, seq_len]
#         """
#         # position_indices = torch.arange(context.size(1), device=context.device).unsqueeze(0)  # 生成位置索引
#         # position_encoded = self.position_embedding(position_indices)  # 获取位置编码
#
#         position_indices = torch.arange(context.size(1), device=context.device).unsqueeze(0)  # 生成位置索引
#         position_encoded = self.position_embedding(position_indices)  # 获取位置编码
#         position_encoded = position_encoded.expand(context.size(0), -1, -1)  # 将 position_encoded 维度扩展为 [B, seq_len, in_dim]
#
#
#         context_with_position = context + position_encoded  # 将位置编码与输入相加得到带位置信息的输入
#
#         context2 = context_with_position
#
#
#         weights = self.proj_v(self.proj(context2)).squeeze(-1)                                       # tanh（）激活函数
#         weights = torch.softmax(weights, dim=-1)  # [B, seq_len]                                         #NRMS中公式9
#
#         weights = self.proj_l(weights)
#
#         return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights  # [B, 1, seq_len], [B, seq_len, dim]




        # weights = self.proj_v(self.proj(context_with_position)).unsqueeze(-1)  # 调整weights的维度为 [B, seq_len1, 1]
        # weights = torch.softmax(weights, dim=-1)
        #
        # weights = self.proj_l(weights)  # [B, seq_len, in_dim // 30]
        #
        #
        # weighted_sum = torch.bmm(weights, context)  # 进行矩阵乘法操作
        #
        # output = weighted_sum



        return output, weights









# class PositionAttention(nn.Module):
#     def __init__(self, in_dim=100, v_size=200, max_seq_len=100):
#         super().__init__()
#
#         self.in_dim = in_dim
#         self.v_size = v_size
#         self.max_seq_len = max_seq_len
#         self.position_embedding = nn.Embedding(max_seq_len, in_dim)  # 位置编码
#
#         self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
#         self.proj_v = nn.Linear(self.v_size, 1)
#
#         self.proj_k = nn.Linear(self.v_size, self.in_dim // 1)  # 修改输出维度
#
#         self.proj_l = nn.Linear(self.v_size, self.in_dim // 10)  # 修改输出维度
#
#     def forward(self, context):
#         """Position Attention
#
#         Args:
#             context (tensor): [B, seq_len, in_dim]
#
#         Returns:
#             outputs, weights: [B, seq_len, in_dim], [B, seq_len]
#         """
#         position_indices = torch.arange(context.size(1), device=context.device).unsqueeze(0)  # 生成位置索引
#         # position_encoded = self.position_embedding(position_indices).expand(context.size(0), context.size(1),
#         #                                                                     context.size(2))  # 调整位置编码维度与context相匹配
#
#         position_indices = torch.arange(context.size(1), device=context.device).unsqueeze(0)  # 生成位置索引
#         position_encoded = self.position_embedding(position_indices)  # 获取位置编码
#
#
#         position_encoded = self.proj_k(position_encoded)  # [B, seq_len, in_dim // 30]
#
#         context_with_position = context + position_encoded  # 将位置编码与输入相加得到带位置信息的输入
#
#         context_with_position = self.proj_l(context_with_position)  # [B, seq_len, in_dim // 30]
#
#
#         weights = self.proj_v(self.proj(context_with_position)).squeeze(-1)
#         weights = torch.softmax(weights, dim=-1)
#
#         weighted_sum = torch.bmm(weights.unsqueeze(1), context)  # 计算加权和
#
#         output = weighted_sum
#
#         return output, weights




# class PositionAttention(nn.Module):
#     def __init__(self, in_dim=100, v_size=200, max_seq_len=100):
#         super().__init__()
#
#         self.in_dim = in_dim
#         self.v_size = v_size
#         self.max_seq_len = max_seq_len
#         self.position_embedding = nn.Embedding(max_seq_len, in_dim)  # 位置编码
#
#         self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
#         self.proj_v = nn.Linear(self.v_size, 1)
#
#     def forward(self, context):
#         """Position Attention
#
#         Args:
#             context (tensor): [B, seq_len, in_dim]
#
#         Returns:
#             outputs, weights: [B, seq_len, in_dim], [B, seq_len]
#         """
#         position_indices = torch.arange(context.size(1), device=context.device).unsqueeze(0)  # 生成位置索引
#         position_encoded = self.position_embedding(position_indices).expand(context.size(0), context.size(1), -1)  # 扩展位置编码维度
#
#         context_with_position = context + position_encoded  # 将位置编码与输入相加得到带位置信息的输入
#
#         weights = self.proj_v(self.proj(context_with_position)).squeeze(-1)
#         weights = torch.softmax(weights, dim=-1)
#
#         return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights

# class PositionAttention(nn.Module):
#     def __init__(self, in_dim=100, v_size=200, max_seq_len=100):
#         super().__init__()
#
#         self.in_dim = in_dim
#         self.v_size = v_size
#         self.max_seq_len = max_seq_len
#         self.position_embedding = nn.Embedding(max_seq_len, in_dim)  # 位置编码
#
#         self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
#         self.proj_v = nn.Linear(self.v_size, 1)
#
#     def forward(self, context):
#         """Position Attention
#
#         Args:
#             context (tensor): [B, seq_len, in_dim]
#
#         Returns:
#             outputs, weights: [B, seq_len, in_dim], [B, seq_len]
#         """
#         position_indices = torch.arange(context.size(1), device=context.device).unsqueeze(0)  # 生成位置索引
#         position_encoded = self.position_embedding(position_indices).expand(context.size(0), context.size(1), -1)  # 扩展位置编码维度
#
#         context_with_position = context + position_encoded  # 将位置编码与输入相加得到带位置信息的输入
#
#         weights = self.proj_v(self.proj(context_with_position)).squeeze(-1)
#         weights = torch.softmax(weights, dim=-1)
#
#         return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights





# class PositionAttention(torch.nn.Module):                    # 按位置注意力机制
#     def __init__(self, in_dim=100, v_size=200):
#         super().__init__()
#
#         self.in_dim = in_dim
#         self.v_size = v_size
#         self.query = nn.Linear(self.in_dim, self.v_size)
#         self.key = nn.Linear(self.in_dim, self.v_size)
#         self.value = nn.Linear(self.in_dim, self.v_size)
#         self.proj = nn.Linear(self.v_size, self.in_dim // 30)  # 修改输出维度
#
#     def forward(self, x):
#         """Self-Attention with Position Encoding
#
#         Args:
#             x (tensor): [B, seq_len, in_dim]
#
#         Returns:
#             output, weights: [B, seq_len, in_dim // 30], [B, seq_len, seq_len]
#         """
#         batch_size, seq_len, _ = x.size()
#
#         # Positional Encoding
#         position_encoding = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len).type_as(x) / 10000
#         position_encoding = torch.stack([torch.sin(position_encoding), torch.cos(position_encoding)], dim=-1)
#         x = x + position_encoding
#
#         query = self.query(x)  # [B, seq_len, v_size]
#         key = self.key(x)  # [B, seq_len, v_size]
#         value = self.value(x)  # [B, seq_len, v_size]
#
#         weights = torch.matmul(query, key.transpose(1, 2)) / (self.v_size ** 0.5)  # [B, seq_len, seq_len]
#         weights = torch.softmax(weights, dim=-1)  # [B, seq_len, seq_len]
#
#         output = torch.matmul(weights, value)  # [B, seq_len, v_size]
#         output = self.proj(output)  # [B, seq_len, in_dim // 30]
#
#         return output, weights




# class PositionAttention(nn.Module):  # 按位置注意力机制
#     def __init__(self, in_dim=100, v_size=200, max_seq_len=1000):
#         super(PositionAttention, self).__init__()
#
#         self.in_dim = in_dim
#         self.v_size = v_size
#         self.query = nn.Linear(self.in_dim, self.v_size)
#         self.key = nn.Linear(self.in_dim, self.v_size)
#         self.value = nn.Linear(self.in_dim, self.v_size)
#         self.proj = nn.Linear(self.v_size, self.in_dim // 30)  # 修改输出维度
#
#         # 位置编码的最大序列长度
#         self.max_seq_len = max_seq_len
#         self.position_encoding = self.generate_position_encoding()
#
#     def generate_position_encoding(self):
#         position_encoding = torch.zeros(self.max_seq_len, self.v_size)
#         position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.v_size, 2).float() * -(math.log(10000.0) / self.v_size))
#         position_encoding[:, 0::2] = torch.sin(position * div_term)
#         position_encoding[:, 1::2] = torch.cos(position * div_term)
#
#         return position_encoding
#
#     def forward(self, x):
#         """Self-Attention
#
#         Args:
#             x (tensor): [B, seq_len, in_dim]
#
#         Returns:
#             output, weights: [B, seq_len, in_dim // 30], [B, seq_len, seq_len]
#         """
#         query = self.query(x)  # [B, seq_len, v_size]
#         key = self.key(x)  # [B, seq_len, v_size]
#         value = self.value(x)  # [B, seq_len, v_size]
#
#         weights = torch.matmul(query, key.transpose(1, 2))  # [B, seq_len, seq_len]
#         weights = torch.softmax(weights, dim=-1)  # [B, seq_len]
#
#         output = torch.matmul(weights, value)  # [B, seq_len, v_size]
#         output = self.proj(output)  # [B, seq_len, in_dim // 30]
#
#         # 添加位置编码
#         output += self.position_encoding[:x.size(1), :].to(x.device)
#
#         output = self.proj(output)  # [B, seq_len, in_dim // 30]
#
#         return output, weights



# class PositionAttention(nn.Module):                                                      # 按位置注意力机制
#     def __init__(self, in_dim=100, v_size=200, max_seq_len=1000):
#         super(PositionAttention, self).__init__()
#
#         self.in_dim = in_dim
#         self.v_size = v_size
#         self.query = nn.Linear(self.in_dim, self.v_size)
#         self.key = nn.Linear(self.in_dim, self.v_size)
#         self.value = nn.Linear(self.in_dim, self.v_size)
#         self.proj = nn.Linear(self.v_size, self.in_dim // 30)  # 修改输出维度
#
#         # 位置编码的最大序列长度
#         self.max_seq_len = max_seq_len
#         self.position_encoding = self.generate_position_encoding()
#
#     def generate_position_encoding(self):
#         position_encoding = torch.zeros(self.max_seq_len, self.v_size)
#         position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.v_size, 2).float() * -(math.log(10000.0) / self.v_size))
#         position_encoding[:, 0::2] = torch.sin(position * div_term)
#         position_encoding[:, 1::2] = torch.cos(position * div_term)
#         position_encoding = position_encoding.unsqueeze(0)  # [1, max_seq_len, v_size]
#         return position_encoding
#
#     def forward(self, x):
#         """Self-Attention
#
#         Args:
#             x (tensor): [B, seq_len, in_dim]
#
#         Returns:
#             output, weights: [B, seq_len, in_dim // 30], [B, seq_len, seq_len]
#         """
#         query = self.query(x)  # [B, seq_len, v_size]
#         key = self.key(x)  # [B, seq_len, v_size]
#         value = self.value(x)  # [B, seq_len, v_size]
#
#         weights = torch.matmul(query, key.transpose(1, 2))  # [B, seq_len, seq_len]
#         weights = torch.softmax(weights, dim=-1)  # [B, seq_len]
#
#         output = torch.matmul(weights, value)  # [B, seq_len, v_size]
#         output = self.proj(output)  # [B, seq_len, in_dim // 30]
#
#         # 添加位置编码
#         output += self.position_encoding[:, :x.size(1), :].to(x.device)
#
#         output = self.proj(output)  # [B, seq_len, in_dim // 30]
#
#         return output, weights



# class PositionAttention(torch.nn.Module):                                               #真·按位置注意力机制
#     def __init__(self, in_dim=100, v_size=200):
#         super().__init__()
#
#         self.in_dim = in_dim
#         self.v_size = v_size
#         self.query = nn.Linear(self.in_dim, self.v_size)
#         self.key = nn.Linear(self.in_dim, self.v_size)
#         self.value = nn.Linear(self.in_dim, self.v_size)
#         self.proj = nn.Linear(self.v_size, self.in_dim // 30)  # 修改输出维度
#
#     def position_encoding(self, context):
#         batch_size, seq_len, _ = context.size()
#         pos_idx = torch.arange(0, self.in_dim, 2, dtype=torch.float, device=context.device) / self.in_dim
#         pos_idx = pos_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, seq_len)
#         pos_enc = torch.zeros(pos_idx.size() + (self.v_size,), dtype=torch.float, device=context.device)
#         div_term = torch.exp(
#             torch.arange(0, self.v_size, 2, dtype=torch.float, device=context.device) * (
#                         -math.log(10000.0) / self.v_size))
#
#         pos_enc[..., 0::2] = torch.sin(pos_idx * div_term)
#         pos_enc[..., 1::2] = torch.cos(pos_idx * div_term)
#
#         return pos_enc
#
#
#
#
#     def forward(self, x):
#         """Self-Attention with Position Encoding
#
#         Args:
#             x (tensor): [B, seq_len, in_dim]
#
#         Returns:
#             output, weights: [B, seq_len, in_dim // 30], [B, seq_len, seq_len, v_size]
#         """
#         pos_enc = self.position_encoding(x)
#         x = x + pos_enc
#
#         query = self.query(x)
#         key = self.key(x)
#         value = self.value(x)
#
#         weights = torch.matmul(query, key.transpose(1, 2))
#         weights = F.softmax(weights, dim=-1)
#
#         output = torch.matmul(weights, value)
#         output = self.proj(output)
#
#         return output, weights







# class AdditiveAttention(torch.nn.Module):                                      #加型注意力机制
#     def __init__(self, in_dim=100, v_size=200):
#         super().__init__()
#
#         self.in_dim = in_dim
#         self.v_size = v_size
#         # self.v = torch.nn.Parameter(torch.rand(self.v_size))
#         self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
#         self.proj_v = nn.Linear(self.v_size, 1)
#
#     def forward(self, context):
#         """Additive Attention
#
#         Args:
#             context (tensor): [B, seq_len, in_dim]
#
#         Returns:
#             outputs, weights: [B, seq_len, out_dim], [B, seq_len]
#         """
#         # weights = self.proj(context) @ self.v
#         weights = self.proj_v(self.proj(context)).squeeze(-1)
#         weights = torch.softmax(weights, dim=-1) # [B, seq_len]
#         return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights # [B, 1, seq_len], [B, seq_len, dim]






# class SelfAttention(torch.nn.Module):                                          #按位置注意力机制
#     def __init__(self, in_dim, v_size):
#         super().__init__()
#
#         self.in_dim = in_dim
#         self.v_size = v_size
#         self.WQ = torch.nn.Linear(in_dim, v_size)
#         self.WK = torch.nn.Linear(in_dim, v_size)
#         self.WV = torch.nn.Linear(in_dim, v_size)
#         self.proj_v = nn.Linear(self.v_size, 1)
#         self.proj = nn.Linear(self.v_size, self.in_dim * 10)  # 修改输出维度
#
#
#     def forward(self, x):
#         """Self-Attention
#
#         Args:
#             x (tensor): [B, seq_len, in_dim]
#
#         Returns:
#             outputs, weights: [B, seq_len, v_size], [B, seq_len]
#         """
#         Q = self.WQ(x)  # [B, seq_len, v_size]
#         K = self.WK(x)  # [B, seq_len, v_size]
#         V = self.WV(x)  # [B, seq_len, v_size]
#
#         weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.v_size)  # [B, seq_len, seq_len]
#         weights = torch.softmax(weights, dim=-1)  # [B, seq_len, seq_len]
#
#         output = torch.matmul(weights, V)  # [B, seq_len, v_size]
#
#         outputs = self.proj_v(output)  # [B, seq_len, 1]
#         outputs = outputs.squeeze(-1)  # [B, seq_len]
#
#         return outputs, weights


#999999999999999999

class SelfAttention(torch.nn.Module):                                               #自注意力机制（按位置注意力机制）
    def __init__(self, in_dim=100, v_size=200):
        super().__init__()

        self.in_dim = in_dim
        self.v_size = v_size
        self.query = nn.Linear(self.in_dim, self.v_size)
        self.key = nn.Linear(self.in_dim, self.v_size)
        self.value = nn.Linear(self.in_dim, self.v_size)
        self.proj = nn.Linear(self.v_size, self.in_dim // 30)  # 修改输出维度

    def forward(self, x):
        """Self-Attention

        Args:
            x (tensor): [B, seq_len, in_dim]

        Returns:
            output, weights: [B, seq_len, in_dim // 30], [B, seq_len, seq_len]
        """
        query = self.query(x)  # [B, seq_len, v_size]
        key = self.key(x)  # [B, seq_len, v_size]
        value = self.value(x)  # [B, seq_len, v_size]

        weights = torch.matmul(query, key.transpose(1, 2))  # [B, seq_len, seq_len]
        #weights = torch.softmax(weights / math.sqrt(self.v_size), dim=-1)
        weights = torch.softmax(weights, dim=-1)  # [B, seq_len]

        output = torch.matmul(weights, value)  # [B, seq_len, v_size]
        output = self.proj(output)  # [B, seq_len, in_dim // 30]

        return output, weights






class LocalAttention(nn.Module):                                                        #局部（邻近）注意力机制
    def __init__(self, in_dim=100, v_size=200):
        super().__init__()

        self.in_dim = in_dim
        self.v_size = v_size
        self.kernel_size = 3
        self.padding = self.kernel_size // 2

        self.proj = nn.Linear(self.in_dim, self.v_size)
        self.conv = nn.Conv1d(self.v_size, 1, self.kernel_size, padding=self.padding)

    def forward(self, context):
        """Local Attention

        Args:
            context (tensor): [B, seq_len, in_dim]

        Returns:
            outputs, weights: [B, seq_len, out_dim], [B, seq_len]
        """
        batch_size, seq_len, _ = context.size()

        projected_context = self.proj(context)  # [B, seq_len, v_size]
        projected_context = projected_context.permute(0, 2, 1)  # [B, v_size, seq_len]

        conv_output = self.conv(projected_context)  # [B, 1, seq_len]
        weights = F.softmax(conv_output.squeeze(1), dim=-1)  # [B, seq_len]

        expanded_weights = weights.unsqueeze(1)  # [B, 1, seq_len]
        attended_context = torch.bmm(expanded_weights, context).squeeze(1)  # [B, in_dim]

        return attended_context, weights




class ScaledDotProductAttention(nn.Module):                                               #缩放点积注意力机制
    def __init__(self, in_dim=100, v_size=200):
        super().__init__()

        self.in_dim = in_dim
        self.v_size = v_size
        # self.v = torch.nn.Parameter(torch.rand(self.v_size))
        self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
        self.proj_v = nn.Linear(self.v_size, 1)

    def forward(self, context):
        """Additive Attention

        Args:
            context (tensor): [B, seq_len, in_dim]

        Returns:
            outputs, weights: [B, seq_len, out_dim], [B, seq_len]
        """
        # weights = self.proj(context) @ self.v
        # weights = torch.matmul(self.proj(context), query.unsqueeze(-1)).squeeze(-1)
        # weights = weights / math.sqrt(self.v_size)  # 缩放点积

        # context = context.unsqueeze(1)
        # weights = torch.matmul(context, context.transpose(-1, -2))
        weights = self.proj_v(self.proj(context)).squeeze(-1)                                      #tanh（）激活函数
        weights = weights / math.sqrt(self.v_size)
        weights = torch.softmax(weights, dim=-1)  # [B, seq_len]                                   #NRMS中公式9
        return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights # [B, 1, seq_len], [B, seq_len, dim]






class AdaptiveAttention(torch.nn.Module):                                #自适应型注意力机制
    def __init__(self, in_dim=100, v_size=200):
        super().__init__()

        self.in_dim = in_dim
        self.proj = nn.Linear(self.in_dim, 1)

    def forward(self, context):
        """Adaptive Attention

        Args:
            context (tensor): [B, seq_len, in_dim]

        Returns:
            outputs, weights: [B, seq_len, out_dim], [B, seq_len]
        """
        attention = self.proj(context).squeeze(-1)  # [B, seq_len]
        weights = torch.softmax(attention, dim=-1)  # [B, seq_len]
        return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights  # [B, 1, seq_len], [B, seq_len]



#999999999999999999


class AdditiveAttention(torch.nn.Module):                                      #加型注意力机制
    def __init__(self, in_dim=100, v_size=200):
        super().__init__()

        self.in_dim = in_dim
        self.v_size = v_size
        # self.v = torch.nn.Parameter(torch.rand(self.v_size))
        self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
        self.proj_v = nn.Linear(self.v_size, 1)

    def forward(self, context):
        """Additive Attention

        Args:
            context (tensor): [B, seq_len, in_dim]

        Returns:
            outputs, weights: [B, seq_len, out_dim], [B, seq_len]
        """
        # weights = self.proj(context) @ self.v
        weights = self.proj_v(self.proj(context)).squeeze(-1)                                           #tanh（）激活函数
        weights = torch.softmax(weights, dim=-1) # [B, seq_len]                                         #NRMS中公式9
        return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights # [B, 1, seq_len], [B, seq_len, dim]




#         Args:
#             nhead: the number of heads in the multiheadattention model
#             in_proj_container: A container of multi-head in-projection linear layers (a.k.a nn.Linear).
#             attention_layer: The attention layer.
#             out_proj: The multi-head out-projection layer (a.k.a nn.Linear).
#         Examples::
#             >>> import torch
#             >>> embed_dim, num_heads, bsz = 10, 5, 64
#             >>> in_proj_container = InProjContainer(torch.nn.Linear(embed_dim, embed_dim),
#                                                     torch.nn.Linear(embed_dim, embed_dim),
#                                                     torch.nn.Linear(embed_dim, embed_dim))
#             >>> MHA = MultiheadAttentionContainer(num_heads,
#                                                   in_proj_container,
#                                                   ScaledDotProduct(),
#                                                   torch.nn.Linear(embed_dim, embed_dim))
#             >>> query = torch.rand((21, bsz, embed_dim))
#             >>> key = value = torch.rand((16, bsz, embed_dim))
#             >>> attn_output, attn_weights = MHA(query, key, value)
#             >>> print(attn_output.shape)
#             >>> torch.Size([21, 64, 10])
#         """
#         super(MultiheadAttentionContainer, self).__init__()
#         self.nhead = nhead
#         self.in_proj_container = in_proj_container
#         self.attention_layer = attention_layer
#         self.out_proj = out_proj
#
#     def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
#                 attn_mask: Optional[torch.Tensor] = None,
#                 bias_k: Optional[torch.Tensor] = None,
#                 bias_v: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         r"""
#         Args:
#             query, key, value (Tensor): map a query and a set of key-value pairs to an output.
#                 See "Attention Is All You Need" for more details.
#             attn_mask, bias_k and bias_v (Tensor, optional): keyword arguments passed to the attention layer.
#                 See the definitions in the attention.
#         Shape:
#             - Inputs:
#             - query: :math:`(L, N, E)`
#             - key: :math:`(S, N, E)`
#             - value: :math:`(S, N, E)`
#             - attn_mask, bias_k and bias_v: same with the shape of the corresponding args in attention layer.
#             - Outputs:
#             - attn_output: :math:`(L, N, E)`
#             - attn_output_weights: :math:`(N * H, L, S)`
#             where where L is the target length, S is the sequence length, H is the number of attention heads,
#                 N is the batch size, and E is the embedding dimension.
#         """
#         tgt_len, src_len, bsz, embed_dim = query.size(
#             -3), key.size(-3), query.size(-2), query.size(-1)
#         q, k, v = self.in_proj_container(query, key, value)
#         assert q.size(-1) % self.nhead == 0, "query's embed_dim must be divisible by the number of heads"
#         head_dim = q.size(-1) // self.nhead
#         q = q.reshape(tgt_len, bsz * self.nhead, head_dim)
#
#         assert k.size(-1) % self.nhead == 0, "key's embed_dim must be divisible by the number of heads"
#         head_dim = k.size(-1) // self.nhead
#         k = k.reshape(src_len, bsz * self.nhead, head_dim)
#
#         assert v.size(-1) % self.nhead == 0, "value's embed_dim must be divisible by the number of heads"
#         head_dim = v.size(-1) // self.nhead
#         v = v.reshape(src_len, bsz * self.nhead, head_dim)
#
#         attn_output, attn_output_weights = self.attention_layer(q, k, v, attn_mask=attn_mask,
#                                                                 bias_k=bias_k, bias_v=bias_v)
#         attn_output = attn_output.reshape(tgt_len, bsz, embed_dim)
#         attn_output = self.out_proj(attn_output)
#         return attn_output, attn_output_weights




class ScaledDotProduct(torch.nn.Module):

    def __init__(self, dropout=0.0):
        r"""Processes a projected query and key-value pair to apply
        scaled dot product attention.
        Args:
            dropout (float): probability of dropping an attention weight.
        Examples::
            >>> SDP = torchtext.models.ScaledDotProduct(0.1)
            >>> q = torch.randn(256, 21, 3)
            >>> k = v = torch.randn(256, 21, 3)
            >>> attn_output, attn_weights = SDP(q, k, v)
            >>> print(attn_output.shape, attn_weights.shape)
            torch.Size([256, 21, 3]) torch.Size([256, 21, 21])
        """
        super(ScaledDotProduct, self).__init__()
        self.dropout = dropout

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                bias_k: Optional[torch.Tensor] = None,
                bias_v: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Uses a scaled dot product with the projected key-value pair to update
        the projected query.
        Args:
            query (Tensor): Projected query
            key (Tensor): Projected key
            value (Tensor): Projected value
            attn_mask (BoolTensor, optional): 3D mask that prevents attention to certain positions.
            bias_k and bias_v: (Tensor, optional): one more key and value sequence to be added at
                sequence dim (dim=-3). Those are used for incremental decoding. Users should provide
                non-None to both arguments in order to activate them.
        Shape:
            - query: :math:`(L, N * H, E / H)`
            - key: :math:`(S, N * H, E / H)`
            - value: :math:`(S, N * H, E / H)`
            - attn_mask: :math:`(N * H, L, S)`, positions with ``True`` are not allowed to attend
                while ``False`` values will be unchanged.
            - bias_k and bias_v:bias: :math:`(1, N * H, E / H)`
            - Output: :math:`(L, N * H, E / H)`, :math:`(N * H, L, S)`
            where L is the target length, S is the source length, H is the number
            of attention heads, N is the batch size, and E is the embedding dimension.
        """
        if bias_k is not None and bias_v is not None:
            assert key.size(-1) == bias_k.size(-1) and key.size(-2) == bias_k.size(-2) and bias_k.size(-3) == 1, \
                "Shape of bias_k is not supported"
            assert value.size(-1) == bias_v.size(-1) and value.size(-2) == bias_v.size(-2) and bias_v.size(-3) == 1, \
                "Shape of bias_v is not supported"
            key = torch.cat([key, bias_k])
            value = torch.cat([value, bias_v])
            if attn_mask is not None:
                _attn_mask = attn_mask
                attn_mask = torch.nn.functional.pad(_attn_mask, (0, 1))

        tgt_len, head_dim = query.size(-3), query.size(-1)
        assert query.size(-1) == key.size(-1) == value.size(
            -1), "The feature dim of query, key, value must be equal."
        assert key.size() == value.size(), "Shape of key, value must match"
        src_len = key.size(-3)
        batch_heads = max(query.size(-2), key.size(-2))

        # Scale query
        query, key, value = query.transpose(
            -2, -3), key.transpose(-2, -3), value.transpose(-2, -3)
        query = query * (head_dim ** -0.5)
        if attn_mask is not None:
            if attn_mask.dim() != 3:
                raise RuntimeError('attn_mask must be a 3D tensor.')
            if (attn_mask.size(-1) != src_len) or (attn_mask.size(-2) != tgt_len) or \
               (attn_mask.size(-3) != 1 and attn_mask.size(-3) != batch_heads):
                raise RuntimeError('The size of the attn_mask is not correct.')
            if attn_mask.dtype != torch.bool:
                raise RuntimeError(
                    'Only bool tensor is supported for attn_mask')

        # Dot product of q, k
        attn_output_weights = torch.matmul(query, key.transpose(-2, -1))
        if attn_mask is not None:
            attn_output_weights.masked_fill_(attn_mask, -1e8,)
        attn_output_weights = torch.nn.functional.softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = torch.nn.functional.dropout(
            attn_output_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_output_weights, value)
        return attn_output.transpose(-2, -3), attn_output_weights


class InProjContainer(torch.nn.Module):
    def __init__(self, query_proj, key_proj, value_proj):
        r"""A in-proj container to process inputs.
        Args:
            query_proj: a proj layer for query.
            key_proj: a proj layer for key.
            value_proj: a proj layer for value.
        """

        super(InProjContainer, self).__init__()
        self.query_proj = query_proj
        self.key_proj = key_proj
        self.value_proj = value_proj

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Projects the input sequences using in-proj layers.
        Args:
            query, key, value (Tensors): sequence to be projected
        Shape:
            - query, key, value: :math:`(S, N, E)`
            - Output: :math:`(S, N, E)`
            where S is the sequence length, N is the batch size, and E is the embedding dimension.
        """
        return self.query_proj(query), self.key_proj(key), self.value_proj(value)


def generate_square_subsequent_mask(nbatch, sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with True.
        Unmasked positions are filled with False.
    Args:
        nbatch: the number of batch size
        sz: the size of square mask
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(
        0, 1).repeat(nbatch, 1, 1)
    return mask
