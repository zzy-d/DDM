import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import  AdditiveAttention
from model.attention import  AdaptiveAttention
from model.attention import  ScaledDotProductAttention
from model.attention import  LocalAttention
from model.attention import  SelfAttention
from model.attention import  PositionAttention

#
# import math

#000
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


class MultiheadAttentionContainer(nn.Module):                                                         #多重多头重构版本
    def __init__(self, embed_size, nhead, dropout=0.1):
        super(MultiheadAttentionContainer, self).__init__()
        self.mh_layer = nn.MultiheadAttention(embed_size, nhead, dropout=dropout)
        self.layernorm = nn.LayerNorm(embed_size)

    def forward(self, x, mask=None):
        residual = x
        output, _ = self.mh_layer(x, x, x, attn_mask=mask, need_weights=False)
        output = residual + output
        output = self.layernorm(output)
        return output

class DocEncoder(nn.Module):
    def __init__(self, hparams, weight=None):
        super(DocEncoder, self).__init__()
        self.hparams = hparams
        if weight is None:
            self.embedding = nn.Embedding(hparams['dct_size'], hparams['embed_size'])
        else:
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
        self.mha1 = MultiheadAttentionContainer(hparams['embed_size'], nhead=hparams['nhead'], dropout=0.1)
        self.mha2 = MultiheadAttentionContainer(hparams['embed_size'], nhead=hparams['nhead'], dropout=0.1)
        self.proj = nn.Linear(hparams['embed_size'], hparams['encoder_size'])

        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])                      #1.加型注意力机制

        self.Adaptive_attn = AdaptiveAttention(hparams['encoder_size'], hparams['v_size'])                      #2.自适应型注意力机制
        self.ScaledDotProduct_attn = ScaledDotProductAttention(hparams['encoder_size'], hparams['v_size'])      #3.缩放点积注意力机制
        self.Local_attn = LocalAttention(hparams['encoder_size'], hparams['v_size'])                            #4.邻近注意力机制
        self.Self_attn = SelfAttention(hparams['encoder_size'], hparams['v_size'])                              #5.按位置注意力机制
        self.Position_attn = PositionAttention(hparams['encoder_size'], hparams['v_size'])                      #6.真·按位置注意力机制

        self.dropout = nn.Dropout(p=0.1)
        self.layernorm = nn.LayerNorm(hparams['embed_size'])

        # self.augru = AUGRU(input_size=hparams['embed_size'],
        #                    hidden_size=hparams['encoder_size'],
        #                    num_layers=2,
        #                    dropout=0.1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)

        output = self.mha1(x)
        output = self.dropout(output)
        output = self.mha2(output)

        output = x + output
        output = self.layernorm(output)

        # output = self.augru(output)

        output = F.dropout(output.permute(1, 0, 2), p=0.2)
        output = self.proj(output)                                        #全连接层

        output, _ = self.additive_attn(output)                            #1.加型注意力机制

        #output, _ = self.Adaptive_attn(output)                            #2.自适应型注意力机制

        #output, _ = self.ScaledDotProduct_attn(output)                    #3.缩放点积注意力机制

        #output, _ = self.Local_attn(output)                                #4.邻近注意力机制

        #output, _ = self.Self_attn(output)                                  #5.按位置注意力机制

        #output, _ = self.Position_attn(output)                              # 6.按位置注意力机制

        return output










#原始版微改

# class MultiheadAttentionContainer(nn.Module):                                                         #仅仅重构多头版本
#     def __init__(self, embed_size, nhead, dropout=0.1):
#         super(MultiheadAttentionContainer, self).__init__()
#         self.mh_layer = nn.MultiheadAttention(embed_size, nhead, dropout=dropout)
#         self.layernorm = nn.LayerNorm(embed_size)
#
#     def forward(self, x, mask=None):
#         residual = x
#
#         output, _ = self.mh_layer(x, x, x, attn_mask=mask, need_weights=False)
#         output = residual + output
#         output = self.layernorm(output)
#
#         return output
#
#
#
# class DocEncoder(nn.Module):
#     def __init__(self, hparams, weight=None):
#         super(DocEncoder, self).__init__()
#         self.hparams = hparams
#         if weight is None:
#             self.embedding = nn.Embedding(hparams['dct_size'], hparams['embed_size'])
#         else:
#             self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
#         self.mha1 = MultiheadAttentionContainer(hparams['embed_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.dropout = nn.Dropout(p=0.1)  # 添加一个0.1的dropout层
#         self.mha2 = MultiheadAttentionContainer(hparams['encoder_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.mha3 = MultiheadAttentionContainer(hparams['encoder_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size'])
#         self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
#
#         self.layernorm = nn.LayerNorm(hparams['embed_size'])
#
#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.permute(1, 0, 2)
#
#         output = self.mha1(x)
#         output = self.dropout(output)  # 在两层之间添加了一个Dropout层，丢失率为0.1
#         output = self.mha2(output)
#
#
#         output = x + output
#
#         output = self.layernorm(output)                                            #这句代码对实验结果影响波动大
#
#         output = F.dropout(output.permute(1, 0, 2), p=0.1)
#         output = self.proj(output)
#         output, _ = self.additive_attn(output)
#
#         return output













#123

# class MultiheadAttentionContainer(nn.Module):                                                         #多重多头重构版本
#     def __init__(self, embed_size, nhead, dropout=0.1):
#         super(MultiheadAttentionContainer, self).__init__()
#         self.mh_layer = nn.MultiheadAttention(embed_size, nhead, dropout=dropout)
#         self.layernorm = nn.LayerNorm(embed_size)
#
#     def forward(self, x, mask=None):
#         residual = x
#         output, _ = self.mh_layer(x, x, x, attn_mask=mask, need_weights=False)
#         output = residual + output
#         output = self.layernorm(output)
#         return output
#
# class DocEncoder(nn.Module):
#     def __init__(self, hparams, weight=None):
#         super(DocEncoder, self).__init__()
#         self.hparams = hparams
#         if weight is None:
#             self.embedding = nn.Embedding(hparams['dct_size'], hparams['embed_size'])
#         else:
#             self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
#         self.mha1 = MultiheadAttentionContainer(hparams['embed_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.mha2 = MultiheadAttentionContainer(hparams['embed_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.proj = nn.Linear(hparams['embed_size'], hparams['encoder_size'])
#
#         self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])                      #1.加型注意力机制
#
#         self.Adaptive_attn = AdaptiveAttention(hparams['encoder_size'], hparams['v_size'])                      #2.自适应型注意力机制
#         self.ScaledDotProduct_attn = ScaledDotProductAttention(hparams['encoder_size'], hparams['v_size'])      #3.缩放点积注意力机制
#         self.Local_attn = LocalAttention(hparams['encoder_size'], hparams['v_size'])                            #4.邻近注意力机制
#         #self.Self_attn = SelfAttention(hparams['encoder_size'], hparams['v_size'])                              #5.按位置注意力机制
#         #self.Self_attn = SelfAttention(hparams['v_size'])
#
#         self.dropout = nn.Dropout(p=0.1)
#         self.layernorm = nn.LayerNorm(hparams['embed_size'])
#
#         # self.augru = AUGRU(input_size=hparams['embed_size'],
#         #                    hidden_size=hparams['encoder_size'],
#         #                    num_layers=2,
#         #                    dropout=0.1)
#
#     def forward(self, x):
#         x = F.dropout(self.embedding(x), 0.1)
#         #x = self.embedding(x)
#         x = x.permute(1, 0, 2)
#
#         output = self.mha1(x)
#         output = self.dropout(output)
#         output = self.mha2(output)
#
#         output = x + output
#         output = self.layernorm(output)
#
#         # output = self.augru(output)
#
#         output = F.dropout(output.permute(1, 0, 2), p=0.1)
#         output = self.proj(output)                                        #全连接层
#
#         output, _ = self.additive_attn(output)                           #1.加型注意力机制
#
#         #output, _ = self.Adaptive_attn(output)                           #2.自适应型注意力机制
#
#
#         #output, _ = self.ScaledDotProduct_attn(output)                   #3.缩放点积注意力机制
#
#         #output, _ = self.Local_attn(output)                              #4.邻近注意力机制
#
#         #output, _ = self.Self_attn(output)                                #5.按位置注意力机制
#
#         return output



#123



# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, in_dim=300, v_size=200):
#         super(ScaledDotProductAttention, self).__init__()
#         self.in_dim = in_dim
#         self.v_size = v_size
#         self.scale_factor = nn.Parameter(torch.sqrt(torch.tensor(in_dim, dtype=torch.float)))
#         self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
#         self.proj_v = nn.Linear(self.v_size, 1)
#
#     def forward(self, context):
#         """Scaled Dot Product Attention
#
#         Args:
#             context (tensor): [B, seq_len, in_dim]
#
#         Returns:
#             outputs, weights: [B, out_dim], [B, seq_len]
#         """
#         weights = torch.bmm(context, context.transpose(1, 2)) / self.scale_factor
#         weights = torch.softmax(weights, dim=-1)
#         outputs = torch.bmm(weights.transpose(1, 2), context).squeeze(1)
#         weights = weights.squeeze(2)
#         return outputs, weights








# class AUGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
#         super(AUGRU, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         # 初始化隐状态张量
#         h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
#         # 前向传播
#         output, hn = self.gru(x, h0)
#         return output

# class MultiheadAttentionContainer(nn.Module): #添加feedforword层
#     def __init__(self, embed_size, nhead, dropout=0.1):
#         super(MultiheadAttentionContainer, self).__init__()
#         self.mh_layer = nn.MultiheadAttention(embed_size, nhead, dropout=dropout)
#         self.layernorm1 = nn.LayerNorm(embed_size)
#         self.layernorm2 = nn.LayerNorm(embed_size)
#
#         # Add feedforward layer here
#         self.fc_layer = nn.Sequential(
#             nn.Linear(embed_size, embed_size * 16),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_size * 16, embed_size),
#             nn.LayerNorm(embed_size),
#         )
#
#
#     def forward(self, x, mask=None):
#         residual = x
#         output, _ = self.mh_layer(x, x, x, attn_mask=mask, need_weights=False)
#         output = residual + output
#         output1 = self.layernorm1(output)
#
#         # Add feedforward layer
#         output = self.fc_layer(output1)
#
#         output = output + output1
#         output = self.layernorm2(output)
#
#         return output
#
# class DocEncoder(nn.Module):
#     def __init__(self, hparams, weight=None):
#         super(DocEncoder, self).__init__()
#         self.hparams = hparams
#         if weight is None:
#             self.embedding = nn.Embedding(hparams['dct_size'], hparams['embed_size'])
#         else:
#             self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
#         self.mha1 = MultiheadAttentionContainer(hparams['embed_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.dropout = nn.Dropout(p=0.1)
#         self.mha2 = MultiheadAttentionContainer(hparams['encoder_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.mha3 = MultiheadAttentionContainer(hparams['encoder_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size'])
#         self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
#
#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.permute(1, 0, 2)
#
#         output = self.mha1(x)
#         output = self.dropout(output)
#         output = self.mha2(output)
#         #output = self.dropout(output)
#         #output = self.mha3(output)
#
#         output = F.dropout(output.permute(1, 0, 2), p=0.2)
#         output = self.proj(output)
#         output, attention_score = self.additive_attn(output)
#         return output


# class MultiheadAttentionContainer(nn.Module):                                                           #添加glu版本
#     def __init__(self, embed_size, nhead, dropout=0.1):
#         super(MultiheadAttentionContainer, self).__init__()
#         self.mh_layer = nn.MultiheadAttention(embed_size, nhead, dropout=dropout)
#         self.layernorm = nn.LayerNorm(embed_size)
#
#         self.linear = nn.Linear(embed_size,  embed_size * 2)  # 添加线性层，输出2倍的embedding_size
#         self.dropout = nn.Dropout(dropout)  # 添加dropout层，丢失率为dropout
#         self.glu = nn.GLU()
#
#     def forward(self, x, mask=None):
#         residual = x
#         output, _ = self.mh_layer(x, x, x, attn_mask=mask, need_weights=False)
#         output = self.linear(output)
#         output = self.glu(output)
#         output = residual + output  # 残差连接
#         output = self.layernorm(output)
#         output = self.dropout(output)
#
#         return output
#
# class DocEncoder(nn.Module):
#     def __init__(self, hparams, weight=None):
#         super(DocEncoder, self).__init__()
#         self.hparams = hparams
#         if weight is None:
#             self.embedding = nn.Embedding(hparams['dct_size'], hparams['embed_size'])
#         else:
#             self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
#         self.mha1 = MultiheadAttentionContainer(hparams['embed_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.dropout = nn.Dropout(p=0.1)  # 添加一个0.1的dropout层
#         self.mha2 = MultiheadAttentionContainer(hparams['encoder_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.mha3 = MultiheadAttentionContainer(hparams['encoder_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size'])
#         self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
#
#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.permute(1, 0, 2)
#
#         output = self.mha1(x)
#         output = self.dropout(output)  # 在两层之间添加了一个Dropout层，丢失率为0.1
#         output = self.mha2(output)
#         #output = self.dropout(output)  # 在两层之间添加了一个Dropout层，丢失率为0.1
#         #output = self.mha3(output)
#
#         output = F.dropout(output.permute(1, 0, 2), p=0.2)
#         output = self.proj(output)
#         output, _ = self.additive_attn(output)
#         return output




"""
class MultiheadAttentionContainer(nn.Module):
    def __init__(self, embed_size, nhead, dropout=0.1):
        super(MultiheadAttentionContainer, self).__init__()
        self.mh_layer = nn.MultiheadAttention(embed_size, nhead, dropout=dropout)
        self.layernorm = nn.LayerNorm(embed_size)

        self.linear1 = nn.Linear(embed_size,  embed_size *4)  # 添加线性层，输出2倍的embedding_size
        self.linear2 = nn.Linear(embed_size, embed_size)  # 添加线性层，输入2倍的embedding_size 输出1倍的embedding_size
        self.activation = nn.Tanh()  # 添加激活函数
        self.dropout = nn.Dropout(dropout)  # 添加dropout层，丢失率为dropout
        self.glu = nn.GLU()

    def forward(self, x, mask=None):
        residual = x
        output, _ = self.mh_layer(x, x, x, attn_mask=mask, need_weights=False)
        output = self.glu(output)
        output = residual + output  # 残差连接
        output = self.layernorm(output)



        # GLU层
        # output = self.linear1(output)
        # gate, output = output.chunk(2, dim=-1)
        # gate = self.activation(gate)
        # output = F.glu(output, dim=-1)
        # output = output * gate
        #output = self.linear2(output)

        output = self.dropout(output)

        return output

class DocEncoder(nn.Module):
    def __init__(self, hparams, weight=None):
        super(DocEncoder, self).__init__()
        self.hparams = hparams
        if weight is None:
            self.embedding = nn.Embedding(hparams['dct_size'], hparams['embed_size'])
        else:
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
        self.mha1 = MultiheadAttentionContainer(hparams['embed_size'], nhead=hparams['nhead'], dropout=0.1)
        self.dropout = nn.Dropout(p=0.1)  # 添加一个0.1的dropout层
        self.mha2 = MultiheadAttentionContainer(hparams['encoder_size'], nhead=hparams['nhead'], dropout=0.1)
        self.mha3 = MultiheadAttentionContainer(hparams['encoder_size'], nhead=hparams['nhead'], dropout=0.1)
        self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size'])
        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)

        output = self.mha1(x)
        output = self.dropout(output)  # 在两层之间添加了一个Dropout层，丢失率为0.1
        output = self.mha2(output)
        #output = self.dropout(output)  # 在两层之间添加了一个Dropout层，丢失率为0.1
        #output = self.mha3(output)

        output = F.dropout(output.permute(1, 0, 2), p=0.2)
        output = self.proj(output)
        output, _ = self.additive_attn(output)
        return output

"""


# class MultiheadAttentionContainer(nn.Module):                                                         #GLU残差，两层多头版本
#     def __init__(self, embed_size, nhead, dropout=0.1):
#         super(MultiheadAttentionContainer, self).__init__()
#         self.mh_layer = nn.MultiheadAttention(embed_size, nhead, dropout=dropout)
#         self.layernorm = nn.LayerNorm(embed_size)
#
#     # def forward(self, x, mask=None):
#     #     residual = x
#     #     x_norm = self.layernorm(x)
#     #     output, _ = self.mh_layer(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
#     #     output = residual + output
#     #     return output
#     def forward(self, x, mask=None):
#         residual = x
#         output, _ = self.mh_layer(x, x, x, attn_mask=mask, need_weights=False)
#         output = nn.Linear(output.size(-1), x.size(-1))(output)
#         # if output.size(-1) % 2 == 1:
#         #     output = output[..., :-1]
#         output = F.glu(output, dim=-1)  # 添加GLU层
#         output = residual + output
#         output = self.layernorm(output)
#         return output
#
#
# class DocEncoder(nn.Module):
#     def __init__(self, hparams, weight=None):
#         super(DocEncoder, self).__init__()
#         self.hparams = hparams
#         if weight is None:
#             self.embedding = nn.Embedding(hparams['dct_size'], hparams['embed_size'])
#         else:
#             self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
#         self.mha1 = MultiheadAttentionContainer(hparams['embed_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.dropout = nn.Dropout(p=0.1)  # 添加一个0.1的dropout层
#         self.mha2 = MultiheadAttentionContainer(hparams['encoder_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.mha3 = MultiheadAttentionContainer(hparams['encoder_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size'])
#         self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
#
#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.permute(1, 0, 2)
#
#         output = self.mha1(x)
#         output = self.dropout(output)  # 在两层之间添加了一个Dropout层，丢失率为0.1
#         output = self.mha2(output)
#         #output = self.dropout(output)  # 在两层之间添加了一个Dropout层，丢失率为0.1
#         #output = self.mha3(output)
#
#         output = F.dropout(output.permute(1, 0, 2), p=0.2)
#         output = self.proj(output)
#         output, _ = self.additive_attn(output)
#         return output



# class MultiheadAttentionContainer(nn.Module):                                                         #仅仅重构多头版本
#     def __init__(self, embed_size, nhead, dropout=0.1):
#         super(MultiheadAttentionContainer, self).__init__()
#         self.mh_layer = nn.MultiheadAttention(embed_size, nhead, dropout=dropout)
#         self.layernorm = nn.LayerNorm(embed_size)
#
#     def forward(self, x, mask=None):
#         residual = x
#         x_norm = self.layernorm(x)
#         output, _ = self.mh_layer(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
#         output = residual + output
#         return output
#
#
# class DocEncoder(nn.Module):
#     def __init__(self, hparams, weight=None):
#         super(DocEncoder, self).__init__()
#         self.hparams = hparams
#         if weight is None:
#             self.embedding = nn.Embedding(hparams['dct_size'], hparams['embed_size'])
#         else:
#             self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
#         self.mha = MultiheadAttentionContainer(hparams['embed_size'], nhead=hparams['nhead'], dropout=0.1)
#         self.proj = nn.Linear(hparams['embed_size'], hparams['encoder_size'])
#         self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
#
#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.permute(1, 0, 2)
#         output = self.mha(x)
#         output = F.dropout(output.permute(1, 0, 2), p=0.2)
#         output = self.proj(output)
#         output, _ = self.additive_attn(output)
#         return output
#





# class DocEncoder(nn.Module):                                                                              #下面是原始版本：
#     def __init__(self, hparams, weight=None) -> None:
#         super(DocEncoder, self).__init__()
#         self.hparams = hparams
#         if weight is None:
#             self.embedding = nn.Embedding(100, 300)
#         else:
#             self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
#         self.mha = nn.MultiheadAttention(hparams['embed_size'], num_heads=hparams['nhead'], dropout=0.1)  #此处可以替换debert
#         self.proj = nn.Linear(hparams['embed_size'], hparams['encoder_size'])
#         self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
#
#
#     def forward(self, x):
#         x = F.dropout(self.embedding(x), 0.2)
#         x = x.permute(1, 0, 2)
#         output, _ = self.mha(x, x, x)                                                                    #同上
#         output = F.dropout(output.permute(1, 0, 2), p=0.2)
#         # output = self.proj(output)
#         output, _ = self.additive_attn(output)
#         return output
