import torch
import torch.nn as nn
import torch.nn.functional as F
from model.doc_encoder import DocEncoder
from model.attention import  AdditiveAttention

from model.attention import  AdaptiveAttention
from model.attention import  ScaledDotProductAttention
from model.attention import  LocalAttention
#from model.attention import  SelfAttention


class NRMSModel(nn.Module):
    def __init__(self, hparams, weight=None):
        super(NRMSModel, self).__init__()
        self.hparams = hparams
        self.doc_encoder = DocEncoder(hparams, weight=weight)
        self.mha = nn.MultiheadAttention(hparams['encoder_size'], hparams['nhead'], dropout=0.1)  #添加debert的位置
        #self.mha = MultiheadAttentionContainer(hparams['encoder_size'], hparams['nhead'], dropout=0.1)

        self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size'])
        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])                  #此为attention.py下面的方法 1.加型注意力机制

        self.Adaptive_attn = AdaptiveAttention(hparams['encoder_size'], hparams['v_size'])                  #2.自适应注意力机制
        self.ScaledDotProduct_attn = ScaledDotProductAttention(hparams['encoder_size'], hparams['v_size'])  #3.缩放点积注意力机制
        self.Local_attn = LocalAttention(hparams['encoder_size'], hparams['v_size'])                        #4.邻近注意力机制
        #self.Self_attn = SelfAttention(hparams['encoder_size'], hparams['v_size'])                          #5.按位置注意力机制

        self.criterion = nn.CrossEntropyLoss()
        self.layernorm = nn.LayerNorm(hparams['encoder_size'])

    def forward(self,
                label,
                impr_index,
                user_index,
                candidate_title_index,
                click_title_index,
                ):
                                                                    #定义模型的前向传播函数，输入参数为label、impr_index、user_index、candidate_title_index、click_title_index
        num_click_docs = click_title_index.shape[1]                 #num_click_docs为click_title_index矩阵的第二维大小
        num_cand_docs = candidate_title_index.shape[1]              #num_cand_docs为candidate_title_index矩阵的第二维大小
        num_user = click_title_index.shape[0]                       #num_user为click_title_index矩阵的第一维大小
        seq_len = click_title_index.shape[2]                        #seq_len为click_title_index矩阵的第三维大小
        clicks = click_title_index.reshape(-1, seq_len)             #将click_title_index重塑为一个N×seq_len的矩阵，并存储在clicks变量中
        cands = candidate_title_index.reshape(-1, seq_len)
        click_embed = self.doc_encoder(clicks)                                      #doc_encoder方法里面包含AdditiveAttention
        cand_embed = self.doc_encoder(cands)
        click_embed = click_embed.reshape(num_user, num_click_docs, -1)             #reshapeclick_embed为num_user×num_click_docs×-1大小的多维数组
        cand_embed = cand_embed.reshape(num_user, num_cand_docs, -1)
        click_embed = click_embed.permute(1, 0, 2)                                  #使用permute函数将click_embed的维度重新排列，使其变为num_click_docs×num_user×-1大小的多维数组
        click_output, _ = self.mha(click_embed, click_embed, click_embed)           #使用MultiheadAttention层，对click_embed进行自注意力计算，得到click_output

        #此处添加残差和layernorm会导致结果急剧下降
        # click_output =click_output+click_embed
        # click_output = self.layernorm(click_output)

        click_output = F.dropout(click_output.permute(1, 0, 2), 0.2)                # 对自注意力计算的结果click_output进行dropout处理

        #click_repr = self.proj(click_output)
        click_output = self.proj(click_output)                                      #源代码没有这句 值得商榷

        click_repr, _ = self.additive_attn(click_output)                            #1.加型注意力机制

        #click_repr, _ = self.Adaptive_attn(click_output)                           #2.自适应注意力机制
        #click_repr, _ = self.ScaledDotProduct_attn(click_output)                   #3.缩放点积注意力机制
        #click_repr, _ = self.Local_attn(click_output)                              #4.邻近注意力机制
        #click_repr, _ = self.Self_attn(click_output)                               #5.按位置注意力机制

        logits = torch.bmm(click_repr.unsqueeze(1), cand_embed.permute(0, 2, 1)).squeeze(1) # [B, 1, hid], [B, 10, hid]
        if label is not None:
            loss = self.criterion(logits, label.long())
            return loss, logits
        return torch.sigmoid(logits)





# class NRMSModel(nn.Module):
#    def __init__(self, hparams, weight=None):
#        super(NRMSModel, self).__init__()
#        self.hparams = hparams
#        self.doc_encoder = DocEncoder(hparams, weight=weight)
#        self.mha = nn.MultiheadAttention(hparams['encoder_size'], hparams['nhead'], dropout=0.1)  #添加debert的位置
#
#        self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size'])
#        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])        #此为attention.py下面的方法
#        self.criterion = nn.CrossEntropyLoss()
#
#     def forward(self,
#                label,
#                impr_index,
#                user_index,
#                candidate_title_index,
#                click_title_index,
#                ):
#         """forward
#
#         Args:
#             clicks (tensor): [num_user, num_click_docs, seq_len]
#             cands (tensor): [num_user, num_candidate_docs, seq_len]
#         """
#        num_click_docs = click_title_index.shape[1]
#        num_cand_docs = candidate_title_index.shape[1]
#        num_user = click_title_index.shape[0]
#        seq_len = click_title_index.shape[2]
#        clicks = click_title_index.reshape(-1, seq_len)
#        cands = candidate_title_index.reshape(-1, seq_len)
#        click_embed = self.doc_encoder(clicks)                                      #doc_encoder方法里面包含AdditiveAttention
#        cand_embed = self.doc_encoder(cands)
#        click_embed = click_embed.reshape(num_user, num_click_docs, -1)
#        cand_embed = cand_embed.reshape(num_user, num_cand_docs, -1)
#        click_embed = click_embed.permute(1, 0, 2)
#        click_output, _ = self.mha(click_embed, click_embed, click_embed)           #此处用到mha，在上面代表MultiheadAttention
#        click_output = F.dropout(click_output.permute(1, 0, 2), 0.2)
#
#        # click_repr = self.proj(click_output)
#        click_repr, _ = self.additive_attn(click_output)
#        logits = torch.bmm(click_repr.unsqueeze(1), cand_embed.permute(0, 2, 1)).squeeze(1) # [B, 1, hid], [B, 10, hid]
#        if label is not None:
#            loss = self.criterion(logits, label.long())
#            return loss, logits
#        return torch.sigmoid(logits)



# class NRMSModel(nn.Module):
#     def __init__(self, hparams, weight=None):
#         super(NRMSModel, self).__init__()
#         self.hparams = hparams
#         self.doc_encoder = DocEncoder(hparams, weight=weight)
#         self.mha = nn.MultiheadAttention(hparams['encoder_size'], hparams['nhead'], dropout=0.1)
#
#         self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size'])
#         self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
#         self.criterion = nn.CrossEntropyLoss()
#
#         self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size'])
#
#     def forward(self,
#                 label,
#                 impr_index,
#                 user_index,
#                 candidate_title_index,
#                 click_title_index,
#                 ):
#         """forward
#
#         Args:
#             clicks (tensor): [num_user, num_click_docs, seq_len]
#             cands (tensor): [num_user, num_candidate_docs, seq_len]
#         """
#         num_click_docs = click_title_index.shape[1]
#         num_cand_docs = candidate_title_index.shape[1]
#         num_user = click_title_index.shape[0]
#         seq_len = click_title_index.shape[2]
#         clicks = click_title_index.reshape(-1, seq_len)
#         cands = candidate_title_index.reshape(-1, seq_len)
#         click_embed = self.doc_encoder(clicks)
#         cand_embed = self.doc_encoder(cands)
#         click_embed = click_embed.reshape(num_user, num_click_docs, -1)
#         cand_embed = cand_embed.reshape(num_user, num_cand_docs, -1)
#         click_embed = click_embed.permute(1, 0, 2)
#         click_output, _ = self.mha(click_embed, click_embed, click_embed)
#         click_output = F.dropout(click_output.permute(1, 0, 2), 0.2)
#
#         # click_repr = self.proj(click_output)
#
#         #click_output = self.proj(click_output)
#         click_repr, _ = self.additive_attn(click_output)
#         logits = torch.bmm(click_repr.unsqueeze(1), cand_embed.permute(0, 2, 1)).squeeze(1) # [B, 1, hid], [B, 10, hid]
#         if label is not None:
#             loss = self.criterion(logits, label.long())
#             return loss, logits
#         return torch.sigmoid(logits)