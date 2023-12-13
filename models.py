# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from transformer import TransformerBlock
from utils import split_last, merge_last

# https://github.com/tatp22/multidim-positional-encoding
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
#  bert_emb에 position에 Summer(PositionalEnocding1D) 쓰는게 더 좋지만(입력 길이 제한 없음) 다음 학습부터
#  일단 pretrain 된 거에는 기존 positional_encoding 사용


class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = 660 # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 8 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 32 # Maximum Length for Positional Embeddings
    #n_segments: int = 2 # Number of Sentence Segments
    pad_idx: int = 0 #500#230

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

    def set_vocab_size(cls, size):
        Config.vocab_size = size


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class PositionalEncoding(nn.Module):

        def __init__(self, d_hid, n_position=256):
            super(PositionalEncoding, self).__init__()

            # Not a parameter
            self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

        def _get_sinusoid_encoding_table(self, n_position, d_hid):
            ''' Sinusoid position encoding table '''
            def get_position_angle_vec(position):
                return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

            sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

            return torch.FloatTensor(sinusoid_table).unsqueeze(0)

        def forward(self, x):
            if x.size(1) !=  self.pos_table[:,:x.size(1)].size(1):
                print(x.size())
                print(self.pos_table[:,:x.size(1)].size())
            return x + self.pos_table[:, :x.size(1)].clone().detach()


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size+1, cfg.dim, padding_idx = cfg.pad_idx) # token embedding
        print('--------cfg vocab size------------')
        print(cfg.vocab_size)
        #self.pos_embed = nn.Embedding(32, cfg.dim) # position embedding

        # drop
        #self.norm = nn.LayerNorm(cfg.dim)#LayerNorm(cfg)
        #self.drop = nn.Dropout(cfg.p_drop_hidden)


    def forward(self, x):
        #seq_len = x.size(1)
        #pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        
        e = self.tok_embed(x) #+ self.pos_embed(pos)
        return e 
        
        #drop
        #return self.drop(self.norm(e))

 
class TokenEmbedding(nn.Embedding):     #토큰을 임베딩 벡터로 변환
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)     #embed_size: 임베딩 벡터의 차원 결정, padding_idx: 패딩토큰 인덱스 지정
        #print("TokenEmbedding vocab_size:", vocab_size)  # vocab_size 출력

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=None):
        super().__init__()

        if max_len is None:
            max_len = 512

        print(d_model)  #999
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()  #pe: 위치 임베딩 행렬. max_len x d_model 크기고 0으로 초기화됨
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)    #0부터 max_len까지의 값 가지는 텐서
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()     #각 차원에 따른 감소율. 위치랑 차원에 따라 다른 주기 ?

        #pe 각각 shape 보기
        #print('pe---------------------------')
        #print(pe)
        #print(pe.shape) #torch.size([1000, 864])
        pe[:, 0::2] = torch.sin(position * div_term)    #짝수 인덱스는 sin 함수로 위치 임베딩 계산
        #print('pe-sin ------------------------------')
        #print(pe)
        #print(pe.size)
        #print(pe.shape) #torch.size([1000, 864])
        pe[:, 1::2] = torch.cos(position * div_term)    #홀수 인덱스는 cos 함수로 위치 임베딩 계산
        #print('pe-cos ------------------------------')
        #print(pe)
        #print(pe.size)
        #print(pe.shape)
        #PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
        #PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))


        pe = pe.unsqueeze(0)    #배치 추가
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe.repeat(x.size(0), 1, 1)
        return pe[:, :x.size(1)]
    
class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        # self.position = PositionalEncoding1D(embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        #self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        #print('--------BERTEmbedding sequence-----------')
        #print(sequence)     #bert_input [1,150]으로 들어옴
        token_embeddings = self.token(sequence)
        #print('token_embeddings-------------------------')
        #print(token_embeddings)
        #print(token_embeddings.shape)   #[1,150,864]
        # positional_embeddings = self.position(sequence)

        x = self.position(token_embeddings)
        return self.dropout(x)



class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=512, n_layers=12, attn_heads=8, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)    #embed_size 768

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
 
    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        #mask = (x != 1111).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)    #패딩 인덱스 1111로 바꿈
        #print('-------------x before mask')
        #print(x)
        #print(x.shape)  #[1,2,15]
        ##############mask = (x != 0).unsqueeze(1).repeat(1, 1, x.size(1), 1)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        #print('----------mask------------')
        #print(mask)         #마스크 어텐션 아래의 shape로 만들어짐
        #print(mask.shape)   #[8, 1,150,150]
        #print('---------------------------')

        #print('--------------x before embedding-------------')
        #print(x)        #bert_input(basic block 1개), [1,2,15]형태
        ##########################x = x.reshape(-1, x.shape[-1])
        #print(x)        #위에꺼에서 괄호 없앰
        #print(x.shape)  #[2,15]로 바뀜
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        ##print('embedded x---------------------')
        #print(x)
        #print(x.shape)  #torch.Size([1, 150, 864])

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)        #x[8, 150, 512], mask=[1,1,4,15]

        return x
    

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model -> NSP 지움
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x):
        x = self.bert(x)
        #print("BERTLM")
        #print(x)
        
        return self.mask_lm(x)
    


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

        self.output = nn.Linear(cfg.dim, cfg.dim)
        #drop
        #self.drop = nn.Dropout(cfg.p_drop_attn)

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        del x
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
 
        del q
        del k

        # masking
        #b,h,s,s = scores.size()
        #indices = torch.triu_indices(s,s,offset=1)
        #scores[:,:,indices[0],indices[1]] = float('-inf')

        b,h,s,s = scores.size()
        masking = np.zeros((s,s))
        for i in range(s):
            for j in range(s):
                if i<j: masking[i][j] = (s+i-j)/s
                else: masking[i][j] = (s-i+j)/s
        masking = torch.Tensor(masking).cuda()      ###에러 ???
        scores = scores * masking

        del masking

        scores = F.softmax(scores, dim=-1)

        #drop
        #scores = self.drop(F.softmax(scores, dim=-1))

        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores

        del v
        del scores


        return self.output(h)


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        #self.norm1 =  nn.LayerNorm(cfg.dim)#LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        #self.norm2 =  nn.LayerNorm(cfg.dim)#LayerNorm(cfg)

        #drop
        #self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        h = self.attn(x)
        
        #drop
        #h = self.norm1(x + self.drop(self.proj(h)))
        #h = self.norm2(h + self.drop(self.pwff(h)))

        #h = self.norm1(x + self.proj(h))
        #h = self.norm2(h + self.pwff(h))
        h = x + self.proj(h)
        h = h + self.pwff(h)
        return h


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks """
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h

class DeepPM(nn.Module):
    """DeepPM model with Trasformer """
    def __init__(self, cfg):
        super().__init__()
        print("CFG PRINT")
        print(cfg)  #config(vocab_size=None, dim=512, n_layers=224, n_heads=8, dim_ff=2048, p_drop_hidden=0.1, p_drop_attn=0.1, max_len=100, pad_idx=628)

                # Initialize the BERT model
        self.bert = BERT(vocab_size=cfg.vocab_size,
                         hidden=cfg.dim,
                         n_layers=cfg.n_layers,
                         attn_heads=cfg.n_heads,
                         dropout=cfg.p_drop_hidden)


        self.pad_idx = cfg.pad_idx
        self.embed = Embeddings(cfg)
        #self.embed = BERTEmbedding(vocab_size=cfg.vocab_size, embed_size=cfg.dim, dropout=cfg.p_drop_hidden)

        self.pre_blocks = nn.ModuleList([Block(cfg) for _ in range(2)])#cfg.n_layers)])
        self.token_blocks = nn.ModuleList([Block(cfg) for _ in range(2)])#cfg.n_layers)])

        #self.pos_embed = PositionalEncoding(cfg.dim, 400)

        #self.pos_embed = nn.Embedding(250, cfg.dim) # position embedding, 1500
        self.instruction_blocks = nn.ModuleList([Block(cfg) for _ in range(4)])#cfg.n_layers)])
        self.prediction = nn.Linear(cfg.dim,1)
    ########error  
    def forward(self, item):

        bb_list = []
        token_len = 0
        for instr, token_inputs in zip(item.block.instrs, item.x):
            bb_list.append(token_inputs)
            ##print("token_inputs")
            #print(token_inputs)
            if len(token_inputs) > token_len :
                token_len = len(token_inputs)

        embed_input = []
        for l in bb_list:
            while len(l) < token_len:
                l.insert(len(l), self.pad_idx)
            embed_input.append(l)

        t_output = self.embed(torch.cuda.LongTensor(embed_input))
        t_output = self.embed.position(t_output)

        n_instr, n_token, n_dim = t_output.size()

        t_output = t_output.view([n_instr*n_token, n_dim])  #1차원으로 바꿈
        t_output = t_output.unsqueeze(0)

        for t_block in self.pre_blocks:
            t_output = t_block(t_output)
    
        t_output = t_output.squeeze(0)
        t_output = t_output.view([n_instr, n_token, n_dim])

        for t_block in self.token_blocks:
            t_output = t_block(t_output)
        
        t_output = t_output[:,0,:]
        while len(t_output.size())<3:
            t_output = t_output.unsqueeze(0)
        i_output = self.pos_embed(t_output)
        del t_output

        for i_block in self.instruction_blocks:
            i_output = i_block(i_output)

        i_output = i_output.squeeze(0)
        i_output = i_output.sum(dim = 0)
        
        # do prediction layer
        out = self.prediction(i_output).squeeze()

        return out
            

