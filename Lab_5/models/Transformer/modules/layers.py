import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.head_dim = dim // num_heads #each head dimension
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(p=attn_drop)
        self.linear_out = nn.Linear(self.dim, self.dim)


    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, num_image_tokens, dim = x.shape
        #線性變換
        q = self.wq(x).view(batch_size, num_image_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, num_image_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, num_image_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # softmax(QK^T / sqrt(d_k))
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(attn)        
        #(batch_size, num_heads, seq_length, head_dim) -> (batch_size, seq_length, num_heads, head_dim)再恢復到形狀 (batch_size, seq_length, dim)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(batch_size, num_image_tokens, dim) 
        out = self.linear_out(out)
        return out

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    