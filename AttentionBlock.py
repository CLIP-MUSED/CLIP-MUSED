import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from visualizer import get_local

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
            #nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    @get_local('attn')
    def forward(self, x):

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class DeepAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()

        
        self.attention = Attention(dim, heads = heads, dim_head = dim_head)
        self.norm1 = nn.BatchNorm1d(dim)
        self.FeedForward = FeedForward(dim, mlp_dim)
        self.norm2 = nn.BatchNorm1d(dim)

        nn.init.constant_(self.norm2.weight, 0)
        nn.init.constant_(self.norm2.bias, 0)
       

    def forward(self, x):

        #residual = x
        n, c, h, w, d = x.size()

        x = x.view(n,c,-1)
        x = x.permute(0,2,1)

        x = self.attention(x)
        x = x.permute(0,2,1)
        x = self.norm1(x)

        x = x.permute(0,2,1)
        x = self.FeedForward(x)
        x = x.permute(0,2,1)
        x = self.norm2(x)
        
        x = x.view(n, c, h, w, d)
        #x = x + residual

        return x

class ShallowAttention(nn.Module):

    def __init__(self, dim1 = 128, dim2 = 128):
        super().__init__()
        
        #dim = 256
        self.mlp1  = nn.Sequential(
            nn.Conv1d(dim1, dim1//2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim1//2, dim1, kernel_size=1)
        )
        #self.norm1 = nn.LayerNorm(dim2)
        self.norm1 = nn.BatchNorm1d(dim1)

        self.mlp2  = nn.Sequential(
            nn.Linear(dim2, dim2//2),
            nn.GELU(),
            nn.Linear(dim2//2, dim2)
        )
        #self.norm2 = nn.LayerNorm(dim2)
        self.norm2 = nn.BatchNorm1d(dim1)

        nn.init.constant_(self.norm2.weight, 0)
        nn.init.constant_(self.norm2.bias, 0)   


    def forward(self, x):
        #print('x', x.size())
        n, c, h, w, d = x.size()
        x = x.view(n, c, -1)
        #print('x1', x.size())

        #x = x + self.mlp1(x)
        #print('x2', x.size())
        x = x + self.norm1(self.mlp1(x))
    
        #x = x + self.mlp2(x)
        #x = self.mlp2(x)
        #x = self.norm2(x)
        x = x + self.norm2(self.mlp2(x))

        x = x.view(n, c, h, w, d)
      
        return x

class DeepAttention_cls(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()

        self.attention = Attention(dim, heads = heads, dim_head = dim_head)
        self.norm1 = nn.BatchNorm1d(dim)
        self.FeedForward = FeedForward(dim, mlp_dim)
        self.norm2 = nn.BatchNorm1d(dim)

        nn.init.constant_(self.norm2.weight, 0)
        nn.init.constant_(self.norm2.bias, 0)
       

    def forward(self, x, cls_token):

        #residual = x
        n, c, h, w, d = x.size()

        x = x.view(n,c,-1)
        x = x.permute(0,2,1)
        x = torch.cat((cls_token, x), dim=1)
        # print(x.size())
        x = self.attention(x)
        x = x.permute(0,2,1)
        x = self.norm1(x)

        x = x.permute(0,2,1)
        x = self.FeedForward(x)
        x = x.permute(0,2,1)
        x = self.norm2(x)
        # print(x.size())
        cls_token = x[:, :, 0:1].permute(0,2,1)
        x = x[:, :, 1:].view(n, c, h, w, d)
        #x = x + residual
        return x, cls_token


class DeepAttention_token(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()

        self.attention = Attention(dim, heads = heads, dim_head = dim_head)
        self.norm1 = nn.BatchNorm1d(dim)
        self.FeedForward = FeedForward(dim, mlp_dim)
        self.norm2 = nn.BatchNorm1d(dim)

        nn.init.constant_(self.norm2.weight, 0)
        nn.init.constant_(self.norm2.bias, 0)
       

    def forward(self, x, hlv_token, llv_token):

        #residual = x
        n, c, h, w, d = x.size()

        x = x.view(n,c,-1)
        x = x.permute(0,2,1)
        x = torch.cat((hlv_token, llv_token, x), dim=1)
        # print(x.size())
        x = self.attention(x)
        x = x.permute(0,2,1)
        x = self.norm1(x)

        x = x.permute(0,2,1)
        x = self.FeedForward(x)
        x = x.permute(0,2,1)
        x = self.norm2(x)
        # print(x.size())
        hlv_token, llv_token = x[:, :, 0:1].permute(0,2,1), x[:, :, 1:2].permute(0,2,1)
        x = x[:, :, 2:].view(n, c, h, w, d)
        #x = x + residual
        return x, hlv_token, llv_token
