import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 dim: int,
                 num_heads: int,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.) -> None:
        """MSA module to aggregate global semantic information

        Args:
            dim (int): dimention of the input features
            num_heads (int): multi-head amount
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.orig_dim = dim
        self.d = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)

        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, 
                x: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.d
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim)) 

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

        
class DropPath(nn.Module):
    def __init__(self, prob: float) -> None:
        super().__init__()
        self.drop_prob = prob

    def forward(self, x):
        if self.drop_prob == 0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)

        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)

        return x * random_tensor


class FFN(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))

        return x

        
class TransformerBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path_rate=0.,
                 mlp_ratio=4.0,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.increase_dim = nn.Linear(49, dim)
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.layer_scale1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ffn = FFN(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop_path_rate)
        self.layer_scale2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor):
        x = self.increase_dim(x)
        x = x + self.drop_path1(self.layer_scale1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.layer_scale2(self.ffn(self.norm2(x))))
        return x


if __name__ == "__main__":
    x = torch.randn((320, 128, 49))

    trans_block = TransformerBlock(128, 8, 0.5, 0.5, 0.5, 4)
    result = trans_block(x)
    print(result.shape)
