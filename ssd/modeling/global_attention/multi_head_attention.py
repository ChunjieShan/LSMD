import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 dim: int,
                 num_heads: int) -> None:
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

        self.Wq = nn.Linear(dim, dim * num_heads)
        self.Wk = nn.Linear(dim, dim * num_heads)
        self.Wv = nn.Linear(dim, dim * num_heads)

        self.proj = nn.Linear(dim * num_heads, self.orig_dim)

        self.attn_drop = nn.Dropout(0.5)
        self.proj_drop = nn.Dropout(0.5)

        self.layer_norm = nn.LayerNorm(dim)

        self.act = nn.ReLU()

    def forward(self, 
                curr_feat: torch.Tensor,
                ref_feat: torch.Tensor):
        bs, n, c = curr_feat.shape

        residual = curr_feat

        q = self.Wq(curr_feat).reshape(bs, n, self.num_heads, -1) 
        k = self.Wk(ref_feat).reshape(bs, n, self.num_heads, -1) 
        v = self.Wv(ref_feat).reshape(bs, n, self.num_heads, -1) 

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        attn = self.attn_drop(F.softmax((q @ k.transpose(-2, -1) * self.d), dim=-1))
        x = attn @ v

        x = x.transpose(1, 2).contiguous().view(bs * n, -1)
        x = self.act(self.proj(x)).reshape(1, -1, self.orig_dim)

        x = x + residual
        x = self.layer_norm(x)
        # x = self.proj_drop(x)

        return x
        

class GlobalPoolRandomSampler(nn.Module):
    def __init__(self,
                 num_imgs : int = 32,
                 global_size: int = 4) -> None:
        super().__init__()
        self.seed = 41
        torch.random.manual_seed(self.seed)
        self.num_imgs = num_imgs
        self.global_size = global_size
        
    def forward(self, x: torch.Tensor):
        rand_seq = torch.randint(0, self.num_imgs, (self.global_size,)) 
        rand_seq, _ = rand_seq.sort()
        global_list = []
        for rand_idx in rand_seq:
            global_list.append(x[rand_idx, ...])            

        return torch.stack(global_list, dim=0)


class FeaturesAggregate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(256, 8)
        self.dims_alignment = nn.ModuleList([
            nn.Linear(100, 256),
            nn.Linear(25, 256),
            nn.Linear(9, 256)
        ])

        self.dims_recover = nn.ModuleList([
            nn.Linear(256, 100),
            nn.Linear(256, 25),
            nn.Linear(256, 9)
        ])

        self.x1_reduce = nn.Conv2d(512, 256, (1, 1), (1, 1))
        self.conv1 = nn.Conv2d(1024, 256, (1, 1), (1, 1))
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, (3, 3), (1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, (1, 1), (1, 1))
        self.bn3 = nn.BatchNorm2d(256)

        self.act = nn.ReLU()

        self.feat_size = [10, 5, 3]
        self.global_pool_sampler = GlobalPoolRandomSampler()

    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                x3: torch.Tensor):
        """Feature Aggregation Module forward function.

        Args:
            x1 (torch.Tensor): (bs, 512, 10, 10)
            x2 (torch.Tensor): (bs, 256, 5, 5)
            x3 (torch.Tensor): (bs, 256, 3, 3)
        """
        # if x1.shape[1] == 512:
        x1 = self.x1_reduce(x1)

        if x1.ndim == 4:
            x1 = x1.flatten(2)
            x2 = x2.flatten(2)
            x3 = x3.flatten(2)

        num_imgs, _, _ = x1.shape 
        xs = [x1, x2, x3]
        total_attn_feat = []

        for i in range(len(xs)):
            x = xs[i]  # features of only one head
            x_align = self.dims_alignment[i](x)  # align to the same dimension

            curr_head_attn_feat = []  # features list for current head

            for idx in range(num_imgs):
                aggregated_feats_list = []
                curr_feat = x_align[idx, ...]  # features of current img
                residual = x[idx, ...]  # for residual connection

                global_pool_feats = self.global_pool_sampler(x_align)  # generating a random global pool
                ref_feat_list = global_pool_feats.unbind(0)  # global pool features

                # Attention module
                for ref_feat in ref_feat_list:
                    # computing attention
                    aggregated_feats = self.attn(curr_feat.unsqueeze(0), ref_feat.unsqueeze(0))
                    aggregated_feats = self.dims_recover[i](aggregated_feats)
                    output = residual + aggregated_feats  # residual
                    aggregated_feats_list.append(output)

                curr_img_attn_feat = torch.cat(aggregated_feats_list, dim=0)
                curr_head_attn_feat.append(curr_img_attn_feat)
            curr_head_attn_feat = torch.stack(curr_head_attn_feat, dim=0)
            curr_head_attn_feat = torch.reshape(curr_head_attn_feat, (curr_head_attn_feat.shape[0], curr_head_attn_feat.shape[1] * curr_head_attn_feat.shape[2], self.feat_size[i], -1))

            curr_head_attn_feat = self.act(self.bn1(self.conv1(curr_head_attn_feat)))
            curr_head_attn_feat = self.act(self.bn2(self.conv2(curr_head_attn_feat)))
            curr_head_attn_feat = self.act(self.bn3(self.conv3(curr_head_attn_feat)))
            # print(total_single_attn_feat.shape)

            total_attn_feat.append(curr_head_attn_feat)
        
        return total_attn_feat



if __name__ == "__main__":
    x1 = torch.randn((32, 256, 10, 10))
    x2 = torch.randn((32, 256, 5, 5))
    x3 = torch.randn((32, 256, 3, 3))
    # x = torch.cat((x1, x2, x3), dim=2)
    # global_sampler = GlobalPoolRandomSampler()
    # print(global_sampler(x).shape)
    agg = FeaturesAggregate()
    torch.onnx.export(
        agg,
        (x1, x2, x3),
        "./aggregate.onnx",
        input_names=["input1", "input2", "input3"]
    )
    # agg(x1, x2, x3)
    # print(len(agg(x1, x2, x3)))
    # ref_feat = torch.randn((1, 256, 256))
    # curr_feat = torch.randn((1, 256, 256))
    # msa = MultiHeadAttention(256, 8)
    # msa(curr_feat, ref_feat)
        
