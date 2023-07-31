import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size,in_c, embed_dim, layer_norm=None, image_embed=None):
        super(PatchEmbed,self).__init__()
        image_size=(image_size,image_size)
        patch_size=(patch_size,patch_size)
        self.in_c = in_c
        self.embed_dim = embed_dim 
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0]//patch_size[0],image_size[1]//image_size[1])
        self.patch_num = self.grid_size[0] ** 2
        self.image_proj = nn.Conv2d(self.in_c, self.embed_dim, kernel_size = patch_size, stride=patch_size)
        self.template_proj = nn.Conv2d(self.in_c, self.embed_dim, kernel_size = patch_size, stride=patch_size)
        self.image_norm = layer_norm(self.embed_dim) if layer_norm else nn.Identity
        self.norm = layer_norm(self.embed_dim) if layer_norm else nn.Identity
        self.image_embed = image_embed

    def forward(self,x):
        #B:batch_size   F:frames  C:channels  H:height  W:width
        
        if self.image_embed:
            # B,F,C,H,W = x.shape
            B,C,H,W = x.shape  
            assert H == self.image_size[0] and W == self.image_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            # x = x.reshape(B,-1,self.image_size[0], self.image_size[0])
            x = self.image_proj(x)
            x = x.flatten(2).transpose(-1, -2)
            x = self.image_norm(x)
        else:
            B,C,H,W = x.shape
            assert H == self.patch_size[0] and W == self.patch_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.patch_size[0]}*{self.patch_size[1]})."
            x = self.template_proj(x)
            x = x.flatten(2).transpose(-1,-2)
            x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim,num_heads=8, qkv_bias=False, qk_scale=None, atten_dropradio=0., proj_dropradio=0.):
        super(Attention,self).__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.scale = qk_scale or self.head_dim ** 0.5
        self.qkv = nn.Linear(dim, dim*3,bias=qkv_bias)
        self.atten_drop = nn.Dropout(atten_dropradio)
        self.proj = nn.Linear(dim,dim,bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_dropradio)

    def forward(self,x):
        B,N,C = x.shape
        #[B,N,C]-->[B,N,3*C]-->[B,N,3,num_heads,embed_dim_per_head]-->[3,B,num_heads,N,embed_dim_per_head]
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        
        #q_dim:[B,num_heads,N,embed_dim_per_head]
        #k_dimL:[B,num_heads,embed_dim_per_head,N]
        #atten_dim:[B,num_heads,N,N]
        attn = (q @ k.transpose(-1,-2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.atten_drop(attn)
        
        #x_dim:[B,num_heads,N,embed_dim_per_head]-->[B,N,num_heads,embed_dim_per_head]
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, dropout=0.):
        super(Mlp,self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self,dim,
                num_heads,
                qkv_bias=None,
                qk_scale=None,
                drop_ratio =0.,
                atten_dropradio=0.,
                proj_dropradio = 0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm):
        super(Block,self).__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.atten = Attention(dim=dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    atten_dropradio=atten_dropradio,
                                    proj_dropradio=proj_dropradio)
        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        hidden_features = int(dim * 0.5)
        self.mlp = Mlp(in_features=dim,hidden_features=hidden_features,out_features=dim,act_layer=act_layer)
    
    def forward(self,x):
        x = x + self.drop_path(self.atten(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,image_size, patch_size, in_c, embed_dim,deepth,num_heads,
                qkv_bias=True,
                qkv_scale=None,
                drop_ratio =0.,
                norm_layer=None,
                embed_layer=PatchEmbed,
                act_layer = None,
                image_embed = None):
        super(VisionTransformer,self).__init__()
        self.image_embed = image_embed
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm,eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(image_size=image_size, patch_size=patch_size, in_c=in_c, 
                                                                                embed_dim=self.embed_dim,
                                                                                layer_norm=norm_layer, 
                                                                                image_embed=self.image_embed)
        num_patches = self.patch_embed.patch_num
        self.pos_embed = PositionalEncoding(embed_dim)
        # self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * .02)
        self.pos_drop = nn.Dropout(drop_ratio)
        self.block = nn.Sequential(*[
            Block(dim = embed_dim, num_heads = num_heads, qkv_bias=qkv_bias, qk_scale=qkv_scale,drop_ratio=0.,
                        atten_dropradio=0.,proj_dropradio=0.,
                        act_layer=act_layer,norm_layer=norm_layer)
            for x in range(deepth)
        ])
        self.norm = norm_layer(embed_dim)
        self.apply(_init_vit_weights)
    

    def forward(self,x):
        x = self.patch_embed(x)
        #Add position information
        x = self.pos_embed(x)
        x = self.block(x)
        x = self.norm(x)
        return x


def _init_vit_weights(m):
    '''
    ViT weight initialization
    :param m: module
    '''
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class PositionalEncoding(nn.Module): 
  "Implement the PE function." 
  def __init__(self, d_model, dropout = 0, max_len=5000): 
    super(PositionalEncoding, self).__init__() 
    self.dropout = nn.Dropout(p=dropout) 
    # Compute the positional encodings once in log space. 
    #https://www.cnblogs.com/xiximayou/p/13343665.html#:~:text=transformer%E4%B8%AD%E7%9A%84%E4%BD%8D%E7%BD%AE%E5%B5%8C%E5%85%A5pytorch%E4%BB%A3%E7%A0%81%20class%20PositionalEncoding%20%28nn.Module%29%3A%20%22Implement%20the%20PE%20function.%22,__init__%20%28self%2C%20d_model%2C%20dropout%2C%20max_len%3D5000%29%3A%20%23d_model%3D512%2Cdropout%3D0.1%2C%20%23max_len%3D5000%E4%BB%A3%E8%A1%A8%E4%BA%8B%E5%85%88%E5%87%86%E5%A4%87%E5%A5%BD%E9%95%BF%E5%BA%A6%E4%B8%BA5000%E7%9A%84%E5%BA%8F%E5%88%97%E7%9A%84%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%EF%BC%8C%E5%85%B6%E5%AE%9E%E6%B2%A1%E5%BF%85%E8%A6%81%EF%BC%8C%20%23%E4%B8%80%E8%88%AC100%E6%88%96%E8%80%85200%E8%B6%B3%E5%A4%9F%E4%BA%86%E3%80%82
    pe = torch.zeros(max_len, d_model) 
    position = torch.arange(0, max_len).unsqueeze(1) 
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
      -(math.log(10000.0) / d_model)) 
    pe[:,0::2] = torch.sin(position * div_term) 
    pe[:,1::2] = torch.cos(position * div_term) 
    pe = pe.unsqueeze(0) 
    self.register_buffer('pe', pe) 
  def forward(self, x): 
    x = x + self.pe[:,:x.size(1)]
    return self.dropout(x)