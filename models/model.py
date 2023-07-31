import torch
import numpy as np
import torch.nn as nn
from models.group_vit import GroupingBlock, AssignAttention
from models.visiontransformer import VisionTransformer

class SiamVIT(nn.Module):
    def __init__(self, image_size, template_size, in_dim, embed_dim, num_heads, deepth, in_c, patch_size,
                 mlp_radio, Frame, norm_layer = nn.LayerNorm):
        super(SiamVIT, self).__init__()
        self.in_c = in_c
        self.in_dim = in_dim
        self.embed_dim  = embed_dim
        self.image_size = image_size
        self.template_size = template_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.deepth = deepth
        self.mlp_radio = mlp_radio
        self.Frame = Frame
        self.norm_layer = nn.LayerNorm(embed_dim)
        self.image_embed = VisionTransformer(image_size = self.image_size, 
                                             patch_size = self.patch_size, 
                                             in_c = self.in_c, 
                                             embed_dim = (self.embed_dim * self.Frame), 
                                             deepth = self.deepth,
                                             num_heads = self.num_heads,
                                             image_embed = True)
        self.template_embed = VisionTransformer(image_size = self.template_size, 
                                                patch_size = self.patch_size, 
                                                embed_dim = self.embed_dim,
                                                in_c = 3, 
                                                deepth = self.deepth,
                                                num_heads = self.num_heads,
                                                image_embed = False)
        # self.grouping_block = GroupingBlock(dim=self.in_dim, 
        #                                     out_dim = self.embed_dim,
        #                                     num_heads=self.num_heads, 
        #                                     num_group_token = 1,
        #                                     num_output_group = 1,
        #                                     norm_layer = nn.LayerNorm,
        #                                     hard=True, 
        #                                     gumbel=True)
        self.assign = AssignAttention( dim = self.in_dim,
                                       num_heads = 1,
                                       qkv_bias = False,
                                       hard = True,
                                       gumbel = True)
        self.Mim  = MiM(channels = in_c)
        self.fc = nn.Linear(self.embed_dim * self.Frame, self.embed_dim)
        self.templates = nn.Parameter(self.makeTemplates(self.patch_size**2).cuda())
     
    def forward(self, image):
        image = self.Mim(image)
        image = self.image_embed(image)
        image = self.fc(image)
        image = self.norm_layer(image)
        attn_list = []
        for item in self.templates:
            item = self.norm_layer(self.template_embed(item.unsqueeze(dim=0)))
            attn_list.append(item.squeeze())
        templates = torch.stack([item for item in attn_list])
        templates = templates.repeat(image.size(0),1,1)
        x, attn = self.assign(image, templates, return_attn = True)
        attn = attn['soft'].flatten(1)
        top_5_pred = torch.topk(attn, 5, dim=1, largest = True, sorted = True, out=None )
        return x, attn, top_5_pred

    def NormalDist4Two(self,x, y, ux, uy, sigmax = 0.55, sigmay = 0.55):
        return 1/(2*np.pi*sigmax*sigmay) * np.exp(-0.5*((x - ux)**2/sigmax**2 + (y - uy)**2/sigmay**2))

    def makeTemplates(self,total):
        templates = torch.zeros([total, 1, self.patch_size, self.patch_size])
        for index in range(total): 
            targetx = index // self.patch_size
            targety = index % self.patch_size

            for x in range(self.patch_size):
                for y in range(self.patch_size):
                    value = self.NormalDist4Two(x,y,targetx,targety)
                    templates[index, 0, x, y] = value
        templates = templates.repeat(1,3,1,1)
        return templates

class MimSubModel(nn.Module):
    def __init__(self, channels, lastedReLU = True):
        super(MimSubModel,self).__init__()
        width = channels * 2

        self.conv1 = nn.Conv2d(channels, width , kernel_size = 1, bias=False )
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size = 3, padding = 1,  groups = width, bias = False )
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, channels, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(True)
        self.lastReLU = lastedReLU

    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identity

        if self.lastReLU:
            out = self.lastReLU(out)
        return out 

class MiM(nn.Module):
    def __init__(self, channels, deepth = 1):
        super(MiM,self).__init__()
        model = []
        for _ in range(deepth - 1):
            model.append(MimSubModel(channels))
        model.append(MimSubModel(channels, False))
        self.submodel = nn.Sequential(*model)
    
    def forward(self,img):
        img = self.submodel(img)
        return img