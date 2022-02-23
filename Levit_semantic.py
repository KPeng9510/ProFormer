# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Copyright 2020 Ross Wightman, Apache-2.0 License
import torch.nn.functional as F
import torch
import itertools
import utils
import pickle
from timm.models.vision_transformer import trunc_normal_
from timm.models.registry import register_model
import torch.nn as nn


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


specification = {
    'LeViT_128S': {
        'C': '128_256_384', 'D': 16, 'N': '4_6_8', 'X': '2_3_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth'},
    'LeViT_128': {
        'C': '128_256_384', 'D': 16, 'N': '4_8_12', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth'},
    'LeViT_192': {
        'C': '192_288_384', 'D': 32, 'N': '3_5_6', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth'},
    'LeViT_256': {
        'C': '256_384_512', 'D': 32, 'N': '4_6_8', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth'},
    'LeViT_384': {
        'C': '384_512_768', 'D': 32, 'N': '6_9_12', 'X': '4_4_4', 'drop_path': 0.1,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-384-9bdaf2e2.pth'},
}

__all__ = [specification.keys()]


@register_model
def LeViT_128S(num_classes=1000, distillation=True,
               pretrained=False, fuse=False):
    return model_factory(**specification['LeViT_128S'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)


@register_model
def LeViT_128(num_classes=1000, distillation=True,
              pretrained=False, fuse=False):
    return model_factory(**specification['LeViT_128'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)


@register_model
def LeViT_192(num_classes=1000, distillation=True,
              pretrained=False, fuse=False):
    return model_factory(**specification['LeViT_192'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)


@register_model
def LeViT_256(num_classes=1000, distillation=True,
              pretrained=False, fuse=False):
    return model_factory(**specification['LeViT_256'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)


@register_model
def LeViT_384(num_classes=1000, distillation=True,
              pretrained=False, fuse=False, in_chans=3):
    return model_factory(**specification['LeViT_384'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse, in_chans=in_chans)


@register_model
def LeViT_384_cascaded(num_classes=1000, distillation=True,
                       pretrained=False, fuse=False, in_chans=3):
    return model_factory_cascaded(**specification['LeViT_384'], num_classes=num_classes,
                                  distillation=distillation, pretrained=pretrained, fuse=fuse, in_chans=in_chans)


FLOPS_COUNTER = 0


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = ((resolution + 2 * pad - dilation *
                          (ks - 1) - 1) // stride + 1) ** 2
        FLOPS_COUNTER += a * b * output_points * (ks ** 2) // groups

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1), w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = resolution ** 2
        FLOPS_COUNTER += a * b * output_points

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)
        global FLOPS_COUNTER
        FLOPS_COUNTER += a * b

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def b16(n, activation, in_chans, resolution=224):
    return torch.nn.Sequential(
        Conv2d_BN(in_chans, n // 8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h, resolution=resolution)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

        global FLOPS_COUNTER
        # queries * keys
        FLOPS_COUNTER += num_heads * (resolution ** 4) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution ** 4)
        # attention * v
        FLOPS_COUNTER += num_heads * self.d * (resolution ** 4)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
        1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class Subsample(torch.nn.Module):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class AttentionSubsample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_ ** 2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)

        self.q = torch.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim, resolution=resolution_))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

        global FLOPS_COUNTER
        # queries * keys
        FLOPS_COUNTER += num_heads * \
                         (resolution ** 2) * (resolution_ ** 2) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution ** 2) * (resolution_ ** 2)
        # attention * v
        FLOPS_COUNTER += num_heads * \
                         (resolution ** 2) * (resolution_ ** 2) * self.d

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
        1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
               (self.attention_biases[:, self.attention_bias_idxs]
                if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x
def compute_causal_constraint(adj, concept_dimension):
    unit = torch.eye(concept_dimension).cuda()
    # print(adj)
    return torch.trace(torch.matrix_power(unit + adj @ adj, 1))
class causal_aggregate(torch.nn.Module):
    def __init__(self, concept_dimension):
        super(causal_aggregate, self).__init__()
        one_matrix = torch.ones([concept_dimension, concept_dimension]).cuda()
        self.adjacency = nn.Parameter(torch.triu(one_matrix).cuda())
        self.adjacency.requires_grad = True
        self.softmax = nn.Softmax(dim=1)
        self.concept_dimension = concept_dimension
    def forward(self, clip):
        # input clip size: batchsize,10,1024
        # print(self.adjacency.requires_grad)

        adjacency = torch.triu(self.softmax(self.adjacency))
        loss = compute_causal_constraint(adjacency, self.concept_dimension)
        return (clip.permute(0, 2, 1) @ adjacency).permute(0,2,1), loss

class LeViT(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=256,
                 patch_size=16,
                 in_chans=3,
                 num_classes=512,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=Hardswish,
                 mlp_activation=Hardswish,
                 distillation=True,
                 drop_path=0):
        super().__init__()
        global FLOPS_COUNTER

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation

        self.patch_embed = hybrid_backbone

        self.blocks = []
        down_ops.append([''])
        resolution = img_size // patch_size
        self.intermediate_mark = []
        self.intermediate_mark_2 = []
        self.blocks_2 = []
        self.count = 0
        self.count_2 = 0
        with open('/home/kpeng/oneshot_metriclearning/transformer-sl-dml/word_embedding_ntu.pkl', 'rb') as f:
            self.world_embedding = pickle.load(f)

        self.world_embedding = torch.from_numpy(self.world_embedding).cuda()
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))
                self.count += 1
                if mr > 0:
                    self.count += 1
            # self.intermediate_mark.append(self.count)

            if do[0] == 'Subsample':
                # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                        resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path))
                    self.count += 1
            self.count += 1
            self.intermediate_mark.append(self.count - 1)
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks_2.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks_2.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))
                self.count_2 += 1
                if mr > 0:
                    self.count_2 += 1
            # self.intermediate_mark.append(self.count)

            if do[0] == 'Subsample':
                # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks_2.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks_2.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path))
                    self.count_2 += 1
                    emb = embed_dim[i+1]
                    print(emb)
            self.count_2 += 1
            self.intermediate_mark_2.append(self.count_2 - 1)
            #print(self.count-1)
            self.blocks_2.append(torch.nn.Linear(emb, emb))
            self.count_2 += 1


        self.blocks = torch.nn.Sequential(*self.blocks)
        self.blocks_2 = torch.nn.Sequential(*self.blocks_2)
        # print(self.intermediate_mark)
        #self.fuse = MLP([512, 21])
        #self.merge = MLP([1024, 512, 512])

        #self.fuse_2 = MLP([384, 21])
        #self.merge_2 = MLP([1024, 512, 512])
        #self.causal_1 = causal_aggregate(64)
        #self.causal_2 = causal_aggregate(16)

        # Classifier head
        self.head = BN_Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.head_2 = BN_Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.head_inter = BN_Linear(
            512, num_classes) if num_classes > 0 else torch.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                512, num_classes) if num_classes > 0 else torch.nn.Identity()

        self.FLOPS = FLOPS_COUNTER
        FLOPS_COUNTER = 0
        self.cosine_embedding_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
        self.f_feature = nn.Linear(512, 768)
        self.f_prototypes = nn.Linear(512, 768)
        self.softmax = nn.Softmax()
        self.f_a = nn.Linear(768*2, 768)
        self.f_r = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.f_feature_2 = nn.Linear(512, 768)
        self.f_prototypes_2 = nn.Linear(768, 768)
        self.softmax_2 = nn.Softmax()
        self.f_a_2 = nn.Linear(768*2, 768)
        self.f_r_2 = nn.Linear(768, 512)
        self.relu_2 = nn.ReLU()
    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}


    def get_augmented_feature(self, features, labels, prototypes):
        curbatch_proto = torch.stack([prototypes[i,:] for i in labels], dim=0)
        lift_proto = self.f_feature(curbatch_proto)
        lift_feature = self.f_feature(features)
        attention = self.softmax(lift_feature*lift_proto)
        out_phi_a = self.relu(self.f_a(torch.cat([lift_proto*attention,lift_feature], dim=-1)))
        return self.relu(features+self.f_r(out_phi_a))
    def get_augmented_feature_with_semantic(self, features, labels, prototypes):
        ori_features = features
        curbatch_proto = torch.stack([prototypes[i,:] for i in labels], dim=0)
        curbatch_proto_semantic = torch.stack([self.world_embedding[i] for i in labels], dim = 0)
        lift_proto = self.f_feature(curbatch_proto)
        lift_feature = self.f_feature(features)
        attention = self.softmax(lift_feature*lift_proto)
        out_phi_a = self.relu(self.f_a(torch.cat([lift_proto*attention,lift_feature], dim=-1)))
        features = self.relu(features+self.f_r(out_phi_a))
        lift_proto = self.f_prototypes_2(curbatch_proto_semantic).squeeze().unsqueeze(1).repeat(1,64,1)
        #print(lift_proto.size())
        lift_feature = self.f_feature_2(features)
        attention = self.softmax_2(lift_feature*lift_proto)
        out_phi_a = self.relu_2(self.f_a_2(torch.cat([lift_proto*attention,lift_feature], dim=-1)))
        features = self.relu(ori_features+features+self.f_r_2(out_phi_a))

        return features

    def get_augmented_feature_with_semantic_2(self, features, labels, prototypes):
        ori_features = features
        curbatch_proto = torch.stack([prototypes[i,:] for i in labels], dim=0)
        curbatch_proto_semantic = torch.stack([self.world_embedding[i] for i in labels], dim = 0)
        lift_proto_semantic = self.f_prototypes(curbatch_proto_semantic).squeeze().unsqueeze(1).repeat(1,64,1)
        lift_proto = self.f_prototypes_2(curbatch_proto)
        lift_feature = self.f_feature(features)
        attention = self.softmax(lift_proto_semantic*lift_feature*lift_proto)
        out_phi_a = self.relu(self.f_a(torch.cat([lift_proto_semantic*lift_proto*attention,lift_feature], dim=-1)))
        features = self.relu(features+self.f_r(out_phi_a))
        
        #print(lift_proto.size())

        return features


    def forward(self, x, prototypes=None, epoch=None, labels=None):
        #print(prototypes)
        #print(epoch)
        if (prototypes !=None) and (epoch>1):
            #print('test')
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            x_2 = x
            count = 0
            c = 0
            for block in self.blocks:
                x = block(x)
                c+=1
                if c == self.intermediate_mark[0]:
                    rep = x
                    #print(rep.size())

            x = torch.mean(x, dim=1, keepdim=False)
            x = self.head(x)
            for block_2 in self.blocks:
                if count == (self.intermediate_mark_2[0]+1):
                    aug_x_2 = self.get_augmented_feature_with_semantic(rep, labels, prototypes)
                    x_2 = block_2(aug_x_2)
                else:
                    if (count > self.intermediate_mark_2[0]+1):
                        x_2 = block_2(x_2)
                count += 1
            x_2 = torch.mean(x_2, dim=1, keepdim=False)
            x_2 = self.head_2(x_2)
            target = torch.ones(x_2.size()[0]).cuda()
            loss = torch.cosine_embedding_loss(x, x_2,target)
            if epoch == 0:
                loss = torch.zeros(x.size()[0]).cuda()
        else:
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            c = 0
            for block in self.blocks:
                x = block(x)
                c += 1
                if c == self.intermediate_mark[0]:
                    rep = x
                    #print(rep.size())
            x = torch.mean(x, dim=1, keepdim=False)
            x = self.head(x)
            loss = torch.zeros(x.size()[0]).cuda()
        return x, loss, rep


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


def model_factory(C, D, X, N, drop_path, weights,
                  num_classes, distillation, pretrained, fuse, in_chans=3):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = Hardswish
    print(in_chans)
    model = LeViT(
        patch_size=16,
        in_chans=in_chans,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', D, embed_dim[0] // D, 4, 2, 2],
            ['Subsample', D, embed_dim[1] // D, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act, in_chans=in_chans),
        num_classes=num_classes,
        drop_path=drop_path,
        distillation=distillation
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    if fuse:
        utils.replace_batchnorm(model)

    return model


def model_factory_cascaded(C, D, X, N, drop_path, weights,
                           num_classes, distillation, pretrained, fuse, in_chans=3):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = Hardswish
    print(in_chans)
    model = LeViT(
        patch_size=16,
        in_chans=in_chans,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', D, embed_dim[0] // D, 4, 2, 2],
            ['Subsample', D, embed_dim[1] // D, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act, in_chans=in_chans),
        num_classes=num_classes,
        drop_path=drop_path,
        distillation=distillation
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    if fuse:
        utils.replace_batchnorm(model)

    return model



if __name__ == '__main__':
    for name in specification:
        net = globals()[name](fuse=True, pretrained=True)
        net.eval()
        net(torch.randn(4, 3, 224, 224))
        print(name,
              net.FLOPS, 'FLOPs',
              sum(p.numel() for p in net.parameters() if p.requires_grad), 'parameters')
