
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

pretrain_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.2.0/pretrain_bisenetm.pth'


def act_func(act='relu6'):
    if act == 'relu6':
        res = nn.ReLU6(inplace=True)
    elif act == 'relu':
        res = nn.ReLU(inplace=True)
    elif act == 'gelu':
        res = nn.GELU()
    elif act == 'none':
        res = nn.Identity()
    else:
        raise NotImplementedError
    return res


@torch.no_grad()
def fuse_bn_conv(bn, conv):
    assert all(p == 0 for p in conv.padding)
    scale = bn.weight / (bn.running_var + bn.eps).sqrt()
    shift = bn.bias - bn.running_mean * scale

    groups = conv.groups
    C_out, C_in_per_group = conv.weight.shape[:2]
    C_out_per_group = C_out // groups
    scale = scale.view(groups, C_in_per_group).repeat_interleave(C_out_per_group, dim=0)
    shift = shift.view(groups, C_in_per_group).repeat_interleave(C_out_per_group, dim=0)

    weight = conv.weight * scale.view(C_out, C_in_per_group, 1, 1)
    bias = torch.zeros(C_out, device=conv.weight.device, dtype=conv.weight.dtype)
    if not conv.bias is None: bias = conv.bias
    bias = bias + (conv.weight.sum(dim=[2, 3]) * shift).sum(dim=1)

    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      stride=conv.stride, padding=0, dilation=conv.dilation,
                      groups=conv.groups, bias=True)
    fused.weight = nn.Parameter(weight.detach())
    fused.bias = nn.Parameter(bias.detach())
    return fused


class ConvBN(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        return feat

    def get_fused(self):
        self.eval()
        return nn.utils.fuse_conv_bn_eval(self.conv, self.bn)


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, act='relu6'):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.act = act_func(act)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.act(feat)
        return feat


class RepConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, groups=1, dilation=1, act='relu6'):
        super(RepConvBNReLU, self).__init__()
        assert ks % 2 == 1
        self.ks, self.dilation = ks, dilation
        self.identity = None
        if in_chan == out_chan and stride == 1:
            self.identity = nn.BatchNorm2d(in_chan)
        padding = (ks // 2) * dilation
        self.dense = ConvBN(in_chan, out_chan, ks=ks, stride=stride, padding=padding, groups=groups, dilation=dilation)
        self.proj = ConvBN(in_chan, out_chan, ks=1, stride=stride, padding=0, groups=groups)
        self.act = act_func(act=act)

    def forward(self, x):
        feat_dense = self.dense(x)
        feat_proj = self.proj(x)
        feat_iden = 0
        if not self.identity is None:
            feat_iden = self.identity(x)
        feat = feat_dense + feat_proj + feat_iden
        feat = self.act(feat)
        return feat

    @torch.no_grad()
    def get_fused(self):
        dense = self.dense.get_fused()
        proj = self.proj.get_fused()

        if not self.identity is None:
            bn = self.identity
            deno = torch.rsqrt(bn.running_var + bn.eps)
            w = bn.weight * deno
            b = bn.bias - bn.weight * bn.running_mean * deno

            dead = bn.running_var < 1e-8
            if bool(dead.any()):
                w = torch.where(dead, torch.zeros_like(w), w)
                dense.weight[:, dead] = 0
                proj.weight[:, dead] = 0

            proj.weight += torch.diag(w)[:, :, None, None]
            proj.bias += b

        ind = (self.ks // 2) * self.dilation
        dense.weight[:, :, ind, ind] += proj.weight[:, :, 0, 0]
        dense.bias += proj.bias
        return nn.Sequential(dense, self.act)


class ResBasicLayer(nn.Module):

    def __init__(self, dim_chan, act='relu6'):
        super(ResBasicLayer, self).__init__()
        self.conv1 = RepConvBNReLU(dim_chan, dim_chan, 3, 1, act=act)
        self.conv2 = RepConvBNReLU(dim_chan, dim_chan, 3, 1, act='none')
        self.act = act_func(act)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        out = feat + x
        out = self.act(out)
        return out

    @torch.no_grad()
    def get_fused(self):
        self.conv1 = self.conv1.get_fused()
        self.conv2 = self.conv2.get_fused()
        return self


class StarLayer(nn.Module):

    def __init__(self, dim_chan, exp_ratio=8, kernel_size=7):
        super(StarLayer, self).__init__()
        mid_chan = dim_chan * exp_ratio
        padding = kernel_size // 2
        self.dwconv1 = ConvBN(dim_chan, dim_chan, kernel_size, 1, padding, groups=dim_chan)
        self.f1 = nn.Conv2d(dim_chan, mid_chan, 1, 1, 0, bias=True)
        self.f2 = nn.Sequential(
                nn.Conv2d(dim_chan, mid_chan, 1, 1, 0, bias=True),
                act_func('relu6'),
        )
        self.g = ConvBN(mid_chan, dim_chan, 1, 1, 0)
        self.dwconv2 = nn.Conv2d(
                dim_chan, dim_chan, kernel_size=kernel_size, stride=1,
                padding=padding, groups=dim_chan, bias=True
        )

    def forward(self, x):
        feat = self.dwconv1(x)
        f1 = self.f1(feat)
        f2 = self.f2(feat)
        feat = self.g(f1 * f2)
        feat = self.dwconv2(feat)
        feat = feat + x
        return feat


class InvertedResidualLayer(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, exp_ratio=1, has_start=False, has_mid=True):
        super(InvertedResidualLayer, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.has_skip = (in_chan == out_chan) and (stride == 1)
        self.dw_start = nn.Identity()
        if has_start:
            self.dw_start = ConvBN(in_chan, in_chan, ks, 1, padding=ks//2, groups=in_chan)
        self.pw_exp = ConvBNReLU(in_chan, mid_chan, 1, 1, 0, act='relu6')
        self.dw_mid = nn.Identity()
        if has_mid:
            self.dw_mid = ConvBNReLU(mid_chan, mid_chan, ks, stride, padding=ks//2,
                                     groups=mid_chan, act='relu6')
        self.pw_proj = ConvBN(mid_chan, out_chan, 1, 1, 0)

    def forward(self, x):
        feat = self.dw_start(x)
        feat = self.pw_exp(feat)
        feat = self.dw_mid(feat)
        feat = self.pw_proj(feat)
        if self.has_skip:
            feat = x + feat
        return feat


class SRQKVAttention(nn.Module):

    def __init__(self, in_chan, N=8, R=8):
        super(SRQKVAttention, self).__init__()
        self.n_heads = N
        self.R = R
        self.head_chan = in_chan // N

        self.conv_q = nn.Conv2d(in_chan, in_chan, 1, 1, 0, bias=True)
        self.conv_kv = nn.Conv2d(in_chan, in_chan * 2, 1, 1, 0, bias=True)
        self.conv_sr = nn.Identity()
        if R > 1:
            self.conv_sr = nn.Sequential(
                    nn.Conv2d(in_chan, in_chan, R, R, 0, bias=True),
                    nn.BatchNorm2d(in_chan))
        self.conv_proj = nn.Conv2d(in_chan, in_chan, 1, 1, 0, bias=True)

    def forward(self, x):
        N, C, H, W = x.size()
        q = self.conv_q(x)
        kv = self.conv_kv(self.conv_sr(x)).reshape(N, 2, self.n_heads, self.head_chan, -1)
        kv = kv.permute(1, 0, 2, 4, 3)
        k, v = kv[0], kv[1]
        q = q.reshape(N, self.n_heads, self.head_chan, -1).permute(0, 1, 3, 2)

        o = F.scaled_dot_product_attention(q, k, v)
        o = o.permute(0, 1, 3, 2).reshape(N, C, H, W)
        o = self.conv_proj(o)
        return o

    @torch.no_grad()
    def get_fused(self, pre_bn=None):
        if self.R > 1:
            conv_sr = nn.utils.fuse_conv_bn_eval(self.conv_sr[0], self.conv_sr[1])
            C = conv_sr.in_channels
            C2 = self.conv_kv.out_channels
            W_kv = self.conv_kv.weight.view(C2, C)
            weight = (W_kv @ conv_sr.weight.view(C, -1)).view(C2, C, self.R, self.R)
            bias = W_kv @ conv_sr.bias + self.conv_kv.bias
            fused = nn.Conv2d(C, C2, self.R, stride=self.R, padding=0, bias=True)
            fused.weight = nn.Parameter(weight)
            fused.bias = nn.Parameter(bias)
            self.conv_sr = nn.Identity()
            self.conv_kv = fused

        if not pre_bn is None:
            self.conv_q = fuse_bn_conv(pre_bn, self.conv_q)
            self.conv_kv = fuse_bn_conv(pre_bn, self.conv_kv)
        return self


class AttnBlock(nn.Module):

    def __init__(self, n_chan, n_heads=8):
        super().__init__()
        self.norm = nn.BatchNorm2d(n_chan)
        self.attn = SRQKVAttention(n_chan, n_heads, R=2)

    def forward(self, x):
        return x + self.attn(self.norm(x))

    @torch.no_grad()
    def get_fused(self):
        self.attn = self.attn.get_fused(pre_bn=self.norm)
        self.norm = nn.Identity()
        return self


class FFN(nn.Module):

    def __init__(self, n_chan, ratio=4, act='gelu'):
        super().__init__()
        mid_chan = int(n_chan * ratio)
        self.norm = nn.BatchNorm2d(n_chan)
        self.mlp = nn.Sequential(
                nn.Conv2d(n_chan, mid_chan, 1, 1, 0, bias=True),
                nn.Conv2d(mid_chan, mid_chan, 3, 1, 1, bias=True, groups=mid_chan),
                act_func(act),
                nn.Conv2d(mid_chan, n_chan, 1, 1, 0, bias=True),
                )

    def forward(self, x):
        return x + self.mlp(self.norm(x))

    @torch.no_grad()
    def get_fused(self):
        self.mlp[0] = fuse_bn_conv(self.norm, self.mlp[0])
        self.norm = nn.Identity()
        return self


class FuseModule(nn.Module):

    def __init__(self, chans, n_chan=128):
        super(FuseModule, self).__init__()
        interp_args = dict(mode='bilinear', align_corners=False)
        chan_l, chan_h, chan_fuse = 128, 32, 64
        chan4, chan8, chan16, chan32 = chans

        self.conv4 = nn.Sequential(
            ConvBNReLU(chan4, chan_h, 1, 1, 0, act='relu6'),
        )
        self.conv8 = nn.Sequential(
            ConvBNReLU(chan8, chan_h, 1, 1, 0, act='relu6'),
            nn.Upsample(scale_factor=2, **interp_args)
        )
        self.conv16 = nn.Sequential(
            ConvBNReLU(chan16, chan_l, 1, 1, 0, act='relu6'),
        )
        self.conv32 = nn.Sequential(
            ConvBNReLU(chan32, chan_l, 1, 1, 0, act='relu6'),
            nn.Upsample(scale_factor=2, **interp_args)
        )
        self.conv_l = nn.Sequential(
            RepConvBNReLU(chan_l, chan_l, 3, 1, act='relu6'),
            nn.Upsample(scale_factor=4, **interp_args)
        )
        self.fuse = nn.Sequential(
            ConvBNReLU(chan_l + chan_h, chan_fuse, 1, 1, 0, act='relu6'),
            RepConvBNReLU(chan_fuse, n_chan, 3, 1, act='relu6'),
        )

    def forward(self, feats):
        feat4, feat8, feat16, feat32 = feats
        feat_h = self.conv4(feat4) + self.conv8(feat8)
        feat_l = self.conv_l(self.conv16(feat16) + self.conv32(feat32))
        feat = torch.cat([feat_h, feat_l], dim=1)
        feat = self.fuse(feat)
        return feat

    @torch.no_grad()
    def get_fused(self):
        self.eval()
        return FuseModuleDeploy(self)


class FuseModuleDeploy(nn.Module):

    def __init__(self, m):
        super(FuseModuleDeploy, self).__init__()
        chan_h = m.conv8[0].conv.out_channels
        chan_l = m.conv16[0].conv.out_channels
        out_chan = m.fuse[0].conv.out_channels

        self.conv4, self.conv8 = m.conv4, m.conv8
        self.conv16, self.conv32 = m.conv16, m.conv32
        self.conv_l = m.conv_l[0].get_fused()
        self.up_l = m.conv_l[1]
        self.fuse1 = m.fuse[1].get_fused()

        conv, bn = m.fuse[0].conv, m.fuse[0].bn
        deno = torch.rsqrt(bn.running_var + bn.eps)
        weight = conv.weight * (bn.weight * deno)[:, None, None, None]
        bias = bn.bias - bn.weight * bn.running_mean * deno

        self.fuse0_h = nn.Conv2d(chan_h, out_chan, 1, 1, 0, bias=True)
        self.fuse0_l = nn.Conv2d(chan_l, out_chan, 1, 1, 0, bias=False)
        self.fuse0_h.weight.data.copy_(weight[:, :chan_h])
        self.fuse0_h.bias.data.copy_(bias)
        self.fuse0_l.weight.data.copy_(weight[:, chan_h:])
        self.act = m.fuse[0].act
        self.fuse0_h.to(conv.weight)
        self.fuse0_l.to(conv.weight)

    def forward(self, feats):
        feat4, feat8, feat16, feat32 = feats
        feat_h = self.conv4(feat4) + self.conv8(feat8)
        feat_l = self.conv_l(self.conv16(feat16) + self.conv32(feat32))
        feat_l = self.up_l(self.fuse0_l(feat_l))
        feat = self.act(self.fuse0_h(feat_h) + feat_l)
        return self.fuse1(feat)


class BiSeNetM(nn.Module):

    def __init__(self, n_classes, aux_mode='train'):
        super(BiSeNetM, self).__init__()
        self.aux_mode = aux_mode
        chan4, chan8, chan16, chan32 = 32, 64, 64, 128

        self.enter4 = ConvBNReLU(3, chan4, 4, 4, 0, act='relu6')
        self.stage4 = nn.Sequential(
            ResBasicLayer(chan4, act='relu6'),
            ResBasicLayer(chan4, act='relu6'),
        )

        self.enter8 = ConvBNReLU(chan4, chan8, 2, 2, 0, act='relu6')
        self.stage8 = nn.Sequential(
            ResBasicLayer(chan8, act='relu6'),
            ResBasicLayer(chan8, act='relu6'),
        )

        self.enter16 = ConvBNReLU(chan8, chan16, 2, 2, 0, act='relu6')
        self.star16 = nn.Sequential(*[StarLayer(chan16, 4, 5) for _ in range(4)])
        self.uib16 = nn.Sequential(
            InvertedResidualLayer(chan16, chan16, 5, 1, 3, True),
            InvertedResidualLayer(chan16, chan16, 3, 1, 2),
            InvertedResidualLayer(chan16, chan16, 3, 1, 2),
            InvertedResidualLayer(chan16, chan16, 3, 1, 2),
            InvertedResidualLayer(chan16, chan16, 3, 1, 2),
            InvertedResidualLayer(chan16, chan16, 3, 1, 4, True, False),
        )

        self.enter32 = ConvBNReLU(chan16, chan32, 2, 2, 0, act='relu6')
        self.star32 = nn.Sequential(*[StarLayer(chan32, 4, 5) for _ in range(4)])
        self.uib32 = nn.Sequential(
            InvertedResidualLayer(chan32, chan32, 3, 1, 6, True),
            InvertedResidualLayer(chan32, chan32, 5, 1, 4, True),
            InvertedResidualLayer(chan32, chan32, 5, 1, 4),
            InvertedResidualLayer(chan32, chan32, 5, 1, 3),
            InvertedResidualLayer(chan32, chan32, 3, 1, 4),
            InvertedResidualLayer(chan32, chan32, 3, 1, 4, False),
        )
        self.transformer = nn.Sequential(
            AttnBlock(chan32, n_heads=4),
            FFN(chan32, ratio=4, act='relu6'),
        )

        self.gather_feats = FuseModule([chan4, chan8, chan16, chan32], 128)
        self.head = nn.Sequential(
            nn.Conv2d(128, n_classes, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        self.init_weights()

    def mix_stage(self, feat, star, uib):
        feat = star[:2](feat) + uib[:3](feat)
        feat = star[2:](feat) + uib[3:](feat)
        return feat

    def forward(self, x):
        feat4 = self.stage4(self.enter4(x))
        feat8 = self.stage8(self.enter8(feat4))
        feat16 = self.mix_stage(self.enter16(feat8), self.star16, self.uib16)
        feat32 = self.mix_stage(self.enter32(feat16), self.star32, self.uib32)
        feat32 = self.transformer(feat32)
        feat = self.gather_feats([feat4, feat8, feat16, feat32])

        logits = self.head(feat)
        if self.aux_mode in ('train', 'eval'):
            return logits,
        elif self.aux_mode == 'pred':
            pred = logits.argmax(dim=1)
            return pred
        else:
            raise NotImplementedError

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.load_pretrain()

    def load_pretrain(self):
        state = modelzoo.load_url(pretrain_url, map_location='cpu')
        state_self = self.state_dict()
        state = {k: v for k, v in state.items()
                 if k in state_self and v.shape == state_self[k].shape}
        self.load_state_dict(state, strict=False)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.dim() == 1:
                    nowd_params.append(param)
                else:
                    wd_params.append(param)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if name in ('gather_feats', 'head'):
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    @torch.no_grad()
    def deploy(self):
        if getattr(self, 'deployed', False): return
        self.deployed = True
        self.eval()
        for stage in (self.stage4, self.stage8, self.transformer):
            for i in range(len(stage)):
                stage[i] = stage[i].get_fused()
        self.gather_feats = self.gather_feats.get_fused()
