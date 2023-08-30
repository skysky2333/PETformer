import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from einops import rearrange


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # print('mu, var', mu.mean(), var.mean())
        # d.append([mu.mean(), var.mean()])
        y = (x - mu) / (var + eps).sqrt()
        weight, bias, y = weight.contiguous(), bias.contiguous(), y.contiguous()  # avoid cuda error
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        # y, var, weight = ctx.saved_variables
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6, requires_grad=True):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels), requires_grad=requires_grad))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels), requires_grad=requires_grad))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class KBAFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, att, selfk, selfg, selfb, selfw):
        B, nset, H, W = att.shape
        KK = selfk ** 2
        selfc = x.shape[1]

        att = att.reshape(B, nset, H * W).transpose(-2, -1)

        ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset = selfk, selfg, selfc, KK, nset
        ctx.x, ctx.att, ctx.selfb, ctx.selfw = x, att, selfb, selfw

        bias = att @ selfb
        attk = att @ selfw

        uf = torch.nn.functional.unfold(x, kernel_size=selfk, padding=selfk // 2)

        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        x = attk @ uf.unsqueeze(-1)  #
        del attk, uf
        x = x.squeeze(-1).reshape(B, H * W, selfc) + bias
        x = x.transpose(-1, -2).reshape(B, selfc, H, W)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, att, selfb, selfw = ctx.x, ctx.att, ctx.selfb, ctx.selfw
        selfk, selfg, selfc, KK, nset = ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset

        B, selfc, H, W = grad_output.size()

        dbias = grad_output.reshape(B, selfc, H * W).transpose(-1, -2)

        dselfb = att.transpose(-2, -1) @ dbias
        datt = dbias @ selfb.transpose(-2, -1)

        attk = att @ selfw
        uf = F.unfold(x, kernel_size=selfk, padding=selfk // 2)
        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        dx = dbias.view(B, H * W, selfg, selfc // selfg, 1)

        dattk = dx @ uf.view(B, H * W, selfg, 1, selfc // selfg * KK)
        duf = attk.transpose(-2, -1) @ dx
        del attk, uf

        dattk = dattk.view(B, H * W, -1)
        datt += dattk @ selfw.transpose(-2, -1)
        dselfw = att.transpose(-2, -1) @ dattk

        duf = duf.permute(0, 2, 3, 4, 1).view(B, -1, H * W)
        dx = F.fold(duf, output_size=(H, W), kernel_size=selfk, padding=selfk // 2)

        datt = datt.transpose(-1, -2).view(B, nset, H, W)

        return dx, datt, None, None, dselfb, dselfw

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class MDTABlock(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTABlock, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class KBABlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, gc=4, nset=32, k=3):
        super(KBABlock, self).__init__()
        self.gc = gc

        hidden_features = int(dim * ffn_expansion_factor)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      groups=hidden_features, bias=bias),
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      groups=hidden_features, bias=bias),
        )

        c = hidden_features
        self.k, self.c = k, c
        self.nset = nset

        self.g = c // gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w, self.b)
        interc = min(dim, 24)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )
        self.conv211 = nn.Conv2d(in_channels=dim, out_channels=self.nset, kernel_size=1)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.ga1 = nn.Parameter(torch.zeros((1, hidden_features, 1, 1)) + 1e-2, requires_grad=True)

    def forward(self, x):
        x1 = self.dwconv(x)

        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv1(x)
        x2 = KBAFunction.apply(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def init_p(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

class PETformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(PETformerBlock, self).__init__()

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        self.kba = KBABlock(dim, ffn_expansion_factor, bias)
        self.mdta = MDTABlock(dim, num_heads, bias)

        self.ca1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.ca2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

    def forward(self, x):
        ca1 = self.ca1(x)
        ca2 = self.ca2(x)
        x1 = self.kba(self.norm1(x)) * ca1
        x2 = self.mdta(self.norm2(x)) * ca2
        return x + x1 + x2


class PETformer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                 heads=[1, 2, 4, 8], ffn_expansion_factor=1.5, bias=False):
        super(PETformer, self).__init__()

        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.initial_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=False)

        # Verify num_blocks and heads have the same length
        assert len(num_blocks) == len(heads), "num_blocks and heads must have the same length"

        self.encoder_levels = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()

        current_dim = dim
        for i, (n_block, n_head) in enumerate(zip(num_blocks, heads)):
            # Create encoder level
            self.encoder_levels.append(
                nn.Sequential(*[
                    PETformerBlock(dim=current_dim, num_heads=n_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                    for _ in range(n_block)
                ])
            )

            if i < len(num_blocks) - 1:
                # Create downsampling layer
                self.downsamples.append(Downsample(current_dim))
                current_dim *= 2

        # Reverse for decoder
        for i, (n_block, n_head) in reversed(list(enumerate(zip(num_blocks[:-1], heads[:-1])))):
            current_dim //= 2
            # Create upsampling layer
            self.upsamples.append(Upsample(current_dim * 2))

            # Create channel reduction layer
            if i > 0:
                self.reduce_chans.append(nn.Conv2d(current_dim * 2, current_dim, kernel_size=1, bias=bias))

                # Create decoder level
                self.decoder_levels.append(
                    nn.Sequential(*[
                        PETformerBlock(dim=current_dim, num_heads=n_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                        for _ in range(n_block)
                    ])
                )

            else:
                # Create decoder level
                self.decoder_levels.append(
                    nn.Sequential(*[
                        PETformerBlock(dim=current_dim * 2, num_heads=n_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
                        for _ in range(n_block)
                    ])
                )

        self.refinement = nn.Sequential(*[
            PETformerBlock(dim=dim * 2, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for _ in range(num_refinement_blocks)])

        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc = self.initial_embed(inp_img)
        enc_outputs = []

        for i, (enc, down) in enumerate(zip(self.encoder_levels[:-1], self.downsamples)):
            out_enc = enc(inp_enc)
            enc_outputs.append(out_enc)
            inp_enc = down(out_enc)

        # Latent space
        latent = self.encoder_levels[-1](inp_enc)

        out_dec = latent

        for dec, up, red_chan, prev_enc in zip(self.decoder_levels[:-1], self.upsamples[:-1], self.reduce_chans, reversed(enc_outputs[1:])):
            inp_dec = up(out_dec)
            inp_dec = torch.cat([inp_dec, prev_enc], 1)
            inp_dec = red_chan(inp_dec)
            out_dec = dec(inp_dec)

        inp_dec = self.upsamples[-1](out_dec)
        inp_dec = torch.cat([inp_dec, enc_outputs[0]], 1)
        out_dec = self.decoder_levels[-1](inp_dec)

        out_dec = self.refinement(out_dec)

        temp_out = self.output(out_dec)

        out_dec = temp_out + inp_img if self.inp_channels == self.out_channels else temp_out + inp_img[:, 2:3, :, :]

        return out_dec