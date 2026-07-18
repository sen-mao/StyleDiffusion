import jittor as jt
from jittor import nn
from jittor.attention import MultiheadAttention


class QuickGELU(nn.Module):
    def execute(self, x):
        return x * jt.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            QuickGELU(),
            nn.Linear(width * 4, width),
        )
        self.ln_2 = nn.LayerNorm(width)

    def attention(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

    def execute(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def execute(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * jt.randn(width))
        self.positional_embedding = nn.Parameter(scale * jt.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * jt.randn(width, output_dim))

    def execute(self, x: jt.Var):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = self.class_embedding.cast(x.dtype).unsqueeze(0).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = jt.concat([cls, x], dim=1)
        x = x + self.positional_embedding.cast(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        return x
