from .act import * # noqa
from .config import * # noqa
from .encoder import * # noqa
from .head import * # noqa
from .layer import * # noqa
from .loader import * # noqa
from .loss import * # noqa
from .network import * # noqa
from .optimizer import * # noqa
from .pooling import * # noqa
from .stage import * # noqa
from .train import * # noqa
from .transform import * # noqa

# Automatically use Marchenko-Pastur initialization for attention layers
import torch
import torch.nn as nn
from graphgps.layer.mp_init import mp_init_weight

# Patch nn.Linear to use MP init
_orig_linear_init = nn.Linear.__init__
def mp_linear_init(self, in_features, out_features, bias=True):
	_orig_linear_init(self, in_features, out_features, bias)
	mp_init_weight(self.weight)
	if self.bias is not None:
		nn.init.zeros_(self.bias)
nn.Linear.__init__ = mp_linear_init

# Patch nn.MultiheadAttention to use MP init
_orig_mha_init = nn.MultiheadAttention.__init__
def mp_mha_init(self, embed_dim, num_heads, **kwargs):
	_orig_mha_init(self, embed_dim, num_heads, **kwargs)
	mp_init_weight(self.in_proj_weight)
	if hasattr(self, 'out_proj') and hasattr(self.out_proj, 'weight'):
		mp_init_weight(self.out_proj.weight)
nn.MultiheadAttention.__init__ = mp_mha_init