import torch
from torch import nn


class VisionConfig:

    def __init__(
        self,
        emb_dim=768,
        num_layer=12,
        batch_size=1,
        num_attention_head=12,
        in_channel=3,
        image_size=224,
        kernel_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_feature = 196,
        num_image_tokens: int = None,):

        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.num_layer = num_layer
        self.num_feature = num_feature
        self.num_attention_head = num_attention_head
        self.image_size = image_size
        self.in_channel = in_channel
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


