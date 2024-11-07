import torch
from torch import nn
import torch.nn.functional as F
from vision_config import VisionConfig

class VisionEmbedding(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.in_channel = config.in_channel
        self.out_channel = self.emb_dim
        self.kernel_size = config.kernel_size
        self.image_size = config.image_size
        self.conv_1 = nn.Conv2d(kernel_size=self.kernel_size,
                                in_channels=self.in_channel,
                                out_channels=self.out_channel,
                                stride=self.kernel_size)
        self.num_feature = (self.image_size // self.kernel_size) ** 2
        self.pos_embedding = nn.Embedding(self.num_feature, self.emb_dim)
        self.register_buffer("pos_emb_index",
                             torch.arange(self.num_feature),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channel, height, width = x.shape
        # x_shape = [batch_size, in_channel, height, width]
        x = self.conv_1(x)
        # x_shape = [batch_size, out_channel, height, width]
        x = x.flatten(2)
        # x_shape = [batch_size, emb_dim, num_feature]
        x = x.transpose(1, 2)
        # x_shape = [batch_size, num_feature, emb_dim]
        x = x + self.pos_embedding(self.pos_emb_index)
        return x

































