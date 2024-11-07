import torch
from torch import nn
from vision_config import VisionConfig
from attention import ResidualTransformer
from embedding import VisionEmbedding
from model.util import LayerNorm

class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embedding = VisionEmbedding(config)
        self.transformer = ResidualTransformer(config)
        self.layer_norm = LayerNorm(config.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x

vision_config = VisionConfig()
vision_model = VisionModel(vision_config)
image_data = torch.rand(1, 3, 224, 224)
x = vision_model(image_data)
print(x.shape)