from torch import nn
from timm.models.vision_transformer import vit_large_patch16_384

from ..base_model import BaseEncoder, BaseModel, LSTMDecoder


class ViTLarge384P16(BaseEncoder):
    def __init__(self):
        model = vit_large_patch16_384(pretrained=True)
        super().__init__(model)

        
class ViTLarge384P16Model(BaseModel):
    def __init__(
        self,
        max_len, 
        embedding_dim, 
        num_features, 
        class_n,
        rate=0.1,
        learning_rate=5e-4,
        max_epochs=50,
        use_sch=False,
        **kwargs
    ):
        cnn = ViTLarge384P16()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super().__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )
        
        
def vit_large_384_p16(
    max_len: int, 
    embedding_dim: int, 
    num_features: int, 
    class_n: int, 
    **kwargs
) -> ViTLarge384P16Model:
    return ViTLarge384P16Model(
        max_len,
        embedding_dim,
        num_features,
        class_n,
        **kwargs
    )
