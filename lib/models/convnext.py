from torch import nn
from timm.models.convnext import convnext_xlarge_in22ft1k

from ..base_model import BaseEncoder, BaseModel, LSTMDecoder


class ConvNext384XLEncoder(BaseEncoder):
    def __init__(self) -> None:
        model = convnext_xlarge_in22ft1k(pretrained=True)
        super(ConvNext384XLEncoder, self).__init__(model)
        

class ConvNext384XLModel(BaseModel):
    def __init__(
        self,
        max_len: int, 
        embedding_dim: int, 
        num_features: int, 
        class_n: int,
        rate: float = 0.1,
        learning_rate: float = 5e-4,
        max_epochs: int = 50,
        use_sch: bool = False,
        **kwargs
    ) -> None:
        cnn = ConvNext384XLEncoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(ConvNext384XLModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )
        
        
def convnext_384_xl_lstm(
    max_len: int, 
    embedding_dim: int, 
    num_features: int, 
    class_n: int, 
    **kwargs
) -> ConvNext384XLModel:
    return ConvNext384XLModel(
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        **kwargs
    )
