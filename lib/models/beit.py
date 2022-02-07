from typing import Sequence, Callable

from torch import nn
from timm.models.beit import beit_large_patch16_224_in22k, beit_large_patch16_384

from ..base_model import BaseEncoder, BaseModel, LSTMDecoder, LSTM22kDecoder
from ..loss import BiTemperedLogisticLoss


class BeitLarge22K224P16Encoder(BaseEncoder):
    def __init__(self) -> None:
        model = beit_large_patch16_224_in22k(pretrained=True)
        super(BeitLarge22K224P16Encoder, self).__init__(model)
        
        
class BeitLarge384P16Encoder(BaseEncoder):
    def __init__(self) -> None:
        model = beit_large_patch16_384(pretrained=True)
        super(BeitLarge384P16Encoder, self).__init__(model)
        
        
class BeitLarge22K224P16Model(BaseModel):
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
        cnn = BeitLarge22K224P16Encoder()
        rnn = LSTM22kDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(BeitLarge22K224P16Model, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )
        
        
class BeitLarge22K224P16BlossModel(BaseModel):
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
        cnn = BeitLarge22K224P16Encoder()
        rnn = LSTM22kDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = BiTemperedLogisticLoss(t1=0.8, t2=1.4, label_smoothing=0.06)
        
        super(BeitLarge22K224P16Model, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )
        
        
class BeitLarge22K224P16TTAModel(BaseModel):
    def __init__(
        self,
        max_len: int, 
        embedding_dim: int, 
        num_features: int, 
        class_n: int,
        tta_transforms: Sequence[Callable],
        rate: float = 0.1,
        learning_rate: float = 5e-4,
        max_epochs: int = 50,
        use_sch: bool = False,
        **kwargs
    ) -> None:
        cnn = BeitLarge22K224P16Encoder()
        rnn = LSTM22kDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(BeitLarge22K224P16TTAModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch,
            is_tta=True, tta_transforms=tta_transforms, **kwargs
        )
        
        
class BeitLarge384P16Model(BaseModel):
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
        cnn = BeitLarge384P16Encoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(BeitLarge384P16Model, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )


def beit_large_22k_224_p16_lstm(
    max_len: int, 
    embedding_dim: int, 
    num_features: int, 
    class_n: int, 
    **kwargs
) -> BeitLarge22K224P16Model:
    return BeitLarge22K224P16Model(
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        **kwargs
    )


def beit_large_22k_224_p16_lstm_bloss(
    max_len: int, 
    embedding_dim: int, 
    num_features: int, 
    class_n: int, 
    **kwargs
) -> BeitLarge22K224P16BlossModel:
    return BeitLarge22K224P16BlossModel(
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        **kwargs
    )


def beit_large_22k_224_p16_lstm_tta(
    max_len: int, 
    embedding_dim: int, 
    num_features: int, 
    class_n: int, 
    tta_transforms: Sequence[Callable],
    **kwargs
) -> BeitLarge22K224P16TTAModel:
    return BeitLarge22K224P16TTAModel(
        max_len, 
        embedding_dim, 
        num_features, 
        class_n,
        tta_transforms,
        **kwargs
    )


def beit_large_384_p16_lstm(
    max_len: int, 
    embedding_dim: int, 
    num_features: int, 
    class_n: int, 
    **kwargs
) -> BeitLarge384P16Model:
    return BeitLarge384P16Model(
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        **kwargs
    )
