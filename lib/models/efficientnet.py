from typing import Sequence, Callable

from torch import nn
from torchvision.models import efficientnet_b0, efficientnet_b7
from torchvision.models import efficientnet_b3
from timm.models.efficientnet import tf_efficientnet_b7_ns, tf_efficientnet_b2_ns
from timm.models.efficientnet import tf_efficientnet_b5_ns

from ..base_model import BaseEncoder, BaseModel, LSTMDecoder
from ..loss import FocalLoss


class Effnetb0Encoder(BaseEncoder):
    def __init__(self) -> None:
        model = efficientnet_b0(pretrained=True)
        super(Effnetb0Encoder, self).__init__(model)
    
    
class Effnetb3Encoder(BaseEncoder):
    def __init__(self) -> None:
        model = efficientnet_b3(pretrained=True)
        super(Effnetb3Encoder, self).__init__(model)


class Effnetb7Encoder(BaseEncoder):
    def __init__(self) -> None:
        model = efficientnet_b7(pretrained=True)
        super(Effnetb7Encoder, self).__init__(model)
    
    
class Effnetb7NSEncoder(BaseEncoder):
    def __init__(self) -> None:
        model = tf_efficientnet_b7_ns(pretrained=True)
        super(Effnetb7NSEncoder, self).__init__(model)
        
        
class Effnetb2NSPlusEncoder(BaseEncoder):
    def __init__(self, drop_path_rate: float = 0.4, drop_rate: float = 0.5) -> None:
        model = tf_efficientnet_b2_ns(
            pretrained=True,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate
        )
        super(Effnetb2NSPlusEncoder, self).__init__(model, freeze=True)
        
        
class Effnetb5NSPlusEncoder(BaseEncoder):
    def __init__(self, drop_path_rate: float = 0.4, drop_rate: float = 0.5) -> None:
        model = tf_efficientnet_b5_ns(
            pretrained=True,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate
        )
        super(Effnetb5NSPlusEncoder, self).__init__(model, freeze=True)
        
        
class Effnetb7NSPlusEncoder(BaseEncoder):
    def __init__(self, drop_path_rate: float = 0.4, drop_rate: float = 0.5) -> None:
        model = tf_efficientnet_b7_ns(
            pretrained=True,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate
        )
        super(Effnetb7NSPlusEncoder, self).__init__(model, freeze=True)


class Effnetb02LSTMModel(BaseModel):
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
        cnn = Effnetb0Encoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(Effnetb02LSTMModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )


class Effnetb32LSTMModel(BaseModel):
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
        cnn = Effnetb3Encoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(Effnetb32LSTMModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )


class Effnetb72LSTMModel(BaseModel):
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
        cnn = Effnetb7Encoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(Effnetb72LSTMModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )
        
    
class Effnetb7NS2LSTMModel(BaseModel):
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
        cnn = Effnetb7NSEncoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(Effnetb7NS2LSTMModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )
        
        
class Effnetb2NSPlus2LSTMModel(BaseModel):
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
        cnn = Effnetb2NSPlusEncoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(Effnetb2NSPlus2LSTMModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )
        
        
class Effnetb2NSPlusFocal2LSTMModel(BaseModel):
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
        cnn = Effnetb2NSPlusEncoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = FocalLoss()
        
        super(Effnetb2NSPlusFocal2LSTMModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )
        
        
class Effnetb2NSPlus2LSTMTTAModel(BaseModel):
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
        cnn = Effnetb2NSPlusEncoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(Effnetb2NSPlus2LSTMTTAModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch,
            is_tta=True, tta_transforms=tta_transforms, **kwargs
        )


class Effnetb5NSPlus2LSTMModel(BaseModel):
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
        cnn = Effnetb5NSPlusEncoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(Effnetb5NSPlus2LSTMModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )
        
        
class Effnetb5NSPlusFocal2LSTMModel(BaseModel):
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
        cnn = Effnetb5NSPlusEncoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = FocalLoss()
        
        super(Effnetb5NSPlusFocal2LSTMModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )
        
        
class Effnetb7NSPlus2LSTMModel(BaseModel):
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
        cnn = Effnetb7NSPlusEncoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(Effnetb7NSPlus2LSTMModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )
        
        
class Effnetb7NSPlusFocal2LSTMModel(BaseModel):
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
        cnn = Effnetb7NSPlusEncoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = FocalLoss()
        
        super(Effnetb7NSPlusFocal2LSTMModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch, **kwargs
        )
        

class Effnetb7NSPlusFocal2LSTMTTAModel(BaseModel):
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
        cnn = Effnetb7NSPlusEncoder()
        rnn = LSTMDecoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = FocalLoss()
        
        super(Effnetb7NSPlusFocal2LSTMTTAModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs, use_sch,
            is_tta=True, tta_transforms=tta_transforms, **kwargs
        )


def efficientnet_b0_lstm(
    max_len: int, 
    embedding_dim: int, 
    num_features: int, 
    class_n: int, 
    **kwargs
) -> Effnetb02LSTMModel:
    return Effnetb02LSTMModel(
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        **kwargs
    )
    
    
def efficientnet_b3_lstm(
    max_len: int,
    embedding_dim: int,
    num_features: int,
    class_n: int,
    **kwargs
) -> Effnetb32LSTMModel:
    return Effnetb32LSTMModel(
        max_len,
        embedding_dim,
        num_features,
        class_n,
        **kwargs
    )
    
    
def efficientnet_b7_lstm(
    max_len: int,
    embedding_dim: int,
    num_features: int,
    class_n: int,
    **kwargs
) -> Effnetb72LSTMModel:
    return Effnetb72LSTMModel(
        max_len,
        embedding_dim,
        num_features,
        class_n,
        **kwargs
    )
    
    
def efficientnet_b7_ns_lstm(
    max_len: int,
    embedding_dim: int,
    num_features: int,
    class_n: int,
    **kwargs
) -> Effnetb7NS2LSTMModel:
    return Effnetb7NS2LSTMModel(
        max_len,
        embedding_dim,
        num_features,
        class_n,
        **kwargs
    )
    
    
def efficientnet_b2_ns_plus_lstm(
    max_len: int,
    embedding_dim: int,
    num_features: int,
    class_n: int,
    **kwargs
) -> Effnetb2NSPlus2LSTMModel:
    return Effnetb2NSPlus2LSTMModel(
        max_len,
        embedding_dim,
        num_features,
        class_n,
        **kwargs
    )
    

def efficientnet_b2_ns_plus_focal_lstm(
    max_len: int,
    embedding_dim: int,
    num_features: int,
    class_n: int,
    **kwargs
) -> Effnetb2NSPlusFocal2LSTMModel:
    return Effnetb2NSPlusFocal2LSTMModel(
        max_len,
        embedding_dim,
        num_features,
        class_n,
        **kwargs
    )
    
    
def efficientnet_b2_ns_plus_lstm_tta(
    max_len: int,
    embedding_dim: int,
    num_features: int,
    class_n: int,
    tta_transforms: Sequence[Callable],
    **kwargs
) -> Effnetb2NSPlus2LSTMTTAModel:
    return Effnetb2NSPlus2LSTMTTAModel(
        max_len,
        embedding_dim,
        num_features,
        class_n,
        tta_transforms,
        **kwargs
    )


def efficientnet_b5_ns_plus_lstm(
    max_len: int,
    embedding_dim: int,
    num_features: int,
    class_n: int,
    **kwargs
) -> Effnetb5NSPlus2LSTMModel:
    return Effnetb5NSPlus2LSTMModel(
        max_len,
        embedding_dim,
        num_features,
        class_n,
        **kwargs
    )
    
    
def efficientnet_b5_ns_plus_focal_lstm(
    max_len: int,
    embedding_dim: int,
    num_features: int,
    class_n: int,
    **kwargs
) -> Effnetb5NSPlusFocal2LSTMModel:
    return Effnetb5NSPlusFocal2LSTMModel(
        max_len,
        embedding_dim,
        num_features,
        class_n,
        **kwargs
    )
    
    
def efficientnet_b7_ns_plus_lstm(
    max_len: int,
    embedding_dim: int,
    num_features: int,
    class_n: int,
    **kwargs
) -> Effnetb7NSPlus2LSTMModel:
    return Effnetb7NSPlus2LSTMModel(
        max_len,
        embedding_dim,
        num_features,
        class_n,
        **kwargs
    )
    
    
def efficientnet_b7_ns_plus_focal_lstm(
    max_len: int,
    embedding_dim: int,
    num_features: int,
    class_n: int,
    **kwargs
) -> Effnetb7NSPlusFocal2LSTMModel:
    return Effnetb7NSPlusFocal2LSTMModel(
        max_len,
        embedding_dim,
        num_features,
        class_n,
        **kwargs
    )
    
    
def efficientnet_b7_ns_plus_focal_lstm_tta(
    max_len: int,
    embedding_dim: int,
    num_features: int,
    class_n: int,
    tta_transforms: Sequence[Callable],
    **kwargs
) -> Effnetb7NSPlusFocal2LSTMTTAModel:
    return Effnetb7NSPlusFocal2LSTMTTAModel(
        max_len,
        embedding_dim,
        num_features,
        class_n,
        tta_transforms,
        **kwargs
    )
