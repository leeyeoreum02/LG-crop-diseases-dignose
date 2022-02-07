from typing import Dict, Union, Tuple, List

from sklearn.metrics import f1_score

import torch
from torch import nn, optim, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


def accuracy_function(real: Tensor, pred: Tensor) -> float: 
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score


class BaseEncoder(nn.Module):
    def __init__(self, model, freeze: bool = False) -> None:
        super(BaseEncoder, self).__init__()
        self.model = model
        
        if freeze:
            self.model.eval()
            # freeze params
            for param in self.model.parameters():
                if isinstance(param, nn.BatchNorm2d):
                    param.requires_grad = False

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.model(inputs)
        return output
    
    
class LSTMDecoder(nn.Module):
    def __init__(self, max_len: int, embedding_dim: int, num_features: int, class_n: int, rate: float) -> None:
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(max_len, embedding_dim)
        self.rnn_fc = nn.Linear(num_features*embedding_dim, 1000)
        self.final_layer = nn.Linear(1000+1000, class_n)  # resnet out_dim + lstm out_dim
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out: Tensor, dec_inp: Tensor) -> Tensor:
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.rnn_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim=1)  # enc_out + hidden 
        fc_input = concat
        output = self.dropout((self.final_layer(fc_input)))
        return output
    
    
class LSTM22kDecoder(nn.Module):
    def __init__(self, max_len: int, embedding_dim: int, num_features: int, class_n: int, rate: float) -> None:
        super(LSTM22kDecoder, self).__init__()
        self.lstm = nn.LSTM(max_len, embedding_dim)
        self.rnn_fc = nn.Linear(num_features*embedding_dim, 1000)
        self.final_layer = nn.Linear(22841, class_n)
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out: Tensor, dec_inp: Tensor) -> Tensor:
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.rnn_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim=1)  # enc_out + hidden 
        fc_input = concat
        output = self.dropout((self.final_layer(fc_input)))
        return output
    
    
class BaseModel(LightningModule):
    def __init__(
        self,
        cnn,
        rnn,
        criterion,
        learning_rate: float = 5e-4,
        max_epochs: int = 50,
        use_sch: bool = False,
        is_tta: bool = False,
        tta_transforms: bool = None,
        is_onehot: bool = False,
    ) -> None:
        super(BaseModel, self).__init__()
        
        self.cnn = cnn
        self.rnn = rnn
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.use_sch = use_sch
        self.is_tta = is_tta
        self.tta_transforms = tta_transforms
        self.is_onehot = is_onehot
        
    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Union[Optimizer, _LRScheduler]]]]:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        if self.use_sch:
            return optimizer
        
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=4, max_epochs=self.max_epochs,
            warmup_start_lr=1e-6, eta_min=1e-6,
        )
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.001)
        return [optimizer], [scheduler]

    def forward(self, img: Tensor, seq: Tensor) -> Tensor:
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)
        
        return output

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, float]:
        img = batch['img']
        csv_feature = batch['csv_feature']
        label = batch['label']
        
        output = self(img, csv_feature)
        loss = self.criterion(output, label)
        score = accuracy_function(label, output)
        
        self.log(
            'train_loss', loss, prog_bar=True, logger=True
        )
        self.log(
            'train_score', score, prog_bar=True, logger=True
        )
        
        return {'loss': loss, 'train_score': score}

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, float]:
        img = batch['img']
        csv_feature = batch['csv_feature']
        label = batch['label']
        
        output = self(img, csv_feature)
        loss = self.criterion(output, label)
        score = accuracy_function(label, output)
        
        self.log(
            'val_loss', loss, prog_bar=True, logger=True
        )
        self.log(
            'val_score', score, prog_bar=True, logger=True
        )
        
        return {'val_loss': loss, 'val_score': score}
    
    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        if self.is_tta:
            return self._tta_predict_step(batch)
        else:
            return self._base_predict_step(batch)
    
    def _base_predict_step(self, batch: Tensor) -> Tensor:
        img = batch['img']
        seq = batch['csv_feature']
        
        output = self(img, seq)
        
        if not self.is_onehot:
            return torch.argmax(output, dim=1)
        
        return output
    
    def _tta_predict_step(self, batch: Tensor) -> Tensor:
        img = batch['img']
        seq = batch['csv_feature']
        
        for i, transformer in enumerate(self.tta_transforms):
            augmented_image = transformer.augment_image(img)
            
            # pass to model
            model_output = self(augmented_image, seq)
            # model_output = torch.argmax(model_output, dim=1).unsqueeze(1)
            
            if i == 0:
                outputs = model_output.clone()
            else:
                # outputs = torch.cat((outputs, model_output), dim=1)
                outputs += model_output
                        
        # outputs = torch.mode(outputs, dim=1)[0]
        if not self.is_onehot:
            return torch.argmax(model_output, dim=1)
        
        return outputs
