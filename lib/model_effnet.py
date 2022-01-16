from sklearn.metrics import f1_score

import torch
from torch import nn, optim
from torchvision.models import efficientnet_b0, efficientnet_b7
from torchvision.models import efficientnet_b3
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import timm
from timm.models.efficientnet import tf_efficientnet_b7_ns

from lib.loss import FocalLoss


def accuracy_function(real, pred):    
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score


class BaseEncoder(nn.Module):
    def __init__(self, model, freeze=False):
        super(BaseEncoder, self).__init__()
        self.model = model
        
        if freeze:
            self.model.eval()
            # freeze params
            for param in self.model.parameters():
                if isinstance(param, nn.BatchNorm2d):
                    param.requires_grad = False

    def forward(self, inputs):
        output = self.model(inputs)
        return output


class Effnetb0_Encoder(BaseEncoder):
    def __init__(self):
        model = efficientnet_b0(pretrained=True)
        super(Effnetb0_Encoder, self).__init__(model)
    
    
class Effnetb3_Encoder(BaseEncoder):
    def __init__(self):
        model = efficientnet_b3(pretrained=True)
        super(Effnetb3_Encoder, self).__init__(model)


class Effnetb7_Encoder(BaseEncoder):
    def __init__(self):
        model = efficientnet_b7(pretrained=True)
        super(Effnetb7_Encoder, self).__init__(model)
    
    
class Effnetb7NS_Encoder(BaseEncoder):
    def __init__(self):
        model = tf_efficientnet_b7_ns(pretrained=True)
        super(Effnetb7NS_Encoder, self).__init__(model)
        
        
class Effnetb7NSPlus_Encoder(BaseEncoder):
    def __init__(self, drop_path_rate=0.4, drop_rate=0.5):
        model = tf_efficientnet_b7_ns(
            pretrained=True,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate
        )
        super(Effnetb7NSPlus_Encoder, self).__init__(model, freeze=True)
        
    
class LSTM_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(LSTM_Decoder, self).__init__()
        self.lstm = nn.LSTM(max_len, embedding_dim)
        self.rnn_fc = nn.Linear(num_features*embedding_dim, 1000)
        self.final_layer = nn.Linear(1000 + 1000, class_n)  # resnet out_dim + lstm out_dim
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out, dec_inp):
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.rnn_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim=1)  # enc_out + hidden 
        fc_input = concat
        output = self.dropout((self.final_layer(fc_input)))
        return output
    
    
class BaseImageModel(LightningModule):
    def __init__(
        self,
        model_name,
        criterion,
        learning_rate=5e-4,
    ):
        super(BaseImageModel, self).__init__()
        
        # !FIXME: Should I add num_classes in create_model and remove self.classifier??
        self.model = timm.create_model(model_name, pretrained=True)
        self.classifier = nn.Linear(1000, 25)
        self.learning_rate = learning_rate
        self.criterion = criterion
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        image, label = batch
        
        output = self(image)
        loss = self.criterion(output, label)
        score = accuracy_function(label, output)
        
        self.log(
            'train_loss', loss, prog_bar=True, logger=True
        )
        self.log(
            'train_score', score, prog_bar=True, logger=True
        )
        
        return {'loss': loss, 'train_score': score}

    def validation_step(self, batch, batch_idx):
        image, label = batch
        
        output = self(image)
        loss = self.criterion(output, label)
        score = accuracy_function(label, output)
        
        self.log(
            'val_loss', loss, prog_bar=True, logger=True
        )
        self.log(
            'val_score', score, prog_bar=True, logger=True
        )
        
        return {'val_loss': loss, 'val_score': score}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        image = batch
        
        output = self(image)
        output = torch.argmax(output, dim=1)
        
        return output
    
    
class Effnetb7NS(BaseImageModel):
    def __init__(self):
        model_name = 'tf_efficientnet_b7_ns'
        criterion = nn.CrossEntropyLoss()
        super(Effnetb7NS, self).__init__(model_name, criterion)
    
    
class BaseModel(LightningModule):
    def __init__(
        self,
        cnn,
        rnn,
        criterion,
        learning_rate=5e-4,
        max_epochs=50,
        is_tta=False,
        tta_transforms=None,
    ):
        super(BaseModel, self).__init__()
        
        self.cnn = cnn
        self.rnn = rnn
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.is_tta = is_tta
        self.tta_transforms = tta_transforms
        
        # self.automatic_optimization = False
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # return optimizer
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=4, max_epochs=self.max_epochs,
            warmup_start_lr=1e-6, eta_min=1e-6,
        )
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.001)
        return [optimizer], [scheduler]

    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)
        
        return output

    def training_step(self, batch, batch_idx):
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
        
        # sch = self.lr_schedulers()
        
        # # step every `n` epochs
        # if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
        #     sch.step()
        
        return {'loss': loss, 'train_score': score}        

    def validation_step(self, batch, batch_idx):
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
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.is_tta:
            return self._tta_predict_step(batch)
        else:
            return self._base_predict_step(batch)
    
    def _base_predict_step(self, batch):
        img = batch['img']
        seq = batch['csv_feature']
        
        output = self(img, seq)
        output = torch.argmax(output, dim=1)
        
        return output
    
    def _tta_predict_step(self, batch):
        img = batch['img']
        seq = batch['csv_feature']
        
        for i, transformer in enumerate(self.tta_transforms):
            augmented_image = transformer.augment_image(img)
            
            # pass to model
            model_output = self(augmented_image, seq)
            model_output = torch.argmax(model_output, dim=1).unsqueeze(1)
            
            if i == 0:
                outputs = model_output.clone()
            else:
                outputs = torch.cat((outputs, model_output), dim=1)
                        
        outputs = torch.mode(outputs, dim=1)
                
        return outputs
            

class Effnetb02LSTMModel(BaseModel):
    def __init__(
        self,
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        rate=0.1,
        learning_rate=5e-4,
    ):  
        cnn = Effnetb0_Encoder()
        rnn = LSTM_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(Effnetb02LSTMModel, self).__init__(cnn, rnn, criterion, learning_rate)


class Effnetb32LSTMModel(BaseModel):
    def __init__(
        self,
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        rate=0.1,
        learning_rate=5e-4,
    ):  
        cnn = Effnetb3_Encoder()
        rnn = LSTM_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(Effnetb32LSTMModel, self).__init__(cnn, rnn, criterion, learning_rate)


class Effnetb72LSTMModel(BaseModel):
    def __init__(
        self,
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        rate=0.1,
        learning_rate=5e-4,
    ):  
        cnn = Effnetb7_Encoder()
        rnn = LSTM_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(Effnetb72LSTMModel, self).__init__(cnn, rnn, criterion, learning_rate)
        
    
class Effnetb7NS2LSTMModel(BaseModel):
    def __init__(
        self,
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        rate=0.1,
        learning_rate=5e-4,
    ):  
        cnn = Effnetb7NS_Encoder()
        rnn = LSTM_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(Effnetb7NS2LSTMModel, self).__init__(cnn, rnn, criterion, learning_rate)
        
        
class Effnetb7NSPlus2LSTMModel(BaseModel):
    def __init__(
        self,
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        rate=0.1,
        learning_rate=5e-4,
        max_epochs=50,
    ):
        cnn = Effnetb7NSPlus_Encoder()
        rnn = LSTM_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = FocalLoss()
        # criterion = nn.CrossEntropyLoss()
        
        super(Effnetb7NSPlus2LSTMModel, self).__init__(cnn, rnn, criterion, learning_rate, max_epochs)
        

class Effnetb7NSPlus2LSTMTTAModel(BaseModel):
    def __init__(
        self,
        max_len, 
        embedding_dim, 
        num_features, 
        class_n,
        tta_transforms,
        rate=0.1,
        learning_rate=5e-4,
        max_epochs=50,
    ):
        cnn = Effnetb7NSPlus_Encoder()
        rnn = LSTM_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = FocalLoss()
        
        super(Effnetb7NSPlus2LSTMTTAModel, self).__init__(
            cnn, rnn, criterion, learning_rate, max_epochs,
            is_tta=True, tta_transforms=tta_transforms
        )
