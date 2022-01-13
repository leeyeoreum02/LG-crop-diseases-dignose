from sklearn.metrics import f1_score

import torch
from torch import nn, optim
from torchvision.models import efficientnet_b0, efficientnet_b7
from torchvision.models import efficientnet_b3
from pytorch_lightning import LightningModule
from timm.models.efficientnet import tf_efficientnet_b7_ns


def accuracy_function(real, pred):    
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score


class Effnetb0_Encoder(nn.Module):
    def __init__(self):
        super(Effnetb0_Encoder, self).__init__()
        self.model = efficientnet_b0(pretrained=True)
    
    def forward(self, inputs):
        output = self.model(inputs)
        return output
    
    
class Effnetb3_Encoder(nn.Module):
    def __init__(self):
        super(Effnetb3_Encoder, self).__init__()
        self.model = efficientnet_b3(pretrained=True)
    
    def forward(self, inputs):
        output = self.model(inputs)
        return output
    

class Effnetb7_Encoder(nn.Module):
    def __init__(self):
        super(Effnetb7_Encoder, self).__init__()
        self.model = efficientnet_b7(pretrained=True)
    
    def forward(self, inputs):
        output = self.model(inputs)
        return output
    
    
class Effnetb7NS_Encoder(nn.Module):
    def __init__(self):
        super(Effnetb7NS_Encoder, self).__init__()
        self.model = tf_efficientnet_b7_ns(pretrained=True)
        
    def forward(self, inputs):
        output = self.model(inputs)
        return output
    
    
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


class Effnetb02LSTMModel(LightningModule):
    def __init__(
        self,
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        rate=0.1,
        learning_rate=5e-4,
    ):
        super().__init__()
        
        self.cnn = Effnetb0_Encoder()
        self.rnn = LSTM_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

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
        img = batch['img']
        seq = batch['csv_feature']
        
        output = self(img, seq)
        output = torch.argmax(output, dim=1)
        
        return output
    
    
class Effnetb32LSTMModel(LightningModule):
    def __init__(
        self,
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        rate=0.1,
        learning_rate=5e-4,
    ):
        super().__init__()
        
        self.cnn = Effnetb3_Encoder()
        self.rnn = LSTM_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

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
        img = batch['img']
        seq = batch['csv_feature']
        
        output = self(img, seq)
        output = torch.argmax(output, dim=1)
        
        return output


class Effnetb72LSTMModel(LightningModule):
    def __init__(
        self,
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        rate=0.1,
        learning_rate=5e-4,
    ):
        super().__init__()
        
        self.cnn = Effnetb7_Encoder()
        self.rnn = LSTM_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

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
        img = batch['img']
        seq = batch['csv_feature']
        
        output = self(img, seq)
        output = torch.argmax(output, dim=1)
        
        return output
    
    
class Effnetb7NS2LSTMModel(LightningModule):
    def __init__(
        self,
        max_len, 
        embedding_dim, 
        num_features, 
        class_n, 
        rate=0.1,
        learning_rate=5e-4,
    ):
        super().__init__()
        
        self.cnn = Effnetb7NS_Encoder()
        self.rnn = LSTM_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

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
        img = batch['img']
        seq = batch['csv_feature']
        
        output = self(img, seq)
        output = torch.argmax(output, dim=1)
        
        return output
