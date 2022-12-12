import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLayerNorm(nn.Module):
    '''Layer normalization built for cnns input'''

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    '''Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    '''

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalLSTM(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalLSTM, self).__init__()

        self.BiLSTM = nn.LSTM(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.BiLSTM(x)
        x = self.dropout(x)
        return x

class AcousticModel(nn.Module):
    '''
        The acoustic model: baseline and MTL share the same class,
        the only difference is the target dimension of the last fc layer
    '''

    def __init__(self, n_cnn_layers, rnn_dim, n_class, n_feats, stride=1, dropout=0.1):
        super(AcousticModel, self).__init__()

        self.n_class = n_class
        if isinstance(n_class, int):
            target_dim = n_class
        else:
            target_dim = n_class[0] * n_class[1]

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, n_feats, 3, stride=stride, padding=3 // 2),
            nn.ReLU()
        )

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(n_feats, n_feats, kernel=3, stride=1, dropout=dropout, n_feats=128)
            for _ in range(n_cnn_layers)
        ])

        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 3))
        self.fully_connected = nn.Linear(n_feats * 64, rnn_dim)

        self.bilstm = nn.Sequential(
            BidirectionalLSTM(rnn_dim=rnn_dim, hidden_size=rnn_dim, dropout=dropout, batch_first=True),
            # BidirectionalLSTM(rnn_dim=2*rnn_dim, hidden_size=rnn_dim, dropout=dropout, batch_first=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, target_dim)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.rescnn_layers(x)
        x = self.maxpooling(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)

        x = self.bilstm(x)
        x = self.classifier(x)

        return F.log_softmax(x, dim=2)