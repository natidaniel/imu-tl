import torch.nn as nn
import torch

class IMULSTM(nn.Module):
    def __init__(self, config):

        super(IMULSTM, self).__init__()
        input_dim = config.get("input_dim")
        num_classes = config.get("num_classes")

        config = config.get("imu-lstm")
        self.n_layers = 1
        self.n_hidden = config.get("lstm_hidden_dim")  # 16

        latent_dim = config.get("latent_dim") # 64

        self.lstm = nn.LSTM(input_dim, self.n_hidden, num_layers=self.n_layers, batch_first=True)
        self.body = nn.Sequential(nn.Linear(self.n_hidden,latent_dim),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(latent_dim, latent_dim),
                                  nn.ReLU(inplace=True)
                                  )
        self.dropout = nn.Dropout(config.get("dropout")) # 0.1
        self.classifier = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(latent_dim, latent_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(latent_dim,  num_classes))
        self.log_softmax = nn.LogSoftmax(dim=1)
        # init
        '''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        '''

    def get_classifier_head_prefix(self, last_layer=False):
        pref = "classifier"
        if last_layer:
            pref = pref + ".{}".format(len(self.classifier)//2+1)
        return pref

    def forward(self, data):
        """
        Forward pass
        :param x:  B X M x T tensor reprensting a batch of size B of  M sensors (measurements) X T time steps (e.g. 128 x 6 x 100)
        :return: B X N weight for each mode per sample
        """
        x = data.get('imu')  # Shape N x S x C with S = sequence length, N = batch size, C = channels
        x, self.hidden = self.lstm(x, self.hidden)
        x = x[:, -1] # last cell
        x = self.body(x)
        x = self.dropout(x)
        x = self.log_softmax(self.classifier(x))
        return x

    def init_hidden(self, batch_size, device):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device)
        self.hidden = (hidden_state, cell_state)
