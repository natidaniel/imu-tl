import torch.nn as nn

class IMUConv(nn.Module):
    def __init__(self, config):

        super(IMUConv, self).__init__()
        input_dim = config.get("input_dim")
        num_classes = config.get("num_classes")
        window_size = config.get("window_size")

        config = config.get("imu-cnn")
        latent_dim = config.get("latent_dim")

        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, latent_dim, kernel_size=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(latent_dim, latent_dim, kernel_size=1), nn.ReLU())

        self.dropout = nn.Dropout(config.get("dropout"))
        self.maxpool = nn.MaxPool1d(2) # Collapse T time steps to T/2
        self.classifier = nn.Sequential(nn.Linear(window_size*(latent_dim//2), latent_dim),
                                        nn.ReLU(),
                                        nn.Linear(latent_dim,  num_classes))
        self.log_softmax = nn.LogSoftmax(dim=1)


        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
        x = data.get('imu').transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.maxpool(x) # return B X C/2 x M
        x = x.view(x.size(0), -1) # B X C/2*M
        x = self.classifier(x)
        x = self.log_softmax(x)
        return x # B X N
