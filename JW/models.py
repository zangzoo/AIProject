# models.py
import torch.nn as nn

class SignTransformer(nn.Module):
    def __init__(self, num_classes,input_dim=1662, d_model=128, nhead=8, nlayers=3):
 
        super().__init__()
        self.fc_in = nn.Linear(input_dim, d_model)
        encoder = nn.TransformerEncoderLayer(d_model, nhead)
        self.trans = nn.TransformerEncoder(encoder, nlayers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        x = self.fc_in(x)            # (B, T, d)
        x = x.permute(1,0,2)         # (T, B, d)
        x = self.trans(x)
        x = x.mean(dim=0)            # (B, d)
        return self.classifier(x)


class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, num_classes, cnn_channels=64, lstm_hidden=128, nlayers=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, nlayers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        x = x.permute(0,2,1)         # (B, D, T)
        x = self.relu(self.conv1(x))
        x = x.permute(0,2,1)         # (B, T, C)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])        # (B, num_classes)

