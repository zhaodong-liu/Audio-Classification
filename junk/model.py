import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from torch import nn
import torch
import torchaudio
import csv




class GenderReco1DCnnModel(nn.Module):
    def __init__(self,n_class=2,n_feats=128,dropout=0.2):
        super(GenderReco1DCnnModel,self).__init__()
        self.layerNorm = nn.LayerNorm(n_feats)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_feats,out_channels=64,kernel_size=8,stride=4,padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64,out_channels=32,kernel_size=4,stride=2,padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2, stride=1,padding=2),
            nn.ReLU(),
        )
 
    def forward(self,embeddings,mask):
 
        x = embeddings[:, :, :]  # embeddings [B,n,D]
        x = self.layerNorm(x)
        x = x.permute(0,2,1)
        x = self.cnn(x)  # []
 
        x = x.view(x.shape[0],-1,x.shape[1])
        x = torch.mean(x, dim=1)
 
 
        out = self.classifier(x)
        out = torch.softmax(out, dim=-1)
 
        return out


import torch
from torch import nn

class GenderReco1DCnnModel(nn.Module):
    def __init__(self, n_class=4, n_feats=40, dropout=0.2):
        super(GenderReco1DCnnModel, self).__init__()
        self.layerNorm = nn.LayerNorm(n_feats)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_feats, out_channels=64, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
        )
        
        # Calculate the output size after Conv1D layers
        def _calc_output_size(L, kernel_size, stride, padding):
            return (L + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        
        # Compute the output length of the sequence after each convolutional layer
        L = 299
        L = _calc_output_size(L, 8, 4, 2)  # After first conv layer
        L = _calc_output_size(L, 4, 2, 2)  # After second conv layer
        L = _calc_output_size(L, 2, 1, 2)  # After third conv layer

        self.classifier = nn.Sequential(
            nn.Linear(16 * L, n_class),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, embeddings):
        x = embeddings.transpose(1, 2)  # Assuming input shape [B, T, F] -> [B, F, T]
        x = self.layerNorm(x)
        x = self.cnn(x)  # [B, C, L]
        x = x.view(x.size(0), -1)  # Flatten the convolution output
        out = self.classifier(x)
        return out




class AudioDataset(Dataset):
    def __init__(self, data_path, labels_path):
        data = np.load(data_path)
        labels = np.load(labels_path)
        self.features = data['features']
        self.labels = labels['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label
    
train_dataset = AudioDataset('train_data.npz', 'train_labels.npz')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


model = GenderReco1DCnnModel(n_class=4, n_feats=40)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10  # 训练轮数
for epoch in range(num_epochs):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'gender_recognition_model.pth')