import torch
import torch.nn as nn
import pdb

# 自定义TextCNN模型
class LSTMClassifier(nn.Module):
    def __init__(self, config):
        super(LSTMClassifier, self).__init__()
        self.config = config

        self.embedding = nn.Embedding.from_pretrained(self.config.glove.vectors,freeze=True)
        self.lstm = nn.LSTM(self.config.embedding_size, self.config.embedding_size,dropout=self.config.dropout,bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.config.embedding_size*2, self.config.output_size)

    def forward(self, input_ids,**kwargs):        
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded)
        output = torch.sum(output,dim=1)
        output = self.fc(output)
        return output
