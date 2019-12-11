
import torch
import torch.nn as nn
import torch.nn.functional as F

class LOL_model(nn.Module):
    def __init__(self, data_size):
        super(LOL_model, self).__init__()
        self.layer1 = nn.Linear(data_size, data_size)
        self.dropout1 = nn.Dropout(.1)
        self.layer2 = nn.Linear(data_size, data_size)
        self.dropout2 = nn.Dropout(.1)
        self.classifier = nn.Linear(data_size, 2)
        
    def forward(self, x, labels=None):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        logits = F.softmax(self.classifier(x), dim=1)
        
        if not labels is None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
            return (loss, logits)

        return logits
