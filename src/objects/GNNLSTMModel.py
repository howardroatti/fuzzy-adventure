import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv

class GNNLSTMModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, num_layers=2):
        super(GNNLSTMModel, self).__init__()
        
        # Camadas GNN para aprendizado espacial
        self.gnn_layers = nn.ModuleList([GraphConv(in_feats, hidden_size) for _ in range(num_layers)])
        
        # LSTM para aprendizado temporal
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Camada de saída
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, g, features):
        h = features
        for gnn in self.gnn_layers:
            h = torch.relu(gnn(g, h))
        
        # Incorporar as características dos nós em uma sequência temporal
        h = h.unsqueeze(0)  # Adicionar dimensão temporal
        _, (hn, _) = self.lstm(h)  # Obter saída LSTM final
        
        # Classificação
        out = self.fc(hn[-1])
        return out

# Função para treinar a GNN-LSTM
def train_gnn_lstm(graph_data, features, labels, num_epochs=100):
    in_feats = features.shape[1]  # Número de características de entrada (ex. 2 para lat/lon)
    hidden_size = 16
    num_classes = len(set(labels.numpy()))

    model = GNNLSTMModel(in_feats, hidden_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Executar a passagem direta do modelo
        pred = model(graph_data, features)
        loss = loss_fn(pred, labels)
        
        loss.backward()
        optimizer.step()

    return model
