import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv

class GNNLSTMModel(nn.Module):
    """
    Modelo de rede neural que combina uma Rede Neural Gráfica (GNN) e uma LSTM para previsão de classes em dados estruturados como grafos.

    Este modelo aplica uma sequência de camadas de convolução gráfica para extrair características dos dados do grafo 
    e, em seguida, utiliza uma LSTM para capturar dependências temporais nas representações extraídas. 

    Atributos:
        input_proj (nn.Linear): Camada linear que projeta as características de entrada para o tamanho oculto.
        gnn_layers (ModuleList): Lista de camadas de convolução gráfica (GraphConv).
        lstm (nn.LSTM): Camada LSTM para processar as características extraídas das GNNs.
        fc (nn.Linear): Camada de saída que mapeia as saídas da LSTM para as classes previstas.
        dropout (nn.Dropout): Camada de dropout para regularização.

    Métodos:
        forward(g, features): Realiza a passagem direta do modelo para calcular as previsões a partir do grafo e características.
    """
    def __init__(self, in_feats, hidden_size, num_classes, num_layers=2):
        """
        Inicializa o modelo GNN-LSTM com as dimensões e parâmetros especificados.

        Parâmetros:
            in_feats (int): Número de características de entrada.
            hidden_size (int): Tamanho da camada oculta.
            num_classes (int): Número de classes de saída.
            num_layers (int, opcional): Número de camadas GNN. Padrão é 2.
        """
        super(GNNLSTMModel, self).__init__()
        
        self.input_proj = nn.Linear(in_feats, hidden_size)
        
        # Camadas GNN
        self.gnn_layers = nn.ModuleList([
            GraphConv(hidden_size, hidden_size, allow_zero_in_degree=True) for _ in range(num_layers)
        ])
        
        # LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Camada de saída
        self.fc = nn.Linear(hidden_size, num_classes)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, features):
        """
        Realiza a passagem direta do modelo para calcular as previsões a partir do grafo e características.

        Parâmetros:
            g (dgl.DGLGraph): O grafo de entrada com suas conexões.
            features (torch.Tensor): Tensor com as características dos nós do grafo.

        Retorno:
            torch.Tensor: Tensor contendo as previsões para cada nó do grafo, com formato [batch_size, num_classes].
        """
        # Projetar características
        h = self.input_proj(features)
        
        # Camadas GNN
        for gnn in self.gnn_layers:
            h = torch.relu(gnn(g, h))
            h = self.dropout(h)
        
        # Remodelar para LSTM - mantendo batch_size correto
        batch_size = features.size(0)
        h = h.view(1, batch_size, -1)  # [1, batch_size, hidden_size]
        
        # LSTM
        output, (hn, _) = self.lstm(h)
        
        # Usar a saída completa ao invés do último estado
        output = output.squeeze(0)  # Remover dimensão do batch
        
        # Classificação
        out = self.fc(output)
        
        return out

def train_gnn_lstm(graph_data, features, labels, num_epochs=100):
    """
    Treina um modelo GNN-LSTM utilizando dados de grafo e características associadas.

    Este método realiza a passagem de treinamento do modelo GNN-LSTM, incluindo a inicialização do modelo, 
    otimização e monitoramento do desempenho. O treinamento é realizado por um número especificado de épocas, 
    com implementação de early stopping para evitar overfitting.

    Parâmetros:
        graph_data (dgl.DGLGraph): O grafo de entrada que contém a estrutura e as arestas entre os nós.
        features (torch.Tensor): Tensor contendo as características dos nós do grafo.
        labels (torch.Tensor): Tensor contendo os rótulos correspondentes para cada nó.
        num_epochs (int, opcional): O número de épocas para treinamento. O padrão é 100.

    Retorno:
        GNNLSTMModel: O modelo treinado, pronto para fazer previsões.
    """
    print("\nInformações do dataset:")
    print(f"Número de nós: {features.size(0)}")
    print(f"Número de features: {features.size(1)}")
    print(f"Número de labels: {labels.size(0)}")
    
    in_feats = features.shape[1]
    hidden_size = 16
    num_classes = len(torch.unique(labels)) + 1  # +1 para incluir possível classe 0
    
    print(f"\nDimensões do modelo:")
    print(f"Features de entrada: {in_feats}")
    print(f"Tamanho oculto: {hidden_size}")
    print(f"Número de classes: {num_classes}")
    
    # Verificar e ajustar labels
    if torch.min(labels) < 0:
        print("Aviso: Labels negativos encontrados!")
        labels = labels - torch.min(labels)  # Normalizar para começar em 0
    
    # Garantir que labels estão dentro do range esperado
    assert torch.max(labels) < num_classes, "Labels maiores que número de classes!"
    
    model = GNNLSTMModel(in_feats, hidden_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Reduzida learning rate
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Adicionar scheduler para learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    best_loss = float('inf')
    patience = 7
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        try:
            # Forward pass
            pred = model(graph_data, features)
            
            # Verificar dimensões
            print(f"\nÉpoca {epoch + 1}:")
            print(f"Pred shape: {pred.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # Garantir que pred e labels têm as mesmas dimensões
            if len(pred) != len(labels):
                print("Ajustando dimensões...")
                labels = labels[:len(pred)]
            
            loss = loss_fn(pred, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Atualizar scheduler
            scheduler.step(loss)
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping na época {epoch}")
                break
                
            if epoch % 5 == 0:
                print(f'Época {epoch}, Loss: {loss.item():.4f}')
                
        except RuntimeError as e:
            print(f"Erro na época {epoch}: {str(e)}")
            raise
            
    return model

# Função auxiliar para verificar dados
def verify_data_integrity(features, labels, graph):
    """
    Verifica a integridade dos dados de entrada, incluindo características, rótulos e estrutura do grafo.

    Este método imprime informações sobre as dimensões das características e rótulos, além do número de nós 
    e arestas no grafo. Ele também verifica se existem valores NaN nas características ou rótulos e imprime 
    avisos se forem encontrados.

    Parâmetros:
        features (torch.Tensor): Tensor contendo as características dos nós do grafo.
        labels (torch.Tensor): Tensor contendo os rótulos correspondentes para cada nó.
        graph (dgl.DGLGraph): O grafo cujas informações de integridade estão sendo verificadas.

    Retorno:
        None
    """
    print("\nVerificando integridade dos dados:")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Número de nós no grafo: {graph.num_nodes()}")
    print(f"Número de arestas no grafo: {graph.num_edges()}")
    print(f"Range de labels: [{torch.min(labels)}, {torch.max(labels)}]")
    
    if torch.isnan(features).any():
        print("AVISO: Features contêm NaN!")
    if torch.isnan(labels).any():
        print("AVISO: Labels contêm NaN!")