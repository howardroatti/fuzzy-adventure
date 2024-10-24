import dgl
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from objects import *
from objects.GNNLSTMModel import train_gnn_lstm

class LocationPredictionSystem:
    def __init__(self, method='gnn'):
        self.method = method
        self.scaler = StandardScaler()  # Escalonador para as coordenadas

        if method == 'markov':
            self.model = MarkovModel()
        elif method == 'clustering':
            self.model = ClusteringModel()
        elif method == 'gnn':
            self.model = None  # Será instanciado durante o treinamento

    def fit(self, data):
        if self.method == 'markov' or self.method == 'clustering':
            self.model.fit(data)
        elif self.method == 'gnn':
            graph_data, features, labels = self._create_graph_data(data)
            # Escalonar as características dos nós
            features = torch.tensor(self.scaler.fit_transform(features), dtype=torch.float32)
            self.model = train_gnn_lstm(graph_data, features, labels)

    def _create_graph_data(self, data):
        """
        Cria o grafo DGL com base em dados de localização sequenciais.
        Args:
            data (list): Lista de dicionários com campos:
                        'coordinate_latitude', 'coordinate_longitude', e 'time'.
        Returns:
            graph_data (DGLGraph): Grafo DGL criado a partir dos dados.
            features (torch.Tensor): Características dos nós (latitude e longitude).
            labels (torch.Tensor): Rótulos para prever o próximo nó.
        """
        # Ordenar os dados por tempo para garantir a sequência correta
        data = sorted(data, key=lambda x: x['time'])

        # Converter os dados em um DataFrame para manipulação mais simples
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])  # Converter para datetime

        # Criar características temporais
        df['hour_of_day'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)

        # Normalizar ou padronizar essas novas características conforme necessário
        temporal_features = df[['hour_of_day', 'day_of_week', 'time_diff']].values

        # Identificar localizações únicas (latitude, longitude)
        unique_locations = df[['coordinate_latitude', 'coordinate_longitude']].drop_duplicates().reset_index(drop=True)
        
        # Criar um mapeamento de localização para índice de nó
        location_to_node = {tuple(loc): i for i, loc in unique_locations.iterrows()}

        # Criar listas de origem e destino para arestas, com base na sequência temporal
        src_nodes, dst_nodes = [], []

        for i in range(len(df) - 1):
            # Verificar se os pontos consecutivos são diferentes (para evitar loops auto-referentes)
            if (df.loc[i, 'coordinate_latitude'], df.loc[i, 'coordinate_longitude']) != \
            (df.loc[i + 1, 'coordinate_latitude'], df.loc[i + 1, 'coordinate_longitude']):
                src = location_to_node[(df.loc[i, 'coordinate_latitude'], df.loc[i, 'coordinate_longitude'])]
                dst = location_to_node[(df.loc[i + 1, 'coordinate_latitude'], df.loc[i + 1, 'coordinate_longitude'])]
                src_nodes.append(src)
                dst_nodes.append(dst)

        # Criar um grafo DGL a partir das arestas
        graph_data = dgl.graph((src_nodes, dst_nodes), num_nodes=len(unique_locations))

        # Atribuir características aos nós (latitude e longitude)
        features = torch.tensor(unique_locations.values, dtype=torch.float32)

        # Preparar rótulos como o nó de destino para cada nó de origem
        labels = torch.tensor(dst_nodes, dtype=torch.long)

        # Combinar características de localização e temporais
        features = np.concatenate([unique_locations.values, temporal_features], axis=1)
        features = torch.tensor(features, dtype=torch.float32)

        return graph_data, features, labels


    def predict(self, current_location):
        if self.method == 'markov' or self.method == 'clustering':
            return self.model.predict(current_location)
        elif self.method == 'gnn':
            # Implementar lógica para prever o próximo nó usando a GNN-LSTM
            pass

def haversine_loss(predicted_coords, true_coords):
    # Implementar a fórmula de Haversine para calcular a distância geográfica
    R = 6371  # Raio da Terra em km
    lat1, lon1 = predicted_coords[:, 0], predicted_coords[:, 1]
    lat2, lon2 = true_coords[:, 0], true_coords[:, 1]

    dlat = torch.radians(lat2 - lat1)
    dlon = torch.radians(lon2 - lon1)

    a = torch.sin(dlat / 2) ** 2 + torch.cos(torch.radians(lat1)) * torch.cos(torch.radians(lat2)) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c
    return torch.mean(distance)
