import dgl
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from objects import *
from objects.gnn_lstm_model import train_gnn_lstm, verify_data_integrity

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
            # Criar grafo e características a partir dos dados
            graph_data, features, labels = self._create_graph_data(data)
            
            # Escalonar as características dos nós
            features = torch.tensor(self.scaler.fit_transform(features), dtype=torch.float32)
            
            # Adicionando impressão para verificar formas
            print("Número de nós no grafo:", graph_data.num_nodes())
            print("Formato das características:", features.shape)

            # Garantir que o número de nós corresponde ao número de características
            if graph_data.num_nodes() != features.shape[0]:
                raise ValueError("O número de nós no grafo não corresponde ao número de características.")
            
            verify_data_integrity(features, labels, graph_data)
            
            # Treinar o modelo GNN com os dados do grafo
            self.model = train_gnn_lstm(graph_data, features, labels)
            
            # Atribuir as features calculadas ao atributo da classe
            self.features = features

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

        # Identificar localizações únicas (latitude, longitude)
        self.unique_locations = df[['coordinate_latitude', 'coordinate_longitude']].drop_duplicates().reset_index(drop=True)
        
        # Criar um mapeamento de localização para índice de nó
        self.location_to_node = {tuple(loc): i for i, loc in self.unique_locations.iterrows()}

        # Criar listas de origem e destino para arestas, com base na sequência temporal
        src_nodes, dst_nodes = [], []
        
        # Inicializar o dicionário para armazenar características temporais
        temporal_features_dict = {}

        for i in range(len(df) - 1):
            # Verificar se os pontos consecutivos são diferentes (para evitar loops auto-referentes)
            if (df.loc[i, 'coordinate_latitude'], df.loc[i, 'coordinate_longitude']) != \
            (df.loc[i + 1, 'coordinate_latitude'], df.loc[i + 1, 'coordinate_longitude']):
                src = self.location_to_node[(df.loc[i, 'coordinate_latitude'], df.loc[i, 'coordinate_longitude'])]
                dst = self.location_to_node[(df.loc[i + 1, 'coordinate_latitude'], df.loc[i + 1, 'coordinate_longitude'])]
                src_nodes.append(src)
                dst_nodes.append(dst)

                # Adicionar características temporais ao dicionário
                if src not in temporal_features_dict:
                    temporal_features_dict[src] = []
                temporal_features_dict[src].append([
                    df.loc[i, 'hour_of_day'],
                    df.loc[i, 'day_of_week'],
                    df.loc[i, 'time_diff']
                ])

        # Criar um grafo DGL a partir das arestas
        graph_data = dgl.graph((src_nodes, dst_nodes), num_nodes=len(self.unique_locations))
        graph_data = dgl.add_self_loop(graph_data)  # Adicionar auto-loops

        # Atribuir características aos nós (latitude e longitude)
        features = torch.tensor(self.unique_locations.values, dtype=torch.float32)

        # Preparar rótulos como o nó de destino para cada nó de origem
        labels = torch.tensor(dst_nodes, dtype=torch.long)

        # Adicionar um rótulo dummy para o último nó
        labels = torch.cat([labels, torch.tensor([-1])])

        # Criar características temporais correspondentes às localizações únicas
        temporal_features = []
        for i in range(len(features)):
            if i in temporal_features_dict:
                # Pegando a média das características temporais do cluster
                mean_features = np.mean(temporal_features_dict[i], axis=0)
                temporal_features.append(mean_features)
            else:
                # Se não houver dados, preencher com zeros
                temporal_features.append([0, 0, 0])

        # Converter para tensor
        temporal_features = torch.tensor(np.array(temporal_features), dtype=torch.float32)

        # Combinar características de localização e temporais
        features = torch.cat((features, temporal_features), dim=1)

        print("Formato das características após combinação:", features.shape)

        return graph_data, features, labels

    def predict(self, current_data_point):
        if self.method == 'markov' or self.method == 'clustering':
            return self.model.predict(np.array([[current_data_point['coordinate_latitude'], current_data_point['coordinate_longitude']]]))
        elif self.method == 'gnn':
            # Extração de características temporais para o ponto atual
            current_time = pd.to_datetime(current_data_point['time'])
            hour_of_day = current_time.hour
            day_of_week = current_time.dayofweek

            # Calcular a diferença de tempo entre o ponto atual e o último ponto de dados do modelo
            if hasattr(self, 'last_time'):
                time_diff = (current_time - self.last_time).total_seconds()
            else:
                time_diff = 0

            # Atualizar o último timestamp para o próximo ponto de previsão
            self.last_time = current_time

            # Etapa 1: Preparar a entrada (coordenadas e características temporais)
            current_location = np.array([[current_data_point['coordinate_latitude'], 
                                          current_data_point['coordinate_longitude'],
                                          hour_of_day, day_of_week, time_diff]])

            # Escalonar as características completas (coordenadas + características temporais)
            current_location = self.scaler.transform(current_location)

            # Criar um novo tensor de características para a localização atual
            new_location_tensor = torch.tensor(current_location, dtype=torch.float32)

            # Criar um novo nó para a localização atual
            new_node_index = len(self.unique_locations)  # Novo índice para o novo nó

            # Adicionar a nova localização ao conjunto de localizações únicas
            self.unique_locations = np.vstack([self.unique_locations, current_location[:, :2]])  # Adiciona apenas as coordenadas
            self.location_to_node[tuple(current_location[0, :2])] = new_node_index

            # Encontrar o nó mais próximo baseado nas coordenadas
            if self.features is not None:
                distances = np.linalg.norm(self.features.numpy()[:, :2] - current_location[:, :2], axis=1)
                nearest_node_index = np.argmin(distances)  # Índice do nó mais próximo

                # Definir as arestas para o novo nó (conectar ao nó mais próximo)
                new_src_nodes = [new_node_index]
                new_dst_nodes = [nearest_node_index]

                # Criar um novo grafo com a localização atual
                graph_data = dgl.graph((new_src_nodes, new_dst_nodes), num_nodes=len(self.unique_locations))

                # Adicionar o novo tensor de características ao conjunto de características existentes
                features = torch.cat((self.features, new_location_tensor), dim=0)

                # Etapa 3: Prever o próximo nó
                with torch.no_grad():
                    predicted_node = self.model(graph_data, features).argmax(dim=1)

                # Etapa 4: Decodificar a previsão
                predicted_coordinates = self.unique_locations[predicted_node].tolist()  # Obter coordenadas do nó previsto

                return predicted_coordinates
            else:
                raise ValueError("As características dos nós não foram inicializadas. Treine o modelo antes de prever.")

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
