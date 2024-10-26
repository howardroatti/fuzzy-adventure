from sklearn.cluster import DBSCAN
from objects.markov_model import MarkovModel

class ClusteringModel:
    """
    Modelo de Agrupamento que utiliza DBSCAN para agrupar localizações e um modelo de Markov para prever transições entre clusters.

    Atributos:
        eps (float): A distância máxima entre dois pontos para serem considerados como parte do mesmo cluster.
        min_samples (int): O número mínimo de pontos necessários para formar um cluster.
        model (DBSCAN): Instância do modelo DBSCAN ajustado aos dados.
        markov_model (MarkovModel): Instância do modelo de Markov que é treinado com os clusters.

    Métodos:
        fit(data): Agrupa localizações e ajusta um modelo de Markov baseado nos clusters.
        predict(current_location): Prevê o próximo cluster usando o modelo de Markov dos clusters.
    """
    def __init__(self, eps=0.01, min_samples=5):
        """
        Inicializa o modelo de agrupamento com os parâmetros especificados.

        Parâmetros:
            eps (float): A distância máxima entre dois pontos para serem considerados como parte do mesmo cluster.
            min_samples (int): O número mínimo de pontos necessários para formar um cluster.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.markov_model = MarkovModel()

    def fit(self, data):
        """
        Agrupa localizações e ajusta um modelo de Markov baseado nos clusters.

        Parâmetros:
            data (list of dict): Lista de dicionários contendo as coordenadas ('coordinate_latitude' e 'coordinate_longitude').

        O método aplica o algoritmo DBSCAN para identificar clusters nas localizações e, em seguida, treina um modelo de Markov com os clusters identificados.
        """
        coordinates = [(entry['coordinate_latitude'], entry['coordinate_longitude']) for entry in data]
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = self.model.fit_predict(coordinates)

        # Adicionar rótulos de clusters aos dados
        for i, entry in enumerate(data):
            entry['cluster'] = labels[i]

        # Construir um modelo de Markov com os clusters como estados
        self.markov_model.fit([{'coordinate_latitude': c[0], 'coordinate_longitude': c[1]} 
                               for c in coordinates if c in self.model.components_])

    def predict(self, current_location):
        """
        Prevê o próximo cluster usando o modelo de Markov dos clusters.

        Parâmetros:
            current_location (tuple): Tupla contendo a latitude e longitude da localização atual.

        Retorno:
            tuple: A próxima localização prevista como uma tupla de (latitude, longitude).

        Levanta:
            ValueError: Se a localização atual não for reconhecida como parte de um cluster.
        """
        cluster_label = self.model.fit_predict([current_location])
        if cluster_label == -1:
            raise ValueError("Localização atual desconhecida.")
        return self.markov_model.predict(current_location)
