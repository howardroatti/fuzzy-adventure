from sklearn.cluster import DBSCAN
from MarkovModel import MarkovModel

class ClusteringModel:
    def __init__(self, eps=0.01, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.markov_model = MarkovModel()

    def fit(self, data):
        """Agrupa localizações e ajusta um modelo de Markov baseado nos clusters."""
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
        """Prevê o próximo cluster usando o modelo de Markov dos clusters."""
        cluster_label = self.model.fit_predict([current_location])
        if cluster_label == -1:
            raise ValueError("Localização atual desconhecida.")
        return self.markov_model.predict(current_location)
