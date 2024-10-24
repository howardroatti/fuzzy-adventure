import numpy as np
from collections import defaultdict

class MarkovModel:
    def __init__(self):
        self.transition_matrix = None
        self.state_mapping = {}
        self.num_states = 0

    def fit(self, data):
        """Constrói a matriz de transição com base nas coordenadas de entrada."""
        # Mapeamento de coordenadas para estados (índices)
        states = list(set((entry['coordinate_latitude'], entry['coordinate_longitude']) for entry in data))
        self.num_states = len(states)
        self.state_mapping = {state: i for i, state in enumerate(states)}

        # Inicializar matriz de transição
        transition_counts = np.zeros((self.num_states, self.num_states))

        # Contar transições
        for i in range(len(data) - 1):
            current_state = self.state_mapping[(data[i]['coordinate_latitude'], data[i]['coordinate_longitude'])]
            next_state = self.state_mapping[(data[i + 1]['coordinate_latitude'], data[i + 1]['coordinate_longitude'])]
            transition_counts[current_state][next_state] += 1

        # Normalizar para obter probabilidades
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        self.transition_matrix = transition_counts / row_sums

    def predict(self, current_location):
        """Prevê a próxima localização com base na matriz de transição."""
        current_state = self.state_mapping.get(current_location)
        if current_state is None:
            raise ValueError("Localização atual desconhecida.")
        next_state = np.argmax(self.transition_matrix[current_state])
        for location, state in self.state_mapping.items():
            if state == next_state:
                return location
