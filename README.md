
# Fuzzy Adventure
## Previsão de Localização Baseada em Dados Geoespaciais e Temporais

### Descrição

Este projeto utiliza um conjunto de dados contendo coordenadas geográficas (latitude, longitude) e timestamps para prever a próxima localização provável de uma pessoa. A solução foi projetada para capturar padrões espaciais e temporais utilizando várias abordagens de modelagem, incluindo Cadeias de Markov, Clustering Não Supervisionado e Redes Neurais Baseadas em Grafos (GNN-LSTM).

### Abordagens Utilizadas

1. **Cadeia de Markov**:
   - Modela as probabilidades de transição entre localizações consecutivas.
   - Útil para padrões simples de transição entre locais conhecidos.

2. **Clustering Não Supervisionado**:
   - Agrupa localizações em regiões próximas geograficamente.
   - Identifica regiões mais comuns para basear previsões.

3. **Rede Neural Baseada em Grafos (GNN-LSTM)**:
   - Utiliza grafos para capturar relações espaciais entre localizações e LSTMs para capturar dependências temporais.
   - Características dos nós incluem latitude, longitude e atributos temporais derivados (hora do dia, dia da semana, etc.).

4. **Atualização Dinâmica do Grafo**:
   - Detecção de anomalias em padrões de movimentação e atualização do grafo com novos nós quando detectadas novas localizações.

### Estrutura do Projeto

- `location_prediction_system.py`: Implementação das classes principais do sistema de previsão.
- `markov_model.py`: Implementação do modelo de Cadeia de Markov.
- `clustering_model.py`: Implementação do modelo de agrupamento (clustering).
- `gnn_lstm_model.py`: Implementação da Rede Neural Baseada em Grafos com LSTM.
- `haversine_loss.py`: Implementação da função de perda baseada na distância geográfica.

### Dependências

- Python 3.8+
- DGL (Deep Graph Library)
- PyTorch
- Scikit-learn
- Pandas
- NumPy

#### Instalação

```bash
pip install dgl torch scikit-learn pandas numpy
```

### Uso

1. **Criação do Sistema de Previsão**:

   ```python
   from location_prediction_system import LocationPredictionSystem

   # Instanciar o sistema com a abordagem desejada
   system = LocationPredictionSystem(method='gnn')
   ```

2. **Treinamento com Dados**:

   ```python
   # Dados de exemplo
   data = [
       {'trp_id': 1, 'coordinate_latitude': 2, 'coordinate_longitude': 3, 'time': '2024-10-31 12:30:35.000'},
       {'trp_id': 2, 'coordinate_latitude': 2, 'coordinate_longitude': 5, 'time': '2024-10-31 12:32:05.000'},
       # Adicione mais dados conforme necessário
   ]

   # Treinar o modelo com os dados
   system.fit(data)
   ```

3. **Previsão**:

   ```python
   # Prever o próximo local para um novo ponto de dados
   next_location = system.predict(current_data_point)
   ```

4. **Uso da Função de Perda Haversine:**:

   ```python
   from haversine_loss import haversine_loss

   # Ponto de previsão e ponto real (latitude, longitude)
   predicted_point = (64.844444, -147.777777)  # Latitude, Longitude previsto
   actual_point = (64.8333333, -147.7666666)    # Latitude, Longitude real

   # Calcular a perda (distância geográfica)
   loss = haversine_loss(predicted_point, actual_point)
   print(f"Perda Haversine: {loss} km")
   ```

### Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

### Licença

Este projeto é licenciado sob a licença MIT.
