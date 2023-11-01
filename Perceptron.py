import random

def initialize_weights(input_size):
    return [random.uniform(-1, 1) for _ in range(input_size + 1)]  # +1 para o peso de bias

def predict(inputs, weights):
    activation = sum(x * w for x, w in zip([1] + inputs, weights))  # 1 como entrada de bias
    return 1 if activation >= 0 else 0

def train(weights, inputs, target, learning_rate):
    prediction = predict(inputs, weights)
    error = target - prediction

    print(f"Entradas: {inputs[1:]}, Pesos: {weights[1:]}, Predição: {prediction}, Alvo: {target}, Erro: {abs(error)}")

    for i in range(len(weights) - 1):
        adjustment = learning_rate * error * inputs[i]
        print(f"Ajustando peso {i}: {weights[i + 1]} + {adjustment}")
        weights[i + 1] += adjustment  # Adicione 1 para ajustar os pesos corretos

# Dados de treinamento
training_data = [
    ([0, 1, 1], 1),
    ([0, 1, 0], 0),
    ([0, 0, 1], 1),
    ([0, 0, 0], 0),
]

input_size = len(training_data[0][0])
learning_rate = 0.1
weights = initialize_weights(input_size)

# Treinamento do Perceptron
epochs = 700
for _ in range(epochs):
    for data in training_data:
        inputs, target = data
        train(weights, inputs, target, learning_rate)
        print("")

# Teste do Perceptron treinado
for data in training_data:
    inputs, _ = data
    prediction = predict(inputs, weights)
    print(f"Entradas: {inputs[1:]} => Predição: {prediction}")
