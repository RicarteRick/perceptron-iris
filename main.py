import numpy as np
import pandas as pd

def step_function(x):
    if x >= 0:
        return 1
    else:
        return -1

def predict_perceptron(weights, x):
    z = np.dot(weights, x)

    return step_function(z)

def train_perceptron(data, learning_rate=0.1, epochs=100):
    # inicializando pesos aleatoriamente
    weights = np.random.uniform(-1, 1, size=data.shape[1] - 1)

    for epoch in range(epochs):
        for i in range(len(data)):
            # pegando os valores
            x = data.iloc[i, :-1].values

            # pegando a label
            y = data.iloc[i, -1]

            # calculando produto escalar (valor1 * peso1 + valor2 * peso2 + ...)
            z = np.dot(weights, x)

            # funcao de ativacao
            predicted_class = step_function(z)

            # atualizando pesos por meio da comparação com o valor real
            error = y - predicted_class
            weights += learning_rate * error * x

    return weights

def main():
    # lendo dataset
    data = pd.read_csv("iris/iris.data", header=None)

    # selecionando características comprimento da sépala (coluna 0) e largura da sépala (coluna 1)
    # e as duas espécies Setosa e Versicolor (primeiras 100 linhas)
    data = data.iloc[:100, [0, 1, 4]]

    # 1 = Setosa
    # -1 = Versicolor
    data[4] = np.where(data[4] == 'Iris-setosa', 1, -1)

    # embaralhando os dados do dataset 
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # treinando perceptron
    trained_weights = train_perceptron(data)
    
    print('trained_weights')
    print(trained_weights)

if __name__ == "__main__":
  main()