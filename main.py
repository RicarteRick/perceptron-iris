import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def step_function(x):
    if x >= 0:
        return 1
    else:
        return -1

def predict_perceptron(weights, x):
    # calculando produto escalar (valor1 * peso1 + valor2 * peso2 + ...)
    z = np.dot(weights, x)

    # chamando funcao de ativacao
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

            predicted_class = predict_perceptron(weights, x)

            # atualizando pesos por meio da comparação com o valor real
            error = y - predicted_class
            weights += learning_rate * error * x

    return weights

def main():
    holdout = 0.2 # 20% dos dados para teste
    
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
    
    # separando dados de treino e teste (hold-out)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=holdout, random_state=42)
    
    # juntando em dataset pra continuar usando a funcao de treino
    train_data = pd.DataFrame(np.column_stack((X_train, y_train)))
    test_data = pd.DataFrame(np.column_stack((X_test, y_test)))

    # treinando perceptron
    trained_weights = train_perceptron(train_data)

    print('pesos treinados:')
    print(trained_weights)

    # testando perceptron
    test_data['predicted_class'] = test_data.apply(lambda x: predict_perceptron(trained_weights, x[:-1]), axis=1)

    # ja contando no codigo pra facilitar
    correct_predictions = 0
    for i in range(len(test_data)):
        if test_data.iloc[i, -1] == test_data.iloc[i, -2]:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)

    print('test_data')
    print(test_data)

    print('acuracia:')
    print(accuracy)

if __name__ == "__main__":
  main()