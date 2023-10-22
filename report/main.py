import os
from matplotlib import patches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics_file(file_path):
    metrics = pd.read_csv(file_path, sep=";")
    return metrics

def generate_individual_accuracy_bar_chart(metrics, epoch_qtd, learning_rate):
    max_accuracy_index = metrics["Acuracia"].idxmax()
    max_accuracy_holdout = metrics.loc[max_accuracy_index, "Hold-out"]
    max_accuracy_value = metrics.loc[max_accuracy_index, "Acuracia"]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics["Hold-out"], metrics["Acuracia"])
    plt.xlabel("Hold-out")
    plt.ylabel("Acuracia")
    plt.title(f"Acuracia por Hold-out. Epochs = {epoch_qtd}, Learning Rate = {learning_rate}")
    plt.xticks(rotation=0)
    plt.savefig(f"accuracy_bar_chart_ep_{epoch_qtd}_lr_{learning_rate}.png")
    plt.close()

    return { 'Combinacao': f'ep_{epoch_qtd}_lr_{learning_rate}', 'Acuracia': max_accuracy_value, 'epoch_qtd': epoch_qtd, 'learning_rate': learning_rate, 'Hold-out': max_accuracy_holdout }

def generate_best_accuracy_bar_chart(data):
    holdout_counts = {}

    for item in data:
        holdout = item['Hold-out']

        if holdout in holdout_counts:
            holdout_counts[holdout] += 1
        else:
            holdout_counts[holdout] = 1


    holdouts = list(holdout_counts.keys())
    counts = [holdout_counts[holdout] for holdout in holdouts]

    plt.figure(figsize=(10, 6))
    plt.bar(holdouts, counts)
    plt.xlabel("Hold-out")
    plt.ylabel("Quantidade de vezes que foi o melhor")
    plt.title("Hold-outs com maiores acurácias dentro da combinação de epochs e learning rate")
    plt.xticks(rotation=0)
    plt.savefig(f"best_accuracy_bar_chart.png")

def generate_accuracy_bar_chart_by_combination(data):
    combinations = [entry['Combinacao'] for entry in data]
    accuracies = [entry['Acuracia'] for entry in data]
    
    holdout_colors = {
        '10-90': 'green',
        '30-70': 'orange',
        '50-50': 'blue'
    }
    colors = [holdout_colors[entry['Hold-out']] for entry in data]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(combinations, accuracies, color=colors)
    plt.xlabel("Combinação")
    plt.ylabel("Acurácia")
    plt.title("Acurácia por Combinação")

    for combination, accuracy in zip(combinations, accuracies):
        plt.text(combination, accuracy, f'{accuracy:.2f}', ha='center', va='bottom')

    # mostrando uma cor para cada hold-out
    legend_patches = [patches.Patch(color=color, label=holdout) for holdout, color in holdout_colors.items()]
    plt.legend(handles=legend_patches, title='Hold-out')

    plt.xticks(rotation=45)
    plt.savefig(f"accuracy_bar_chart_by_combination.png")

def generate_weights_scatter_plot(metrics, epoch_qtd, learning_rate):
    weights_data = metrics["Pesos treinados"].apply(lambda x: eval(x))
    plt.figure(figsize=(8, 6))
    plt.scatter(weights_data.apply(lambda x: x[0]), weights_data.apply(lambda x: x[1]), c=metrics["Acuracia"], cmap='viridis')
    plt.xlabel("Peso 1")
    plt.ylabel("Peso 2")
    plt.colorbar(label="Acurácia")
    
    plt.title(f"Dispersão dos pesos treinados. Epochs = {epoch_qtd}, Learning Rate = {learning_rate}")
    
    plt.savefig(f"weights_scatter_plot_ep_{epoch_qtd}_lr_{learning_rate}.png")
    plt.close()

def main():
    output_folder = '../output'
    folder_name = ''
    metrics_files = []
    best_accuracies = [] # { 'Combinacao': '', 'Acuracia': 0, 'epoch_qtd': 0, 'learning_rate': 0, Hold-out: '' }

    # iterando pelos arquivos de metricas para criar graficos individuais e pegar as maiores acuracias dentre as combinações
    for epoch_qtd in [10, 100, 1000]:
        for learning_rate in [0.1, 0.2, 0.3]:
            folder_name = f'{output_folder}/ep_{epoch_qtd}_lr_{learning_rate}'

            file_name = f'{folder_name}/metrics.out'

            metrics_files.append(file_name)

            metrics_data = load_metrics_file(file_name)

            # grafico individual de acuracia por hold-out
            best_accuracies.append(generate_individual_accuracy_bar_chart(metrics_data, epoch_qtd, learning_rate))

            # grafico de dispersão dos pesos
            generate_weights_scatter_plot(metrics_data, epoch_qtd, learning_rate)

    for data in best_accuracies:
        print(data)

    # grafico que mostra a quantidade de vezes que cada hold-out foi o melhor
    generate_best_accuracy_bar_chart(best_accuracies)

    # grafico que mostra a acuracia por combinação de epochs e learning rate
    generate_accuracy_bar_chart_by_combination(best_accuracies)

if __name__ == "__main__":
    main()