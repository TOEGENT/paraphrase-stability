"""
Задача
    имея raw_stability.jsonl с ключами
        prompt; k; N; k_over_N
    Вычленить k и построить гистограмму (unique_k,freq)


"""

from pathlib import Path
import json
import matplotlib.pyplot as plt

def make_uniqueK_over_k_bar(dataset_path):
    path = dataset_path

    with path.open("r",encoding="utf-8") as f:
        k_data = []
        for line in f:
            data_row = json.loads(line.strip())
            k_data.append(data_row["k"])
        data_x_histo = sorted(list(set(k_data)))
        data_y_histo = [k_data.count(k) for k in data_x_histo]
        plt.bar(data_x_histo, data_y_histo, width=0.8, edgecolor="black")
        plt.xlabel("unique_k")
        plt.ylabel("freq")
        plt.show()
