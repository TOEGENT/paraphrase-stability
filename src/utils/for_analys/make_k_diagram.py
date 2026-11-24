"""
Задача
    имея raw_stability.jsonl с ключами
        prompt; k; N; k_over_N
    Вычленить k и построить гистограмму (example_number,k)


"""

from pathlib import Path
import json
import matplotlib.pyplot as plt

def make_k_diagram(dataset_path):
    path = dataset_path

    with path.open("r",encoding="utf-8") as f:
        lines = f.readlines()
        hist_x_data = list(range(len(lines)))
        hist_y_data = []
        for line in lines:
            data_row = json.loads(line.strip())
            k = data_row["k"]
            hist_y_data.append(k)
        print(len(hist_x_data),len(hist_y_data))

        plt.scatter(hist_x_data, hist_y_data,s=1)

        plt.ylim(0, 25)
        plt.xlabel("ex_num")
        plt.ylabel("k")
        plt.show()
