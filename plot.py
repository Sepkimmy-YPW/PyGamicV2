# import math
import json
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QFileDialog

def readFile():
    path, _ = QFileDialog.getOpenFileName(
        None, 
        "Choose a score pack", 
        ".", 
        "Json files (*.json);;All Files (*.*)"
    )
    if path == '':
        return None, None, False
    else:
        try:
            with open(path, 'r', encoding='utf-8') as fw:
                input_json = json.load(fw)
            score_list = input_json["score"]
            list_length = len(score_list)
            x_list = [x for x in range(list_length)]
            return np.array(x_list), np.array(score_list), True
        except:
            pass

def main():
    plt.rcParams['font.sans-serif'] = 'Times new roman'
    plt.figure()
    # plt.title("Evolution Process")
    # plt.xlabel("Iteration Number")
    # plt.ylabel("Fitness Value")
    length_list = []
    while 1:
        x, y, goon = readFile()
        if not goon:
            break
        else:
            # plt.plot(x, 1.0 - y)
            plt.plot(x, y - 40.0)
            length_list.append(len(x))
    if len(length_list) == 0:
        return
    max_length = max(length_list)
    step_number = int(max_length / 60) + 2
    labels = [str(i * 60) for i in range(step_number)]
    plt.xticks(range(0, step_number * 60, 60), labels=labels)
    plt.grid(axis='x')
    plt.ylim((0, 60))
    plt.show()


if __name__ == "__main__":
    main()