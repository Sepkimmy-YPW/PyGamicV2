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
        return None, None, None, None, False
    else:
        try:
            with open(path, 'r', encoding='utf-8') as fw:
                input_json = json.load(fw)
            
            tsa_turning_angle_list = input_json["tsa_turning_angle"]
            energy_list = input_json["energy"]
            dead_count_list = input_json["dead_count"]

            list_length = len(energy_list)
            x_list = [x for x in range(list_length)]
            return np.array(x_list), np.array(tsa_turning_angle_list), np.array(energy_list), np.array(dead_count_list), True
        except:
            pass

def main():
    plt.rcParams['font.sans-serif'] = 'Times new roman'
    fig, ax = plt.subplots()
    # ax.set_xlabel('folding angle/rad')
    # # ax.set_ylabel('/rad')
    # # ax.set_title('主连接角与第一折叠角的关系曲线')
    ax.set_xmargin(0)
    ax.set_ymargin(0)

    # plt.title("Evolution Process")
    # plt.xlabel("Iteration Number")
    # plt.ylabel("Fitness Value")
    length_list = []
    while 1:
        x, y1, y2, y3, goon = readFile()
        if not goon:
            break
        else:
            # plt.plot(x, 1.0 - y)
            l1, = ax.plot(x, y1, 'b--')
            l2, = ax.plot(x, y2, 'r-')
            new_x = [0]
            new_y3 = [0]
            for i in range(1, len(x)):
                if y3[i] == 0:
                    new_x.append(i)
                    new_y3.append(y3[i - 1])
            new_x.append(x[-1])
            new_y3.append(y3[-1])
            l3, = ax.plot(new_x, new_y3, 'g-')
            length_list.append(len(x))
    if len(length_list) == 0:
        return
    max_length = max(length_list)
    step_number = int(max_length / 100) + 2
    labels = [str(i * 100) for i in range(step_number)]
    plt.xticks(range(0, step_number * 100, 100), labels=labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis='x')
    plt.legend(handles=[l1, l2, l3], labels=["TSA turning angle (rad)", "Energy of origami (mJ)", "Termination count"], loc="upper left", fontsize=14)
    # plt.ylim((0, 60))
    plt.show()


if __name__ == "__main__":
    main()