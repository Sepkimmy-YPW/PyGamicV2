import os
import numpy as np
from copy import deepcopy
# import random
# import math
# import matplotlib.pyplot as plt
# from units import UnitPackParserReverse
# import dxfgrabber
from phys_sim17 import OrigamiSimulator
from utils import *
import json
import pandas as pd
import multiprocessing

START = 0
END = 1
X = 0
Y = 1
VALLEY = 0
MOUNTAIN = 1
BORDER = 2

bonus_val = 1.
fail_reward = -bonus_val
encourage_reward = 1.0
episodes = 2001
record_episode = 2000

seed = 3407

string_number = 2
direct_reward = False

cut_num = 0
SIM = 1
origami = "mountain11"
new_method = False
calculate_number = False
initialize = True
cut_nodes = True

candidate_methods = []
time = 1
step = 0
maximum_valid = SIM

NAME = f"SBS-{origami}-{SIM}sim-{string_number}string-{episodes}episodes"

if calculate_number:
    NAME = f"CALNUM-{origami}-{SIM}sim-{string_number}string-{episodes}episodes"

if initialize:
    NAME += "-initialize"

if new_method:
    NAME += "-precut"

if cut_nodes:
    NAME += "-cutnodes"

file_path = f"./threadingResult/train-" + NAME
try:
    os.makedirs(file_path)
except:
    pass

maximum_thread = 4

FOLDING_MAXIMUM = 0.95

BUFFER_INIT = 5000

def workerMultisim(mlist, method, pointer, p_points, p_connections, max_edge, units, max_size, total_bias):
    ori_sim = OrigamiSimulator(use_gui=False, origami_name=origami, fast_simulation=True)

    ori_sim.P_number = len(p_points)
    ori_sim.P_candidate = p_points
    ori_sim.P_candidate_connection = p_connections
    
    ori_sim.method = method

    ori_sim.startOnlyTSA(units, max_size, total_bias, max_edge)
    ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
    ori_sim.check_force = False
    ori_sim.initializeRunning()
    
    while 1:
        ori_sim.step()
        if ori_sim.folding_angle_reach_pi[0] or (ori_sim.dead_count >= 100 and not ori_sim.can_rotate) or (ori_sim.dead_count >= 40 and ori_sim.can_rotate):
            if ori_sim.sim_mode == ori_sim.TSA_SIM and ori_sim.can_rotate:
                if ori_sim.recorded_folding_percent[-1] > FOLDING_MAXIMUM:
                    break
                else:
                    if ori_sim.recorded_folding_percent[-1] < np.mean(np.array(ori_sim.recorded_folding_percent[-51: -1])):
                        break
            else:
                break
        
    if not ori_sim.can_rotate:
        print("Batch: " + str(pointer) + ", Value: Error Actuation")
        mlist[pointer] = -1.
    else:
        if ori_sim.dead_count >= 500:
            value = -1.
            print("Batch: " + str(pointer) + ", Value: " + str(value))
            mlist[pointer] = value
        else:
            folding_percent = ori_sim.recorded_folding_percent[-1]
            folding_speed = (ori_sim.recorded_folding_percent[-1] - ori_sim.recorded_folding_percent[0]) / (ori_sim.recorded_t[-1] - ori_sim.recorded_t[0])
            # value = folding_speed * folding_percent
            if folding_percent < FOLDING_MAXIMUM:
                value = folding_percent
            else:
                value = folding_speed + folding_percent

            print("Batch: " + str(pointer) + ", Value: " + str(value))
            
            mlist[pointer] = value

class Env:
    def __init__(self) -> None:
        self.max_edge = 4
        self.output_reward_buffer = True

        # 获取点和线段信息
        with open(f"./descriptionData/{origami}.json", 'r', encoding='utf-8') as fw:
            input_json = json.load(fw)
        self.kps = []
        self.lines = []
        self.units = []
        for i in range(len(input_json["kps"])):
            self.kps.append(input_json["kps"][i])
        for i in range(len(input_json["lines"])):
            self.lines.append(Crease(
                input_json["lines"][i][START], input_json["lines"][i][END], BORDER 
            ))
            self.lines[i].crease_type = input_json["line_features"][i]["type"]
            self.lines[i].level = input_json["line_features"][i]["level"]
            self.lines[i].coeff = input_json["line_features"][i]["coeff"]
            self.lines[i].recover_level = input_json["line_features"][i]["recover_level"]
            self.lines[i].hard = input_json["line_features"][i]["hard"]
            self.lines[i].hard_angle = input_json["line_features"][i]["hard_angle"]
        for i in range(len(input_json["units"])):
            self.units.append(Unit())
            kps = input_json["units"][i]
            for j in range(0, -len(kps), -1):
                crease_type = BORDER
                current_kp = [kps[j][X], kps[j][Y]]
                next_kp = [kps[j - 1][X], kps[j - 1][Y]]
                for line in self.lines:
                    if (line[START] == current_kp and line[END] == next_kp) or \
                        (line[END] == current_kp and line[START] == next_kp):
                        crease_type = line.getType()
                        break
                self.units[i].addCrease(Crease(
                    current_kp, next_kp, crease_type
                ))

        # calculate max length of view
        self.max_size, max_x, max_y = getMaxDistance(self.kps)
        self.total_bias = getTotalBias(self.units)

        self.unit_number = len(self.units)
        self.best_reward = 0.0
        self.node_num = 1

        self.string_number = string_number

        self.P_candidators = input_json["P_candidators"]["points"]
        self.P_candidators_connections = input_json["P_candidators"]["connections"]

        self.P_points = [
            np.array(self.P_candidators[i]) for i in range(len(self.P_points))
        ]

        self.O_points = [
            np.array(self.units[i].getCenter()) for i in range(self.unit_number)
        ]

    def getRewardList(self, methods):
        valid_number = len(methods)

        if valid_number >= 1: 
            print("There are " + str(valid_number) + " valid cases, using multi-process technology")
            initial_fitness_list = [0. for _ in range(valid_number)]

            mlist = multiprocessing.Manager().list(initial_fitness_list)

            p_list = []

            pointer = 0

            while pointer < len(methods):
                p = multiprocessing.Process(target=workerMultisim, args=(
                        mlist, methods[pointer], pointer,
                        self.P_points, self.P_candidators_connections, self.max_edge, self.units, self.max_size, self.total_bias
                    )
                )
                p_list.append(p)
                pointer += 1
            
            process_id = 0

            total_process_number = maximum_thread

            current_process_number = 0

            while process_id < len(p_list):
                while current_process_number < total_process_number:
                    p_list[process_id].start()
                    current_process_number += 1
                    process_id += 1
                    if current_process_number == total_process_number or process_id == len(p_list):
                        break

                while current_process_number > 0:
                    p_list[process_id - current_process_number].join()
                    current_process_number -= 1

            reward_list = list(mlist)

        return reward_list
    
def train(env: Env, buf=0):
    try:
        path = os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}_true_reward.json")
        with open(path, 'r', encoding='utf-8') as fw:
            input_json = json.load(fw)
    except:
        path = os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}.json")
        with open(path, 'r', encoding='utf-8') as fw:
            input_json = json.load(fw)

    total_number = input_json["number"]
    methods = input_json["method"]
    # maximum_thread = maximum_thread
    sub_methods = []
    sub_ids = []
    best_reward = fail_reward
    sim_step = []
    tension_scores = []
    reward_scores = []

    print(f"----Training using candidate: {path}. \nTotal number: {total_number}----")
    for step in range(0, total_number):
        sim_step.append(step)
        tension_scores.append(methods[step]["score"])
        try:
            reward = methods[step]["reward"]
            reward_scores.append(reward)
            # if reward == -1.0:
            #     raise RuntimeError
            if reward >= best_reward:
                best_reward = reward
                best_trajectory = methods[step]["method"]
                output_trajectory = deepcopy(best_trajectory)
                output_trajectory["location"] = [[] for _ in range(env.string_number)]
                for i in range(len(best_trajectory['id'])):
                    for j in range(len(best_trajectory['id'][i])):
                        if best_trajectory['type'][i][j] == 'A':
                            output_trajectory['location'][i].append(env.P_points[best_trajectory['id'][i][j]].tolist())
                        else:
                            output_trajectory['location'][i].append(env.O_points[best_trajectory['id'][i][j]].tolist())
                with open(os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}_best.json"), 'w', encoding="utf-8") as f:
                    json.dump(output_trajectory, f, indent=4)
            print(f"Reward is directly acquired. Step: {step}, Best reward: {best_reward}")
            print(best_trajectory)
            data = [[sim_step[i], tension_scores[i], reward_scores[i]] for i in range(len(sim_step))]
            test = pd.DataFrame(columns=["step", "tension_score", "true_reward"], data=data)
            test.to_csv(os.path.join(file_path, f'result_train_time_{time}_candidate_buf_{buf}.csv'))
        except:
            if len(sub_methods) < maximum_thread and step < total_number - 1:
                sub_methods.append(methods[step]["method"])
                sub_ids.append(step)
                level = [len(ele) for ele in methods[step]["method"]["id"]]
                print(f"Add id {step} with tension score {tension_scores[-1]} and level {level}")
                if len(sub_methods) < maximum_thread:
                    continue
                else:
                    reward_list = env.getRewardList(sub_methods)
                    for i in range(len(sub_methods)):
                        methods[sub_ids[i]]["reward"] = reward_list[i]
                        reward_scores.append(reward_list[i])
                        if reward_list[i] >= best_reward:
                            best_reward = reward_list[i]
                            best_trajectory = methods[sub_ids[i]]["method"]
                            output_trajectory = deepcopy(best_trajectory)
                            output_trajectory["location"] = [[] for _ in range(env.string_number)]
                            for i in range(len(best_trajectory['id'])):
                                for j in range(len(best_trajectory['id'][i])):
                                    if best_trajectory['type'][i][j] == 'A':
                                        output_trajectory['location'][i].append(env.P_points[best_trajectory['id'][i][j]])
                                    else:
                                        output_trajectory['location'][i].append(env.O_points[best_trajectory['id'][i][j]])
                            with open(os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}_best.json"), 'w', encoding="utf-8") as f:
                                json.dump(best_trajectory, f, indent=4)

                    print(f"Step: {step}, Best reward: {best_reward}")
                    print(best_trajectory)   
                    
                    with open(os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}_true_reward.json"), 'w', encoding="utf-8") as f:
                        json.dump({
                            "method": methods,
                            "number": total_number,
                            "simulated_number": step
                        }, f, indent=4)    
                    sub_methods.clear()
                    sub_ids.clear()

                    data = [[sim_step[i], tension_scores[i], reward_scores[i]] for i in range(len(sim_step))]
                    test = pd.DataFrame(columns=["step", "tension_score", "true_reward"], data=data)
                    test.to_csv(os.path.join(file_path, f'result_train_time_{time}_candidate_buf_{buf}.csv'))

env = Env()

if __name__ == "__main__":
    for i in range(0, 3):
        train(env, int(i * BUFFER_INIT))
