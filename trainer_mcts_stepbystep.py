import os
import numpy as np
from copy import deepcopy
import random
import math
import matplotlib.pyplot as plt
from units import UnitPackParserReverse
import dxfgrabber
# from phys_sim11 import OrigamiSimulator
from utils import *
import json
import pandas as pd

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

fixed_action_list_initial = [
    [],
    [],
    []
]

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
time = 0
step = 0
maximum_valid = SIM

BUFFER_SIZE = 5000

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

def pad_and_get_mask(lists):
    lens = [len(l) for l in lists]
    max_len = max(lens)
    arr = np.zeros((len(lists), max_len), float)
    mask = np.arange(max_len) < np.array(lens)[:, None]
    arr[mask] = np.concatenate(lists)
    return np.ma.array(arr, mask=~mask)

def moving_average(a, n):
    if len(a) <= n:
        return a
    ret = np.cumsum(a, dtype=float, axis=-1)
    ret[n:] = ret[n:] - ret[:-n]
    return (ret[n - 1:] / n).tolist()

def plot_scores(scores, steps=None, window=100, label=None, fail_reward=0.):
    avg_scores = deepcopy(scores)
    if steps is not None:
        for i in range(len(scores)):
            avg_scores[i] = np.interp(np.arange(steps[i][-1]), steps[i], avg_scores[i])
    if len(scores) > 1:
        avg_scores = pad_and_get_mask(avg_scores)
        scores = avg_scores.mean(axis=0)
        scores_l = avg_scores.mean(axis=0) - avg_scores.std(axis=0)
        scores_h = avg_scores.mean(axis=0) + avg_scores.std(axis=0)
        idx = list(range(len(scores)))
        plt.fill_between(idx, scores_l, scores_h, where=scores_h > scores_l, interpolate=True, alpha=0.25)
    else:
        scores = avg_scores[0]
    plt.plot(scores, label=label)

def visualize(step, title, log_dict):
    train_window, loss_window, q_window = 1, 100, 100
    plt.figure(figsize=(19, 6))

    # plot train and eval returns
    plt.subplot(1, 3, 1)
    plt.title('Frame %s. Reward: %s. Max: %s.' % (step, round(log_dict['reward_returns'][-1][-1], 2), round(log_dict['max_reward_returns'][-1][-1], 2)))
    plot_scores(log_dict['reward_returns'], log_dict['train_steps'], window=1, label='avg_reward', fail_reward=fail_reward)
    plot_scores(log_dict['max_reward_returns'], log_dict['train_steps'], window=1, label='maximum_reward', fail_reward=fail_reward)
    # plt.title('frame %s. train_return: %s' % (step, np.mean(log_dict['train_returns'][-1][-train_window:])))
    # if min([len(log_dict['eval_steps'][i]) for i in range(len(log_dict['eval_steps']))]) > 0:
    #     plot_scores(log_dict['eval_returns'], log_dict['eval_steps'], window=1, label='eval') 
    # plot_scores(log_dict['train_returns'], log_dict['train_steps'], window=100, label='train_reward')
    
    # plot_scores(log_dict['max_reward_returns'], log_dict['train_steps'], window=10, label='max_reward')
    plt.legend(loc='lower right')
    plt.xlabel('step')

    # # plot td losses
    plt.subplot(1, 3, 2)
    plt.title('Cut nodes: %s.' % (log_dict['cut_number'][-1][-1]))
    plot_scores(log_dict['cut_number'], log_dict['train_steps'], window=1, label='cut_node_number')
    # plot_scores(log_dict['losses'], window=loss_window, label='loss')
    plt.xlabel('step')

    plt.subplot(1, 3, 3)
    # # plot q values
    plt.title('Path number: %s / %s.' % (log_dict['valid_number'][-1][-1], log_dict['number'][-1][-1]))
    plot_scores(log_dict['valid_number'], log_dict['train_steps'], window=1, label='negative_diversity_number')
    plot_scores(log_dict['number'], log_dict['train_steps'], window=1, label='total_number')
    # plot_scores(log_dict['Qs'], window=q_window, label='q_values')
    # plt.xlabel('step')
    plt.legend(loc='lower right')
    plt.xlabel('step')

    plt.suptitle(title, fontsize=16)
    plt.savefig(f'{NAME}.png')
    plt.close()

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

        try:
            with open(os.path.join(file_path, "simulation_buffer.json"), 'r', encoding='utf-8') as fw:
                self.existing_id_reward = json.load(fw)["id_reward"]
        except:
            self.existing_id_reward = []

        # calculate max length of view

        self.string_number = string_number

        self.P_candidators = input_json["P_candidators"]["points"]
        self.P_points = [
            np.array(self.P_candidators[i]) for i in range(len(self.P_candidators))
        ]
        self.mid_x = max_x / 2.
        self.mid_y = max_y / 2.

        self.O_points = [
            np.array(self.units[i].getCenter()) for i in range(self.unit_number)
        ]
        
        self.state_number = self.string_number * (self.unit_number + 2)
        self.action_number = len(self.P_points) + self.unit_number

        self.current_state = np.array([0. for _ in range(self.state_number)])

        self.current_stand_point_id = -1.
        self.old_stand_point_id = -1.

        self.current_trajectory = [[] for _ in range(self.string_number)]
        self.current_string_id = 1.
        self.current_valid_pass = 0
        self.temp_done = 1
        # self.done = 0
        self.reward = 0.
        
        self.previous_side = 0

        self.fail_reward = fail_reward
        self.encourage_reward = encourage_reward
        self.avg_reward = [self.fail_reward for _ in range(100)]
        self.scores = []
        self.steps = []

        self.calculateValidActions()

        self.basic_trajectory_list = []
        self.backup_current_state_list = []
        self.backup_node_status_list = []
        self.backup_temp_done_list = []
        self.backup_current_string_id_list = []
        self.backup_current_point_num_list = []
        self.backup_valid_pass_list = []
        self.backup_old_stand_point_id_list = []
        self.backup_current_stand_point_id_list = []
        self.backup_previous_side_list = []
        
        # Create Root
        if not calculate_number:
            self.root_node = Node(0, len(self.P_points), 0, 0)
        else:
            self.root_node = Node(deepcopy(self.current_state), len(self.P_points), 0, 0)
        
        # self.action_number = max(max(self.valid_action_num), len(self.P_points))
    def initialize(self):
        if not calculate_number:
            self.root_node = Node(0, len(self.P_points), 0, 0)
        else:
            self.root_node = Node(deepcopy(self.current_state), len(self.P_points), 0, 0)

    def treePolicy(self, node: Node, scalar):
        global cut_num
        depth = 0
        while not node.done:
            if len(node.children) == 0:
                self.node_num += 1
                # if self.node_num % 1000 == 0:
                #     print(f"Create node with id: {self.node_num}")
                new_node = self.expand(node)
                if new_method:
                    if self.reward < 0:
                        level = [len(self.current_trajectory[i]) for i in range(self.string_number)]
                        new_node.reward = self.reward
                        # print(f"Node with level {level} is cut due to minus reward")
                    else:
                        return new_node
                else:
                    return new_node
                # return self.expand(node)
            # elif random.uniform(0, 1) > 1. and node.existBestChild():
            #     node, action = node.bestChild(scalar, self.best_reward / r_s)
            #     next_state, self.reward, done, truncated = self.step(action, 0, False)
            else:
                if node.done:
                    a = 1
                if not node.fullyExpanded():
                    self.node_num += 1
                    # if self.node_num % 1000 == 0:
                    #     print(f"Create node with id: {self.node_num}")
                    new_node = self.expand(node)
                    if new_method:
                        if self.reward < 0:
                            level = [len(self.current_trajectory[i]) for i in range(self.string_number)]
                            new_node.reward = self.reward
                        else:
                            return new_node
                    else:
                        return new_node
                else:
                    if cut_nodes:
                        if node.existBestChild(self.best_reward):
                            node, action = node.bestChild(scalar, self.best_reward)
                            next_state, self.reward, done, truncated = self.step(action, 0, False)
                            depth += 1
                        else:
                            level = [len(self.current_trajectory[i]) for i in range(self.string_number)]
                            cut_num += node.maximum_child
                            while len(node.children):
                                del(node.children[0])
                            node.children = None
                            # print(f"Node with level {level} is cut due to average reward {node.reward / (node.visits - 1)} < {self.best_reward}, cut_number: {cut_num}")
                            node.done = 1
                            
                            if depth > 0:
                                node = node.parent
                                depth -= 1
                                if node != None:
                                    self.deeppopup()
                                else:
                                    # print("Root node ends!")
                                    return node
                            else:
                                if node.parent != None:
                                    self.deeppopup()
                                return node
                    else:
                        if new_method:
                            if node.existBestChild(0):
                                node, action = node.bestChild(scalar, 0)
                                depth += 1
                            else:                            
                                if depth > 0:
                                    node = node.parent
                                    depth -= 1
                                    if node != None:
                                        self.deeppopup()
                                    else:
                                        # print("Root node ends!")
                                        return node
                                else:
                                    if node.parent != None:
                                        self.deeppopup()
                                    return node
                        else:
                            node, action = node.bestChild(scalar, -math.inf)
                            depth += 1
                        next_state, self.reward, done, truncated = self.step(action, 0, False)
                        
        return node
    
    def expand(self, node: Node):
        tried_children = [c.action_id for c in node.children]
        action_range = node.maximum_child
        notried_children = []
        for i in range(action_range):
            if i not in tried_children:
                notried_children.append(i)
        action = np.random.choice(np.array(notried_children), 1).item()
        next_state, self.reward, done, truncated = self.step(action, 0, False)
        if not calculate_number:
            node.addChild(
                self.current_stand_point_id, 
                self.valid_action_num[self.current_stand_point_id] if not self.temp_done else len(self.P_points),
                done, action
            )
        else:
            node.addChild(
                deepcopy(next_state), 
                self.valid_action_num[self.current_stand_point_id] if not self.temp_done else len(self.P_points),
                done, action
            )
        return node.children[-1]

    def backward(self, node, reward, num_visit):
        while node is not None:
            node.visits += num_visit
            node.reward += reward
            node = node.parent

    def calculateValidActions(self):
        self.valid_matrix = np.zeros((self.action_number, self.action_number), int)
        self.valid_action_num = np.zeros(self.action_number, int)

        self.node_status = np.array([0. for i in range(self.action_number)])
        self.status_number = self.action_number

        for old_stand_point_id in range(self.action_number):
            for current_stand_point_id in range(old_stand_point_id, self.action_number):
                if old_stand_point_id < len(self.P_points) and current_stand_point_id < len(self.P_points):
                    self.valid_matrix[old_stand_point_id][current_stand_point_id] = -10
                    self.valid_matrix[current_stand_point_id][old_stand_point_id] = -10
                else:
                    old_stand_point = self.P_points[old_stand_point_id] if old_stand_point_id < len(self.P_points) else self.O_points[old_stand_point_id - len(self.P_points)]
                    new_stand_point = self.P_points[current_stand_point_id] if current_stand_point_id < len(self.P_points) else self.O_points[current_stand_point_id - len(self.P_points)]
                    intersection_ids = self.calculateIntersectionWithCreases(old_stand_point, new_stand_point, self.lines)
                    valid = 0
                    if len(intersection_ids) > 0:
                        intersection_crease_type = [0, 0, 0]
                        for id in intersection_ids:
                            intersection_crease_type[self.lines[id].getType()] += 1
                        # check valid
                        if intersection_crease_type[BORDER] > 0 and intersection_crease_type[MOUNTAIN] == 0 and intersection_crease_type[VALLEY] == 0:
                            valid = 1
                            side = 0
                        elif intersection_crease_type[BORDER] == 0 and intersection_crease_type[MOUNTAIN] > 0 and intersection_crease_type[VALLEY] == 0:
                            valid = 1
                            side = -1
                        elif intersection_crease_type[BORDER] == 0 and intersection_crease_type[MOUNTAIN] == 0 and intersection_crease_type[VALLEY] > 0:
                            valid = 1
                            side = 1
                    if valid:
                        self.valid_matrix[old_stand_point_id][current_stand_point_id] = side
                        self.valid_matrix[current_stand_point_id][old_stand_point_id] = side
                        self.valid_action_num[old_stand_point_id] += 1
                        self.valid_action_num[current_stand_point_id] += 1
                    else:
                        if old_stand_point_id == current_stand_point_id:
                            self.valid_matrix[old_stand_point_id][current_stand_point_id] = -2
                            self.valid_action_num[old_stand_point_id] += 1
                        else:
                            self.valid_matrix[old_stand_point_id][current_stand_point_id] = -10
                            self.valid_matrix[current_stand_point_id][old_stand_point_id] = -10

    def packTrajectory(self, trajectories):
        length = len(trajectories)
        dict = {
            "type": [[] for _ in range(length)],
            "id": [[] for _ in range(length)],
            "reverse": [[] for _ in range(length)]
        }
        for k in range(length):
            trajectory = trajectories[k]
            for i in range(len(trajectory)):
                if trajectory[i][0] < len(self.P_points):
                    dict["type"][k].append("A")
                    dict["id"][k].append(trajectory[i][0])
                else:
                    dict["type"][k].append("B")
                    dict["id"][k].append(trajectory[i][0] - len(self.P_points))
            exist_non_zero_side = 0
            for i in range(len(trajectory)):
                if trajectory[i][1] != 0:
                    exist_non_zero_side = trajectory[i][1]
                    break
            if exist_non_zero_side:
                for j in range(len(trajectory)):
                    if (j - i) % 2 == 0:
                        dict["reverse"][k].append(int(trajectory[i][1]))
                    else:
                        dict["reverse"][k].append(int(-trajectory[i][1]))
                dict["reverse"][k][0] = int(dict["reverse"][k][1])
            else:
                for j in range(len(trajectory)):
                    if j % 2 == 0:
                        dict["reverse"][k].append(-1)
                    else:
                        dict["reverse"][k].append(1)
                dict["reverse"][k][0] = 1
        return dict
    
    def calculateIntersectionWithCreases(self, P_choice, O_choice, creases):
        ids = []
        for i in range(len(creases)):
            crease_start_point = creases[i][START]
            crease_end_point = creases[i][END]
            vec1_2D = np.array([crease_end_point[X] - crease_start_point[X], crease_end_point[Y] - crease_start_point[Y]])
            vec1_relevant1_2D = np.array([P_choice[X] - crease_start_point[X], P_choice[Y] - crease_start_point[Y]])
            vec1_relevant2_2D = np.array([O_choice[X] - crease_start_point[X], O_choice[Y] - crease_start_point[Y]])
            result1 = np.cross(vec1_2D, vec1_relevant1_2D).item() * np.cross(vec1_2D, vec1_relevant2_2D).item()
            vec2_2D = np.array([O_choice[X] - P_choice[X], O_choice[Y] - P_choice[Y]])
            vec2_relevant1_2D = np.array([crease_start_point[X] - P_choice[X], crease_start_point[Y] - P_choice[Y]])
            vec2_relevant2_2D = np.array([crease_end_point[X] - P_choice[X], crease_end_point[Y] - P_choice[Y]])
            result2 = np.cross(vec2_2D, vec2_relevant1_2D).item() * np.cross(vec2_2D, vec2_relevant2_2D).item()
            if result1 < 0.0 and result2 < 0.0:
                ids.append(i)
        return ids

    def reset(self):
        self.current_state = np.array([0. for _ in range(self.state_number)])
        self.current_stand_point_id = -1
        self.old_stand_point_id = -1.
        self.current_valid_pass = 0
        self.current_trajectory = [[] for _ in range(self.string_number)]
        self.new_trajectory = [[] for _ in range(self.string_number)]
        # self.avg_reward = [self.fail_reward for _ in range(100)]

        self.node_status = np.array([0 for _ in range(self.status_number)])

        self.current_string_id = 1.
        self.current_point_num = 0
        self.previous_side = 0
        self.temp_done = 1
        self.done = 0
        self.reward = 0.

        self.basic_trajectory_list = []
        self.backup_current_state_list = []
        self.backup_node_status_list = []
        self.backup_temp_done_list = []
        self.backup_current_string_id_list = []
        self.backup_current_point_num_list = []
        self.backup_valid_pass_list = []
        self.backup_old_stand_point_id_list = []
        self.backup_current_stand_point_id_list = []
        self.backup_previous_side_list = []

        # self.current_state[0] = 6.
        # self.current_stand_point_id = 5
        # self.old_stand_point_id = 5
        # self.current_trajectory[0].append((5, 0))
        # self.node_status[5] = 1
        # self.current_point_num += 1
        # self.temp_done = 0
        self.deepbackup()
        return np.append(self.current_state, self.current_string_id), 0
    
    def sample(self, precise=False):
        if not precise:
            return np.random.choice(self.action_number, 1).item()
        else:
            if self.temp_done:
                return np.random.choice(len(self.P_points), 1).item()
            else:
                valid_number = self.valid_action_num[self.current_stand_point_id]
                return np.random.choice(valid_number, 1).item()

    def getRange(self):
        if self.temp_done:
            return len(self.P_points)
        else:
            return self.valid_action_num[self.current_stand_point_id]
        
    def equalMethod(self, method1, method2):
        equal = False
        for id1 in method1:
            equal = False
            for id2 in method2:
                if id1 == id2:  
                    equal = True
                    break
            if not equal:
                break
        equal1 = equal
        for id2 in method2:
            equal = False
            for id1 in method1:
                if id1 == id2:  
                    equal = True
                    break
            if not equal:
                break
        return equal and equal1
    
    def existInBuffer(self, method):
        for k in range(len(self.existing_id_reward)):
            if self.equalMethod(method["id"], self.existing_id_reward[k]["id"]):
                return k
        return -1

    def output(self, episode, method, reward):
        if self.output_reward_buffer:
            self.scores.append(reward)
            self.steps.append(episode)
            total_string = deepcopy(method)
            total_string["score"] = reward

            try:
                with open(os.path.join(file_path, "result_step_" + str(episode) + "_score_" + str(round(reward, 2))) + ".json", 'w', encoding="utf-8") as f:
                    json.dump(total_string, f, indent=4)
            except:
                pass
            
            score_list = {
                "score": self.scores,
                "steps": self.steps
            }

            try:
                with open(os.path.join(file_path, "score.json"), 'w', encoding="utf-8") as f:
                    json.dump(score_list, f, indent=4)
            except:
                pass

    # def getSimReward(self, method):
    #     ori_sim = OrigamiSimulator(use_gui=False)

    #     ori_sim.string_total_information = methodToTotalInformation(method, self.P_points, self.O_points)
    #     ori_sim.pref_pack = {
    #         "tsa_resolution": len(self.P_points),
    #         "tsa_radius": self.panel_size
    #     }

    #     ori_sim.startOnlyTSA(self.units, self.max_size, self.total_bias, self.max_edge)
    #     ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
    #     ori_sim.initializeRunning()
        
    #     step = 1
    #     while 1:
    #         ori_sim.step()
    #         if ori_sim.folding_angle_reach_pi[0] or (ori_sim.dead_count >= 1000 and not ori_sim.can_rotate) or (ori_sim.dead_count >= 200 and ori_sim.can_rotate):
    #             break
    #         if step % 225 == 0:
    #             if abs(ori_sim.recorded_folding_percent[-1]) < 1e-2:
    #                 ori_sim.can_rotate = False
    #                 break
    #         step += 1
            
    #     if not ori_sim.can_rotate:
    #         print("Simulation done, fail reward")
    #         reward = self.fail_reward
    #     else:
    #         folding_percent = ori_sim.recorded_folding_percent[-1]
    #         folding_speed = (ori_sim.recorded_folding_percent[-1] - ori_sim.recorded_folding_percent[0]) / (ori_sim.recorded_t[-1] - ori_sim.recorded_t[0])

    #         value = folding_speed / (0.9 - folding_percent)
    #         print(f"Simulation done, reward: {value}")
            
    #         reward = value
    #     return reward
    
    def getReward(self, trajectory, episode, bonus = 1.):
        reward = 0.0
        method = self.packTrajectory(trajectory)
        # print(method)

        index = self.existInBuffer(method)
        if index >= 0:
            # print("existing reward")
            reward = self.existing_id_reward[index]["reward"]
            self.output(episode, method, reward)
            return reward
    
        # reward = self.getSimReward(method)
        reward = bonus_val * sum([len(trajectory[i]) for i in range(len(trajectory))])
        self.output(episode, method, reward)
        
        if self.output_reward_buffer:
            self.existing_id_reward.append({
                "id": deepcopy(method["id"]),
                "reward": reward
            })
            try:
                with open(os.path.join(file_path, "simulation_buffer.json"), 'w', encoding="utf-8") as f:
                    json.dump({
                        "id_reward": self.existing_id_reward
                    }, f, indent=4)
            except:
                pass

        return reward

    def getStateIndex(self):
        return self.current_point_num + int(self.current_string_id - 1) * (self.unit_number + 2)
    
    def beginRouting(self):
        self.temp_done = 0
        self.current_valid_pass = 0

    def updateStateAndTrajectory(self, action, dir, origin_action):
        self.current_stand_point_id = action
        self.current_state[self.getStateIndex()] = action + 1

        if dir != -2:
            self.current_point_num += 1
            self.node_status[action] += 1
            self.current_trajectory[(int(self.current_string_id) - 1)].append((action, dir))
        else:
            self.node_status[action] = -1
        
    def getActionIndex(self, action_id):
        if self.current_stand_point_id == -1:
            return action_id, 0
        pointer = -1
        valid_vector = self.valid_matrix[self.current_stand_point_id]
        for i in range(self.status_number):
            if valid_vector[i] != -10:
                pointer += 1
                if pointer == action_id:
                    return i, valid_vector[i]
        return -1, -10

    def endString(self):
        self.temp_done = 1
        self.current_string_id += 1.
        self.current_point_num = 0

    def backup(self):
        self.basic_trajectory = deepcopy(self.current_trajectory)
        self.backup_current_state = deepcopy(self.current_state)
        self.backup_node_status = deepcopy(self.node_status)
        self.backup_temp_done = self.temp_done
        self.backup_current_string_id = self.current_string_id
        self.backup_current_point_num = self.current_point_num
        self.backup_valid_pass = self.current_valid_pass
        self.backup_old_stand_point_id = self.old_stand_point_id
        self.backup_current_stand_point_id = self.current_stand_point_id
        self.backup_previous_side = self.previous_side
    
    def deepbackup(self):
        self.basic_trajectory_list.append(deepcopy(self.current_trajectory))
        self.backup_current_state_list.append(deepcopy(self.current_state))
        self.backup_node_status_list.append(deepcopy(self.node_status))
        self.backup_temp_done_list.append(self.temp_done)
        self.backup_current_string_id_list.append(self.current_string_id)
        self.backup_current_point_num_list.append(self.current_point_num)
        self.backup_valid_pass_list.append(self.current_valid_pass)
        self.backup_old_stand_point_id_list.append(self.old_stand_point_id)
        self.backup_current_stand_point_id_list.append(self.current_stand_point_id)
        self.backup_previous_side_list.append(self.previous_side)
    
    def deeppopup(self):
        del(self.basic_trajectory_list[-1])
        del(self.backup_current_state_list[-1])
        del(self.backup_node_status_list[-1])
        del(self.backup_temp_done_list[-1])
        del(self.backup_valid_pass_list[-1])
        del(self.backup_current_string_id_list[-1])
        del(self.backup_current_point_num_list[-1])
        del(self.backup_old_stand_point_id_list[-1])
        del(self.backup_current_stand_point_id_list[-1])
        del(self.backup_previous_side_list[-1])
        self.current_trajectory = deepcopy(self.basic_trajectory_list[-1])
        self.current_state = deepcopy(self.backup_current_state_list[-1])
        self.node_status = deepcopy(self.backup_node_status_list[-1])
        self.temp_done = self.backup_temp_done_list[-1]
        self.current_valid_pass = self.backup_valid_pass_list[-1]
        self.current_string_id = self.backup_current_string_id_list[-1]
        self.current_point_num = self.backup_current_point_num_list[-1]
        self.old_stand_point_id = self.backup_old_stand_point_id_list[-1]
        self.current_stand_point_id = self.backup_current_stand_point_id_list[-1]
        self.previous_side = self.backup_previous_side_list[-1]


    
    def takeFixedActions(self, fixed_action_list):
        while 1:
            old_string_id = self.current_string_id
            for action_id in fixed_action_list[int(self.current_string_id)]:
                _ = self.step(action_id, 0, False)
            if self.current_string_id <= old_string_id:
                break

    def popup(self):
        self.current_trajectory = deepcopy(self.basic_trajectory)
        self.current_state = deepcopy(self.backup_current_state)
        self.node_status = deepcopy(self.backup_node_status)
        self.temp_done = self.backup_temp_done
        self.current_valid_pass = self.backup_valid_pass
        self.current_string_id = self.backup_current_string_id
        self.current_point_num = self.backup_current_point_num
        self.old_stand_point_id = self.backup_old_stand_point_id
        self.current_stand_point_id = self.backup_current_stand_point_id
        self.previous_side = self.backup_previous_side

    def rollout(self, number, front_node):
        self.backup()

        methods = []
        invalid_list = []

        

        for _ in range(number):
            self.fake_step = 0
            need_to_check_front_node = True
            while 1:
                if not self.fake_step:
                    if self.temp_done:
                        choose_time = np.zeros(len(self.P_points))
                    else:
                        choose_time = np.zeros(self.valid_action_num[self.current_stand_point_id])
                action = self.sample(True)

                # if need_to_check_front_node:
                #     exist, child = front_node.existChildWithAction(action)
                #     if exist:
                #         node_done = child.done
                #     else:
                #         need_to_check_front_node = False

                _, reward, done, _ = self.step(action, 0, False)

                # if need_to_check_front_node and node_done != done: 
                #     self.deeppopup()
                #     self.fake_step = 1
                #     choose_time[action] = self.fail_reward
                # else:
                #     choose_time[action] = reward
                #     if need_to_check_front_node:
                #         front_node = child

                if (done and reward > 0) or (choose_time < 0).all():
                    if (choose_time < 0).all():
                        invalid_list.append(0)
                    else:
                        invalid_list.append(1)
                    break

            methods.append(deepcopy(self.current_trajectory))

            self.popup()
        
        return methods, invalid_list

    def getStringNumber(self, trajectories):
        string_number_correct = [0. for _ in range(len(trajectories))]
        for h in range(len(trajectories)):
            strings = trajectories[h]
            string_number_correct_temp = 1.
            for trajectory in strings:
                if len(trajectory) == 0:
                    string_number_correct_temp = 0.
                    break
            string_number_correct[h] = string_number_correct_temp
        return string_number_correct

    def getDiversion(self, trajectories):
        div = [0. for _ in range(len(trajectories))]
        for h in range(len(trajectories)):
            strings = trajectories[h]
            # external_div = []
            for trajectory in strings:
                string_length = len(trajectory)
                div_one_string = 0.0
                for i in range(string_length - 1):
                    current_point_id = trajectory[i][0]
                    next_point_id = trajectory[i + 1][0]
                    current_point = self.P_points[current_point_id] if current_point_id < len(self.P_points) else self.O_points[current_point_id - len(self.P_points)]
                    next_point = self.P_points[next_point_id] if next_point_id < len(self.P_points) else self.O_points[next_point_id - len(self.P_points)]
                    if i != 0:
                        v1 = np.array([current_point[X] - self.mid_x, current_point[Y] - self.mid_y])
                        v2 = np.array([next_point[X] - current_point[X], next_point[Y] - current_point[Y]])
                        div_one_string += v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
                        # if i == 1 or (i == string_length - 2 and next_point_id <= len(self.P_points)):
                        #     external_div.append([div_one_string[X], div_one_string[Y]])
                        # div[h] += 1 if v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2) >= 0 else -1
                    if i != string_length - 2 or (i == string_length - 2 and next_point_id > len(self.P_points)):
                        v1 = np.array([next_point[X] - self.mid_x, next_point[Y] - self.mid_y])
                        v2 = -np.array([next_point[X] - current_point[X], next_point[Y] - current_point[Y]])
                        div_one_string += v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
                        # if i == 0 or (i == string_length - 2 and next_point_id > len(self.P_points)):
                        #     external_div.append([div_one_string[X], div_one_string[Y]])

                        # div[h] += 1 if v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2) >= 0 else -1
                if div_one_string > 0:
                    div[h] = div_one_string
                    break
                else:
                    div[h] += div_one_string
        return div

    def getReplicatedPoint(self, trajectories):
        ratio = [0. for _ in range(len(trajectories))]
        go_over_all_units = [0. for _ in range(len(trajectories))]
        for h in range(len(trajectories)):
            strings = trajectories[h]
            duplicated_number = [0 for _ in range(len(self.P_points) + self.unit_number)]
            total_point_number = 0
            for trajectory in strings:
                string_length = len(trajectory)
                total_point_number += string_length
                if trajectory[0][0] < len(self.P_points):
                    total_point_number -= 1
                if trajectory[-1][0] < len(self.P_points):
                    total_point_number -= 1
                for i in range(string_length):
                    duplicated_number[trajectory[i][0]] += 1
            coef = sum([1 if duplicated_number[i] > 0 else 0 for i in range(len(self.P_points), len(duplicated_number))])
            go_over_all_units[h] = coef / (self.unit_number)
            x = coef / total_point_number
            if self.string_number == 1:
                ratio[h] = 1
            else:
                ratio[h] = self.string_number / (self.string_number - 1.) * (x - 1. / self.string_number)
        return ratio, go_over_all_units

    def getAList(self, trajectories):
        As = [0 for _ in range(len(trajectories))]
        for h in range(len(trajectories)):
            strings = trajectories[h]
            duplicated_number = [0 for _ in range(len(self.P_points))]
            for k in range(len(strings)):
                trajectory = strings[k]
                if len(trajectory) > 0:
                    if trajectory[0][0] < len(self.P_points):
                        duplicated_number[trajectory[0][0]] = 1
                    if trajectory[-1][0] < len(self.P_points):
                        duplicated_number[trajectory[-1][0]] = 1
            As[h] = sum(duplicated_number)
        return [1. if As[h] >= 3 else 0. for h in range(len(trajectories))]

    def getEqualLength(self, trajectories):
        ratio = [0. for _ in range(len(trajectories))]
        for h in range(len(trajectories)):
            strings = trajectories[h]
            string_length = np.array([len(trajectory) for trajectory in strings])
            std = string_length.std()
            ratio[h] = 1 / (1 + std)
        return ratio

    def getRewardList(self, methods, invalid_list, step):
        # reward_list = []
        # for method in methods:
        #     reward_list.append(self.getReward(method, step))
        string_number_correct_list = self.getStringNumber(methods)
        true_reward_list = np.zeros_like(string_number_correct_list)
        for i in range(len(methods)):
            if not string_number_correct_list[i]:
                true_reward_list[i] = fail_reward
            else:
                reward_list = (-np.array(self.getDiversion(methods)))
                ratio_list, go_over_all_units_list = np.array(self.getReplicatedPoint(methods))
                if self.string_number > 1:
                    As_list = np.array(self.getAList(methods))
                else:
                    As_list = np.array([1])
                np_invalid_list = np.array(invalid_list)
                equal_length_list = np.array(self.getEqualLength(methods))
                
                true_reward_list = np.zeros_like(reward_list)
                for i in range(len(methods)):
                    if reward_list[i] < 0:
                        true_reward_list[i] = fail_reward
                    elif np_invalid_list[i] < 1.:
                        true_reward_list[i] = fail_reward
                    else:
                        true_reward_list[i] = (reward_list[i] * ratio_list[i] + go_over_all_units_list[i] * self.unit_number + equal_length_list[i]) * As_list[i]
                # true_reward_list = (reward_list * ratio_list * As_list * np_invalid_list).tolist()
                # true_reward_list = (reward_list).tolist()
        return true_reward_list
    
    def getTrajectoryLength(self):
        return sum([len(self.current_trajectory[i]) for i in range(self.string_number)])
    
    def step(self, action, step, rollout=True):
        self.fake_step = 0
        if self.temp_done:
            # choose P is correct
            if action < len(self.P_points):
                self.beginRouting()
                self.updateStateAndTrajectory(action, 0, action)
                if rollout:
                    methods = self.rollout(8)
                    reward_list = self.getRewardList(methods, step)
                    reward = sum(reward_list) / 8.
                else:
                    reward = self.getTrajectoryLength()
                if not rollout:
                    self.deepbackup()
                return np.append(self.current_state, self.current_string_id), reward, 0, 0
            else:
                self.fake_step = 1
                next_state = np.append(self.current_state, self.current_string_id)
                next_state[self.getStateIndex()] = action + 1
                return next_state, self.fail_reward, 1, 0
        else:
            self.old_stand_point_id = self.current_stand_point_id
            true_action_index, side = self.getActionIndex(action)

            # if self.previous_side != 0:
            #     has_solution = False
            #     for ele in self.valid_matrix[self.old_stand_point_id]:
            #         if ele == -self.previous_side:
            #             has_solution = True
            #             break
            #     if has_solution and side != ele:
            #         self.fake_step = 1
            #         next_state = np.append(self.current_state, self.current_string_id)
            #         next_state[self.getStateIndex()] = action + 1
            #         return next_state, self.fail_reward, 1, 0
            id_start = int((self.current_string_id - 1) * (self.unit_number + 2))
            id_end = int(self.current_string_id * (self.unit_number + 2))
            for i in range(id_start, id_end):
                # same id
                id = self.current_state[i]
                if id == 0.0:
                    break
                if id > len(self.P_points) and int(id) == true_action_index + 1 and self.current_state[i + 1] != 0.0:
                    self.fake_step = 1
                    next_state = np.append(self.current_state, self.current_string_id)
                    next_state[self.getStateIndex()] = true_action_index + 1
                    return next_state, self.fail_reward, 1, 0
            if true_action_index == -1:
                self.fake_step = 1
                next_state = np.append(self.current_state, self.current_string_id)
                next_state[self.getStateIndex()] = true_action_index + 1
                return next_state, self.fail_reward, 1, 0
            # choose P or O is correct, but needs validation
            
            # not pass the fixed point, otherwise fixed
            if self.node_status[true_action_index] == -1.:
                # can't be the same side
                if side + self.previous_side != 0 and self.current_valid_pass:
                    self.fake_step = 1
                    next_state = np.append(self.current_state, self.current_string_id)
                    next_state[self.getStateIndex()] = true_action_index + 1
                    return next_state, self.fail_reward, 1, 0
                # return np.append(self.current_state, self.current_string_id), self.fail_reward * (sum([len(trajectory) for trajectory in self.current_trajectory]) + 1), 1, 0
                self.updateStateAndTrajectory(true_action_index, side, action)
                # end one string
                if self.current_string_id < self.string_number: 
                    self.endString()
                    if rollout:
                        methods = self.rollout(8)
                        reward_list = self.getRewardList(methods, step)
                        reward = sum(reward_list) / 8.
                    else:
                        reward = self.getTrajectoryLength()
                    if not rollout:
                        self.deepbackup()
                    self.takeFixedActions(fixed_action_list_initial)
                    return np.append(self.current_state, self.current_string_id), reward, 0, 0
                else:
                    self.done = 1
                    # simulator here
                    if direct_reward:
                        self.reward = self.getReward(self.current_trajectory, step)
                    else:
                        self.reward = 1.
                    self.new_trajectory = deepcopy(self.current_trajectory)
                    if not rollout:
                        self.deepbackup()
                    return np.append(self.current_state, self.current_string_id), self.reward, 1, 0
            
            # 2 same Os, fixed
            elif self.old_stand_point_id == true_action_index:
                # not fixed the pass point
                passed = False
                for string in self.current_trajectory:
                    for index in range(len(string)):
                        if string[index][0] == true_action_index and index != len(string) - 1:
                            passed = True
                            break
                    if passed:
                        break
                if passed:
                    self.fake_step = 1
                    next_state = np.append(self.current_state, self.current_string_id)
                    next_state[self.getStateIndex()] = true_action_index + 1
                    return next_state, self.fail_reward, 1, 0
                self.updateStateAndTrajectory(true_action_index, side, action)
                # end one string
                if self.current_string_id < self.string_number: 
                    self.endString()
                    if rollout:
                        methods = self.rollout(8)
                        reward_list = self.getRewardList(methods, step)
                        reward = sum(reward_list) / 8.
                    else:
                        reward = self.getTrajectoryLength()
                    if not rollout:
                        self.deepbackup()
                    self.takeFixedActions(fixed_action_list_initial)
                    return np.append(self.current_state, self.current_string_id), reward, 0, 0
                else:
                    self.done = 1
                    # simulator here
                    if direct_reward:
                        self.reward = self.getReward(self.current_trajectory, step)
                    else:
                        self.reward = 1.
                    self.new_trajectory = deepcopy(self.current_trajectory)
                    if not rollout:
                        self.deepbackup()
                    return np.append(self.current_state, self.current_string_id), self.reward, 1, 0
            else:
                if self.current_valid_pass:
                    if side == 0:
                        # done
                        self.updateStateAndTrajectory(true_action_index, side, action)
                        if self.current_string_id < self.string_number:
                            self.endString()
                            if rollout:
                                methods = self.rollout(8)
                                reward_list = self.getRewardList(methods, step)
                                reward = sum(reward_list) / 8.
                            else:
                                reward = self.getTrajectoryLength()
                            if not rollout:
                                self.deepbackup()
                            self.takeFixedActions(fixed_action_list_initial)
                            return np.append(self.current_state, self.current_string_id), reward, 0, 0
                        else:
                            self.done = 1
                            # simulator here
                            if direct_reward:
                                self.reward = self.getReward(self.current_trajectory, step)
                            else:
                                self.reward = 1.
                            self.new_trajectory = deepcopy(self.current_trajectory)
                            if not rollout:
                                self.deepbackup()
                            return np.append(self.current_state, self.current_string_id), self.reward, 1, 0   
                    elif side + self.previous_side != 0:
                        self.fake_step = 1
                        next_state = np.append(self.current_state, self.current_string_id)
                        next_state[self.getStateIndex()] = true_action_index + 1
                        return next_state, self.fail_reward, 1, 0
                    else:
                        self.previous_side = side
                        self.updateStateAndTrajectory(true_action_index, side, action)
                        if rollout:
                            methods = self.rollout(8)
                            reward_list = self.getRewardList(methods, step)
                            reward = sum(reward_list) / 8.
                        else:
                            reward = self.getTrajectoryLength()
                        if not rollout:
                            self.deepbackup()
                        return np.append(self.current_state, self.current_string_id), reward, 0, 0
                else:
                    if side == 0:
                        # no valid pass
                        if true_action_index < len(self.P_points):
                            self.fake_step = 1
                            next_state = np.append(self.current_state, self.current_string_id)
                            next_state[self.getStateIndex()] = true_action_index + 1
                            return next_state, self.fail_reward, 1, 0
                        else:
                            self.updateStateAndTrajectory(true_action_index, 0, action)
                            if rollout:
                                methods = self.rollout(8)
                                reward_list = self.getRewardList(methods, step)
                                reward = sum(reward_list) / 8.
                            else:
                                reward = self.getTrajectoryLength()
                            if not rollout:
                                self.deepbackup()
                            return np.append(self.current_state, self.current_string_id), reward, 0, 0
                    else:
                        self.updateStateAndTrajectory(true_action_index, side, action)
                        self.current_valid_pass = 1
                        self.previous_side = side
                        if rollout:
                            methods = self.rollout(8)
                            reward_list = self.getRewardList(methods, step)
                            reward = sum(reward_list) / 8.
                        else:
                            reward = self.getTrajectoryLength()
                        if not rollout:
                            self.deepbackup()
                        return np.append(self.current_state, self.current_string_id), reward, 0, 0

def eval(env, agent, episode):
    state, _ = env.reset()
    done, truncated = False, False

    while not (done or truncated):
        state, reward, done, truncated = env.step(agent.get_action(state).item(), episode)
   
    if reward > 0:
        return env.getReward(env.new_trajectory, episode)
    else:
        return reward

def get_epsilon(step, eps_min, eps_max, eps_steps, warmup_steps):
    if step < warmup_steps:
        return eps_max
    elif step > eps_steps:
        return eps_min
    else:
        return eps_max - (eps_max - eps_min) / (eps_steps - warmup_steps) * (step - warmup_steps)
    
def outputBestTrajectory(env, best_trajectory, best_reward):
    try:
        env.output(step, best_trajectory, best_reward)
        print(best_trajectory)
    except:
        pass
    try:
        sorted_method = sorted(candidate_methods, key=lambda x: x['score'], reverse=True)
        with open(os.path.join(file_path, "result_train_time_" + str(time) + "_candidate.json"), 'w', encoding="utf-8") as f:
            json.dump({
                "method": sorted_method,
                "number": len(sorted_method)
            }, f, indent=4)
        print([f"{len(sorted_method)} methods become candidators"])
    except:
        pass

def setSeeds(seed):
    random.seed(seed + time)
    np.random.seed(seed + time)
    os.environ["PYTHONHASHSEED"] = str(seed + time) 

def setup():
    global cut_num, roller
    cut_num = 0
    roller = 0
    

def train(env: Env):
    global cut_num, roller, step, candidate_methods, time
    
    log_dict = {
        "train_steps": [],
        "train_returns": [],
        "reward_returns": [],
        "max_reward_returns": [],
        "cut_number": [],
        "valid_number": [],
        "number": []
    }
    
    while time < 1:
        time += 1
        env.initialize()
        setSeeds(seed + time)
        setup()

        log_dict["train_steps"].append([0])
        log_dict["train_returns"].append([fail_reward])
        log_dict["reward_returns"].append([fail_reward])
        log_dict["max_reward_returns"].append([fail_reward])
        log_dict["cut_number"].append([0])
        log_dict["valid_number"].append([0])
        log_dict["number"].append([0])

        best_reward = env.fail_reward
        best_trajectory = None

        bonus = 1.
        valid_method_num = 0
        method_num = 0
        step = 0
        epsilon_bonus = 1.
        dead_count = 0

        total_pointers = []

        if calculate_number:
            while 1:
                step += 1
                env.reset()
                current_node = env.root_node
            
                eps = 1.

                front_node = env.treePolicy(current_node, eps * bonus)
                level = [len(env.current_trajectory[i]) for i in range(env.string_number)]

                if front_node == None or (current_node.parent == None and current_node.children == None):
                    log_dict["train_steps"][-1].append(step)
                    log_dict["train_returns"][-1].append(reward)
                    log_dict["reward_returns"][-1].append(sum(env.avg_reward) / 100.)
                    log_dict["max_reward_returns"][-1].append(best_reward)
                    log_dict["cut_number"][-1].append(cut_num)
                    log_dict["valid_number"][-1].append(valid_method_num)
                    log_dict["number"][-1].append(method_num)
                    print(f"Root node has done, total step: {step}, cut number: {cut_num}, valid number: {valid_method_num} / {method_num}")
                    try:
                        visualize(step, "String-routing path training", log_dict)
                        data = [[log_dict['train_steps'][-1][i], log_dict['reward_returns'][-1][i], log_dict['max_reward_returns'][-1][i], log_dict["cut_number"][-1][i], log_dict["valid_number"][-1][i]] for i in range(len(log_dict['train_steps'][-1]))]
                        test = pd.DataFrame(columns=["step", "avg_reward", "maximum_reward", "cut", "valid"], data=data)
                        test.to_csv(f'{NAME}.csv')
                    except:
                        pass
                    try:
                        env.output(step, best_trajectory, best_reward)
                        print(best_trajectory)
                    except:
                        pass
                    return log_dict

                if not front_node.done:
                    methods, invalid_list = env.rollout(maximum_valid, front_node)
                    reward_list = env.getRewardList(methods, invalid_list, step)
                    total_reward = sum(reward_list)
                    reward = total_reward / maximum_valid

                    for i in range(maximum_valid):
                        env.avg_reward[roller % 100] = reward_list[i]
                        roller += 1
                        if reward_list[i] > best_reward:
                            best_reward = reward_list[i]
                            best_trajectory = env.packTrajectory(methods[i])

                    env.backward(front_node, reward, 1)

                else:
                    if env.reward < 0:
                        reward = env.reward
                    else:
                        
                        if front_node.visits != 1:
                            reward = front_node.reward / (front_node.visits - 1)
                        else:
                            method_num += 1
                            reward_list = env.getRewardList([deepcopy(env.current_trajectory)], [1], step)
                            reward = reward_list[0]
                            env.avg_reward[roller % 100] = reward
                            roller += 1
                            if reward > best_reward:
                                best_reward = reward
                                best_trajectory = env.packTrajectory(env.new_trajectory)
                            if reward > 0:
                                valid_method_num += 1
            
                    env.backward(front_node, reward, 1)

                if step % record_episode == 0:
                    log_dict["train_steps"][-1].append(step)
                    log_dict["train_returns"][-1].append(reward)
                    log_dict["reward_returns"][-1].append(sum(env.avg_reward) / 100.)
                    log_dict["max_reward_returns"][-1].append(best_reward)
                    log_dict["cut_number"][-1].append(cut_num)
                    log_dict["valid_number"][-1].append(valid_method_num)
                    log_dict["number"][-1].append(method_num)

                env.best_reward = best_reward
                bonus = best_reward if best_reward > 1. else 1.

                if step % record_episode == 0:
                    print(f"Step: {step}, Best reward: {round(best_reward, 2)}, Bonus/epsilon: {round(bonus, 2)}/{round(eps, 2)}, Level: {level}, Valid method number: {valid_method_num} / {method_num}")
                    print(f"Best: {best_trajectory}")
                    try:
                        visualize(step, "String-routing Path Training - Negative Diversion Training", log_dict)
                        data = [[log_dict['train_steps'][-1][i], log_dict['reward_returns'][-1][i], log_dict['max_reward_returns'][-1][i], log_dict["cut_number"][-1][i], log_dict["valid_number"][-1][i]] for i in range(len(log_dict['train_steps'][-1]))]
                        test = pd.DataFrame(columns=["step", "avg_reward", "maximum_reward", "cut", "valid"], data=data)
                        test.to_csv(f'{NAME}.csv')
                    except:
                        pass
        else:
            for trial in range(1000):
                fixed_action_list = deepcopy(fixed_action_list_initial)
                if trial >= 1:
                    temp_pointer = env.root_node
                    total_pointers.append(temp_pointer)
                while env.root_node.parent != None:
                    env.root_node = env.root_node.parent
                if initialize:
                    env.root_node.initializeTree()
                for pointer in total_pointers:
                    pointer.reward = fail_reward
                
                previous_best = best_reward

                if dead_count >= 10:
                    print("No improvement, early stop.")
                    return log_dict
                
                while 1:
                    step += 1
                    env.reset()
                    env.takeFixedActions(fixed_action_list)
                    current_string_id = int(env.current_string_id)
                    current_node = env.root_node

                    if current_node.parent == None and (current_node.children == None or (current_node.fullyExpanded() and not current_node.existBestChild(env.best_reward))):
                        log_dict["train_steps"][-1].append(step)
                        log_dict["train_returns"][-1].append(reward)
                        log_dict["reward_returns"][-1].append(sum(env.avg_reward) / 100.)
                        log_dict["max_reward_returns"][-1].append(best_reward)
                        log_dict["cut_number"][-1].append(cut_num)
                        log_dict["valid_number"][-1].append(valid_method_num)
                        log_dict["number"][-1].append(method_num)
                        print(f"Root node has done, total step: {step}, cut number: {cut_num}, valid number: {valid_method_num} / {method_num}")
                        try:
                            visualize(step, "String-routing path training", log_dict)
                            data = [[log_dict['train_steps'][-1][i], log_dict['reward_returns'][-1][i], log_dict['max_reward_returns'][-1][i], log_dict["cut_number"][-1][i], log_dict["valid_number"][-1][i]] for i in range(len(log_dict['train_steps'][-1]))]
                            test = pd.DataFrame(columns=["step", "avg_reward", "maximum_reward", "cut", "valid"], data=data)
                            test.to_csv(f'{NAME}.csv')
                        except:
                            pass
                        try:
                            env.output(step, best_trajectory, best_reward)
                            print(best_trajectory)
                        except:
                            pass
                        try:
                            sorted_method = sorted(candidate_methods, key=lambda x: x['score'], reverse=True)
                            buf = 0
                            while buf < len(sorted_method):
                                with open(os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}.json"), 'w', encoding="utf-8") as f:
                                    json.dump({
                                        "method": sorted_method[buf: buf + BUFFER_SIZE],
                                        "number": BUFFER_SIZE
                                    }, f, indent=4)
                                buf += BUFFER_SIZE
                            print([f"{len(sorted_method)} methods become candidators"])
                        except:
                            pass
                        return log_dict
                    
                    if current_node.done or (current_node.fullyExpanded() and not current_node.existBestChild(env.best_reward)):
                        print(f"Trial {trial} has done, Total step: {step}, Cut number: {cut_num}, Valid number: {valid_method_num} / {method_num}")
                        step -= 1

                        if best_reward <= previous_best:
                            epsilon_bonus *= 1.25
                            dead_count += 1
                        else:
                            epsilon_bonus = 1.
                            dead_count = 0
                        try:
                            visualize(step, "String-routing path training", log_dict)
                            data = [[log_dict['train_steps'][-1][i], log_dict['reward_returns'][-1][i], log_dict['max_reward_returns'][-1][i], log_dict["cut_number"][-1][i], log_dict["valid_number"][-1][i]] for i in range(len(log_dict['train_steps'][-1]))]
                            test = pd.DataFrame(columns=["step", "avg_reward", "maximum_reward", "cut", "valid"], data=data)
                            test.to_csv(f'{NAME}.csv')
                        except:
                            pass
                        try:
                            env.output(step, best_trajectory, best_reward)
                            print(best_trajectory)
                        except:
                            pass
                        try:
                            sorted_method = sorted(candidate_methods, key=lambda x: x['score'], reverse=True)
                            buf = 0
                            while buf < len(sorted_method):
                                with open(os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}.json"), 'w', encoding="utf-8") as f:
                                    json.dump({
                                        "method": sorted_method[buf: buf + BUFFER_SIZE],
                                        "number": BUFFER_SIZE
                                    }, f, indent=4)
                                buf += BUFFER_SIZE
                            print([f"{len(sorted_method)} methods become candidators"])
                        except:
                            pass
                        break
                    
                    # eps = (1. - len(fixed_action_list[current_string_id]) / (env.state_number)) * epsilon_bonus * (1. - ((step - 1) % episodes) / episodes)
                    # print(eps)
                    eps = epsilon_bonus * (1. - ((step - 1) % (episodes - 1)) / (episodes - 1))

                    front_node = env.treePolicy(current_node, eps * bonus)
                    level = [len(env.current_trajectory[i]) for i in range(env.string_number)]

                    if front_node.parent == None and (front_node.children == None or (front_node.fullyExpanded() and not front_node.existBestChild(env.best_reward))):
                        log_dict["train_steps"][-1].append(step)
                        log_dict["train_returns"][-1].append(reward)
                        log_dict["reward_returns"][-1].append(sum(env.avg_reward) / 100.)
                        log_dict["max_reward_returns"][-1].append(best_reward)
                        log_dict["cut_number"][-1].append(cut_num)
                        log_dict["valid_number"][-1].append(valid_method_num)
                        log_dict["number"][-1].append(method_num)
                        print(f"Root node has done, total step: {step}, cut number: {cut_num}, valid number: {valid_method_num} / {method_num}")
                        try:
                            visualize(step, "String-routing path training", log_dict)
                            data = [[log_dict['train_steps'][-1][i], log_dict['reward_returns'][-1][i], log_dict['max_reward_returns'][-1][i], log_dict["cut_number"][-1][i], log_dict["valid_number"][-1][i]] for i in range(len(log_dict['train_steps'][-1]))]
                            test = pd.DataFrame(columns=["step", "avg_reward", "maximum_reward", "cut", "valid"], data=data)
                            test.to_csv(f'{NAME}.csv')
                        except:
                            pass
                        try:
                            env.output(step, best_trajectory, best_reward)
                            print(best_trajectory)
                        except:
                            pass
                        try:
                            sorted_method = sorted(candidate_methods, key=lambda x: x['score'], reverse=True)
                            buf = 0
                            while buf < len(sorted_method):
                                with open(os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}.json"), 'w', encoding="utf-8") as f:
                                    json.dump({
                                        "method": sorted_method[buf: buf + BUFFER_SIZE],
                                        "number": BUFFER_SIZE
                                    }, f, indent=4)
                                buf += BUFFER_SIZE
                            print([f"{len(sorted_method)} methods become candidators"])
                        except:
                            pass
                        return log_dict
                    
                    if front_node.children == None or (front_node.fullyExpanded() and not front_node.existBestChild(env.best_reward)):
                        print(f"Trial {trial} has done, Total step: {step}, Cut number: {cut_num}, Valid number: {valid_method_num} / {method_num}")
                        step -= 1

                        if best_reward <= previous_best:
                            epsilon_bonus *= 1.25
                            dead_count += 1
                        else:
                            epsilon_bonus = 1.
                            dead_count = 0
                        try:
                            visualize(step, "String-routing path training", log_dict)
                            data = [[log_dict['train_steps'][-1][i], log_dict['reward_returns'][-1][i], log_dict['max_reward_returns'][-1][i], log_dict["cut_number"][-1][i], log_dict["valid_number"][-1][i]] for i in range(len(log_dict['train_steps'][-1]))]
                            test = pd.DataFrame(columns=["step", "avg_reward", "maximum_reward", "cut", "valid"], data=data)
                            test.to_csv(f'{NAME}.csv')
                        except:
                            pass
                        try:
                            env.output(step, best_trajectory, best_reward)
                            print(best_trajectory)
                        except:
                            pass
                        try:
                            sorted_method = sorted(candidate_methods, key=lambda x: x['score'], reverse=True)
                            buf = 0
                            while buf < len(sorted_method):
                                with open(os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}.json"), 'w', encoding="utf-8") as f:
                                    json.dump({
                                        "method": sorted_method[buf: buf + BUFFER_SIZE],
                                        "number": BUFFER_SIZE
                                    }, f, indent=4)
                                buf += BUFFER_SIZE
                            print([f"{len(sorted_method)} methods become candidators"])
                        except:
                            pass
                        break
                
                    # if current_node.parent != None and current_node.done:
                    #     print(f"Trial {trial} has done, total step: {step}, cut number: {cut_num}, valid number: {valid_method_num}")
                    #     step -= 1
                    #     # visualize(step, "String-routing path training", log_dict)
                    #     try:
                    #         env.output(step, best_trajectory, best_reward)
                    #         print(best_trajectory)
                    #     except:
                    #         pass
                    #     try:
                    #         sorted_method = sorted(candidate_methods, key=lambda x: x['score'], reverse=True)
                    #         with open(os.path.join(file_path, "result_train_time_" + str(time) + "_candidate.json"), 'w', encoding="utf-8") as f:
                    #             json.dump({
                    #                 "method": sorted_method,
                    #                 "number": len(sorted_method)
                    #             }, f, indent=4)
                    #         print([f"{len(sorted_method)} methods become candidators"])
                    #     except:
                    #         pass
                    #     break

                    if not front_node.done:
                        methods, invalid_list = env.rollout(maximum_valid, front_node)
                        reward_list = env.getRewardList(methods, invalid_list, step)
                        total_reward = sum(reward_list)
                        for i in range(maximum_valid):
                            env.avg_reward[roller % 100] = reward_list[i]
                            roller += 1
                            reward = total_reward / maximum_valid
                            if reward_list[i] > best_reward:
                                best_reward = reward_list[i]
                                best_trajectory = env.packTrajectory(methods[i])
                                try:
                                    env.output(step, best_trajectory, best_reward)
                                except:
                                    pass
                            if reward_list[i] > 0:
                                equal = False
                                current_method = env.packTrajectory(methods[i])
                                for k in range(len(candidate_methods)):
                                    # if env.equalMethod(current_method["id"], candidate_methods[k]["method"]["id"]):
                                    if abs(reward_list[i] - candidate_methods[k]["score"]) < 1e-5:
                                        # if reward_list[i] != candidate_methods[k]["score"]:
                                        #     a = 1
                                        equal = True
                                        break
                                if not equal:
                                    # valid_method_num += 1
                                    candidate_methods.append({
                                        "method": current_method,
                                        "score": reward_list[i] 
                                    })
                                    # if len(candidate_methods) >= 3000:
                                    #     candidate_methods = sorted(candidate_methods[: 3000], key=lambda x: x['score'], reverse=True)
                                    #     del(candidate_methods[-1])

                        env.backward(front_node, reward, 1)

                    else:
                        
                        if env.reward < 0:
                            reward = env.reward
                        else:
                            if front_node.visits != 1:
                                reward = front_node.reward / (front_node.visits - 1)
                            else:
                                method_num += 1
                                reward_list = env.getRewardList([deepcopy(env.current_trajectory)], [1], step)
                                reward = reward_list[0]
                                env.avg_reward[roller % 100] = reward
                                roller += 1
                                if reward > best_reward:
                                    best_reward = reward
                                    best_trajectory = env.packTrajectory(env.new_trajectory)
                                    try:
                                        env.output(step, best_trajectory, best_reward)
                                        # print(f"Best: {best_trajectory}")
                                    except:
                                        pass
                                if reward > 0:
                                    equal = False
                                    current_method = env.packTrajectory(env.current_trajectory)
                                    for k in range(len(candidate_methods)):
                                        # if env.equalMethod(current_method["id"], candidate_methods[k]["method"]["id"]):
                                        if abs(reward - candidate_methods[k]["score"]) < 1e-5:
                                            # if not env.equalMethod(current_method["id"], candidate_methods[k]["method"]["id"]):
                                            #     a = 1
                                            equal = True
                                            break
                                    if not equal:
                                        valid_method_num += 1
                                        candidate_methods.append({
                                            "method": current_method,
                                            "score": reward
                                        })
                                        # if len(candidate_methods) >= 3000:
                                        #     candidate_methods = sorted(candidate_methods[: 3000], key=lambda x: x['score'], reverse=True)
                                        #     del(candidate_methods[-1])

                        env.backward(front_node, reward, 1)

                    env.best_reward = best_reward
                    bonus = best_reward if best_reward > 1. else 1.

                    if step % record_episode == 0:
                        log_dict["train_steps"][-1].append(step)
                        log_dict["train_returns"][-1].append(reward)
                        log_dict["reward_returns"][-1].append(sum(env.avg_reward) / 100.)
                        log_dict["max_reward_returns"][-1].append(best_reward)
                        log_dict["cut_number"][-1].append(cut_num)
                        log_dict["valid_number"][-1].append(valid_method_num)
                        log_dict["number"][-1].append(method_num)
                        print(f"Step: {step}, Best reward: {round(best_reward, 2)}, Bonus/epsilon: {round(bonus, 2)}/{round(eps, 2)}, Level: {level}, Valid method number: {valid_method_num} / {method_num}")
                        print(f"Best: {best_trajectory}")
                        try:
                            visualize(step, "String-routing Path Training - Negative Diversion Training", log_dict)
                            data = [[log_dict['train_steps'][-1][i], log_dict['reward_returns'][-1][i], log_dict['max_reward_returns'][-1][i], log_dict["cut_number"][-1][i], log_dict["valid_number"][-1][i]] for i in range(len(log_dict['train_steps'][-1]))]
                            test = pd.DataFrame(columns=["step", "avg_reward", "maximum_reward", "cut", "valid"], data=data)
                            test.to_csv(f'{NAME}.csv')
                        except:
                            pass
                        
                    if step % episodes == 0:
                        # step in
                        print([f"Sub-trial: {trial}-{current_string_id}-{len(fixed_action_list[current_string_id])} ends with best reward: {best_reward}"])
                        env.root_node, action_id = env.root_node.bestChild(0.0, -math.inf)
                        # env.root_node.parent = None
                        fixed_action_list[current_string_id].append(action_id)
                        print(f"Choose action: {action_id}, Current fixed policy list: {fixed_action_list}")
                        # if env.root_node.done:
                        #     print(f"Trial {trial} has done, Total step: {step}, Cut number: {cut_num}, Valid number: {valid_method_num} / {method_num}")
                        #     try:
                        #         visualize(step, "String-routing Path Training - Negative Diversion Training", log_dict)
                        #         data = [[log_dict['train_steps'][-1][i], log_dict['reward_returns'][-1][i], log_dict['max_reward_returns'][-1][i], log_dict["cut_number"][-1][i], log_dict["valid_number"][-1][i]] for i in range(len(log_dict['train_steps'][-1]))]
                        #         test = pd.DataFrame(columns=["step", "avg_reward", "maximum_reward", "cut", "valid"], data=data)
                        #         test.to_csv(f'{NAME}.csv')
                        #     except:
                        #         pass
                        #     try:
                        #         env.output(step, best_trajectory, best_reward)
                        #         print(best_trajectory)
                        #     except:
                        #         pass
                        #     try:
                        #         sorted_method = sorted(candidate_methods, key=lambda x: x['score'], reverse=True)
                        #         buf = 0
                        #         while buf < len(sorted_method):
                        #             with open(os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}.json"), 'w', encoding="utf-8") as f:
                        #                 json.dump({
                        #                     "method": sorted_method[buf: buf + BUFFER_SIZE],
                        #                     "number": BUFFER_SIZE
                        #                 }, f, indent=4)
                        #             buf += BUFFER_SIZE
                        #         print([f"{len(sorted_method)} methods become candidators"])
                        #     except:
                        #         pass

                # try:
                #     sorted_method = sorted(candidate_methods, key=lambda x: x['score'], reverse=True)
                #     buf = 0
                #     while buf < len(sorted_method):
                #         with open(os.path.join(file_path, f"result_train_time_{time}_candidate_buf_{buf}.json"), 'w', encoding="utf-8") as f:
                #             json.dump({
                #                 "method": sorted_method[buf: buf + BUFFER_SIZE],
                #                 "number": BUFFER_SIZE
                #             }, f, indent=4)
                #         buf += BUFFER_SIZE
                #     print([f"{len(sorted_method)} methods become candidators"])
                # except:
                #     pass
    return log_dict

env = Env()

if __name__ == "__main__":
    log_dict = train(env)
    data = [[log_dict['train_steps'][-1][i], log_dict['reward_returns'][-1][i], log_dict['max_reward_returns'][-1][i], log_dict["cut_number"][-1][i], log_dict["valid_number"][-1][i]] for i in range(len(log_dict['train_steps'][-1]))]
    test = pd.DataFrame(columns=["step", "avg_reward", "maximum_reward", "cut", "valid"], data=data)
    test.to_csv(f'{NAME}.csv')