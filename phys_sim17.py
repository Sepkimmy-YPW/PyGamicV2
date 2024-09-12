import taichi as ti
import taichi.math as tm

from ori_sim_sys import *
from desc import *
from units import *
import json
import os
import time
FOLDING_MAXIMUM = 0.95

#----折纸信息初始化开始----#
data_type = ti.f64
ti.init(arch=ti.cpu, default_fp=data_type, fast_math=True, advanced_optimization=True)

@ti.data_oriented
class OrigamiSimulator:
    def __init__(self, use_gui=True, debug_mode=False, origami_name="default", fast_simulation=False) -> None:
        self.use_gui = use_gui
        self.debug_mode = debug_mode

        if use_gui:
            self.window = ti.ui.Window("Origami Simulation", (800, 800), vsync=True)
            self.gui = self.window.get_gui()
            self.canvas = self.window.get_canvas()
            self.canvas.set_background_color((1., 1., 1.))
            self.scene = self.window.get_scene()
            self.camera = ti.ui.Camera()

        self.dxfg = DxfDirectGrabber()

        # Simulation Scenary
        self.FOLD_SIM = 0
        self.TSA_SIM = 1

        # String information
        self.string_total_information = []
        self.fast_simulation_mode = fast_simulation

        # Preference pack
        self.pref_pack = None

        self.origami_name = origami_name

        self.sparse_solver = ti.linalg.SparseSolver(data_type, "LDLT")

        self.ITER = 32

        self.BIAS = 0.1

        self.print = 1

        if not self.fast_simulation_mode:
            self.time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            try:
                os.makedirs(f"./physResult/{self.origami_name}-{self.time}")
            except:
                pass

    def pointInUnit(self, point):
        length = len(self.units)
        for unit_id in range(length):
            unit = self.units[unit_id]
            kps = unit.getSeqPoint()
            if pointInPolygon(point, kps):
                return unit_id
        return None
    
    # start part 1
    def commonStart_1(self, unit_edge_max):
        # Calculate TSA information
        self.unit_edge_max = unit_edge_max
        # Constraint number
        self.completed_constraint_number = len(self.method["id"])
        
        # 计算线段数
        self.tsa_string_number = 0
        for i in range(self.completed_constraint_number):
            self.tsa_string_number += len(self.method["id"][i]) - 1 # will multi 2

        # find string head
        self.string_head = [self.method["id"][i][0] for i in range(self.completed_constraint_number)]

        # find string end
        self.string_end = [self.method["id"][i][-1] if self.method["type"][i][-1] == 'A' else None for i in range(self.completed_constraint_number)]

        self.rotation = False
        # # get device information
        # if self.pref_pack != None:
        #     self.panel_resolution = self.pref_pack["tsa_resolution"]
        #     self.panel_size = self.pref_pack["tsa_radius"]
        # else:
        #     self.panel_resolution = 6
        #     self.panel_size = 100.
                    
        # 构造折纸系统
        self.ori_sim = OrigamiSimulationSystem(unit_edge_max)
        for ele in self.units:
            self.ori_sim.addUnit(ele)
        self.ori_sim.mesh() #构造三角剖分
        self.ori_sim.fillBlankIndices() # fill all blank indice with -1
    
    def commonStart_2(self, unit_edge_max):
        ori_sim = self.ori_sim

        self.kps = ori_sim.kps                                           # all keypoints of origami
        self.creases = ori_sim.getNewLines()                             # all creases of origami
        self.tri_indices = ori_sim.tri_indices                           # all triangle indices of origami
        self.kp_num = len(ori_sim.kps)                                   # total number of keypoints
        self.indices_num = len(ori_sim.tri_indices)                      # total number of triangles indices
        self.div_indices_num = int(self.indices_num / 3)                 # total_number of triangles
        self.unit_indices_num = len(ori_sim.indices)                     # total number of units
        self.line_total_indice_num = len(ori_sim.line_indices)           # total number of lines
        self.bending_pairs_num = len(ori_sim.bending_pairs)              # total number of bending pairs
        self.crease_pairs_num = len(ori_sim.crease_pairs)                # total number of crease pairs
        self.facet_bending_pairs_num = len(ori_sim.facet_bending_pairs)  # total number of facet bending pairs
        self.facet_crease_pairs_num = len(ori_sim.facet_crease_pairs)    # total number of facet crease pairs
        self.mass_list = ori_sim.mass_list                               # mass list of kps

        # self.point_mass = ori_sim.point_mass #节点质量
        self.gravitational_acc = ti.Vector([0., 0., -9810.]) #全局重力场
        self.string_k = 4. #绳的轴向弹性模量
        self.miu = 0.0
        self.string_length_decrease_step = 0.08
        self.max_stretch_length = 1.
        self.bending_k = self.ori_sim.bending_k
        self.bending_param = ti.field(data_type, shape=1)
        self.facet_bending_param = ti.field(data_type, shape=1)
        
        self.bending_param[0] = self.bending_k
        self.facet_k = self.bending_k
        self.facet_bending_param[0] = self.facet_k

        self.collision_indice = 1e-5
        self.collision_d = 1e-4

        self.barrier = 17. / 18. * math.pi
        self.barrier_left = math.pi - self.barrier
        self.barrier_energy_maximum = -self.collision_indice * self.barrier_left ** 2 * tm.log(self.collision_d / self.barrier_left)
        self.barrier_maximum = self.collision_indice * (2 * self.barrier_left * math.log(self.collision_d / self.barrier_left) - self.barrier_left ** 2 / self.collision_d)
        self.barrier_df_maximum = 2 * self.collision_indice * (2 * self.barrier_left / self.collision_d + self.barrier_left ** 2 / (2 * self.collision_d ** 2) - math.log(self.collision_d / self.barrier_left))

        self.ground_barrier = 1.
        self.ground_collision_indice = 5.
        self.ground_barrier_energy_maximum = -self.ground_collision_indice * self.ground_barrier ** 2 * tm.log(self.collision_d / self.ground_barrier)
        self.ground_force_maximum = self.ground_collision_indice * (2 * self.ground_barrier * tm.log(self.collision_d / self.ground_barrier) + self.ground_barrier ** 2 / self.collision_d)
        self.df_ground_force_maximum = self.ground_collision_indice * (4. * self.ground_barrier / self.collision_d + 2 * tm.log(self.collision_d / self.ground_barrier) - self.ground_barrier ** 2 / self.collision_d ** 2)

        self.ground_miu = 0.6
        self.velocity_barrier = 1e-2

        self.enable_add_folding_angle = 0. #启用折角增加的仿真模式
        self.enable_tsa_rotate = 0 #启用TSA驱动的仿真模式

        self.n = 201 #仿真的时间间隔
        self.dt = 0.1 / self.n #仿真的时间间隔
        self.substeps = round(1. / 250. // self.dt) #子步长，用于渲染
        self.linear_search_step = 8
        self.basic_dt = self.substeps * self.dt
        self.now_t = 0.
        self.iter_h = ti.field(ti.i32, shape=1)
        self.dt_bonus = ti.field(data_type, shape=1)
        

        self.check_force = False
        self.counter = 0

        self.folding_angle = 0.0 #当前的目标折叠角度

        self.energy = 0.0 #当前系统能量

        #折纸参数
        self.d_hole = 2. #折纸上所打通孔的直径
        self.h_hole = 1.2 #通孔高度
        self.beta = self.h_hole / math.sqrt(self.h_hole**2 + self.d_hole**2)

        #折纸初始高度
        self.origami_z_bias = 50.

        #折纸最大折叠能量
        self.total_energy_maximum = ti.field(data_type, shape=1)
        self.total_energy = ti.field(data_type, shape=1)
        self.max_force = ti.field(data_type, shape=1)

        #最大末端作用力
        self.end_force = ti.field(data_type, shape=1)

        #----define parameters for taichi----#
        self.string_params = ti.field(dtype=data_type, shape=1)
        self.masses = ti.field(dtype=data_type, shape=self.kp_num) # 质量信息
        self.original_vertices = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) # 原始点坐标
        self.vertices = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) # 点坐标
        self.vertices_color = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) # 点坐标颜色
        self.unit_indices = ti.Vector.field(unit_edge_max, dtype=int, shape=self.unit_indices_num) # 每个单元的索引信息

        self.ground_vertices = ti.Vector.field(3, dtype=data_type, shape=144) # 点坐标
        self.ground_indices = ti.field(int, shape=726) # 点索引
        self.ground_vertices_color = ti.Vector.field(3, dtype=data_type, shape=144)

        self.sequence_level = ti.field(int, shape=2) # max, min
        self.folding_micro_step = ti.field(data_type, shape=1) # step calculated by sequence_level max and min

        self.unit_center_initial_point = ti.Vector.field(3, dtype=data_type, shape=self.unit_indices_num) # 每个单元的初始中心点位置
        self.unit_center = ti.Vector.field(3, dtype=data_type, shape=self.unit_indices_num) # 每个单元的中心点位置
        self.indices = ti.field(int, shape=self.indices_num) #三角面索引信息

        self.bending_pairs = ti.field(dtype=int, shape=(self.bending_pairs_num, 2)) #弯曲对索引信息
        self.crease_pairs = ti.field(dtype=int, shape=(self.crease_pairs_num, 2)) #折痕对索引信息
        self.previous_dir = ti.field(dtype=data_type, shape=self.bending_pairs_num)
        self.line_pairs = ti.field(dtype=int, shape=(self.line_total_indice_num, 2)) #线段索引信息，用于初始化渲染

        self.crease_angle = ti.field(dtype=data_type, shape=self.bending_pairs_num)
        self.backup_crease_angle = ti.field(dtype=data_type, shape=self.bending_pairs_num)
        self.energy_buffer = ti.field(dtype=data_type, shape=self.linear_search_step)

        self.crease_folding_angle = ti.field(dtype=data_type, shape=self.crease_pairs_num) #折痕折角
        self.crease_folding_accumulate = ti.field(dtype=data_type, shape=self.crease_pairs_num) #补偿折角
        self.bending_pairs_area = ti.field(dtype=data_type, shape=(self.bending_pairs_num, 2)) #弯曲对面积信息
        self.crease_initial_length = ti.field(dtype=data_type, shape=self.crease_pairs_num) #折痕长度

        self.crease_type = ti.field(dtype=int, shape=self.crease_pairs_num) #折痕类型信息，与折痕对一一对应
        self.crease_level = ti.field(dtype=int, shape=self.crease_pairs_num)
        self.crease_coeff = ti.field(dtype=data_type, shape=self.crease_pairs_num)

        self.connection_matrix = ti.field(dtype=data_type, shape=(self.kp_num, self.kp_num)) #关键点之间的连接矩阵

        self.line_color = ti.Vector.field(3, dtype=data_type, shape=self.line_total_indice_num*2) #线段颜色，用于渲染
        self.line_vertex = ti.Vector.field(3, dtype=data_type, shape=self.line_total_indice_num*2) #线段顶点位置，用于渲染

        self.P_number = len(self.P_candidate_connection)
        self.border_vertex = ti.Vector.field(3, dtype=data_type, shape=self.P_number + 1)
        #----TSA constraints----#

        self.constraint_number = len(self.method["id"]) #约束的数量
        

        #根据约束数量，确定约束起始点和终止点位置， 若初始点一致，则识别为TSA
        if self.constraint_number == 0 or self.P_number == 0:
            self.constraint_start_point = ti.Vector.field(3, dtype=data_type, shape=1)
            self.constraint_start_point_candidate_id = ti.field(dtype=int, shape=self.constraint_number)

            self.constraint_start_point_candidate = ti.Vector.field(3, dtype=data_type, shape=1)
            self.constraint_start_point_candidate_connection = ti.field(dtype=int, shape=1)

            self.constraint_height = ti.field(dtype=data_type, shape=1)
            
            self.string_number_each = ti.field(dtype=int, shape=1)
            self.string_length_decrease = ti.field(dtype=data_type, shape=1) #当前绳子的减少长度

            self.constraint_end_point_existence = ti.field(dtype=bool, shape=1)
            self.constraint_end_point = ti.Vector.field(3, dtype=data_type, shape=1)
            self.constraint_end_point_candidate_id = ti.field(dtype=int, shape=1)

            # 绳信息
            self.string_vertex = ti.Vector.field(3, dtype=data_type, shape=1)
            self.max_control_length = 1
            self.endpoint_vertex = ti.Vector.field(3, dtype=data_type, shape=1)
            #---#
            self.unit_control = ti.Vector.field(1, dtype=int, shape=1)
            self.unit_control.fill(-1)

            self.initial_length_per_string = ti.Vector.field(1, dtype=data_type, shape=1)
            self.current_length_per_string = ti.Vector.field(1, dtype=data_type, shape=1)
            self.intersection_points = ti.Vector.field(3, dtype=data_type, shape=1)
            self.intersection_infos = ti.Vector.field(5, dtype=data_type, shape=1)

            self.intersection_flag = ti.Vector.field(1, dtype=int, shape=1)
            self.intersection_flag.fill(0)

            #穿线方向信息，-1表示从下往上穿，1表示从上往下穿
            self.hole_dir = ti.Vector.field(1, dtype=data_type, shape=1)
            self.hole_dir.fill(0.)
            # 约束绳的长度信息
            self.constraint_initial_length = ti.field(data_type, shape=1)
            self.constraint_length = ti.field(data_type, shape=1)
            self.backup_constraint_length = ti.field(data_type, shape=1)
            self.points = ti.Vector.field(3, dtype=data_type, shape=1)
        else:
            self.constraint_start_point = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number)
            self.constraint_start_point_candidate_id = ti.field(dtype=int, shape=self.constraint_number)

            self.constraint_start_point_candidate = ti.Vector.field(3, dtype=data_type, shape=self.P_number)
            self.constraint_start_point_candidate_connection = ti.field(dtype=int, shape=self.P_number)
            
            self.constraint_height = ti.field(dtype=data_type, shape=self.P_number)

            self.string_number_each = ti.field(dtype=int, shape=self.constraint_number)
            self.string_length_decrease = ti.field(dtype=data_type, shape=self.constraint_number) #当前绳子的减少长度

            self.constraint_end_point_existence = ti.field(dtype=bool, shape=self.constraint_number)
            self.constraint_end_point = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number)
            self.constraint_end_point_candidate_id = ti.field(dtype=int, shape=self.constraint_number)

            # 要控制的点的索引信息，至多控制单元数目个数的点, TYPE B的点
            self.max_control_length = max([len(ele) for ele in self.method["id"]])
            self.unit_control = ti.Vector.field(self.max_control_length, dtype=int, shape=self.constraint_number)
            self.unit_control.fill(-1)

            self.initial_length_per_string = ti.Vector.field(self.max_control_length, dtype=data_type, shape=self.constraint_number)
            self.current_length_per_string = ti.Vector.field(self.max_control_length, dtype=data_type, shape=self.constraint_number)
            self.intersection_points = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number * self.max_control_length)
            self.intersection_infos = ti.Vector.field(5, dtype=data_type, shape=self.constraint_number * self.max_control_length)

            self.intersection_flag = ti.Vector.field(self.max_control_length, dtype=int, shape=self.constraint_number)
            self.intersection_flag.fill(0)

            #穿线方向信息，-1表示从下往上穿，1表示从上往下穿
            self.hole_dir = ti.Vector.field(self.max_control_length, dtype=data_type, shape=self.constraint_number)
            self.hole_dir.fill(0.)

            # 绳信息
            self.string_vertex = ti.Vector.field(3, dtype=data_type, shape=int(self.tsa_string_number * 4))
            
            self.endpoint_vertex = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number)
            #---#
            
            # 约束绳的长度信息
            self.constraint_initial_length = ti.field(data_type, shape=self.constraint_number)
            self.constraint_length = ti.field(data_type, shape=self.constraint_number)
            self.backup_constraint_length = ti.field(data_type, shape=self.constraint_number)
            self.points = ti.Vector.field(3, dtype=data_type, shape=1)

        # 受力顶点，渲染时用
        self.force_vertex = ti.Vector.field(3, dtype=data_type, shape=2*self.kp_num)

        # 有可能所有单元都是三角形，故没有面折痕，根据特定条件初始化面折痕信息
        if self.facet_bending_pairs_num > 0:
            self.facet_bending_pairs = ti.field(dtype=int, shape=(self.facet_bending_pairs_num, 2))
            self.facet_crease_pairs = ti.field(dtype=int, shape=(self.facet_crease_pairs_num, 2))
            self.facet_bending_pairs_area = ti.field(dtype=data_type, shape=(self.facet_bending_pairs_num, 2)) #弯曲对面积信息
            self.facet_crease_initial_length = ti.field(dtype=data_type, shape=self.facet_bending_pairs_num) #折痕长度

        else:
            self.facet_bending_pairs = ti.field(dtype=int, shape=(1, 2))
            self.facet_crease_pairs = ti.field(dtype=int, shape=(1, 2))
            self.facet_bending_pairs_area = ti.field(dtype=data_type, shape=(1, 2)) #弯曲对面积信息
            self.facet_crease_initial_length = ti.field(dtype=data_type, shape=1) #折痕长度
        
        #----simulator information----#
        self.x = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的位置
        self.v = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的速度
        self.dv = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的加速度
        self.force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的力
        self.record_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #跟踪的某一类型的力
        self.print_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #打印的某一类型的力
        self.u = ti.Vector.field(2, dtype=data_type, shape=self.kp_num)

        if self.constraint_number:
            self.dldx_force = ti.Vector.field(3, dtype=data_type, shape=(self.constraint_number, self.kp_num)) #string
            self.dldx_friction_force = ti.Vector.field(3, dtype=data_type, shape=(self.constraint_number, self.max_control_length, self.kp_num)) #string
        else:
            self.dldx_force = ti.Vector.field(3, dtype=data_type, shape=(1, 1)) #string
            self.dldx_friction_force = ti.Vector.field(3, dtype=data_type, shape=(1, 1, 1)) #string

        self.stvk_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.bending_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.facet_bending_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.viscosity_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.string_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.preventing_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.ground_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.df_ground_force = ti.field(dtype=data_type, shape=self.kp_num)
        self.ground_friction = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)

        # self.viscousity = 1e-3
        self.enable_ground = False

        self.back_up_x = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.back_up_v = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.back_up_start_point = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number + 1)

        self.start_start_point = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number + 1)
        self.start_x = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的位置

        self.epsilon_v = ti.field(data_type, shape=1)
        self.epsilon_string = ti.field(data_type, shape=1)
    
    def commonStart_3(self):
        self.current_t = 0.0
        self.half_dt = self.dt * 0.5

        self.folding_step = tm.pi * 17.85 / 18.0
        self.folding_max = tm.pi * 17.85 / 18.0

        self.half_max_size = self.max_size * 0.5

        self.folding_angle_reach_pi = ti.field(dtype=bool, shape=1)

        self.past_move_indice = 0.0

        self.stable_state = 0

        self.dm = ti.Matrix.field(3, 3, dtype=data_type, shape=self.div_indices_num)
        
        self.A = ti.field(dtype=data_type, shape=self.div_indices_num)

        self.mu = 3. / 2.6

        self.landa = 3. * 0.3 / (1 + 0.3) / (1 - 0.6)

        self.lame_k = 1000.

        # derivative
        self.dDs = ti.Matrix.field(3, 3, dtype=data_type, shape=(3, 3))
        self.b = ti.field(data_type, shape=3 * self.kp_num)
        self.K_element = ti.Matrix.field(9, 9, data_type, shape=self.div_indices_num)
        self.K_element_bending = ti.Matrix.field(12, 12, data_type, shape=self.bending_pairs_num+self.facet_bending_pairs_num)

        self.triplets = ti.Vector.field(3, dtype=int, shape=self.div_indices_num)
        self.triplets_bending = ti.Vector.field(4, dtype=int, shape=self.bending_pairs_num+self.facet_bending_pairs_num)

        if self.constraint_number == 0 or self.P_number == 0:
            self.outer_P = ti.Vector.field(3, dtype=data_type, shape=1)
            self.outer_Q = ti.Vector.field(3, dtype=data_type, shape=1)
            self.outer_PI = ti.Vector.field(3, dtype=data_type, shape=1)
            self.outer_QI = ti.Vector.field(3, dtype=data_type, shape=1)
            self.exist_grad_number = ti.field(dtype=int, shape=1)
        else:
            self.outer_P = ti.Vector.field(3, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.outer_Q = ti.Vector.field(3, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.outer_PI = ti.Vector.field(3, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.outer_QI = ti.Vector.field(3, dtype=data_type, shape=self.max_control_length * self.constraint_number)
            self.exist_grad_number = ti.field(dtype=int, shape=1)
        
        trips = 27 * self.kp_num ** 2 + 81 * (self.bending_pairs_num+self.facet_bending_pairs_num) + (self.constraint_number * self.max_control_length) * 27 * self.kp_num ** 2
        trips = 3 * self.kp_num + 81 * self.div_indices_num + 144 * (self.bending_pairs_num+self.facet_bending_pairs_num) + (self.constraint_number * (self.max_control_length + 1)) * 9 * self.kp_num ** 2

        self.AK = ti.linalg.SparseMatrixBuilder(3 * self.kp_num, 3 * self.kp_num, max_num_triplets=trips, dtype=data_type)

        self.u0 = ti.field(data_type, shape=3 * self.kp_num) # solution

        self.lames_bonus = ti.field(data_type, shape=2) # mu, landa

        # flat the line_indices
        for i in range(len(self.ori_sim.line_indices)):
            self.ori_sim.line_indices[i] = [self.ori_sim.line_indices[i][START][START], self.ori_sim.line_indices[i][START][END], self.ori_sim.line_indices[i][END]]
        

        
    def startOnlyTSA(self, units, max_size, total_bias, unit_edge_max):
        self.units = units
        self.max_size = max_size
        self.total_bias = total_bias
        self.commonStart_1(unit_edge_max)
        self.sim_mode = self.TSA_SIM
        self.sequence_level_initial = 0
        self.commonStart_2(unit_edge_max)
        self.enable_ground = False
        self.gravitational_acc = tm.vec3([0., 0., -9810.]) 
        self.commonStart_3()
    
    def start(self, filepath, unit_edge_max, sim_type):
        # 获取点和线段信息
        with open("./descriptionData/" + filepath + ".json", 'r', encoding='utf-8') as fw:
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
        self.method = input_json["strings"]
        try:
            self.P_candidate = input_json["P_candidators"]["points"]
            self.P_candidate_connection = input_json["P_candidators"]["connections"]
        except:
            self.P_candidate = [[]]
            self.P_candidate_connection =[]
        # calculate max length of view
        self.max_size, max_x, max_y = getMaxDistance(self.kps)
        self.total_bias = getTotalBias(self.units)
        self.commonStart_1(unit_edge_max)
    
        self.commonStart_2(unit_edge_max)
        self.sim_mode = sim_type
        if sim_type == self.FOLD_SIM:
            self.enable_ground = False
            self.gravitational_acc = tm.vec3([0., 0., 0.])
        else:
            self.enable_ground = True
            self.gravitational_acc = tm.vec3([0., 0., -9810.])  
        self.commonStart_3()

    @ti.kernel
    def modify_parameters(self, lame_k: data_type):
        self.lames_bonus[0] = lame_k * self.mu
        self.lames_bonus[1] = lame_k * self.landa

    @ti.kernel
    def initialize(     self, 
                        indices:                    ti.types.ndarray(), 
                        kps:                        ti.types.ndarray(), 
                        mass_list:                  ti.types.ndarray(),
                        tri_indices:                ti.types.ndarray(),
                        connection_matrix:          ti.types.ndarray(),
                        bending_pairs:              ti.types.ndarray(),
                        crease_pairs:               ti.types.ndarray(),
                        line_indices:               ti.types.ndarray(),
                        facet_bending_pairs:        ti.types.ndarray(),
                        facet_crease_pairs:         ti.types.ndarray(),
                        sim_mode:                   int,
                        string_number_each:         ti.types.ndarray(),
                        string_total_information:   ti.types.ndarray(),
                        tsa_end:                    ti.types.ndarray(),
                        original_kps:               ti.types.ndarray(), 
                        tb_lines:                   ti.types.ndarray(),
                        lame_k:                     data_type,
                        p_candidate:                ti.types.ndarray(),
                        p_candidate_connections:    ti.types.ndarray(),
                        epsilon_v:                  data_type,
                        epsilon_string:             data_type
        ):
        self.epsilon_v[0] = epsilon_v
        self.epsilon_string[0] = epsilon_string
        # 定义初始点候选集及连接关系
        if self.constraint_number > 0:
            for i in ti.ndrange(self.P_number):
                self.constraint_start_point_candidate[i][X] = p_candidate[i, X]
                self.constraint_start_point_candidate[i][Y] = p_candidate[i, Y]
                self.constraint_start_point_candidate[i][Z] = p_candidate[i, Z]
                self.constraint_start_point_candidate_connection[i] = p_candidate_connections[i]
                if p_candidate_connections[i] >= 0:
                    self.constraint_height[i] = p_candidate[i, Z] - self.origami_z_bias
                    # print(self.constraint_height[i])
                else:
                    self.constraint_height[i] = 0.0

        for i, j in ti.ndrange(12, 12):
            self.ground_vertices[12 * i + j] = [-550. + 100. * j, -550. + 100. * i, 0.95 * self.ground_barrier]
            if j % 2 or i % 2:
                self.ground_vertices_color[12 * i + j] = [.6, .6, .6]
            else:
                self.ground_vertices_color[12 * i + j] = [.8, .8, .8]

        for i, j in ti.ndrange(11, 11):
            if (i + j) % 2:
                self.ground_indices[66 * i + 6 * j + 0] = 12 * i + j
                self.ground_indices[66 * i + 6 * j + 1] = 12 * i + j + 1
                self.ground_indices[66 * i + 6 * j + 2] = 12 * i + j + 1 + 12
                self.ground_indices[66 * i + 6 * j + 3] = 12 * i + j
                self.ground_indices[66 * i + 6 * j + 4] = 12 * i + j + 1 + 12
                self.ground_indices[66 * i + 6 * j + 5] = 12 * i + j + 12
            else:
                self.ground_indices[66 * i + 6 * j + 0] = 12 * i + j
                self.ground_indices[66 * i + 6 * j + 1] = 12 * i + j + 1
                self.ground_indices[66 * i + 6 * j + 2] = 12 * i + j + 12
                self.ground_indices[66 * i + 6 * j + 3] = 12 * i + j + 1
                self.ground_indices[66 * i + 6 * j + 4] = 12 * i + j + 1 + 12
                self.ground_indices[66 * i + 6 * j + 5] = 12 * i + j + 12


        # 初始化单元索引
        for i in ti.ndrange(self.unit_indices_num):
            for j in ti.ndrange(self.unit_edge_max):
                self.unit_indices[i][j] = indices[i, j]
        # 初始化节点位置与质量
        for i in ti.ndrange(self.kp_num):
            self.original_vertices[i] = [kps[i, X], kps[i, Y], kps[i, Z]]
            self.masses[i] = mass_list[i]
        # 初始化三角面索引
        for i in ti.ndrange(self.indices_num):
            self.indices[i] = tri_indices[i]
        # # 初始化连接矩阵
        for i, j in ti.ndrange(self.kp_num, self.kp_num):
            self.connection_matrix[i, j] = connection_matrix[i, j]
        # 初始化弯曲对和折痕对
        for i, j in ti.ndrange(self.bending_pairs_num, 2):
            self.bending_pairs[i, j] = bending_pairs[i, j]
            self.crease_pairs[i, j] = crease_pairs[i, j]

        for i in ti.ndrange(self.bending_pairs_num):
            # 初始化弯曲对和折痕对的面积
            cs = self.original_vertices[self.crease_pairs[i, 0]]
            ce = self.original_vertices[self.crease_pairs[i, 1]]
            p1 = self.original_vertices[self.bending_pairs[i, 0]]
            p2 = self.original_vertices[self.bending_pairs[i, 1]]
            a1 = ((ce - cs).cross(p1 - cs)).norm()
            a2 = ((p2 - cs).cross(ce - cs)).norm()
            self.bending_pairs_area[i, 0] = a1
            self.bending_pairs_area[i, 1] = a2
            self.crease_initial_length[i] = (ce - cs).norm()

        # 初始化线段对
        for i, j in ti.ndrange(self.line_total_indice_num, 2):
            self.line_pairs[i, j] = line_indices[i, j]
        # 初始化面折痕对
        for i, j in ti.ndrange(self.facet_bending_pairs_num, 2):
            self.facet_bending_pairs[i, j] = facet_bending_pairs[i, j]
            self.facet_crease_pairs[i, j] = facet_crease_pairs[i, j]

        for i in ti.ndrange(self.facet_bending_pairs_num):
            # 初始化弯曲对和折痕对的面积
            cs = self.original_vertices[self.facet_crease_pairs[i, 0]]
            ce = self.original_vertices[self.facet_crease_pairs[i, 1]]
            p1 = self.original_vertices[self.facet_bending_pairs[i, 0]]
            p2 = self.original_vertices[self.facet_bending_pairs[i, 1]]
            a1 = ((ce - cs).cross(p1 - cs)).norm()
            a2 = ((p2 - cs).cross(ce - cs)).norm()
            # assert a1 > 0 and a2 > 0
            self.facet_bending_pairs_area[i, 0] = a1
            self.facet_bending_pairs_area[i, 1] = a2
            self.facet_crease_initial_length[i] = (ce - cs).norm()

        #初始化折痕折角
        for i in ti.ndrange(self.crease_pairs_num):
            self.crease_angle[i] = 0.0
            self.crease_folding_angle[i] = 0.0
            self.crease_folding_accumulate[i] = 0.0
            self.previous_dir[i] = 0.0

        # 初始化折痕类型
        for i in ti.ndrange(self.crease_pairs_num):
            for j in ti.ndrange(self.line_total_indice_num):
                if crease_pairs[i, 0] == line_indices[j, 0] and crease_pairs[i, 1] == line_indices[j, 1]:
                    self.crease_type[i] = line_indices[j, 2]
                    break

        # ti.loop_config(serialize=True)
        self.sequence_level[0] = 0
        self.sequence_level[1] = 0
        if sim_mode == self.FOLD_SIM:
            # 初始化折叠等级和系数
            for i in ti.ndrange(self.crease_pairs_num):
                kp1 = [original_kps[crease_pairs[i, 0], X], original_kps[crease_pairs[i, 0], Y]]
                kp2 = [original_kps[crease_pairs[i, 1], X], original_kps[crease_pairs[i, 1], Y]]
                for j in ti.ndrange(tb_lines.shape[0]):
                    kp11 = [tb_lines[j, 0], tb_lines[j, 1]]
                    kp22 = [tb_lines[j, 2], tb_lines[j, 3]]
                    if (((kp1[X] - kp11[X]) ** 2 + (kp1[Y] - kp11[Y]) ** 2) <= 16. and \
                        ((kp2[X] - kp22[X]) ** 2 + (kp2[Y] - kp22[Y]) ** 2) <= 16.) or \
                       (((kp1[X] - kp22[X]) ** 2 + (kp1[Y] - kp22[Y]) ** 2) <= 16. and \
                        ((kp2[X] - kp11[X]) ** 2 + (kp2[Y] - kp11[Y]) ** 2) <= 16.):

                        self.crease_level[i] = int(tb_lines[j, 4])
                        self.crease_coeff[i] = tb_lines[j, 5]
                        break
                if self.crease_level[i] > self.sequence_level[0]:
                    self.sequence_level[0] = self.crease_level[i]
                if self.crease_level[i] < self.sequence_level[1]:
                    self.sequence_level[1] = self.crease_level[i]
            self.folding_micro_step[0] = tm.pi / 180.0 / (1 + self.sequence_level[0] - self.sequence_level[1])
        # ti.loop_config(serialize=False)

        # 初始化渲染的线的颜色信息
        for i in ti.ndrange(self.line_total_indice_num):
            if line_indices[i, 2] == BORDER:
                self.line_color[2 * i] = [0, 0, 0]
                self.line_color[2 * i + 1] = [0, 0, 0]
            elif line_indices[i, 2] == VALLEY:
                self.line_color[2 * i] = [0, 0.17, 0.83]
                self.line_color[2 * i + 1] = [0, 0.17, 0.83]
            elif line_indices[i, 2] == MOUNTAIN:
                self.line_color[2 * i] = [0.75, 0.2, 0.05]
                self.line_color[2 * i + 1] = [0.75, 0.2, 0.05]
            else:
                self.line_color[2 * i] = [0.5, 0.5, 0.5]
                self.line_color[2 * i + 1] = [0.5, 0.5, 0.5]

        # 计算折纸最大能量
        self.total_energy_maximum[0] = 0.0
        self.iter_h[0] = 0
        for i in ti.ndrange(self.crease_pairs_num):
            index1 = crease_pairs[i, 0]
            index2 = crease_pairs[i, 1]
            # length = (self.original_vertices[index2] - self.original_vertices[index1]).norm()
            self.total_energy_maximum[0] += 100.

        # 暂不考虑ABAB, 初始化控制单元和穿孔方向
        for i in ti.ndrange(self.constraint_number):
            index = 0
            self.string_number_each[i] = string_number_each[i]
            for j in ti.ndrange(string_number_each[i] - 1):
                if string_total_information[i, j + 1, 0] != 0:
                    self.unit_control[i][index] = string_total_information[i, j + 1, 1]
                    self.hole_dir[i][index] = string_total_information[i, j + 1, 2]
                    index += 1

        # calculate initial center point
        for i in ti.ndrange(self.unit_indices_num):
            unit_indice = self.unit_indices[i]
            center_point = self.calculateCenterPoint3DWithVerticeUnitId(unit_indice)
            self.unit_center_initial_point[i] = center_point
            self.unit_center[i] = center_point

        # 计算初始点
        if self.P_number > 0:
            for i in ti.ndrange(self.constraint_number):
                start_id = string_total_information[i, 0, 1]
                self.string_length_decrease[i] = 0.0
                self.constraint_start_point[i] = self.constraint_start_point_candidate[start_id]
                self.constraint_start_point_candidate_id[i] = start_id
                # print(self.constraint_start_point_candidate_id[i])

                if self.constraint_start_point_candidate_connection[start_id] >= 0:
                    unit_indice = self.unit_indices[self.constraint_start_point_candidate_connection[start_id]]
                    center_point = self.calculateCenterPoint3DWithVerticeUnitId(unit_indice)
                    self.constraint_start_point[i] = center_point + tm.vec3([0., 0., self.constraint_height[start_id]])
                    # print(self.constraint_start_point[i])
        
        # 计算末端点信息
        if self.P_number > 0:
            for i in ti.ndrange(self.completed_constraint_number):
                if tsa_end[i] != -1:
                    id = int(tsa_end[i])
                    self.constraint_end_point[i] = self.constraint_start_point_candidate[id]
                    self.constraint_end_point_candidate_id[i] = id
                    self.constraint_end_point_existence[i] = True
                    if self.constraint_start_point_candidate_connection[id] >= 0:
                        unit_indice = self.unit_indices[self.constraint_start_point_candidate_connection[id]]
                        center_point = self.calculateCenterPoint3DWithVerticeUnitId(unit_indice)
                        self.constraint_end_point[i] = center_point + tm.vec3([0., 0., self.constraint_height[id]])
                else:
                    self.constraint_end_point[i] = [0.0, 0.0, 0.0]
                    self.constraint_end_point_existence[i] = False

        # calculate initial length
        for i in ti.ndrange(self.constraint_number):
            self.constraint_initial_length[i] = 0.0
            start_point = self.constraint_start_point[i]
            for j in ti.ndrange(self.max_control_length):
                if self.unit_control[i][j] != -1:
                    length = (self.unit_center_initial_point[self.unit_control[i][j]] - start_point).norm()
                    self.constraint_initial_length[i] += length
                    self.initial_length_per_string[i][j] = length
                    start_point = self.unit_center_initial_point[self.unit_control[i][j]]
                else:
                    break
            if self.constraint_end_point_existence[i]:
                self.constraint_initial_length[i] += (self.constraint_end_point[i] - start_point).norm()
            self.constraint_length[i] = self.constraint_initial_length[i]
            self.backup_constraint_length[i] = self.constraint_initial_length[i]

        # 初始化结束条件
        self.folding_angle_reach_pi[0] = False
        # self.stable_state = 0
        # self.past_move_indice = 0.0

        # self.can_rotate = False
        # self.tsa_turning_angle = 0.0

        # # 初始化线的信息
        # self.string_params[0] = self.string_k

        # # 初始化边缘信息
        for i in ti.ndrange(self.P_number):
            self.border_vertex[i] = self.constraint_start_point_candidate[i]
        # for i in ti.ndrange(30):
        #     self.border_vertex[2 * i] = [tm.cos(i / 15. * tm.pi) * self.panel_size, tm.sin(i / 15. * tm.pi) * self.panel_size, self.origami_z_bias]
        #     self.border_vertex[2 * i + 1] = [tm.cos((i + 1) / 15. * tm.pi) * self.panel_size, tm.sin((i + 1) / 15. * tm.pi) * self.panel_size, self.origami_z_bias]

        # 拉梅常数增益
        self.lames_bonus[0] = lame_k * self.mu
        self.lames_bonus[1] = lame_k * self.landa

        # 初始化形状微分矩阵
        self.dDs[0, 0] = ti.Matrix.rows([[-1., -1., 0], [0, 0, 0], [0, 0, 0]])
        self.dDs[0, 1] = ti.Matrix.rows([[0, 0, 0], [-1., -1., 0], [0, 0, 0]])
        self.dDs[0, 2] = ti.Matrix.rows([[0, 0, 0], [0, 0, 0], [-1., -1., 0]])
        self.dDs[1, 0] = ti.Matrix.rows([[1., 0, 0], [0, 0, 0], [0, 0, 0]])
        self.dDs[1, 1] = ti.Matrix.rows([[0, 0, 0], [1., 0, 0], [0, 0, 0]])
        self.dDs[1, 2] = ti.Matrix.rows([[0, 0, 0], [0, 0, 0], [1., 0, 0]])
        self.dDs[2, 0] = ti.Matrix.rows([[0, 1., 0], [0, 0, 0], [0, 0, 0]])
        self.dDs[2, 1] = ti.Matrix.rows([[0, 0, 0], [0, 1., 0], [0, 0, 0]])
        self.dDs[2, 2] = ti.Matrix.rows([[0, 0, 0], [0, 0, 0], [0, 1., 0]])
    
        for i in ti.ndrange(self.kp_num):
            self.x[i] = self.original_vertices[i]
            self.v[i] = [0., 0., 0.]
            self.dv[i] = [0., 0., 0.]

        # precompute dm and A
        for i in ti.ndrange(self.div_indices_num):
            x0 = self.original_vertices[self.indices[3 * i]]
            x1 = self.original_vertices[self.indices[3 * i + 1]]
            x2 = self.original_vertices[self.indices[3 * i + 2]]
            dm = ti.Matrix.cols([x1 - x0, x2 - x0, tm.normalize((x1 - x0).cross(x2 - x0))])
            self.dm[i] = tm.inverse(dm)
            self.A[i] = 0.5 * tm.determinant(dm)

    @ti.kernel
    def fill_line_vertex(self):
        for i in ti.ndrange(self.line_total_indice_num):
            indice1 = self.line_pairs[i, 0]
            indice2 = self.line_pairs[i, 1]
            self.line_vertex[2 * i] = self.vertices[indice1]
            self.line_vertex[2 * i + 1] = self.vertices[indice2]

    @ti.func
    def getBendingEnergy(self, cs, ce, p1, p2, k, theta, crease_type, debug=False, enable_dynamic_change=False, a1=1., a2=1., L=1., id=-1):
        xc = ce - cs

        f11 = p1 - cs
        f22 = p2 - cs
        n1 = xc.cross(f11)
        n2 = f22.cross(xc)

        energy = 0.0

        n1_norm = n1.norm()
        n2_norm = n2.norm()

        multi_n1_n2 = n1_norm * n2_norm

        dir = n1.cross(n2).dot(xc)
        val = n1.dot(n2)

        norm_val = val / multi_n1_n2

        current_theta = 0.0
        if norm_val >= 1.0:
            val = multi_n1_n2
            norm_val = 1.0
            current_theta = 0.0
        elif norm_val <= -1.0:
            val = -multi_n1_n2
            norm_val = -1.0
            current_theta = tm.pi
        else:
            current_theta = tm.acos(norm_val)
    
        # 求折叠角
        n_value = 0.
        backup_n_value = 0.
        # current_theta = tm.acos(norm_val)
    
        # 求折叠角
        if not enable_dynamic_change:
            # if index == -1:
            if crease_type == MOUNTAIN:
                if dir >= 0.:
                    n_value = current_theta + theta 
                    backup_n_value = n_value
                    t11 = current_theta - self.barrier
                    t22 = tm.pi + self.collision_d - current_theta
                    if t11 >= 0:
                        n_value -= self.collision_indice * (2 * t11 * tm.log(t22 / self.barrier_left) - t11 ** 2 / t22)
                        energy += -self.collision_indice * t11 ** 2 * tm.log(t22 / self.barrier_left)
                    # if id == 0:
                    #     print(f"1, {dir}, {norm_val}, {t11}, {backup_n_value}")
                else:
                    if id != -1 and self.previous_dir[id] >= 0 and norm_val <= -0.5: # 180~270
                        n_value = theta + 2. * tm.pi - current_theta
                        backup_n_value = n_value
                        n_value -= self.barrier_maximum - backup_n_value * self.barrier_df_maximum
                        energy += self.barrier_energy_maximum + (-2. * self.barrier_maximum + (tm.pi - current_theta) * self.barrier_df_maximum) * (tm.pi - current_theta) * 0.5
                        # if id == 0:
                        #     print(f"1, {dir}, {norm_val}, {backup_n_value}")
                    else:
                        n_value = theta - current_theta
                        backup_n_value = n_value
            else:
                if dir <= 0.:
                    n_value = -current_theta + theta
                    backup_n_value = n_value
                    t11 = current_theta - self.barrier
                    t22 = tm.pi + self.collision_d - current_theta
                    if t11 >= 0:
                        n_value += self.collision_indice * (2 * t11 * tm.log(t22 / self.barrier_left) - t11 ** 2 / t22)
                        energy += -self.collision_indice * t11 ** 2 * tm.log(t22 / self.barrier_left)
                else:
                    if id != -1 and self.previous_dir[id] <= 0 and norm_val <= -0.5:
                        n_value = theta - 2. * tm.pi + current_theta
                        backup_n_value = n_value
                        n_value += self.barrier_maximum + backup_n_value * self.barrier_df_maximum
                        energy += self.barrier_energy_maximum + (-2. * self.barrier_maximum + (tm.pi - current_theta) * self.barrier_df_maximum) * (tm.pi - current_theta) * 0.5
                    else:
                        n_value = theta + current_theta       
                        backup_n_value = n_value     
        else:
            if dir >= 0.:
                if id != -1 and self.previous_dir[id] <= 0 and norm_val <= -0.5:
                    n_value = theta - 2. * tm.pi + current_theta
                    backup_n_value = n_value
                    n_value += self.barrier_maximum + backup_n_value * self.barrier_df_maximum
                    energy += self.barrier_energy_maximum + (-2. * self.barrier_maximum + (tm.pi - current_theta) * self.barrier_df_maximum) * (tm.pi - current_theta) * 0.5
                else:
                    n_value = current_theta + theta 
                    backup_n_value = n_value
                    t11 = current_theta - self.barrier
                    t22 = tm.pi + self.collision_d - current_theta
                    if t11 >= 0:
                        n_value -= self.collision_indice * (2 * t11 * tm.log(t22 / self.barrier_left) - t11 ** 2 / t22)
                        energy += -self.collision_indice * t11 ** 2 * tm.log(t22 / self.barrier_left)
            else:
                if id != -1 and self.previous_dir[id] >= 0 and norm_val <= -0.5:
                    n_value = theta + 2. * tm.pi - current_theta
                    backup_n_value = n_value
                    n_value -= self.barrier_maximum - backup_n_value * self.barrier_df_maximum
                    energy += self.barrier_energy_maximum + (-2. * self.barrier_maximum + (tm.pi - current_theta) * self.barrier_df_maximum) * (tm.pi - current_theta) * 0.5
                else:
                    n_value = -current_theta + theta
                    backup_n_value = n_value
                    t11 = current_theta - self.barrier
                    t22 = tm.pi + self.collision_d - current_theta
                    if t11 >= 0:
                        n_value += self.collision_indice * (2 * t11 * tm.log(t22 / self.barrier_left) - t11 ** 2 / t22)
                        energy += -self.collision_indice * t11 ** 2 * tm.log(t22 / self.barrier_left)

        energy *= k * L
        energy += 0.5 * k * L * backup_n_value ** 2

        return energy
    
    @ti.func
    def getSkewMatrix(self, x):
        return ti.Matrix.cols([[0., x[Z], -x[Y]], [-x[Z], 0., x[X]], [x[Y], -x[X], 0.]])
    
    @ti.func
    def getDthetaDx(self, x0, x1, x2, x3, theta):
        s1 = x1 - x0
        cr = x2 - x0
        s2 = x3 - x0

        e = -cr / cr.norm() # valley crease is positive

        v1 = cr.cross(s2)
        v2 = s1.cross(cr)

        v1_norm = v1.norm()
        v2_norm = v2.norm()

        n1 = v1 / v1_norm
        n2 = v2 / v2_norm

        proj_v1 = ti.Matrix.identity(data_type, 3) - n1.outer_product(n1)
        proj_v2 = ti.Matrix.identity(data_type, 3) - n2.outer_product(n2)

        dv1dx0 = self.getSkewMatrix(x3 - x2)
        # dv1dx1 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        dv1dx2 = self.getSkewMatrix(x0 - x3)
        dv1dx3 = self.getSkewMatrix(x2 - x0)

        dv2dx0 = self.getSkewMatrix(x2 - x1)
        dv2dx1 = self.getSkewMatrix(x0 - x2)
        dv2dx2 = self.getSkewMatrix(x1 - x0)
        # dv2dx3 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

        v1_skew = self.getSkewMatrix(v1)
        v2_skew = self.getSkewMatrix(v2)

        cos_term_x0 = (dv2dx0 @ proj_v2 @ v1_skew - dv1dx0 @ proj_v1 @ v2_skew) @ e
        cos_term_x1 = (dv2dx1 @ proj_v2 @ v1_skew) @ e
        cos_term_x2 = (dv2dx2 @ proj_v2 @ v1_skew - dv1dx2 @ proj_v1 @ v2_skew) @ e
        cos_term_x3 = (-dv1dx3 @ proj_v1 @ v2_skew) @ e

        sin_term_x0 = dv1dx0 @ proj_v1 @ v2 + dv2dx0 @ proj_v2 @ v1
        sin_term_x1 = dv2dx1 @ proj_v2 @ v1
        sin_term_x2 = dv1dx2 @ proj_v1 @ v2 + dv2dx2 @ proj_v2 @ v1
        sin_term_x3 = dv1dx3 @ proj_v1 @ v2

        k = 1. / (v1_norm * v2_norm)

        dthetadx0 = k * (tm.cos(theta) * cos_term_x0 + tm.sin(theta) * sin_term_x0)
        dthetadx1 = k * (tm.cos(theta) * cos_term_x1 + tm.sin(theta) * sin_term_x1)
        dthetadx2 = k * (tm.cos(theta) * cos_term_x2 + tm.sin(theta) * sin_term_x2)
        dthetadx3 = k * (tm.cos(theta) * cos_term_x3 + tm.sin(theta) * sin_term_x3)

        return dthetadx0, dthetadx1, dthetadx2, dthetadx3
    
    @ti.func
    def getBendingForce(self, cs, ce, p1, p2, k, theta, crease_type, debug=False, enable_dynamic_change=False, a1=1., a2=1., L=1., id=-1):
        # 求折痕的信息
        xc = ce - cs
        # xc_norm = xc.norm()

        energy = 0.0

        # 求单元法向量
        f11 = p1 - cs
        f22 = p2 - cs
        n1 = xc.cross(f11)
        n2 = f22.cross(xc)
        # self.n1[0] = n1
        # self.n2[0] = n2

        n1_norm = n1.norm()
        n2_norm = n2.norm()

        multi_n1_n2 = n1_norm * n2_norm

        dir = n1.cross(n2).dot(xc)

        val = n1.dot(n2)

        norm_val = val / multi_n1_n2

        current_theta = 0.0
        if norm_val >= 1.0:
            val = multi_n1_n2
            norm_val = 1.0
            current_theta = 0.0
        elif norm_val <= -1.0:
            val = -multi_n1_n2
            norm_val = -1.0
            current_theta = tm.pi
        else:
            current_theta = tm.acos(norm_val)

        n_value = 0.
        backup_n_value = 0.
        signed_current_theta = 0.
       
        # 求折叠角
        if not enable_dynamic_change:
            # if index == -1:
            if crease_type == MOUNTAIN:
                if dir >= 0.:
                    self.crease_angle[id] = 1.
                    signed_current_theta = -current_theta
                    n_value = theta - signed_current_theta
                    backup_n_value = n_value
                    t11 = current_theta - self.barrier
                    t22 = tm.pi + self.collision_d - current_theta
                    if t11 >= 0:
                        n_value -= self.collision_indice * (2 * t11 * tm.log(t22 / self.barrier_left) - t11 ** 2 / t22)
                        energy += -self.collision_indice * t11 ** 2 * tm.log(t22 / self.barrier_left)
                else:
                    if id != -1 and self.previous_dir[id] >= 0 and norm_val <= -0.5: # 180~270
                        self.crease_angle[id] = 1.
                        signed_current_theta = current_theta - 2. * tm.pi
                        n_value = theta - signed_current_theta
                        backup_n_value = n_value
                        n_value -= self.barrier_maximum - (tm.pi - current_theta) * self.barrier_df_maximum
                        energy += self.barrier_energy_maximum + (-2. * self.barrier_maximum + (tm.pi - current_theta) * self.barrier_df_maximum) * (tm.pi - current_theta) * 0.5
                    else:
                        self.crease_angle[id] = -1.
                        signed_current_theta = current_theta
                        n_value = theta - signed_current_theta
                        backup_n_value = n_value
            else:
                if dir <= 0.:
                    self.crease_angle[id] = 1.
                    signed_current_theta = current_theta
                    n_value = theta - current_theta
                    backup_n_value = n_value
                    t11 = current_theta - self.barrier
                    t22 = tm.pi + self.collision_d - current_theta
                    if t11 >= 0:
                        n_value += self.collision_indice * (2 * t11 * tm.log(t22 / self.barrier_left) - t11 ** 2 / t22)
                        energy += -self.collision_indice * t11 ** 2 * tm.log(t22 / self.barrier_left)
                else:
                    if id != -1 and self.previous_dir[id] <= 0 and norm_val <= -0.5:
                        self.crease_angle[id] = 1.
                        signed_current_theta = 2. * tm.pi - current_theta
                        n_value = theta - signed_current_theta
                        backup_n_value = n_value
                        n_value += self.barrier_maximum - (tm.pi - current_theta) * self.barrier_df_maximum
                        energy += self.barrier_energy_maximum + (-2. * self.barrier_maximum + (tm.pi - current_theta) * self.barrier_df_maximum) * (tm.pi - current_theta) * 0.5
                    else:
                        self.crease_angle[id] = -1.
                        signed_current_theta = -current_theta
                        n_value = theta - signed_current_theta       
                        backup_n_value = n_value     
        else:
            if dir >= 0.:
                if id != -1 and self.previous_dir[id] <= 0 and norm_val <= -0.5:
                    if crease_type == VALLEY:
                        self.crease_angle[id] = 1.
                    else:
                        self.crease_angle[id] = -1.
                    signed_current_theta = 2. * tm.pi - current_theta
                    n_value = theta - signed_current_theta
                    backup_n_value = n_value
                    n_value += self.barrier_maximum - (tm.pi - current_theta) * self.barrier_df_maximum
                    energy += self.barrier_energy_maximum + (-2. * self.barrier_maximum + (tm.pi - current_theta) * self.barrier_df_maximum) * (tm.pi - current_theta) * 0.5
                else:
                    if id != -1:
                        if crease_type == VALLEY:
                            self.crease_angle[id] = -1.
                        else:
                            self.crease_angle[id] = 1.
                    signed_current_theta = -current_theta        
                    n_value = theta - signed_current_theta
                    backup_n_value = n_value
                    t11 = current_theta - self.barrier
                    t22 = tm.pi + self.collision_d - current_theta
                    if t11 >= 0:
                        n_value -= self.collision_indice * (2 * t11 * tm.log(t22 / self.barrier_left) - t11 ** 2 / t22)
                        energy += -self.collision_indice * t11 ** 2 * tm.log(t22 / self.barrier_left)
            else:
                if id != -1 and self.previous_dir[id] >= 0 and norm_val <= -0.5:
                    if crease_type == VALLEY:
                        self.crease_angle[id] = -1.
                    else:
                        self.crease_angle[id] = 1.
                    signed_current_theta = current_theta - 2. * tm.pi
                    n_value = theta - signed_current_theta
                    backup_n_value = n_value
                    n_value -= self.barrier_maximum - (tm.pi - current_theta) * self.barrier_df_maximum
                    energy += self.barrier_energy_maximum + (-2. * self.barrier_maximum + (tm.pi - current_theta) * self.barrier_df_maximum) * (tm.pi - current_theta) * 0.5
                else:
                    if id != -1:
                        if crease_type == VALLEY:
                            self.crease_angle[id] = 1.
                        else:
                            self.crease_angle[id] = -1.
                    signed_current_theta = current_theta
                    n_value = theta - signed_current_theta
                    backup_n_value = n_value
                    t11 = current_theta - self.barrier
                    t22 = tm.pi + self.collision_d - current_theta
                    if t11 >= 0:
                        n_value += self.collision_indice * (2 * t11 * tm.log(t22 / self.barrier_left) - t11 ** 2 / t22)
                        energy += -self.collision_indice * t11 ** 2 * tm.log(t22 / self.barrier_left)
        
        dqdx0, dqdx1, dqdx2, dqdx3 = self.getDthetaDx(cs, p2, ce, p1, signed_current_theta)
        
        # 求折叠角与目标之差
        if abs(current_theta) >= self.barrier:
            self.folding_angle_reach_pi[0] = True

        # 计算折痕等效弯曲系数
        k_crease = k * L

        #计算力
        csf = k_crease * n_value * dqdx0
        rpf2 = k_crease * n_value * dqdx1
        cef = k_crease * n_value * dqdx2
        rpf1 = k_crease * n_value * dqdx3

        #计算能量
        energy *= k_crease 
        energy += 0.5 * k_crease * backup_n_value ** 2

        return csf, cef, rpf1, rpf2, energy, dqdx0, dqdx1, dqdx2, dqdx3, current_theta / tm.pi, dir

    @ti.func
    def get_position_with_index(self, index: int):
        return self.x[index]

    @ti.func
    def get_velocity_with_index(self, index: int):
        return self.v[index]
    
    @ti.func
    def getViscousity(self, posvel, other_posvel, i, j, viscosity):
        ret = tm.vec3([0., 0., 0.])
        if abs(self.connection_matrix[i, j] - self.ori_sim.spring_k) < 1e-5:
            direction = other_posvel - posvel
            ret = viscosity * direction
        return ret

    @ti.func
    def calculateCenterPoint3DWithUnitId(self, unit_kps):
        center_accumulate = tm.vec3([0., 0., 0.])
        area_accumulate = 0.0
        for i in ti.ndrange((0, self.unit_edge_max)):
            if unit_kps[i] != -1:
                center_accumulate += self.get_position_with_index(unit_kps[i])
                area_accumulate += 1.
        return center_accumulate / area_accumulate
    
    @ti.func
    def calculateCenterPoint3DWithVerticeUnitId(self, unit_kps):
        center_accumulate = tm.vec3([0., 0., 0.])
        area_accumulate = 0.0
        for i in ti.ndrange((0, self.unit_edge_max)):
            if unit_kps[i] != -1:
                center_accumulate += self.original_vertices[unit_kps[i]]
                area_accumulate += 1.
        return center_accumulate / area_accumulate

    @ti.func
    def calculatedDs(self, x0, x1, x2, n, A):
        basic = 1. / A * (ti.Matrix.identity(data_type, 3) - n.outer_product(n) / (A ** 2))
        dndx0 = basic @ self.getSkewMatrix(x2 - x1)
        self.dDs[0, 0][0, 2] = dndx0[0, 0]
        self.dDs[0, 0][1, 2] = dndx0[1, 0]
        self.dDs[0, 0][2, 2] = dndx0[2, 0]

        self.dDs[0, 1][0, 2] = dndx0[0, 1]
        self.dDs[0, 1][1, 2] = dndx0[1, 1]
        self.dDs[0, 1][2, 2] = dndx0[2, 1]
        
        self.dDs[0, 2][0, 2] = dndx0[0, 2]
        self.dDs[0, 2][1, 2] = dndx0[1, 2]
        self.dDs[0, 2][2, 2] = dndx0[2, 2]

        dndx1 = basic @ self.getSkewMatrix(x0 - x2)
        self.dDs[1, 0][0, 2] = dndx1[0, 0]
        self.dDs[1, 0][1, 2] = dndx1[1, 0]
        self.dDs[1, 0][2, 2] = dndx1[2, 0]

        self.dDs[1, 1][0, 2] = dndx1[0, 1]
        self.dDs[1, 1][1, 2] = dndx1[1, 1]
        self.dDs[1, 1][2, 2] = dndx1[2, 1]

        self.dDs[1, 2][0, 2] = dndx1[0, 2]
        self.dDs[1, 2][1, 2] = dndx1[1, 2]
        self.dDs[1, 2][2, 2] = dndx1[2, 2]

        dndx2 = basic @ self.getSkewMatrix(x1 - x0)
        self.dDs[2, 0][0, 2] = dndx2[0, 0]
        self.dDs[2, 0][1, 2] = dndx2[1, 0]
        self.dDs[2, 0][2, 2] = dndx2[2, 0]
        
        self.dDs[2, 1][0, 2] = dndx2[0, 1]
        self.dDs[2, 1][1, 2] = dndx2[1, 1]
        self.dDs[2, 1][2, 2] = dndx2[2, 1]
        
        self.dDs[2, 2][0, 2] = dndx2[0, 2]
        self.dDs[2, 2][1, 2] = dndx2[1, 2]
        self.dDs[2, 2][2, 2] = dndx2[2, 2]

        return dndx1, dndx2

    @ti.func
    def calculateKpNumWithUnitId(self, unit_kps):
        kp_len = 0
        for i in ti.ndrange(len(unit_kps)):
            if unit_kps[i] != -1:
                kp_len += 1
        return kp_len
    
    # @ti.func
    # def getAxisForce(self, force_dir, delta_length):
    #     force = force_dir / force_dir.norm(1e-6) * self.string_params[0] * delta_length
    #     return force
    
    @ti.func
    def calculateNormalVectorWithUnitId(self, unit_kps):
        n = tm.vec3([0.0, 0.0, 0.0])
        kp_len = 0
        for i in ti.ndrange(len(unit_kps)):
            if unit_kps[i] != -1:
                kp_len += 1
        for i in ti.ndrange((1, kp_len - 1)):
            v1 = self.get_position_with_index(unit_kps[i]) - self.get_position_with_index(unit_kps[0])
            v2 = self.get_position_with_index(unit_kps[i + 1]) - self.get_position_with_index(unit_kps[0])
            n += v1.cross(v2)
        return tm.normalize(n)

    @ti.func
    def calculateUnitCenter(self):
        for i in range(self.constraint_number):
            for j in range(self.max_control_length):
                unit_id = self.unit_control[i][j]
                if unit_id != -1:
                    self.unit_center[unit_id] = self.calculateCenterPoint3DWithUnitId(self.unit_indices[unit_id])     
                else:
                    break
    
    @ti.func
    def calculateStringLength(self, i):
        self.constraint_length[i] = 0.0
        start_point = self.constraint_start_point[i]
        current_id = 0
        for k in ti.ndrange(self.max_control_length):
            if self.unit_control[i][k] != -1:
                current_id = k
                self.intersection_points[k + i * self.max_control_length] = [0., 0., 1000.]
                self.intersection_flag[i][k] = 0
                kp_id = self.unit_indices[self.unit_control[i][k]]
                end_point = self.calculateCenterPoint3DWithUnitId(kp_id)
                force_dir = end_point - start_point
                unit_kp_num = self.calculateKpNumWithUnitId(kp_id)
                hole_direction = self.hole_dir[i][k]
                nm = self.calculateNormalVectorWithUnitId(kp_id)
                penetration = force_dir.dot(nm) * hole_direction / force_dir.norm(1e-6)
                # print(penetration)
                # if 0: #penetration
                if penetration > 0: #penetration
                    facet_force_dir = force_dir - force_dir.dot(nm) * nm
                    outer_point = end_point - facet_force_dir / facet_force_dir.norm(1e-6) * self.max_size
                    for l in range(unit_kp_num):
                        cur_x = self.get_position_with_index(kp_id[l])
                        next_x = self.get_position_with_index(kp_id[(l + 1) % unit_kp_num])
                        flag1 = (end_point - cur_x).cross(outer_point - cur_x)
                        flag2 = (end_point - next_x).cross(outer_point - next_x)
                        flag3 = (end_point - next_x).cross(end_point - cur_x)
                        flag4 = (outer_point - next_x).cross(outer_point - cur_x)
                        facet_force_dir_norm = tm.normalize(facet_force_dir)
                        if flag1.dot(flag2) < 0 and flag3.dot(flag4) < 0:
                            v1 = next_x - end_point
                            v2 = cur_x - end_point
                            vertical1 = (v1 - v1.dot(facet_force_dir_norm) * facet_force_dir_norm)
                            vertical2 = (v2 - v2.dot(facet_force_dir_norm) * facet_force_dir_norm)
                            if vertical1.dot(vertical2) < 0:
                                t = vertical2.norm() / (vertical1.norm()  + vertical2.norm())
                                intersect = cur_x + (next_x - cur_x) * t
                                r = (intersect - end_point)
                                k1 = v2.dot(r) / r.norm() ** 2
                                k2 = v1.dot(r) / r.norm() ** 2
                                self.intersection_points[k + i * self.max_control_length] = intersect
                                self.intersection_infos[k + i * self.max_control_length] = ti.Vector([t, k1, k2, l, (l + 1) % unit_kp_num])
                                self.intersection_flag[i][k] = 1
                                length = (intersect - start_point).norm() + (intersect - end_point).norm()
                                self.constraint_length[i] += length
                                self.current_length_per_string[i][k] = length
                                break
                else:
                    length = (end_point - start_point).norm()
                    self.constraint_length[i] += length
                    self.current_length_per_string[i][k] = length
                start_point = end_point
            else:
                if self.constraint_end_point_existence[i]:
                    # print("start ok")
                    index = self.max_control_length - 1
                    self.intersection_points[index + i * self.max_control_length] = [0., 0., 1000.]
                    self.intersection_flag[i][index] = 0
                    kp_id = self.unit_indices[self.unit_control[i][current_id]]
                    end_point = self.constraint_end_point[i]
                    force_dir = end_point - start_point
                    unit_kp_num = self.calculateKpNumWithUnitId(kp_id)
                    hole_direction = self.hole_dir[i][current_id]
                    nm = self.calculateNormalVectorWithUnitId(kp_id)
                    penetration = force_dir.dot(nm) * hole_direction / force_dir.norm(1e-6)
                    # print(penetration)
                    # if 0: #penetration
                    if penetration > 0: #penetration
                        facet_force_dir = force_dir - force_dir.dot(nm) * nm
                        outer_point = start_point + facet_force_dir / facet_force_dir.norm(1e-6) * self.max_size
                        for l in range(unit_kp_num):
                            cur_x = self.get_position_with_index(kp_id[l])
                            next_x = self.get_position_with_index(kp_id[(l + 1) % unit_kp_num])
                            flag1 = (start_point - cur_x).cross(outer_point - cur_x)
                            flag2 = (start_point - next_x).cross(outer_point - next_x)
                            flag3 = (start_point - next_x).cross(start_point - cur_x)
                            flag4 = (outer_point - next_x).cross(outer_point - cur_x)
                            facet_force_dir_norm = tm.normalize(facet_force_dir)
                            if flag1.dot(flag2) < 0 and flag3.dot(flag4) < 0:
                                v1 = next_x - start_point
                                v2 = cur_x - start_point
                                vertical1 = (v1 - v1.dot(facet_force_dir_norm) * facet_force_dir_norm)
                                vertical2 = (v2 - v2.dot(facet_force_dir_norm) * facet_force_dir_norm)
                                if vertical1.dot(vertical2) < 0:
                                    t = vertical2.norm() / (vertical1.norm()  + vertical2.norm())
                                    intersect = cur_x + (next_x - cur_x) * t
                                    r = (intersect - start_point)
                                    k1 = v2.dot(r) / r.norm() ** 2
                                    k2 = v1.dot(r) / r.norm() ** 2
                                    self.intersection_points[index + i * self.max_control_length] = intersect
                                    self.intersection_infos[index + i * self.max_control_length] = ti.Vector([t, k1, k2, l, (l + 1) % unit_kp_num])
                                    self.intersection_flag[i][index] = 1
                                    length = (intersect - start_point).norm() + (intersect - end_point).norm()
                                    self.constraint_length[i] += length
                                    self.current_length_per_string[i][index] = length
                                    # print("end intersection ok")
                                    break
                    else:        
                        length = (self.constraint_end_point[i] - start_point).norm()
                        self.constraint_length[i] += length
                        self.current_length_per_string[i][index] = length
                        # print("end ok")
                break

    @ti.kernel
    def backupStringLength(self):
        for i in ti.ndrange(self.constraint_number):
            self.initial_length_per_string[i] = self.current_length_per_string[i]
    
    @ti.func
    def calculateConstraintPoint(self):
        for i in ti.ndrange(self.constraint_number):
            candidate_id = self.constraint_start_point_candidate_id[i]
            unit_id = self.constraint_start_point_candidate_connection[candidate_id]
            print(unit_id)
            if unit_id >= 0:
                unit_indice = self.unit_indices[unit_id]
                height = self.constraint_height[candidate_id]
                center = self.calculateCenterPoint3DWithUnitId(unit_indice)
                n = self.calculateNormalVectorWithUnitId(unit_indice)
                self.constraint_start_point[i] = center + n * height
                print(self.constraint_start_point[i])
            if self.constraint_end_point_existence[i]:
                candidate_id = self.constraint_end_point_candidate_id[i]
                unit_id = self.constraint_start_point_candidate_connection[candidate_id]
                if unit_id >= 0:
                    unit_indice = self.unit_indices[unit_id]
                    height = self.constraint_height[candidate_id]
                    center = self.calculateCenterPoint3DWithUnitId(unit_indice)
                    n = self.calculateNormalVectorWithUnitId(unit_indice)
                    self.constraint_end_point[i] = center + n * height
                    # print(self.constraint_end_point[i])

    @ti.func
    def getEnergy(self, sim_mode, theta, gravitational_acc, facet_k, enable_ground):
        total_energy = 0.0
        for i in ti.ndrange(self.div_indices_num):
            x0 = self.get_position_with_index(self.indices[3 * i])
            x1 = self.get_position_with_index(self.indices[3 * i + 1])
            x2 = self.get_position_with_index(self.indices[3 * i + 2])
            n = (x1 - x0).cross(x2 - x0)
            # print(i, n)
            ds = ti.Matrix.cols([x1 - x0, x2 - x0, n / n.norm(1e-6)])
            f = ds @ self.dm[i]
            # print(i, f)
            green_tensor = 0.5 * (f.transpose() @ f - ti.Matrix.identity(data_type, 3))
            energy_tensor = 0.0
            for j, k in ti.ndrange(3, 3):
                energy_tensor += green_tensor[j, k] ** 2
            psi = self.lames_bonus[0] * energy_tensor + self.lames_bonus[1] * 0.5 * ti.Matrix.trace(green_tensor) ** 2
            total_energy += psi * self.A[i]
        if self.print:
            print(f"After stvk calculate energy function: energy: {total_energy}")
        if sim_mode == self.FOLD_SIM:
            for i in ti.ndrange(self.crease_pairs_num):
                crease_start_index = self.crease_pairs[i, 0]
                crease_end_index = self.crease_pairs[i, 1]
                related_p1 = self.bending_pairs[i, 0]
                related_p2 = self.bending_pairs[i, 1]

                target_folding_angle = 0.0
                percent_low = (self.sequence_level[0] - self.crease_level[i]) / (self.sequence_level[0] - self.sequence_level[1] + 1.)
                percent_high = (self.sequence_level[0] - self.crease_level[i] + 1.) / (self.sequence_level[0] - self.sequence_level[1] + 1.)
                percent_theta = abs(theta) / tm.pi
                if percent_theta < percent_low:
                    target_folding_angle = 0.0
                elif percent_theta > percent_high:
                    target_folding_angle = tm.pi
                else:
                    coeff = self.crease_coeff[i]
                    target_folding_angle = (percent_theta - percent_low) / (percent_high - percent_low) * tm.pi
                    target_folding_angle = 2. * tm.atan2(coeff * tm.tan(target_folding_angle * 0.5), 1.)
                if self.crease_type[i]:
                    target_folding_angle = -target_folding_angle
            
                energy = self.getBendingEnergy(
                    self.get_position_with_index(crease_start_index), self.get_position_with_index(crease_end_index), 
                    self.get_position_with_index(related_p1), self.get_position_with_index(related_p2),
                    self.bending_param[0], target_folding_angle, self.crease_type[i], False, False, 
                    self.bending_pairs_area[i, 0], self.bending_pairs_area[i, 1], self.crease_initial_length[i], i
                )
                # if i == 0:
                #     print(f"E, id:{i}, energy:{energy}, previous_dir:{self.previous_dir[i]}, angle:{self.crease_angle[i]}, target:{target_folding_angle}")
                total_energy += energy

        else:
            for i in ti.ndrange(self.crease_pairs_num):
                energy = self.getBendingEnergy(
                    self.get_position_with_index(self.crease_pairs[i, 0]), self.get_position_with_index(self.crease_pairs[i, 1]), 
                    self.get_position_with_index(self.bending_pairs[i, 0]), self.get_position_with_index(self.bending_pairs[i, 1]),
                    self.bending_param[0], 0.0, self.crease_type[i], False, True, 
                    self.bending_pairs_area[i, 0], self.bending_pairs_area[i, 1], self.crease_initial_length[i], i
                )
                total_energy += energy

        if self.print:
            print(f"After bending calculate energy function: energy: {total_energy}")

        if self.facet_bending_pairs_num > 0:
            for i in ti.ndrange(self.facet_crease_pairs_num):
                energy = self.getBendingEnergy(
                    self.get_position_with_index(self.facet_crease_pairs[i, 0]), self.get_position_with_index(self.facet_crease_pairs[i, 1]), 
                    self.get_position_with_index(self.facet_bending_pairs[i, 0]), self.get_position_with_index(self.facet_bending_pairs[i, 1]), 
                    facet_k, 0.0, 0, False, True, 
                    self.facet_bending_pairs_area[i, 0], self.facet_bending_pairs_area[i, 1], self.facet_crease_initial_length[i]
                )
                total_energy += energy

        if self.print:
            print(f"After facet calculate energy function: energy: {total_energy}")

        if sim_mode:
            if enable_ground:
                c = self.epsilon_v[0]
                for i in ti.ndrange(self.kp_num):
                    if self.x[i][Z] < self.ground_barrier:
                        barrier_exceed = self.x[i][Z] - self.ground_barrier
                        barrier_left = self.x[i][Z] + self.collision_d
                        if self.x[i][Z] > 0.:
                            total_energy += -self.ground_collision_indice * barrier_exceed ** 2 * tm.log(barrier_left / self.ground_barrier)
                            # self.ground_force[i] = self.ground_collision_indice * (2 * barrier_exceed * tm.log(barrier_left / self.ground_barrier) + barrier_exceed ** 2 / barrier_left) * tm.vec3([0., 0., 1.])
                        else:
                            total_energy += self.ground_barrier_energy_maximum - (2. * self.ground_force_maximum + self.x[i][Z] * self.df_ground_force_maximum) * self.x[i][Z] * 0.5
                            # self.ground_force[i] = (self.ground_force_maximum + self.x[i][Z] * self.df_ground_force_maximum) * tm.vec3([0., 0., 1.])
                        if self.print:
                            print(f"After {i} ground calculate energy function: energy: {total_energy}")
                        self.u[i] = [self.x[i][X] - self.back_up_x[i][X], self.x[i][Y] - self.back_up_x[i][Y]]
                        N = self.ground_force[i][Z]
                        ui_norm = self.u[i].norm()
                        total_energy += self.ground_miu * N * self.friction_f0(ui_norm, c)
                        if self.print:
                            print(f"After {i} ground friction calculate energy function: energy: {total_energy}, {c}, {N}, {ui_norm}")

            if self.P_number > 0:
                self.calculateUnitCenter()
                self.calculateConstraintPoint()

                for i in ti.ndrange(self.constraint_number):
                    self.calculateStringLength(i)
                    delta_length = self.constraint_length[i] - self.constraint_initial_length[i] + self.string_length_decrease[i]
                    print(delta_length)
                    if delta_length > 0:
                        total_energy += 0.5 * self.string_params[0] * delta_length ** 2
                        for j in ti.ndrange(self.max_control_length):
                            if self.unit_control[i][j] != -1:
                                current_delta_length = self.current_length_per_string[i][j] - self.initial_length_per_string[i][j]
                                total_energy += 0.5 * self.miu * (current_delta_length) ** 2

                            else: #end
                                if self.constraint_end_point_existence[i]:
                                    index = self.max_control_length - 1
                                    current_delta_length = self.current_length_per_string[i][index] - self.initial_length_per_string[i][index]
                                    total_energy += 0.5 * self.miu * (current_delta_length) ** 2
                    
        if self.print:
            print(f"After string calculate energy function: energy: {total_energy}")

        for i in ti.ndrange(self.kp_num):
            total_energy += 0.5 * self.masses[i] * (self.get_velocity_with_index(i).norm()) ** 2 - self.masses[i] * gravitational_acc[Z] * self.get_position_with_index(i)[Z]
        
        if self.print:
            print(f"After merge energy function: energy: {total_energy}")

        return total_energy

    @ti.func
    def stvkForce(self):
        for i in ti.ndrange(self.div_indices_num):
            i0 = self.indices[3 * i]
            i1 = self.indices[3 * i + 1]
            i2 = self.indices[3 * i + 2]

            x0 = self.get_position_with_index(i0)
            x1 = self.get_position_with_index(i1)
            x2 = self.get_position_with_index(i2)
            n = (x1 - x0).cross(x2 - x0)
            dndx1, dndx2 = self.calculatedDs(x0, x1, x2, n, n.norm(1e-6))

            ds = ti.Matrix.cols([x1 - x0, x2 - x0, n / n.norm(1e-6)])
            f = ds @ self.dm[i]

            #stvk model
            green_tensor = 0.5 * (f.transpose() @ f - ti.Matrix.identity(data_type, 3))
            energy_tensor = 0.0
            for j, k in ti.ndrange(3, 3):
                energy_tensor += green_tensor[j, k] ** 2
            psi = self.lames_bonus[0] * energy_tensor + self.lames_bonus[1] * 0.5 * ti.Matrix.trace(green_tensor) ** 2

            #force
            piola_temp = 2.0 * self.lames_bonus[0] * green_tensor + self.lames_bonus[1] * ti.Matrix.trace(green_tensor) * ti.Matrix.identity(data_type, 3)
            piola = f @ piola_temp 
            dm = ti.Vector([self.dm[i][2, 0], self.dm[i][2, 1], self.dm[i][2, 2]])
            H = -self.A[i] * (piola @ self.dm[i].transpose() + ti.Matrix.cols([dndx1.transpose() @ piola @ dm, dndx2.transpose() @ piola @ dm, [0., 0., 0.]]))

            # if i == 0:
            #     print(H)

            f1 = tm.vec3([H[0, 0], H[1, 0], H[2, 0]])
            f2 = tm.vec3([H[0, 1], H[1, 1], H[2, 1]])
            f0 = -f1 - f2

            self.stvk_force[i0] += f0
            self.stvk_force[i1] += f1
            self.stvk_force[i2] += f2

            # print(i, f0, f1, f2)

            # derivative
            
            for j, k in ti.ndrange(3, 3):
                dF = self.dDs[j, k] @ self.dm[i]
                dE = 0.5 * (dF.transpose() @ f + f.transpose() @ dF)
                dP = dF @ piola_temp + f @ (2.0 * self.lames_bonus[0] * dE + self.lames_bonus[1] * ti.Matrix.trace(dE) * ti.Matrix.identity(data_type, 3))
                dH = -self.A[i] * (dP @ self.dm[i].transpose() + ti.Matrix.cols([dndx1.transpose() @ dP @ dm, dndx2.transpose() @ dP @ dm, [0., 0., 0.]]))
                df1 = tm.vec3([dH[0, 0], dH[1, 0], dH[2, 0]])
                df2 = tm.vec3([dH[0, 1], dH[1, 1], dH[2, 1]])
                df0 = -df1 - df2
                # if i == 0:
                #     print(f"{j}, {k}, {dH}")
                for l in ti.static(range(3)):
                    self.K_element[i][0 + l, 3 * j + k] = df0[l]
                    self.K_element[i][3 + l, 3 * j + k] = df1[l]
                    self.K_element[i][6 + l, 3 * j + k] = df2[l]
                
            self.triplets[i] = [3 * i0, 3 * i1, 3 * i2]
            self.total_energy[0] += psi * self.A[i]

    @ti.func
    def calculateTargetAngle(self, i, theta):
        target_folding_angle = 0.0
        percent_low = (self.sequence_level[0] - self.crease_level[i]) / (self.sequence_level[0] - self.sequence_level[1] + 1.)
        percent_high = (self.sequence_level[0] - self.crease_level[i] + 1.) / (self.sequence_level[0] - self.sequence_level[1] + 1.)
        percent_theta = abs(theta) / tm.pi

        if percent_theta < percent_low:
            target_folding_angle = 0.0
        elif percent_theta > percent_high:
            target_folding_angle = tm.pi
        else:
            coeff = self.crease_coeff[i]
            target_folding_angle = (percent_theta - percent_low) / (percent_high - percent_low) * tm.pi
            target_folding_angle = 2. * tm.atan2(coeff * tm.tan(target_folding_angle * 0.5), 1.)

        if self.crease_type[i]:
            target_folding_angle = -target_folding_angle
        
        return target_folding_angle
    
    @ti.func
    def fillBendingMatrix(self, i, total_k, dqdx0, dqdx1, dqdx2, dqdx3):
        block_00 = total_k * (dqdx0.outer_product(dqdx0))
        block_01 = total_k * (dqdx0.outer_product(dqdx1))
        block_02 = total_k * (dqdx0.outer_product(dqdx2))
        block_03 = total_k * (dqdx0.outer_product(dqdx3))

        block_10 = total_k * (dqdx1.outer_product(dqdx0))
        block_11 = total_k * (dqdx1.outer_product(dqdx1))
        block_12 = total_k * (dqdx1.outer_product(dqdx2))
        block_13 = total_k * (dqdx1.outer_product(dqdx3))

        block_20 = total_k * (dqdx2.outer_product(dqdx0))
        block_21 = total_k * (dqdx2.outer_product(dqdx1))
        block_22 = total_k * (dqdx2.outer_product(dqdx2))
        block_23 = total_k * (dqdx2.outer_product(dqdx3))

        block_30 = total_k * (dqdx3.outer_product(dqdx0))
        block_31 = total_k * (dqdx3.outer_product(dqdx1))
        block_32 = total_k * (dqdx3.outer_product(dqdx2))
        block_33 = total_k * (dqdx3.outer_product(dqdx3))

        for j, k in ti.ndrange(3, 3):
            self.K_element_bending[i][j, k + 0] = block_00[j, k]
            self.K_element_bending[i][j, k + 3] = block_01[j, k]
            self.K_element_bending[i][j, k + 6] = block_02[j, k]
            self.K_element_bending[i][j, k + 9] = block_03[j, k]

            self.K_element_bending[i][j + 3, k + 0] = block_10[j, k]
            self.K_element_bending[i][j + 3, k + 3] = block_11[j, k]
            self.K_element_bending[i][j + 3, k + 6] = block_12[j, k]
            self.K_element_bending[i][j + 3, k + 9] = block_13[j, k]

            self.K_element_bending[i][j + 6, k + 0] = block_20[j, k]
            self.K_element_bending[i][j + 6, k + 3] = block_21[j, k]
            self.K_element_bending[i][j + 6, k + 6] = block_22[j, k]
            self.K_element_bending[i][j + 6, k + 9] = block_23[j, k]

            self.K_element_bending[i][j + 9, k + 0] = block_30[j, k]
            self.K_element_bending[i][j + 9, k + 3] = block_31[j, k]
            self.K_element_bending[i][j + 9, k + 6] = block_32[j, k]
            self.K_element_bending[i][j + 9, k + 9] = block_33[j, k]

    @ti.func
    def bendingForceFoldSim(self, theta: data_type):
        for i in ti.ndrange(self.crease_pairs_num):
            crease_start_index = self.crease_pairs[i, 0]
            crease_end_index = self.crease_pairs[i, 1]
            related_p1 = self.bending_pairs[i, 0]
            related_p2 = self.bending_pairs[i, 1]

            target_folding_angle = self.calculateTargetAngle(i, theta)

            csf, cef, rpf1, rpf2, energy, dqdx0, dqdx1, dqdx2, dqdx3, n_value, dir = self.getBendingForce(
                self.get_position_with_index(crease_start_index), self.get_position_with_index(crease_end_index), 
                self.get_position_with_index(related_p1), self.get_position_with_index(related_p2),
                self.bending_param[0], target_folding_angle, self.crease_type[i], False, False, 
                self.bending_pairs_area[i, 0], self.bending_pairs_area[i, 1], self.crease_initial_length[i], i
            )
            # 增加至force
            self.bending_force[crease_start_index] += csf
            self.bending_force[crease_end_index] += cef
            self.bending_force[related_p1] += rpf1
            self.bending_force[related_p2] += rpf2

            self.crease_angle[i] *= n_value
            self.total_energy[0] += energy

            # print(f"F, id:{i}, energy:{energy}, previous_dir:{self.previous_dir[i]}, angle:{self.crease_angle[i]}, target:{target_folding_angle}")

            self.triplets_bending[i] = [3 * crease_start_index, 3 * related_p2, 3 * crease_end_index, 3 * related_p1]
            
            bending_k = -self.bending_param[0] * self.crease_initial_length[i]

            barrier_k = 0.
            barrier_exceed = n_value * tm.pi - self.barrier
            barrier_left = tm.pi + self.collision_d - n_value * tm.pi

            if (self.crease_type[i] == VALLEY):
                if dir <= 0: # 0~180
                    if barrier_exceed > 0.:
                        barrier_k = bending_k * self.collision_indice * 2. * (barrier_exceed ** 2 / (2. * barrier_left) ** 2 + 2. * barrier_exceed / barrier_left - tm.log(barrier_left / barrier_exceed))
                    self.previous_dir[i] = dir 
                else:
                    if self.previous_dir[i] <= 0 and n_value >= 0.5: # collision, 180~270
                        barrier_k = bending_k * self.barrier_df_maximum
                    else: # -90~0
                        self.previous_dir[i] = dir 
            else:
                if dir >= 0: #-180~0
                    if barrier_exceed > 0.:
                        barrier_k = bending_k * self.collision_indice * 2. * (barrier_exceed ** 2 / (2. * barrier_left) ** 2 + 2. * barrier_exceed / barrier_left - tm.log(barrier_left / barrier_exceed))
                    self.previous_dir[i] = dir  
                else:
                    if self.previous_dir[i] >= 0 and n_value >= 0.5: # collision, -270~-180
                        barrier_k = bending_k * self.barrier_df_maximum
                    else: # 0~90
                        self.previous_dir[i] = dir 

            total_k = bending_k + barrier_k

            self.fillBendingMatrix(i, total_k, dqdx0, dqdx1, dqdx2, dqdx3)

    @ti.func
    def bendingForceTSASim(self):
        for i in ti.ndrange(self.crease_pairs_num):
            crease_start_index = self.crease_pairs[i, 0]
            crease_end_index = self.crease_pairs[i, 1]
            related_p1 = self.bending_pairs[i, 0]
            related_p2 = self.bending_pairs[i, 1]

            csf, cef, rpf1, rpf2, energy, dqdx0, dqdx1, dqdx2, dqdx3, n_value, dir = self.getBendingForce(
                self.get_position_with_index(crease_start_index), self.get_position_with_index(crease_end_index), 
                self.get_position_with_index(related_p1), self.get_position_with_index(related_p2),
                self.bending_param[0], 0.0, self.crease_type[i], False, True, 
                self.bending_pairs_area[i, 0], self.bending_pairs_area[i, 1], self.crease_initial_length[i], i
            )
            
            # 增加至force
            self.bending_force[crease_start_index] += csf
            self.bending_force[crease_end_index] += cef
            self.bending_force[related_p1] += rpf1
            self.bending_force[related_p2] += rpf2
        
            self.crease_angle[i] *= n_value
            self.total_energy[0] += energy

            self.triplets_bending[i] = [3 * crease_start_index, 3 * related_p2, 3 * crease_end_index, 3 * related_p1]

            bending_k = -self.bending_param[0] * self.crease_initial_length[i]

            barrier_k = 0.
            barrier_exceed = n_value * tm.pi - self.barrier
            barrier_left = tm.pi + self.collision_d - n_value * tm.pi

            if (dir < 0):
                if self.previous_dir[i] >= 0 and n_value >= 0.5: # collision
                    barrier_k = bending_k * self.barrier_df_maximum
                else:
                    if barrier_exceed > 0.:
                        barrier_k = bending_k * self.collision_indice * 2. * (barrier_exceed ** 2 / (2. * barrier_left) ** 2 + 2. * barrier_exceed / barrier_left - tm.log(barrier_left / barrier_exceed))
                    self.previous_dir[i] = dir 
            else:
                if self.previous_dir[i] < 0 and n_value >= 0.5: # collision
                    barrier_k = bending_k * self.barrier_df_maximum
                else:
                    if barrier_exceed > 0.:
                        barrier_k = bending_k * self.collision_indice * 2. * (barrier_exceed ** 2 / (2. * barrier_left) ** 2 + 2. * barrier_exceed / barrier_left - tm.log(barrier_left / barrier_exceed))
                    self.previous_dir[i] = dir 

            total_k = bending_k + barrier_k

            self.fillBendingMatrix(i, total_k, dqdx0, dqdx1, dqdx2, dqdx3)
    
    @ti.func
    def facetBendingForce(self, facet_k: data_type):
        for i in ti.ndrange(self.facet_crease_pairs_num):
            crease_start_index = self.facet_crease_pairs[i, 0]
            crease_end_index = self.facet_crease_pairs[i, 1]
            related_p1 = self.facet_bending_pairs[i, 0]
            related_p2 = self.facet_bending_pairs[i, 1]

            csf, cef, rpf1, rpf2, energy, dqdx0, dqdx1, dqdx2, dqdx3, n_value, dir = self.getBendingForce(
                self.get_position_with_index(crease_start_index), self.get_position_with_index(crease_end_index),
                self.get_position_with_index(related_p1), self.get_position_with_index(related_p2),
                facet_k, 0.0, 0, False, True, 
                self.facet_bending_pairs_area[i, 0], self.facet_bending_pairs_area[i, 1], self.facet_crease_initial_length[i]
            )
            
            # 增加至force
            self.facet_bending_force[crease_start_index] += csf
            self.facet_bending_force[crease_end_index] += cef
            self.facet_bending_force[related_p1] += rpf1
            self.facet_bending_force[related_p2] += rpf2

            self.total_energy[0] += energy

            index = i + self.bending_pairs_num
            self.triplets_bending[index] = [3 * crease_start_index, 3 * related_p2, 3 * crease_end_index, 3 * related_p1]

            bending_k = -facet_k * self.facet_crease_initial_length[i]

            total_k = bending_k

            self.fillBendingMatrix(index, total_k, dqdx0, dqdx1, dqdx2, dqdx3)
    
    @ti.func
    def viscosityForce(self, viscosity: data_type):
        for i, j in ti.ndrange(self.kp_num, self.kp_num):
            self.viscosity_force[i] += self.getViscousity(self.get_velocity_with_index(i), self.get_velocity_with_index(j), i, j, viscosity)

    @ti.func
    def getStringEnergy(self, delta_length, epsilon):
        value = 0.5 * delta_length ** 2
        if delta_length < epsilon:
            value = -delta_length ** 4 / (12. * epsilon ** 2) + delta_length ** 3 / (3. * epsilon) + epsilon * delta_length / 3. - epsilon ** 2 / 12.
        return self.string_params[0] * value

    @ti.func
    def getStringForce(self, delta_length, epsilon):
        value = delta_length
        if delta_length < epsilon:
            value = -delta_length ** 3 / (3. * epsilon ** 2) + delta_length ** 2 / epsilon + epsilon / 3.
        return self.string_params[0] * value
    
    @ti.func
    def getStringDF(self, delta_length, epsilon):
        value = 1.
        if delta_length < epsilon:
            value = -delta_length ** 2 / (epsilon ** 2) + 2 * delta_length / epsilon
        return tm.sqrt(self.string_params[0] * value)
    
    @ti.func
    def stringForce(self):
        # for i, j in ti.ndrange(self.constraint_number, self.kp_num):
        #     self.dldx_force[i, j] = [0., 0., 0.]
        self.exist_grad_number[0] = 0
        self.calculateUnitCenter()
        self.calculateConstraintPoint()
        for i in ti.ndrange(self.constraint_number):
            self.calculateStringLength(i)
            delta_length = self.constraint_length[i] - self.constraint_initial_length[i] + self.string_length_decrease[i]
            # print(delta_length)
            if delta_length > 0: # add force
                # print(i)   
                # self.delta_length_of_string[0] = delta_length
                self.total_energy[0] += self.getStringEnergy(delta_length, self.epsilon_string[0])
                force = self.getStringForce(delta_length, self.epsilon_string[0])
                bonus = self.getStringDF(delta_length, self.epsilon_string[0])

                # current_delta_length = self.current_length_per_string[i][0] - self.initial_length_per_string[i][0]
                # friction_force = self.miu * current_delta_length
                
                start_point = self.constraint_start_point[i]
                candidate_id = self.constraint_start_point_candidate_id[i]
                unit_connection_id = self.constraint_start_point_candidate_connection[candidate_id]
                if unit_connection_id >= 0:
                    kp_id = self.unit_indices[unit_connection_id]
                    unit_kp_num = self.calculateKpNumWithUnitId(kp_id)
                    center = self.calculateCenterPoint3DWithUnitId(kp_id)
                    if self.intersection_flag[i][0]:
                        center = self.intersection_points[i * self.max_control_length]
                    f_basic = (1. / unit_kp_num * tm.normalize(center - start_point))
                    for k in range(unit_kp_num):
                        self.string_force[kp_id[k]] += f_basic * (force + friction_force)
                        self.dldx_force[i, kp_id[k]] += f_basic * bonus
                        # self.dldx_friction_force[i, 0, kp_id[k]] += f_basic * bonus

                # print(f"{i}, force: {force}")

                for j in ti.ndrange(self.max_control_length):
                    if self.unit_control[i][j] != -1:
                        current_delta_length = self.current_length_per_string[i][j] - self.initial_length_per_string[i][j]
                        friction_force = self.miu * current_delta_length
                        self.total_energy[0] += 0.5 * self.miu * (current_delta_length) ** 2

                        kp_id = self.unit_indices[self.unit_control[i][j]]
                        unit_kp_num = self.calculateKpNumWithUnitId(kp_id)
                        hole_direction = self.hole_dir[i][j]
                        nm = self.calculateNormalVectorWithUnitId(kp_id)
                        end_point = self.calculateCenterPoint3DWithUnitId(kp_id)
                        force_dir = end_point - start_point
                        # axis_force = tm.vec3([0., 0., 0.])
                            
                        if j == 0: # 1 point
                            self.outer_P[self.exist_grad_number[0]] = start_point
                            self.outer_PI[self.exist_grad_number[0]] = start_point

                            penetration = force_dir.dot(nm) * hole_direction / force_dir.norm(1e-6)
                            # print(f"{i}, {j}, penetration: {penetration}")
                            if penetration >= -self.beta: #penetration
                                if penetration > 0:
                                    t = self.intersection_infos[j + i * self.max_control_length][0]
                                    intersect = self.intersection_points[j + i * self.max_control_length]
                                    self.outer_PI[self.exist_grad_number[0]] = intersect
                                    
                                    index1 = int(self.intersection_infos[j + i * self.max_control_length][3])
                                    index2 = int(self.intersection_infos[j + i * self.max_control_length][4])

                                    id1 = kp_id[index1]
                                    id2 = kp_id[index2]
                                        
                                    f_basic = (tm.normalize(start_point - self.outer_PI[self.exist_grad_number[0]]) + tm.normalize(end_point - self.outer_PI[self.exist_grad_number[0]]))
                                    f1 = (1 - t) * f_basic
                                    f2 = t * f_basic

                                    self.string_force[id1] += f1 * (force + friction_force)
                                    self.string_force[id2] += f2 * (force + friction_force)
                                    self.dldx_force[i, id1] += f1 * bonus
                                    self.dldx_force[i, id2] += f2 * bonus

                                    # self.dldx_friction_force[i, j, id1] += f1 * bonus
                                    # self.dldx_friction_force[i, j, id2] += f2 * bonus

                            f_basic = 1. / unit_kp_num * tm.normalize(self.outer_PI[self.exist_grad_number[0]] - end_point) 
                            for k in range(unit_kp_num):
                                self.string_force[kp_id[k]] += f_basic * (force + friction_force)
                                self.dldx_force[i, kp_id[k]] += f_basic * bonus
                                # self.dldx_friction_force[i, j, kp_id[k]] += f_basic
                               
                            self.outer_Q[self.exist_grad_number[0]] = end_point
                            self.outer_QI[self.exist_grad_number[0]] = end_point
                            
                            self.exist_grad_number[0] += 1

                        else:
                            self.outer_P[self.exist_grad_number[0]] = start_point
                            self.outer_PI[self.exist_grad_number[0]] = start_point
                            self.outer_Q[self.exist_grad_number[0] - 1] = end_point
                            self.outer_QI[self.exist_grad_number[0] - 1] = end_point

                            before_unit_id = self.unit_control[i][j - 1]
                            before_kp_id = self.unit_indices[before_unit_id]
                            before_kp_num = self.calculateKpNumWithUnitId(before_kp_id)
                            before_n = self.calculateNormalVectorWithUnitId(before_kp_id)

                            penetration = force_dir.dot(nm) * hole_direction / force_dir.norm(1e-6)
                            # print(f"{i}, {j}, penetration: {penetration}")
                            if penetration >= -self.beta: #penetration
                                tight_indice = (penetration + self.beta) / self.beta

                            if penetration > 0: #与前一块板有穿透
                                t = self.intersection_infos[j + i * self.max_control_length][0]
                                # print(t)
                                intersect = self.intersection_points[j + i * self.max_control_length]
                                self.outer_PI[self.exist_grad_number[0]] = intersect
                                self.outer_QI[self.exist_grad_number[0] - 1] = intersect
                                
                                index1 = int(self.intersection_infos[j + i * self.max_control_length][3])
                                index2 = int(self.intersection_infos[j + i * self.max_control_length][4])

                                id1 = kp_id[index1]
                                id2 = kp_id[index2]

                                f_basic = (tm.normalize(start_point - self.outer_PI[self.exist_grad_number[0]]) + tm.normalize(end_point - self.outer_PI[self.exist_grad_number[0]]))
                                # print(f"fbasic: {f_basic}")
                                f1 = (1 - t) * f_basic
                                f2 = t * f_basic

                                self.string_force[id1] += f1 * (force + friction_force)
                                self.string_force[id2] += f2 * (force + friction_force)
                                self.dldx_force[i, id1] += f1 * bonus
                                self.dldx_force[i, id2] += f2 * bonus
                                # self.dldx_friction_force[i, j, id1] += f1
                                # self.dldx_friction_force[i, j, id2] += f2


                            f_basic = 1. / before_kp_num * tm.normalize(self.outer_QI[self.exist_grad_number[0] - 1] - start_point)
                            
                            for k in range(before_kp_num):
                                self.string_force[before_kp_id[k]] += f_basic * (force + friction_force)
                                self.dldx_force[i, before_kp_id[k]] += f_basic * bonus
                                # self.dldx_friction_force[i, j, before_kp_id[k]] += f_basic

                            f_basic = 1. / unit_kp_num * tm.normalize(self.outer_PI[self.exist_grad_number[0]] - end_point) 
                            # print(f"fbasic: {f_basic}")
                            for k in range(unit_kp_num):
                                self.string_force[kp_id[k]] += f_basic * (force + friction_force)
                                self.dldx_force[i, kp_id[k]] += f_basic * bonus
                                # self.dldx_friction_force[i, j, kp_id[k]] += f_basic

                            self.outer_Q[self.exist_grad_number[0]] = end_point
                            self.outer_QI[self.exist_grad_number[0]] = end_point
                        
                            self.exist_grad_number[0] += 1
                                
                        start_point = end_point

                    else: #end
                        if self.constraint_end_point_existence[i]:
                            index = self.max_control_length - 1
                            # current_delta_length = self.current_length_per_string[i][index] - self.initial_length_per_string[i][index]
                            # friction_force = self.miu * current_delta_length
                            # self.total_energy[0] += 0.5 * self.miu * (current_delta_length) ** 2

                            # print("have end")
                            self.outer_Q[self.exist_grad_number[0] - 1] = self.constraint_end_point[i]
                            self.outer_QI[self.exist_grad_number[0] - 1] = self.constraint_end_point[i]

                            before_unit_id = self.unit_control[i][j - 1]
                            force_dir = self.constraint_end_point[i] - start_point
                            
                            before_kp_id = self.unit_indices[before_unit_id]
                            before_kp_num = self.calculateKpNumWithUnitId(before_kp_id)
                            before_n = self.calculateNormalVectorWithUnitId(self.unit_indices[before_unit_id])

                            #引导力
                            hole_direction = self.hole_dir[i][j - 1]

                            penetration = force_dir.dot(before_n) * hole_direction / force_dir.norm(1e-6)
                            # print(penetration)
                            if penetration >= -self.beta: #penetration
                                if penetration > 0:
                                    intersect = self.intersection_points[index + i * self.max_control_length]
                                    self.outer_QI[self.exist_grad_number[0] - 1] = intersect
                                    # print(intersect)

                                    t = self.intersection_infos[index + i * self.max_control_length][0]
                                    # print(t)
                                    
                                    index1 = int(self.intersection_infos[index + i * self.max_control_length][3])
                                    index2 = int(self.intersection_infos[index + i * self.max_control_length][4])

                                    id1 = before_kp_id[index1]
                                    id2 = before_kp_id[index2]

                                    f_basic = (tm.normalize(start_point - self.outer_QI[self.exist_grad_number[0] - 1]) + tm.normalize(self.constraint_end_point[i] - self.outer_QI[self.exist_grad_number[0] - 1]))
                                    f1 = (1 - t) * f_basic
                                    f2 = t * f_basic
                                    self.string_force[id1] += f1 * (force + friction_force)
                                    self.string_force[id2] += f2 * (force + friction_force)

                                    self.dldx_force[i, id1] += f1 * bonus
                                    self.dldx_force[i, id2] += f2 * bonus
                                    # self.dldx_friction_force[i, index, id1] += f1
                                    # self.dldx_friction_force[i, index, id2] += f2
                                    # self.bonus_string_Q[self.exist_grad_number[0] - 1][index1] = (1 - t) * self.string_params[0] / before_kp_num
                                    # self.bonus_string_Q[self.exist_grad_number[0] - 1][index2] = t * self.string_params[0] / before_kp_num

                            self.end_force[0] = force

                            # avg_vel = tm.vec3([.0, .0, .0])
                            # for k in range(before_kp_num):
                            #     avg_vel += self.get_velocity_with_index(before_kp_id[k]) / before_kp_num
                            # avg_vel /= avg_vel.norm(1e-6)

                            # if penetration <= 0:
                            f_basic = 1. / before_kp_num * tm.normalize(self.outer_QI[self.exist_grad_number[0] - 1] - start_point)
                            # print(f_basic)


                            for k in range(before_kp_num):
                                self.string_force[before_kp_id[k]] += f_basic * (force + friction_force)
                                self.dldx_force[i, before_kp_id[k]] += f_basic * bonus
                                # self.dldx_friction_force[i, index, before_kp_id[k]] += f_basic

                            candidate_id = self.constraint_end_point_candidate_id[i]
                            unit_connection_id = self.constraint_start_point_candidate_connection[candidate_id]
                            if unit_connection_id >= 0:
                                kp_id = self.unit_indices[unit_connection_id]
                                unit_kp_num = self.calculateKpNumWithUnitId(kp_id)
                                center = self.calculateCenterPoint3DWithUnitId(kp_id)
                                if self.intersection_flag[i][index]:
                                    center = self.intersection_points[index + i * self.max_control_length]
                                f_basic = 1. / unit_kp_num * tm.normalize(center - self.constraint_end_point[i]) 
                                for k in range(unit_kp_num):
                                    self.string_force[kp_id[k]] += f_basic * (force + friction_force)
                                    self.dldx_force[i, kp_id[k]] += f_basic * bonus
                                    # self.dldx_friction_force[i, index, kp_id[k]] += f_basic
                        break

    @ti.func
    def groundForce(self):
        for i in ti.ndrange(self.kp_num):
            if self.x[i][Z] < self.ground_barrier:
                barrier_exceed = self.x[i][Z] - self.ground_barrier
                barrier_left = self.x[i][Z] + self.collision_d
                if self.x[i][Z] > 0.:
                    self.total_energy[0] += -self.ground_collision_indice * barrier_exceed ** 2 * tm.log(barrier_left / self.ground_barrier)
                    self.ground_force[i] = self.ground_collision_indice * (2 * barrier_exceed * tm.log(barrier_left / self.ground_barrier) + barrier_exceed ** 2 / barrier_left) * tm.vec3([0., 0., 1.])
                    self.df_ground_force[i] = self.ground_collision_indice * (4. * barrier_exceed / barrier_left + 2 * tm.log(barrier_left / self.ground_barrier) - barrier_exceed ** 2 / barrier_left ** 2)
                else:
                    self.total_energy[0] += self.ground_barrier_energy_maximum - (2. * self.ground_force_maximum + self.x[i][Z] * self.df_ground_force_maximum) * self.x[i][Z] * 0.5
                    self.df_ground_force[i] = self.df_ground_force_maximum
                    self.ground_force[i] = (self.ground_force_maximum + self.x[i][Z] * self.df_ground_force_maximum) * tm.vec3([0., 0., 1.])

    @ti.func
    def friction_f0(self, x, c):
        value = 0.0
        if x >= c:
            value = x
        else:
            value = -x ** 3 / (3. * c ** 2) + x ** 2 / c + c / 3.
        return value

    @ti.func
    def friction_f1(self, x, c):
        value = 1.0
        if x < c:
            value = -x ** 2 / (c ** 2) + 2. * x / c
        return value
    
    @ti.func
    def friction_f2(self, x, c):
        value = 0.0
        if x < c:
            value = -2 * x / (c ** 2) + 2. / c
        return value

    @ti.func
    def frictionForce(self):
        c = self.epsilon_v[0]
        for i in ti.ndrange(self.kp_num):
            if self.x[i][Z] < self.ground_barrier:
                self.u[i] = [self.x[i][X] - self.back_up_x[i][X], self.x[i][Y] - self.back_up_x[i][Y]]
                N = self.ground_force[i][Z]
                ui_norm = self.u[i].norm()
                self.total_energy[0] += self.ground_miu * N * self.friction_f0(ui_norm, c)
                if ui_norm > 1e-6:
                    self.ground_friction[i] = -self.ground_miu * N * self.friction_f1(ui_norm, c) * tm.vec3([self.u[i][X], self.u[i][Y], 0.]) / ui_norm
                if self.print:
                    print(f"After {i} ground friction: energy: {self.total_energy[0]}, {c}, {N}, {ui_norm}")

    @ti.func
    def clearForce(self):
        self.total_energy[0] = 0.0
        self.max_force[0] = 0.0
        for i in ti.ndrange(self.kp_num):
            self.stvk_force[i] = [0., 0., 0.]
            self.bending_force[i] = [0., 0., 0.]
            self.facet_bending_force[i] = [0., 0., 0.]
            self.viscosity_force[i] = [0., 0., 0.]
            self.string_force[i] = [0., 0., 0.]
            self.record_force[i] = [0., 0., 0.]
            self.preventing_force[i] = [0., 0., 0.]
            self.ground_force[i] = [0., 0., 0.]
            self.ground_friction[i] = [0., 0., 0.]
            self.u[i] = [0., 0.]
            self.df_ground_force[i] = 0.
        for i, j in ti.ndrange(self.constraint_number, self.kp_num):
            self.dldx_force[i, j] = [0., 0., 0.]
        for i, j, k in ti.ndrange(self.constraint_number, self.max_control_length, self.kp_num):
            self.dldx_friction_force[i, j, k] = [0., 0., 0.]

    @ti.func
    def mergeForce(self, gravitational_acc: tm.vec3):
        for i in ti.ndrange(self.kp_num): 
            self.force[i] = self.stvk_force[i] + self.bending_force[i] + self.facet_bending_force[i] + self.viscosity_force[i] + self.string_force[i] + self.ground_force[i] + self.ground_friction[i]
            if self.stvk_force[i].norm() > self.max_force[0]:
                self.max_force[0] = self.stvk_force[i].norm()
            self.force[i] += gravitational_acc * self.masses[i]
            self.total_energy[0] += 0.5 * self.masses[i] * (self.get_velocity_with_index(i).norm()) ** 2 - self.masses[i] * gravitational_acc[Z] * self.get_position_with_index(i)[Z]

    @ti.kernel
    def fill_K(self, h: data_type, builder: ti.types.sparse_matrix_builder(), sim_mode: bool):
        for index in ti.ndrange(3 * self.kp_num):
            builder[index, index] += self.masses[index // 3] / (h ** 2)

        for i, j, k, jj, kk in ti.ndrange(self.div_indices_num, 3, 3, 3, 3):
            builder[self.triplets[i][j] + jj, self.triplets[i][k] + kk] += -self.K_element[i][3 * j + jj, 3 * k + kk]
        
        for ii, ij, ik, ijj, ikk in ti.ndrange(self.bending_pairs_num + self.facet_bending_pairs_num, 4, 4, 3, 3):
            builder[self.triplets_bending[ii][ij] + ijj, self.triplets_bending[ii][ik] + ikk] += -self.K_element_bending[ii][3 * ij + ijj, 3 * ik + ikk]

        if sim_mode:
            # for ii, i in ti.ndrange(self.constraint_number, self.kp_num):
            #     print(self.dldx_force[ii, i])
                
            for ii, i, j, jj, kk in ti.ndrange(self.constraint_number, self.kp_num, self.kp_num, 3, 3):
                builder[i * 3 + jj, j * 3 + kk] += self.dldx_force[ii, i].outer_product(self.dldx_force[ii, j])[jj, kk]
            
            # for ii, i, j, k, jj, kk in ti.ndrange(self.constraint_number, self.max_control_length, self.kp_num, self.kp_num, 3, 3):
            #     builder[j * 3 + jj, k * 3 + kk] += self.miu * self.dldx_friction_force[ii, i, j].outer_product(self.dldx_friction_force[ii, i, k])[jj, kk]

            c = self.epsilon_v[0]
            for index in ti.ndrange(self.kp_num):
                if self.x[index][Z] < self.ground_barrier:
                    builder[index * 3 + 2, index * 3 + 2] += -self.df_ground_force[index]
                    ui_norm = self.u[index].norm()
                    if ui_norm > 1e-6:
                        friction_matrix = self.ground_miu * self.ground_force[index][Z] * (((self.friction_f2(ui_norm, c) * ui_norm - self.friction_f1(ui_norm, c)) / ui_norm ** 3) * self.u[index].outer_product(self.u[index]) + self.friction_f1(ui_norm, c) / ui_norm * ti.Matrix.identity(data_type, 2))
                        for jj, kk in ti.ndrange(2, 2):
                            builder[index * 3 + jj, index * 3 + kk] += friction_matrix[jj, kk]
   
    @ti.kernel
    def fill_b(self, h: data_type):
        for i in ti.ndrange(3 * self.kp_num):
            self.b[i] = -self.masses[i // 3] * (self.x[i // 3][i % 3] - self.back_up_x[i // 3][i % 3] - h * self.back_up_v[i // 3][i % 3]) / (h ** 2) + self.force[i // 3][i % 3]

    @ti.kernel
    def step_x(self, time_step: data_type):
        for i in ti.ndrange(3 * self.kp_num):
            self.x[i // 3][i % 3] = self.x[i // 3][i % 3] + self.u0[i] * time_step
                
    @ti.kernel
    def step_xv(self, time_step: data_type, sim_mode: bool, theta: data_type, gravitational_acc: tm.vec3, facet_k: data_type, basic_energy: data_type) -> data_type:
        # bonus = 1.
        if sim_mode == self.FOLD_SIM:
            for i in ti.ndrange(self.kp_num):
                self.v[i] = (self.x[i] - self.back_up_x[i]) / time_step * 0.5
        else:
            for i in ti.ndrange(self.kp_num):
                self.v[i] = (self.x[i] - self.back_up_x[i]) / time_step * 0.9999

        # energy1 = self.getEnergy(sim_mode, theta, gravitational_acc, facet_k)
        self.total_energy[0] = basic_energy
        return 1.
        # return time_step

    @ti.func
    def backup_v(self):
        for i in ti.ndrange(self.kp_num):
            self.back_up_v[i] = self.v[i]

    @ti.func
    def backup_startpoint(self):
        for i in ti.ndrange(self.constraint_number):
            self.back_up_start_point[i] = self.constraint_start_point[i]

    @ti.func
    def backup_angle(self):
        for i in ti.ndrange(self.crease_pairs_num):
            self.backup_crease_angle[i] = self.crease_angle[i]

    @ti.func
    def backup_x(self):
        for i in ti.ndrange(self.kp_num):
            self.back_up_x[i] = self.x[i]
    
    @ti.func
    def backup_startx(self):
        for i in ti.ndrange(self.kp_num):
            self.start_x[i] = self.x[i]
        for i in ti.ndrange(self.constraint_number):
            self.start_start_point[i] = self.constraint_start_point[i]

    @ti.kernel
    def backup_xv(self):
        self.backup_v()
        self.backup_x()

    @ti.kernel
    def update_vertices(self):
        for i in ti.ndrange(self.kp_num):
            self.vertices[i] = self.get_position_with_index(i)
            force = self.force[i].norm()
            if force > self.lames_bonus[0]:
                self.vertices_color[i] = [1., 0.5, 0.5]
            elif force > self.lames_bonus[0] * 0.5:
                self.vertices_color[i] = [0.5 + 0.5 * (force * 2. / self.lames_bonus[0] - 1.), 1. - 0.5 * (force * 2. / self.lames_bonus[0] - 1.), 0.5]
            else:
                self.vertices_color[i] = [0.75 - 0.25 * (force * 2. / self.lames_bonus[0]), 1., 1. - 0.5 * (force * 2. / self.lames_bonus[0])]
    
    @ti.kernel
    def update_string_vertices(self):
        # i = 0
        # k = 0
        if self.constraint_number:
            current_string_index = 0
            endpoint_index = 0
            for i in ti.ndrange(self.constraint_number):
                self.string_vertex[current_string_index] = self.constraint_start_point[i]
                current_string_index += 1
                current_string_id = 0
                for k in ti.ndrange(self.max_control_length):
                    if self.unit_control[i][k] != -1:
                        current_string_id = k
                        kp_id = self.unit_indices[self.unit_control[i][k]]
                        hole_direction = self.hole_dir[i][k]
                        nm = self.calculateNormalVectorWithUnitId(kp_id)
                        before_kp_id = kp_id
                        before_n = tm.vec3([0., 0., 0.])
                        if k >= 1:
                            before_kp_id = self.unit_indices[self.unit_control[i][k - 1]]
                            before_n = self.calculateNormalVectorWithUnitId(before_kp_id)

                        if self.intersection_flag[i][k]:
                            if before_n.norm() > 0:
                                self.string_vertex[current_string_index] = self.intersection_points[k + i * self.max_control_length] + self.BIAS * hole_direction * (nm + before_n) * 2. / ((nm + before_n).norm()) ** 2
                                current_string_index += 1
                                self.string_vertex[current_string_index] = self.intersection_points[k + i * self.max_control_length] + self.BIAS * hole_direction * (nm + before_n) * 2. / ((nm + before_n).norm()) ** 2
                                current_string_index += 1
                            else:
                                self.string_vertex[current_string_index] = self.intersection_points[k + i * self.max_control_length] + self.BIAS * hole_direction * nm
                                current_string_index += 1
                                self.string_vertex[current_string_index] = self.intersection_points[k + i * self.max_control_length] + self.BIAS * hole_direction * nm
                                current_string_index += 1
                    
                        self.string_vertex[current_string_index] = self.unit_center[self.unit_control[i][k]] + self.BIAS * hole_direction * nm
                        current_string_index += 1
                        
                        if self.unit_control[i][k + 1] != -1 or self.constraint_end_point_existence[i]:
                            self.string_vertex[current_string_index] = self.unit_center[self.unit_control[i][k]] - self.BIAS * hole_direction * nm
                            current_string_index += 1
                        
                        # current_string_index += 1
                        self.endpoint_vertex[endpoint_index] = self.unit_center[self.unit_control[i][k]]
                    else:
                        break
                if self.constraint_end_point_existence[i]:
                    # print(current_string_index)
                    # print("-0-")
                    hole_direction = self.hole_dir[i][current_string_id]
                    before_kp_id = self.unit_indices[self.unit_control[i][current_string_id]]
                    before_n = self.calculateNormalVectorWithUnitId(before_kp_id)
                    # print("1")
                    if self.intersection_flag[i][self.max_control_length - 1]:
                        # print("2")
                        self.string_vertex[current_string_index] = self.intersection_points[self.max_control_length - 1 + i * self.max_control_length] - self.BIAS * hole_direction * before_n
                        current_string_index += 1
                        self.string_vertex[current_string_index] = self.intersection_points[self.max_control_length - 1 + i * self.max_control_length] - self.BIAS * hole_direction * before_n
                        current_string_index += 1
                    self.string_vertex[current_string_index] = self.constraint_end_point[i]
                    current_string_index += 1
                    # print("3")
                    self.endpoint_vertex[endpoint_index] = self.constraint_end_point[i]
                    endpoint_index += 1
                else:
                    # print(k)
                    # self.endpoint_vertex[endpoint_index] = self.unit_center[self.unit_control[i][k - 1]]
                    endpoint_index += 1
            for i in ti.ndrange((current_string_index, 4 * self.tsa_string_number)):
                self.string_vertex[i] = [0., 0., 1000.]

    @ti.kernel
    def line_search(self, t: data_type, base_energy: data_type, dx_norm: data_type, sim_mode: bool, theta: data_type, gravitational_acc: tm.vec3, facet_k: data_type, enable_ground: bool):
        self.dt_bonus[0] = 2. * self.dt_bonus[0] if self.dt_bonus[0] <= 0.5 else 1.
        self.backup_startx()
        new_energy = base_energy
        # old_energy = math.inf
        # normal = 0
        while 1:
            for i in ti.ndrange(3 * self.kp_num):
                self.x[i // 3][i % 3] = self.start_x[i // 3][i % 3] + self.u0[i] * self.dt_bonus[0]
            for i in ti.ndrange(self.kp_num):
                self.v[i] = (self.x[i] - self.back_up_x[i] - t * self.back_up_v[i]) / t
            # print(self.unit_indices[0][0])
            new_energy = self.getEnergy(sim_mode, theta, gravitational_acc, facet_k, enable_ground)

            if new_energy > base_energy + 1e-12:
                if 1:
                    print("Err", self.dt_bonus[0], new_energy, base_energy)
                self.dt_bonus[0] *= 0.5
                if self.dt_bonus[0] < 1e-12:
                    break
            else:
                if 1:
                    print("Good", self.dt_bonus[0], new_energy, base_energy)
                break

    def initializeRunning(self):
        # turn to numpy structures to initialize the kernel
        numpy_indices                       = np.array(self.ori_sim.indices, dtype=np.int32)
        numpy_kps                           = np.array(self.kps) - np.array(self.total_bias + [-self.origami_z_bias])
        numpy_original_kps                  = np.array(self.kps)
        numpy_mass_list                     = np.array(self.mass_list)
        numpy_tri_indices                   = np.array(self.tri_indices, dtype=np.int32)
        numpy_connection_matrix             = np.array(self.ori_sim.connection_matrix)
        numpy_bending_pairs                 = np.array(self.ori_sim.bending_pairs, dtype=np.int32)
        numpy_crease_pairs                  = np.array(self.ori_sim.crease_pairs, dtype=np.int32)
        numpy_line_indices                  = np.array(self.ori_sim.line_indices, dtype=np.int32)
        if len(self.ori_sim.facet_bending_pairs) == 0:
            numpy_facet_bending_pairs           = np.array([[0, 0]], dtype=np.int32)
            numpy_facet_crease_pairs            = np.array([[0, 0]], dtype=np.int32)
        else:
            numpy_facet_bending_pairs           = np.array(self.ori_sim.facet_bending_pairs, dtype=np.int32)
            numpy_facet_crease_pairs            = np.array(self.ori_sim.facet_crease_pairs, dtype=np.int32)
        # construct tb_line information which contains start, end, level and coeff
        if self.sim_mode == self.FOLD_SIM:
            tb_line = []
            for line in self.lines:
                tb_line.append([line[START][X], line[START][Y], line[END][X], line[END][Y], line.level, line.coeff])
        else:
            tb_line = [[]]
        numpy_tb_line                       = np.array(tb_line)
        # construct string information including string number in each completed constraint and every id and dir
        numpy_string_number                 = np.array([len(self.method["id"][i]) for i in range(self.constraint_number)], dtype=np.int32)
        if numpy_string_number.size != 0:
            max_string_number = max(numpy_string_number)
        else:
            max_string_number = 0
            numpy_string_number = np.array([1])
        parsed_string_information = []
        
        for i in range(self.constraint_number):
            parsed_string = []
            for j in range(numpy_string_number[i]):
                parsed_string.append([0 if self.method["type"][i][j] == 'A' else 1, self.method["id"][i][j], self.method["reverse"][i][j]])
                # parsed_string.append([0 if self.string_total_information[i][j].point_type == 'A' else 1, self.string_total_information[i][j].id, self.string_total_information[i][j].dir])
            for j in range(numpy_string_number[i], max_string_number):
                parsed_string.append([0, -1, 0])
            parsed_string_information.append(parsed_string)
        numpy_parsed_string_information     = np.array(parsed_string_information, dtype=np.int32) if len(parsed_string_information) != 0 else np.array([[[0, -1, 0]]], dtype=np.int32)
        numpy_string_end                    = np.array([ele if ele != None else -1 for ele in self.string_end])

        if self.P_number > 0:
            numpy_p_candidator = np.array(self.P_candidate) - np.array(self.total_bias + [-self.origami_z_bias])
        else:
            numpy_p_candidator = np.array([[]])
        numpy_p_candidator_connection = np.array(self.P_candidate_connection, dtype=np.int32)
        # parameters reset
        self.dead_count = 0
        self.recorded_t = []
        self.recorded_string_decrease_length_control = []
        self.recorded_string_decrease_length = [[] for _ in range(self.constraint_number)]
        self.recorded_max_force = []
        self.recorded_folding_percent = []
        self.recorded_maximum_folding_percent = []
        self.recorded_minimum_folding_percent = []
        self.stable_state = 0
        self.past_move_indice = 0.0
        self.folding_percent = 0.0

        self.tolerance = 1.

        self.can_rotate = False

        self.dt_bonus[0] = 1.

        if self.sim_mode == self.FOLD_SIM:
            self.n = 6 #仿真的时间间隔
            self.dt = 0.1 / self.n #仿真的时间间隔
            self.substeps = round(1. / 60. / self.dt) #子步长，用于渲染
            self.basic_dt = self.substeps * self.dt
            self.now_t = 0.
            self.lame_k = 15.
            self.bending_param[0] = 0.1
            self.facet_bending_param[0] = 0.1
            if self.use_gui:
                self.camera.position(0., -1.5 * self.max_size, 1.5 * self.max_size + self.origami_z_bias)
                self.camera.up(0.2, 0.4, 0.9)
                self.camera.lookat(0, 0, self.origami_z_bias)
            self.viscousity = 0.0
            self.ITER = 6
            self.enable_ground = False
        else:
            self.n = 24 #仿真的时间间隔
            self.dt = 0.1 / self.n #仿真的时间间隔
            self.substeps = round(1. / 120. / self.dt) #子步长，用于渲染
            self.basic_dt = self.substeps * self.dt
            self.now_t = 0.
            self.lame_k = 1000.
            self.string_params[0] = 100. #绳的轴向弹性模量
            self.bending_param[0] = 0.1
            self.facet_bending_param[0] = 0.1
            self.miu = 0.0
            if self.use_gui:
                self.camera.position(0, -400, 400 + self.origami_z_bias)
                self.camera.lookat(0, 0, self.origami_z_bias)
            self.viscousity = 0.0
            self.ITER = 12
            self.enable_ground = True

        epsilon_v = self.velocity_barrier * self.dt
        epsilon_string = 1e-1 * self.dt
        # initialize!
        self.initialize(
            numpy_indices, numpy_kps, numpy_mass_list, numpy_tri_indices, 
            numpy_connection_matrix, 
            numpy_bending_pairs, numpy_crease_pairs, 
            numpy_line_indices, numpy_facet_bending_pairs, numpy_facet_crease_pairs, self.sim_mode, numpy_string_number, 
            numpy_parsed_string_information, numpy_string_end, numpy_original_kps, numpy_tb_line, self.lame_k,
            numpy_p_candidator, numpy_p_candidator_connection, epsilon_v, epsilon_string
        )


    def step(self):
        self.update_folding_target()
        facet_k = (self.lame_k * 0.2 + 1.) * self.facet_bending_param[0]
        if self.facet_bending_pairs_num > 0:
            while self.now_t < self.basic_dt:
                self.backup_xv()
                self.dt_bonus[0] = 1.
                for k in range(self.ITER):
                    self.Fc()
                    self.F2(facet_k)
                    self.Fm(self.gravitational_acc)
                    
                    # Solve Equations
                    self.fill_K(self.dt, self.AK, self.sim_mode)
                    A = self.AK.build()
                    self.sparse_solver.compute(A)
                    if not k:
                        backup_energy = self.total_energy[0]

                    self.fill_b(self.dt)
                    dx = self.sparse_solver.solve(self.b)
                    self.u0.from_numpy(dx)

                    dx_norm = np.linalg.norm(dx)

                    self.line_search(self.dt, self.total_energy[0], dx_norm, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, self.enable_ground)

                    if dx_norm < 1e-3:
                        break

                dt = self.step_xv(self.dt, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, backup_energy)
                self.now_t += self.dt
                self.current_t += self.dt

            self.now_t -= self.basic_dt
        else:
            while self.now_t < self.basic_dt:
                self.backup_xv()

                self.dt_bonus[0] = 1.
                for k in range(self.ITER):
                    self.Fc()
                    self.F2(-1.)
                    self.Fm(self.gravitational_acc)
                
                    # Solve Equations
                    self.fill_K(self.dt, self.AK, self.sim_mode)
                    A = self.AK.build()
                    self.sparse_solver.compute(A)
                    if not k:
                        backup_energy = self.total_energy[0]
                    # print(self.total_energy[0])

                    self.fill_b(self.dt)
                    dx = self.sparse_solver.solve(self.b)
                    self.u0.from_numpy(dx)

                    dx_norm = np.linalg.norm(dx)
                    # print(dx_norm)

                    self.line_search(self.dt, self.total_energy[0], dx_norm, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, self.enable_ground)

                    if dx_norm < 1e-3:
                        break

                dt = self.step_xv(self.dt, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, backup_energy)

                self.now_t += self.dt
                self.current_t += self.dt
            
        self.now_t -= self.basic_dt

        if not self.can_rotate:
            self.dead_count += 1
            self.folding_percent = 0.0

            for i in range(self.bending_pairs_num):
                self.folding_percent += self.crease_angle[i]

            self.folding_percent /= self.bending_pairs_num
            move_indice = 0.0
            for i in range(self.kp_num):
                move_indice += self.v[i].norm()
            
            # print(self.current_t)
            move_indice /= self.kp_num
            if self.debug_mode:
                print("Max force: " + str(round(self.max_force[0], 1)) + ", Move indice: " + str(round(move_indice, 1)) + ", Current time: " + str(round(self.current_t, 3)) + ", Stable state: " + str(self.stable_state) + ", Parameters: " + str(self.string_params[0]) + ", "+ str(self.n))
            if move_indice < 1.0:
                self.stable_state += 1
            else:
                self.stable_state = 0

            if self.stable_state >= 10:
                self.can_rotate = True
                self.dead_count = 0
                print("Actuation ok")

            self.past_move_indice = move_indice

        else:
            if not self.rotation:
                self.dead_count += 1
            else:
                self.dead_count = 0

            self.recorded_string_decrease_length_control.append(max([self.string_length_decrease[j] for j in range(self.constraint_number)]))
            for i in range(self.constraint_number):
                self.recorded_string_decrease_length[i].append(self.constraint_initial_length[i] - self.constraint_length[i])
            self.folding_percent = 0.0
            max_folding_percent = -1.0
            min_folding_percent = 1.0
            # print(self.crease_angle)
            for i in range(self.bending_pairs_num):
                self.folding_percent += self.crease_angle[i]
                if self.crease_angle[i] > max_folding_percent:
                    max_folding_percent = self.crease_angle[i]
                if self.crease_angle[i] < min_folding_percent:
                    min_folding_percent = self.crease_angle[i]
            self.folding_percent /= self.bending_pairs_num
            self.recorded_folding_percent.append(self.folding_percent)
            self.recorded_maximum_folding_percent.append(max_folding_percent)
            self.recorded_minimum_folding_percent.append(min_folding_percent)
            self.recorded_max_force.append(max([self.string_params[0] * (self.string_length_decrease[j] - self.recorded_string_decrease_length[j][-1]) for j in range(self.constraint_number)]))
            self.recorded_t.append(self.current_t)
        # print(self.lames_bonus[0], self.lames_bonus[1])

    def deal_with_key(self, key):
        self.dt_bonus[0] = 1.
        if key == 'r':
            self.initializeRunning()
            self.current_t = 0

            if self.sim_mode == self.FOLD_SIM:
                self.gravitational_acc[Z] = 0.
                self.enable_ground = False
                self.tsa_turning_angle = 0.0
                self.enable_tsa_rotate = 0.0
                self.folding_angle = 0.0
            else:
                self.gravitational_acc[Z] = -9810.
                self.enable_ground = True
                self.folding_angle = 0.0
                self.enable_add_folding_angle = 0.0

        elif key == 'c':
            if self.sim_mode == self.FOLD_SIM:
                self.sim_mode = self.TSA_SIM
            else:
                self.sim_mode = self.FOLD_SIM

            self.initializeRunning()
            self.current_t = 0

            if self.sim_mode == self.TSA_SIM:
                self.gravitational_acc[Z] = -9810.
                self.enable_ground = True
                self.folding_angle = 0.0
                self.enable_add_folding_angle = 0.0
            else:
                self.gravitational_acc[Z] = 0.
                self.enable_ground = False
                # self.string_length_decrease = 0.0
                self.enable_tsa_rotate = 0.0
                self.folding_angle = 0.0
        else:
            if self.sim_mode == self.FOLD_SIM:
                if key == 'u': 
                    self.folding_angle += self.folding_step
                    if self.folding_angle >= self.folding_max:
                        self.folding_angle = self.folding_max
                
                elif key == 'j': 
                    self.folding_angle -= self.folding_step
                    if self.folding_angle <= 0:
                        self.folding_angle = 0

                elif key == 'i': 
                    self.enable_add_folding_angle = self.folding_micro_step[0]
                
                elif key == 'k': 
                    self.enable_add_folding_angle = 0.0
                
                elif key == 'm': 
                    self.enable_add_folding_angle = -self.folding_micro_step[0]
            
            else:
                if key == 'i': 
                    self.enable_tsa_rotate = self.string_length_decrease_step

                elif key == 'k': 
                    self.enable_tsa_rotate = 0.0
            
                elif key == 'm': 
                    self.enable_tsa_rotate = -self.string_length_decrease_step
    
    def deal_with_motion(self, motion):
        pass

    def update_folding_target(self):
        if self.sim_mode == self.FOLD_SIM:
            self.folding_angle += self.enable_add_folding_angle
            if self.folding_angle >= self.folding_max:
                self.folding_angle = self.folding_max
            if self.folding_angle <= 0.0:
                self.folding_angle = 0.0

        else:
            if self.constraint_number > 0:
                self.rotation = False

                if self.can_rotate:
                    max_initial_length = max([self.string_number_each[i] for i in range(self.constraint_number)])
                    for i in range(self.constraint_number):
                        if (self.constraint_initial_length[i] - self.constraint_length[i] + self.tolerance >= self.string_length_decrease[i]) or (self.enable_tsa_rotate < 0):
                            self.rotation = True
                        else:
                            self.rotation = False
                            break
                    if self.rotation:
                        for ele in range(self.constraint_number):
                            self.string_length_decrease[ele] += self.enable_tsa_rotate * self.string_number_each[ele] / max_initial_length

    @ti.kernel
    def Fc(self, facet_k: data_type):
        self.clearForce()
        #--------#
        self.stvkForce()
        if self.print:
            print(f"After stvk, energy: {self.total_energy[0]}")
        if facet_k > 0:
            self.facetBendingForce(facet_k)
            if self.print:
                print(f"After facet, energy: {self.total_energy[0]}")

    @ti.kernel  
    def F1(self, folding_angle: data_type):
        self.bendingForceFoldSim(folding_angle)

    @ti.kernel  
    def F2(self, enable_ground: bool):
        self.bendingForceTSASim()
        if self.print:
            print(f"After bending, energy: {self.total_energy[0]}")
        if enable_ground:
            self.groundForce()
            if self.print:
                print(f"After ground, energy: {self.total_energy[0]}")
            self.frictionForce()
            if self.print:
                print(f"After ground friction, energy: {self.total_energy[0]}")
        if self.P_number > 0:
            self.stringForce()
            if self.print:
                print(f"After string, energy: {self.total_energy[0]}")
        
    
    @ti.kernel
    def Fm(self, gravitational_acc: tm.vec3):
        self.mergeForce(gravitational_acc)

    def run(self, force_dir=False, check_force_dir=True):
        self.initializeRunning()
        image_id = 0
        while self.window.running:
            if self.window.get_event(ti.ui.PRESS):
                self.deal_with_key(self.window.event.key)

            self.update_folding_target()

            # if self.folding_angle_reach_pi[0] or (self.dead_count >= 500 and not self.can_rotate) or (self.dead_count >= 200 and self.can_rotate):
            #     if self.sim_mode == self.TSA_SIM and self.can_rotate:
            #         if self.recorded_folding_percent[-1] > FOLDING_MAXIMUM:
            #             break
            #         else:
            #             if self.recorded_folding_percent[-1] <= np.mean(np.array(self.recorded_folding_percent[-51: -1])):
            #                 break

            if self.sim_mode == self.FOLD_SIM:
                facet_k = (self.lame_k * 0.2 + 1.) * self.facet_bending_param[0]
                if self.facet_bending_pairs_num > 0:
                    while self.now_t < self.basic_dt:
                        self.backup_xv()
                        
                        self.dt_bonus[0] = 1.
                        # backup_energy = 0.0
                        for k in range(self.ITER):
                            self.Fc(facet_k)
                            self.F1(self.folding_angle)
                            self.Fm(self.gravitational_acc)

                            # Solve Equations
                            self.fill_K(self.dt, self.AK, self.sim_mode)
                            A = self.AK.build()
                            self.sparse_solver.compute(A)
                            # print(A)

                            # for i in range(3 * self.kp_num):
                            #     for j in range(i, 3 * self.kp_num):
                            #         s = abs(A[i, j] - A[j, i])
                            #         if s > 1e-2:
                            #             a = 1
                            #             for ii in range(self.div_indices_num):
                            #                 for k in range(9):
                            #                     for l in range(k, 9):
                            #                         g = abs(self.K_element[ii][k, l] - self.K_element[ii][l, k])
                            #                         if g > 1e-2:
                            #                             print(self.K_element[ii])
                            if not k:
                                backup_energy = self.total_energy[0]

                            self.fill_b(self.dt)
                            dx = self.sparse_solver.solve(self.b)
                            self.u0.from_numpy(dx)

                            dx_norm = np.linalg.norm(dx)

                            self.line_search(self.dt, self.total_energy[0], dx_norm, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, self.enable_ground)

                            if dx_norm < 1e-3:
                                break

                        dt = self.step_xv(self.dt, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, backup_energy)
                        self.now_t += self.dt
                        self.current_t += self.dt

                    self.now_t -= self.basic_dt
                else:
                    while self.now_t < self.basic_dt:
                        self.backup_xv()
                        # backup_energy = 0.0
                        self.dt_bonus[0] = 1.
                        for k in range(self.ITER):
                            self.Fc(-1.)
                            self.F1(self.folding_angle)
                            self.Fm(self.gravitational_acc)

                            # Solve Equations
                            self.fill_K(self.dt, self.AK, self.sim_mode)
                            A = self.AK.build()
                            self.sparse_solver.compute(A)
                            if not k:
                                backup_energy = self.total_energy[0]

                            self.fill_b(self.dt)
                            dx = self.sparse_solver.solve(self.b)

                            self.u0.from_numpy(dx)

                            dx_norm = np.linalg.norm(dx)

                            self.line_search(self.dt, self.total_energy[0], dx_norm, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, self.enable_ground)

                            if dx_norm < 1e-3:
                                break

                        dt = self.step_xv(self.dt, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, backup_energy)
                        self.now_t += self.dt
                        self.current_t += self.dt
                    self.now_t -= self.basic_dt
            else:
                facet_k = (self.lame_k * 0.2 + 1.) * self.facet_bending_param[0]
                if self.facet_bending_pairs_num > 0:
                    while self.now_t < self.basic_dt:
                        self.backup_xv()
                        
                        self.dt_bonus[0] = 1.
                        
                        for k in range(self.ITER):
                            
                            self.Fc(facet_k)
                            self.F2(self.enable_ground)
                            self.Fm(self.gravitational_acc)

                            # print(self.initial_length_per_string)
                            # print(self.current_length_per_string)
                            
                            # Solve Equations
                            self.fill_K(self.dt, self.AK, self.sim_mode)
                            A = self.AK.build()
                            self.sparse_solver.compute(A)
                            if not k:
                                backup_energy = self.total_energy[0]

                            self.fill_b(self.dt)
                            dx = self.sparse_solver.solve(self.b)
                            self.u0.from_numpy(dx)
                            print("----")
                            print(self.b)
                            print(self.u0)
                            for i in range(3 * self.kp_num):
                                for j in range(i, 3 * self.kp_num):
                                    s = abs(A[i, j] - A[j, i])
                                    if s > 1e-2:
                                        a = 1
                                        for ii in range(self.div_indices_num):
                                            for k in range(9):
                                                for l in range(k, 9):
                                                    g = abs(self.K_element[ii][k, l] - self.K_element[ii][l, k])
                                                    if g > 1e-2:
                                                        print(self.K_element[ii])

                            dx_norm = np.linalg.norm(dx)

                            self.line_search(self.dt, self.total_energy[0], dx_norm, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, self.enable_ground)
                            # print(self.x)

                            if dx_norm < 1e-3:
                                break
                        
                        if k == self.ITER - 1:
                            a = 1

                        dt = self.step_xv(self.dt, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, backup_energy)
                        self.backupStringLength()
                        self.now_t += self.dt
                        self.current_t += self.dt

                    self.now_t -= self.basic_dt
                else:
                    while self.now_t < self.basic_dt:
                        self.backup_xv()

                        self.dt_bonus[0] = 1.
                        for k in range(self.ITER):
                            self.Fc(-1.)
                            self.F2(self.enable_ground)
                            self.Fm(self.gravitational_acc)
                        
                            # Solve Equations
                            self.fill_K(self.dt, self.AK, self.sim_mode)
                            A = self.AK.build()
                            self.sparse_solver.compute(A)
                            if not k:
                                backup_energy = self.total_energy[0]
                            # print(self.total_energy[0])

                            self.fill_b(self.dt)
                            dx = self.sparse_solver.solve(self.b)
                            self.u0.from_numpy(dx)

                            dx_norm = np.linalg.norm(dx)
                            # print(dx_norm)

                            self.line_search(self.dt, self.total_energy[0], dx_norm, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, self.enable_ground)

                            if dx_norm < 1e-3:
                                break

                        dt = self.step_xv(self.dt, self.sim_mode, self.folding_angle, self.gravitational_acc, facet_k, backup_energy)
                        self.backupStringLength()
                        self.now_t += self.dt
                        self.current_t += self.dt
                        
                    self.now_t -= self.basic_dt
                
            self.update_vertices() 

            if self.sim_mode == self.TSA_SIM:
                if self.constraint_number:
                    if not self.can_rotate:
                        self.dead_count += 1
                        self.folding_percent = 0.0
                        for i in range(self.bending_pairs_num):
                            self.folding_percent += self.crease_angle[i]
                        self.folding_percent /= self.bending_pairs_num
                        move_indice = 0.0
                        for i in range(self.kp_num):
                            move_indice += self.v[i].norm()
                        
                        move_indice /= self.kp_num
                        if 1:
                            print("Move indice: " + str(round(move_indice, 1)) + ", Current time: " + str(round(self.current_t, 3)) + ", Stable state: " + str(self.stable_state))
                        if move_indice < 1.0:
                            self.stable_state += 1
                        else:
                            self.stable_state = 0

                        if self.stable_state >= 10:
                            self.can_rotate = True
                            self.dead_count = 0
                            print("Actuation ok")

                        self.past_move_indice = move_indice

                    if self.can_rotate:
                        if not self.rotation:
                            self.dead_count += 1
                        else:
                            self.dead_count = 0
                        self.recorded_string_decrease_length_control.append(max([self.string_length_decrease[j] for j in range(self.constraint_number)]))
                        for i in range(self.constraint_number):
                            self.recorded_string_decrease_length[i].append(self.constraint_initial_length[i] - self.constraint_length[i])
                        self.folding_percent = 0.0
                        max_folding_percent = -1.0
                        min_folding_percent = 1.0
                        # print(self.crease_type)
                        for i in range(self.bending_pairs_num):
                            self.folding_percent += self.crease_angle[i]
                            if self.crease_angle[i] > max_folding_percent:
                                max_folding_percent = self.crease_angle[i]
                            if self.crease_angle[i] < min_folding_percent:
                                min_folding_percent = self.crease_angle[i]
                        self.folding_percent /= self.bending_pairs_num
                        self.recorded_folding_percent.append(self.folding_percent)
                        self.recorded_maximum_folding_percent.append(max_folding_percent)
                        self.recorded_minimum_folding_percent.append(min_folding_percent)
                        self.recorded_max_force.append(self.max_force[0])
                        self.recorded_t.append(self.current_t)

            self.camera.track_user_inputs(self.window, movement_speed=0.23, hold_key=ti.ui.RMB)
            self.scene.set_camera(self.camera)

            self.scene.point_light(pos=(0., 0., 3. * self.max_size + 3 * self.origami_z_bias), color=(0.9, 0.9, 0.9))
            self.scene.ambient_light((0.5, 0.5, 0.5))

            if self.enable_ground:
                self.scene.mesh(vertices=self.ground_vertices, indices=self.ground_indices, per_vertex_color=self.ground_vertices_color, two_sided=True)

            self.scene.mesh(self.vertices,
                    indices=self.indices,
                    per_vertex_color=self.vertices_color,
                    two_sided=True)
            
            self.fill_line_vertex()
            self.scene.lines(vertices=self.line_vertex,
                        width=2,
                        per_vertex_color=self.line_color)
            
            self.gui.text(f"System time: {round(self.current_t, 3)}s")
            if self.sim_mode == self.TSA_SIM:
                for i in range(self.constraint_number):
                    self.gui.text(text=f"Delta_length[{i}]: " + str(round(self.constraint_initial_length[i] - self.constraint_length[i], 2)))
                    self.gui.text(text=f"String length decrease[{i}]: " + str(round(self.string_length_decrease[i], 2)))
                self.gui.slider_float('Total folding percent', round(self.folding_percent, 4), -1., 1.)
                self.gui.slider_float('Total folding energy', round(self.total_energy[0], 4), 0., self.total_energy_maximum[0])
                self.gui.slider_int('Dead count', self.dead_count, 0, 500)
                self.enable_ground = self.gui.slider_int('Enable ground', self.enable_ground, 0, 1)
                # self.gui.slider_float('Total folding energy', round(self.total_energy[0], 2), 0.0, round(self.total_energy_maximum[0], 2))
                self.update_string_vertices()

                if self.constraint_number and self.P_number:
                    self.scene.lines(vertices=self.string_vertex, width=1, color=(.6, .03, .8))
                    self.scene.particles(centers=self.constraint_start_point, radius=1, color=(.6, .03, .8))
                    self.scene.particles(centers=self.endpoint_vertex, radius=1., color=(.6, .03, .8))
                    # self.scene.particles(centers=self.intersection_points, radius=0.4, color=(.6, .03, .8))

                self.string_params[0] = self.gui.slider_float('String_k', self.string_params[0], 0.0, 10000.0)

            else:
                self.folding_angle = self.gui.slider_float('Folding angle', self.folding_angle, 0.0, self.folding_max)
                self.gui.slider_float('Total folding energy', round(self.total_energy[0], 4), 0., self.total_energy_maximum[0])

            self.gui.slider_int('Step number', self.n, 1, 6000)
            self.lame_k = self.gui.slider_float('Lame_k', self.lame_k, 1., 2000.)
            self.tolerance = self.gui.slider_float('Tolerance', self.tolerance, 1., 30.)
            self.bending_param[0] = self.gui.slider_float('Bending_k', self.bending_param[0], 0.01, 1.5)
            self.facet_bending_param[0] = self.bending_param[0]

            self.lames_bonus[0] = self.lame_k * self.mu
            self.lames_bonus[1] = self.lame_k * self.landa
            
            self.canvas.scene(self.scene)
            if not self.fast_simulation_mode:
                self.window.save_image(f'./physResult/{self.origami_name}-{self.time}/' + str(image_id).zfill(8) + '.png')
                image_id += 1
            self.window.show()

if __name__ == "__main__":
    ori_name = "phys_sim"
    ori = OrigamiSimulator(origami_name=ori_name, fast_simulation=True)
    ori.start(ori_name, 4, ori.FOLD_SIM)
    ori.run(False, False)
    ori.window.destroy()
