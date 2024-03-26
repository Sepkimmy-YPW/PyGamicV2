import taichi as ti
import taichi.math as tm

from ori_sim_sys import *
from desc import *
from units import *

from cdftool import GraphModel, MCTS

#----折纸信息初始化开始----#
data_type = ti.f64
ti.init(arch=ti.cpu, default_fp=data_type, fast_math=False, advanced_optimization=False, verbose=False)

@ti.data_oriented
class OrigamiSimulator:
    def __init__(self, use_gui=True, debug_mode=False) -> None:
        self.use_gui = use_gui
        self.debug_mode = debug_mode

        if use_gui:
            self.window = ti.ui.Window("Origami Simulation", (700, 700), vsync=True)
            self.gui = self.window.get_gui()
            self.canvas = self.window.get_canvas()
            self.canvas.set_background_color((0.93, 0.97, 1.))
            self.scene = self.window.get_scene()
            self.camera = ti.ui.Camera()

        self.dxfg = DxfDirectGrabber()

        self.FOLD_SIM = 0
        self.TSA_SIM = 1

        self.string_total_information = []

        self.pref_pack = None

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
        # self.tsa_string_number = 0
        self.completed_constraint_number = len(self.string_total_information)
        # for i in range(self.completed_constraint_number):
        #     self.tsa_string_number += len(self.string_total_information[i]) - 1 # will multi 2

        # self.tsa_visual_string_number = self.tsa_string_number
        # self.tsa_root = []
        # self.string_root = []
        # self.tsa_end = []
        # # find tsa root
        # for i in range(self.completed_constraint_number):
        #     same_flag = 2
        #     id_1 = self.string_total_information[i][0].id
        #     if id_1 in [root[0] for root in self.tsa_root]:
        #         continue
        #     for j in range(i + 1, self.completed_constraint_number):
        #         id_2 = self.string_total_information[j][0].id
        #         if id_1 == id_2:
        #             same_flag += 2
        #     if same_flag > 2:
        #         self.tsa_root.append([self.string_total_information[i][0].id, int(same_flag / 2)])
        #         self.tsa_visual_string_number += same_flag # will multi 2
        #     else:
        #         self.string_root.append([self.string_total_information[i][0].id, 1])
        # # find tsa end
        # for i in range(self.completed_constraint_number):
        #     tsa_end = self.string_total_information[i][-1]
        #     if tsa_end.point_type == 'A':
        #         self.tsa_end.append(tsa_end.id)
        #     else:
        #         self.tsa_end.append(None)

        # # recalculate id and point_position
        # if self.pref_pack != None:
        #     self.panel_resolution = self.pref_pack["tsa_resolution"]
        #     self.panel_size = self.pref_pack["tsa_radius"]
        # else:
        #     self.panel_resolution = 72
        #     self.panel_size = 100.

        # for completed_constraint_string in self.string_total_information:
        #     for tsa_point in completed_constraint_string:
        #         if tsa_point.point_type == 'A':
        #             tsa_point.point[X] -= self.total_bias[X]
        #             tsa_point.point[Y] -= self.total_bias[Y]
        #         else:
        #             unit_id = self.pointInUnit(tsa_point.point)
        #             tsa_point.id = unit_id
        # 计算其他参数
        self.tsa_string_number = 0
        for i in range(len(self.string_total_information)):
            self.tsa_string_number += len(self.string_total_information[i]) - 1 # will multi 2

        self.tsa_visual_string_number = self.tsa_string_number
        self.tsa_root = []
        self.string_root = []
        self.tsa_end = []
        # find tsa root
        for i in range(len(self.string_total_information)):
            same_flag = 2
            id_1 = self.string_total_information[i][0].id
            if id_1 in [root[0] for root in self.tsa_root]:
                continue
            for j in range(i + 1, len(self.string_total_information)):
                id_2 = self.string_total_information[j][0].id
                if id_1 == id_2:
                    same_flag += 2
            if same_flag > 2:
                self.tsa_root.append([self.string_total_information[i][0].id, int(same_flag / 2)])
                self.tsa_visual_string_number += same_flag # will multi 2
            else:
                self.string_root.append([self.string_total_information[i][0].id, 1])
        # find tsa end
        for i in range(len(self.string_total_information)):
            if self.string_total_information[i][-1].point_type == 'A':
                self.tsa_end.append(self.string_total_information[i][-1].id)
            else:
                self.tsa_end.append(None)

        # recalculate id and point_position
        if self.pref_pack != None:
            self.panel_resolution = self.pref_pack["tsa_resolution"]
            self.panel_size = self.pref_pack["tsa_radius"]
        else:
            self.panel_resolution = 72
            self.panel_size = 100.

        for ele in self.string_total_information:
            for tsa_point in ele:
                if tsa_point.point_type == 'A':
                    tsa_point.point[X] -= self.total_bias[X]
                    tsa_point.point[Y] -= self.total_bias[Y]
                else:
                    unit_id = self.pointInUnit(tsa_point.point)
                    tsa_point.id = unit_id
        #----折纸信息初始化结束----#
                    
        # 构造折纸系统
        self.ori_sim = OrigamiSimulationSystem(unit_edge_max)
        for ele in self.units:
            self.ori_sim.addUnit(ele)
        self.ori_sim.mesh() #构造三角剖分
        self.ori_sim.fillBlankIndices() # fill all blank indice with -1
    
    def commonStart_2(self, unit_edge_max):
        ori_sim = self.ori_sim

        self.kps = ori_sim.kps                                           # all keypoints of origami
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
        self.string_k = .5 #绳的轴向弹性模量
        self.shearing_k = 4. #绳的抗剪强度
        self.miu = 6. #摩擦系数
        self.rotation_step = tm.pi / 36.0
        self.max_stretch_length = 1.
        self.facet_k = 1. * self.ori_sim.bending_k

        self.enable_add_folding_angle = 0. #启用折角增加的仿真模式
        self.enable_tsa_rotate = 0 #启用TSA驱动的仿真模式

        self.n = 50 #仿真的时间间隔
        self.dt = 0.1 / self.n #仿真的时间间隔
        self.substeps = round(1. / 250. // self.dt) #子步长，用于渲染
        self.linear_search_step = 32

        self.folding_angle = 0.0 #当前的目标折叠角度
        self.tsa_turning_angle = 0.0 #当前tsa旋转角度

        self.energy = 0.0 #当前系统能量

        self.NO_QUASI = True
        self.QUASI = False

        # tsa参数
        self.d1 = 10. #TSA的旋转直径
        self.ds = 1. #TSA的绳宽

        #折纸参数
        self.d_hole = 4. #折纸上所打通孔的直径
        self.h_hole = 2. #通孔高度
        self.beta = self.h_hole / math.sqrt(self.h_hole**2 + self.d_hole**2)

        #折纸初始高度
        self.origami_z_bias = 10.

        #折纸最大折叠能量
        self.total_energy_maximum = ti.field(data_type, shape=1)
        self.total_energy = ti.field(data_type, shape=1)
        self.max_force = ti.field(data_type, shape=1)

        #最大末端作用力
        self.end_force = ti.field(data_type, shape=1)

        #----define parameters for taichi----#
        self.string_params = ti.field(dtype=data_type, shape=2)
        self.masses = ti.field(dtype=data_type, shape=self.kp_num) # 质量信息
        self.original_vertices = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) # 原始点坐标
        self.vertices = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) # 点坐标
        self.unit_indices = ti.Vector.field(unit_edge_max, dtype=int, shape=self.unit_indices_num) # 每个单元的索引信息

        self.sequence_level = ti.field(int, shape=2) # max, min
        self.folding_micro_step = ti.field(data_type, shape=1) # step calculated by sequence_level max and min

        self.unit_center_initial_point = ti.Vector.field(3, dtype=data_type, shape=self.unit_indices_num) # 每个单元的初始中心点位置
        self.unit_center = ti.Vector.field(3, dtype=data_type, shape=self.unit_indices_num) # 每个单元的中心点位置
        self.points = ti.Vector.field(3, dtype=data_type, shape=unit_edge_max) #计算中心点坐标的暂存数
        self.indices = ti.field(int, shape=self.indices_num) #三角面索引信息

        self.bending_pairs = ti.field(dtype=int, shape=(self.bending_pairs_num, 2)) #弯曲对索引信息
        self.crease_pairs = ti.field(dtype=int, shape=(self.crease_pairs_num, 2)) #折痕对索引信息
        self.line_pairs = ti.field(dtype=int, shape=(self.line_total_indice_num, 2)) #线段索引信息，用于初始化渲染

        self.crease_angle = ti.field(dtype=data_type, shape=self.bending_pairs_num)
        self.backup_crease_angle = ti.field(dtype=data_type, shape=self.bending_pairs_num)
        self.energy_buffer = ti.field(dtype=data_type, shape=self.linear_search_step)
        self.linear_search_start = ti.field(dtype=data_type, shape=1)

        self.crease_folding_angle = ti.field(dtype=data_type, shape=self.crease_pairs_num) #折痕折角
        self.crease_folding_accumulate = ti.field(dtype=data_type, shape=self.crease_pairs_num) #补偿折角

        self.crease_type = ti.field(dtype=int, shape=self.crease_pairs_num) #折痕类型信息，与折痕对一一对应
        self.crease_level = ti.field(dtype=int, shape=self.crease_pairs_num)
        self.crease_coeff = ti.field(dtype=data_type, shape=self.crease_pairs_num)

        self.connection_matrix = ti.field(dtype=data_type, shape=(self.kp_num, self.kp_num)) #关键点之间的连接矩阵

        self.line_color = ti.Vector.field(3, dtype=data_type, shape=self.line_total_indice_num*2) #线段颜色，用于渲染
        self.line_vertex = ti.Vector.field(3, dtype=data_type, shape=self.line_total_indice_num*2) #线段顶点位置，用于渲染

        self.border_vertex = ti.Vector.field(3, dtype=data_type, shape=60)
        #----TSA constraints----#

        self.constraint_number = len(self.string_total_information) #约束的数量
        self.actuation_number = len(self.tsa_root)
        self.total_root_number = len(self.tsa_root) + len(self.string_root)

        #根据约束数量，确定约束起始点和终止点位置， 若初始点一致，则识别为TSA
        if self.constraint_number == 0:
            self.constraint_start_point = ti.Vector.field(3, dtype=data_type, shape=1)
            self.constraint_point_number = ti.field(dtype=int, shape=1)
            self.string_dis = ti.field(dtype=data_type, shape=(1, 1))
            self.string_dis.fill(-1)
            self.first_calculated = ti.field(dtype=bool, shape=(1, 1))
            self.string_dis.fill(True)

            self.constraint_angle = ti.field(dtype=data_type, shape=1)
            self.constraint_angle_enable = ti.field(dtype=bool, shape=1)
            self.constraint_angle_enable.fill(False)

            self.constraint_start_point_duplicate = ti.Vector.field(3, dtype=data_type, shape=1)
            self.visual_constraint_start_point = ti.Vector.field(3, dtype=data_type, shape=1)
            
            self.direction = ti.Vector.field(3, dtype=data_type, shape=1)

            self.constraint_end_point_existence = ti.field(dtype=bool, shape=1)
            self.constraint_end_point = ti.Vector.field(3, dtype=data_type, shape=1)

            # TSA交叉点
            self.intersection_point = ti.Vector.field(3, dtype=data_type, shape=1)

            # 视觉上的绳信息
            # self.visual_string_vertex = ti.Vector.field(3, dtype=data_type, shape=1)

            # 绳信息
            self.string_vertex = ti.Vector.field(3, dtype=data_type, shape=1)
            self.max_control_length = 1
            self.endpoint_vertex = ti.Vector.field(3, dtype=data_type, shape=1)
            #---#
            self.unit_control = ti.field(dtype=int, shape=(1, 1))
            #穿线方向信息，-1表示从下往上穿，1表示从上往下穿
            self.hole_dir = ti.field(dtype=data_type, shape=(1, 1))
            # 约束绳的长度信息
            self.constraint_initial_length = ti.field(data_type, shape=1)
            self.constraint_length = ti.field(data_type, shape=1)
            self.backup_constraint_length = ti.field(data_type, shape=1)
        else:
            self.constraint_start_point = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number)
            self.constraint_point_number = ti.field(dtype=int, shape=self.total_root_number)
            self.string_dis = ti.field(dtype=data_type, shape=(self.constraint_number, self.constraint_number))
            self.string_dis.fill(-1)
            self.first_calculated = ti.field(dtype=bool, shape=(self.constraint_number, self.constraint_number))
            self.string_dis.fill(True)

            self.constraint_angle = ti.field(dtype=data_type, shape=self.total_root_number)
            self.constraint_angle_enable = ti.field(dtype=bool, shape=self.total_root_number)
            self.constraint_angle_enable.fill(False)

            self.constraint_start_point_duplicate = ti.Vector.field(3, dtype=data_type, shape=self.total_root_number)

            self.direction = ti.Vector.field(3, dtype=data_type, shape=self.total_root_number)

            self.constraint_end_point_existence = ti.field(dtype=bool, shape=self.constraint_number)
            self.constraint_end_point = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number)

            # 要控制的点的索引信息，至多控制单元数目个数的点, TYPE B的点
            self.max_control_length = max([len(ele) for ele in self.string_total_information])
            self.unit_control = ti.field(dtype=int, shape=(self.constraint_number, self.max_control_length))
            self.unit_control.fill(-1)

            #穿线方向信息，-1表示从下往上穿，1表示从上往下穿
            self.hole_dir = ti.field(dtype=data_type, shape=(self.constraint_number, self.max_control_length))
            self.hole_dir.fill(0.)

            #TSA旋转打结后，新的渲染点位
            self.visual_constraint_start_point = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number)

            # TSA交叉点
            self.intersection_point = ti.Vector.field(3, dtype=data_type, shape=max(1, sum([self.tsa_root[i][1] - 1 for i in range(self.actuation_number)])))

            # TSA是否交叉
            self.have_intersection = ti.field(dtype=bool, shape=max(1, self.actuation_number))

            # 绳信息
            self.string_vertex = ti.Vector.field(3, dtype=data_type, shape=int(self.tsa_visual_string_number * 2))
            
            self.endpoint_vertex = ti.Vector.field(3, dtype=data_type, shape=self.constraint_number)
            #---#
            
            # 约束绳的长度信息
            self.constraint_initial_length = ti.field(data_type, shape=self.constraint_number)
            self.constraint_length = ti.field(data_type, shape=self.constraint_number)
            self.backup_constraint_length = ti.field(data_type, shape=self.constraint_number)

        # 受力顶点，渲染时用
        self.force_vertex = ti.Vector.field(3, dtype=data_type, shape=2*self.kp_num)

        # 有可能所有单元都是三角形，故没有面折痕，根据特定条件初始化面折痕信息
        if self.facet_bending_pairs_num > 0:
            self.facet_bending_pairs = ti.field(dtype=int, shape=(self.facet_bending_pairs_num, 2))
            self.facet_crease_pairs = ti.field(dtype=int, shape=(self.facet_crease_pairs_num, 2))
        else:
            self.facet_bending_pairs = ti.field(dtype=int, shape=(1, 2))
            self.facet_crease_pairs = ti.field(dtype=int, shape=(1, 2))
        
        #----simulator information----#
        self.x = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的位置
        self.v = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的速度
        self.dv = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的加速度
        self.force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #点的力
        self.record_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #跟踪的某一类型的力
        self.print_force = ti.Vector.field(3, dtype=data_type, shape=self.kp_num) #打印的某一类型的力
        self.viscousity = 1.e-3
        self.enable_ground = False

        self.back_up_x = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
        self.back_up_v = ti.Vector.field(3, dtype=data_type, shape=self.kp_num)
    
    def commonStart_3(self):
        self.current_t = 0.0
        self.half_dt = self.dt / 2.0

        self.folding_step = tm.pi * 0.99
        self.folding_max = tm.pi * 0.99

        self.half_max_size = self.max_size / 2.0
        self.r1 = self.d1 / 2.0

        self.folding_angle_reach_pi = ti.field(dtype=bool, shape=1)

        self.past_move_indice = 0.0

        self.stable_state = 0

        self.dm = ti.Matrix.field(3, 3, dtype=data_type, shape=self.div_indices_num)
        
        self.A = ti.field(dtype=data_type, shape=self.div_indices_num)

        self.mu = 3. / 2.6

        self.landa = 3. * 0.3 / (1 + 0.3) / (1 - 0.6)

        self.lame_k = 1.

        # derivative
        self.dDs = ti.Matrix.field(3, 3, dtype=data_type, shape=(3, 3))
        # self.dF = ti.Matrix.field(3, 3, dtype=data_type, shape=())
        # self.dE = ti.Matrix.field(3, 3, dtype=data_type, shape=()) # E = 0.5 * (F ^ T * F - I)
        # self.dP = ti.Matrix.field(3, 3, dtype=data_type, shape=()) # P = F(2 * mu * E + landa * tr(E)I)
        # self.dH = ti.Matrix.field(3, 3, dtype=data_type, shape=()) # H = -AP * dm ^ (-T)

        self.b = ti.field(data_type, shape=3 * self.kp_num)
        self.K_element = ti.Matrix.field(9, 9, data_type, shape=self.div_indices_num)
        self.K_element_bending = ti.Matrix.field(12, 12, data_type, shape=self.bending_pairs_num+self.facet_bending_pairs_num)
        self.triplets = ti.Vector.field(3, dtype=int, shape=self.div_indices_num)
        self.triplets_bending = ti.Vector.field(4, dtype=int, shape=self.bending_pairs_num+self.facet_bending_pairs_num)

        self.bending_params = ti.field(data_type, shape=7) #c1,c2,t1,t2,a1,a2,dir
        self.n1 = ti.Vector.field(3, data_type, shape=1)
        self.n2 = ti.Vector.field(3, data_type, shape=1)

        max_triplets = 9 * self.kp_num ** 2
        for indice in range(self.kp_num):
            connect_list = [0 if i != indice else 1 for i in range(self.kp_num)]
            for i in range(self.div_indices_num):
                indice_1 = self.tri_indices[3 * i + 0]
                indice_2 = self.tri_indices[3 * i + 1]
                indice_3 = self.tri_indices[3 * i + 2]
                if indice_1 == indice:
                    connect_list[indice_2] = 1
                    connect_list[indice_3] = 1
                elif indice_2 == indice:
                    connect_list[indice_1] = 1
                    connect_list[indice_3] = 1
                elif indice_3 == indice:
                    connect_list[indice_1] = 1
                    connect_list[indice_2] = 1
            zero_connection_number = connect_list.count(0)
            max_triplets -= 9 * zero_connection_number

        self.AK = ti.linalg.SparseMatrixBuilder(3 * self.kp_num, 3 * self.kp_num, max_num_triplets=9 * self.kp_num ** 2 + 160 * (self.bending_pairs_num+self.facet_bending_pairs_num) ** 2, dtype=data_type)

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

        self.sequence_level_initial = 0

        self.commonStart_2(unit_edge_max)

        self.sim_mode = self.TSA_SIM

        self.enable_ground = False
        self.gravitational_acc = [0., 0., -9810.]     
            
        self.commonStart_3()
    
    def start(self, filepath, unit_edge_max, sim_type):
        # 获取点和线段信息
        self.dxfg.readFile("./dxfResult/" + filepath + ".dxf")

        # 获取折纸单元信息
        unit_parser = UnitPackParserReverse(
                        tsp=[0.0, 0.0],
                        kps=self.dxfg.kps,
                        lines=self.dxfg.lines,
                        lines_type=self.dxfg.lines_type
                    )

        self.lines = unit_parser.getLine()

        unit_parser.setMaximumNumberOfEdgeInAllUnit(unit_edge_max) #For every units, there exists at most 4 edges
        self.units = unit_parser.getUnits()

        # calculate max length of view
        self.max_size = unit_parser.getMaxDistance()
        self.total_bias = unit_parser.getTotalBias(units=self.units)

        self.commonStart_1(unit_edge_max)

        self.tb = TreeBasedOrigamiGraph(self.ori_sim.kps, self.ori_sim.getNewLines())
        self.tb.calculateTreeBasedGraph()
        self.sequence_level_initial = int(self.tb.lines[0].level)

        self.commonStart_2(unit_edge_max)

        self.sim_mode = sim_type
        if sim_type == self.FOLD_SIM:
            self.enable_ground = False
            self.gravitational_acc = [0., 0., 0.]
        else:
            self.enable_ground = False
            self.gravitational_acc = [0., 0., -9810.]     
            
        self.commonStart_3()
    
    @ti.func
    def calculateStringDistanceParallel(self, cur_angle: data_type):
        accumulate_index = 0
        for i in range(self.actuation_number):
            dup_time = self.constraint_point_number[i]
            for j in range(accumulate_index, accumulate_index + dup_time):
                for k in range(j + 1, accumulate_index + dup_time):
                    x1 = self.constraint_start_point[j]
                    x2 = self.constraint_start_point[k]
                    s1 = self.unit_center[self.unit_control[j, 0]] - x1
                    s2 = self.unit_center[self.unit_control[k, 0]] - x2
                    n1 = s1.cross(s2)
                    dt2 = ti.Matrix.cols([x1 - x2, -s1, n1])
                    dk = ti.Matrix.cols([s2, -s1, x1 - x2])
                    d = ti.Matrix.cols([s2, -s1, n1])
                    D = d.determinant()
                    val = dk.determinant() / D
                    if dt2.determinant() / D > 0:
                        if self.first_calculated[j, k]:
                            self.string_dis[j, k] = val
                            self.first_calculated[j, k] = False
                            self.constraint_angle_enable[i] = False
                        else:
                            if val * self.string_dis[j, k] < .0:
                                self.constraint_angle[i] = cur_angle
                                self.constraint_angle_enable[i] = True
                            else:
                                self.string_dis[j, k] = val
                    else:
                        self.first_calculated[j, k] = True
    @ti.kernel
    def calculateStringDistance(self, cur_angle: data_type):
        self.calculateStringDistanceParallel(cur_angle)

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
                        tb_lines:                   ti.types.ndarray(),
                        string_number_each:         ti.types.ndarray(),
                        string_total_information:   ti.types.ndarray(),
                        tsa_end:                    ti.types.ndarray(),
                        sequence_initial:           int,
                        tsa_root:                   ti.types.ndarray(),
                        string_root:                ti.types.ndarray(),
        ):
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
        # 初始化连接矩阵
        for i, j in ti.ndrange(self.kp_num, self.kp_num):
            self.connection_matrix[i, j] = connection_matrix[i, j]
        # 初始化弯曲对和折痕对
        for i, j in ti.ndrange(self.bending_pairs_num, 2):
            self.bending_pairs[i, j] = bending_pairs[i, j]
            self.crease_pairs[i, j] = crease_pairs[i, j]
        # 初始化线段对
        for i, j in ti.ndrange(self.line_total_indice_num, 2):
            self.line_pairs[i, j] = line_indices[i, j]
        # 初始化面折痕对
        for i, j in ti.ndrange(self.facet_bending_pairs_num, 2):
            self.facet_bending_pairs[i, j] = facet_bending_pairs[i, j]
            self.facet_crease_pairs[i, j] = facet_crease_pairs[i, j]
        #初始化折痕折角
        for i in ti.ndrange(self.crease_pairs_num):
            self.crease_folding_angle[i] = 0.0
            self.crease_folding_accumulate[i] = 0.0
        # 初始化折痕类型
        for i in ti.ndrange(self.crease_pairs_num):
            for j in ti.ndrange(self.line_total_indice_num):
                if crease_pairs[i, 0] == line_indices[j, 0] and crease_pairs[i, 1] == line_indices[j, 1]:
                    self.crease_type[i] = line_indices[j, 2]
                    break

        ti.loop_config(serialize=True)
        self.sequence_level[0] = sequence_initial
        self.sequence_level[1] = sequence_initial
        if self.sim_mode == self.FOLD_SIM:
            # 初始化折叠等级和系数
            for i in ti.ndrange(self.crease_pairs_num):
                index1 = crease_pairs[i, 0]
                index2 = crease_pairs[i, 1]
                for true_index in ti.ndrange(self.line_total_indice_num):
                    index_11 = tb_lines[true_index, 0]
                    index_22 = tb_lines[true_index, 1]
                    if (index1 == index_11 and index2 == index_22) or (index2 == index_11 and index1 == index_22):
                        self.crease_level[i] = int(tb_lines[true_index, 2])
                        self.crease_coeff[i] = tb_lines[true_index, 3]
                        break
                if self.crease_level[i] > self.sequence_level[0]:
                    self.sequence_level[0] = self.crease_level[i]
                if self.crease_level[i] < self.sequence_level[1]:
                    self.sequence_level[1] = self.crease_level[i]
            self.folding_micro_step[0] = tm.pi / 180.0 / (1 + self.sequence_level[0] - self.sequence_level[1])
        ti.loop_config(serialize=False)

        # 初始化线性搜索初始点
        self.linear_search_start[0] = 1.0

        # 初始化渲染的线的颜色信息
        for i in ti.ndrange(self.line_total_indice_num):
            if line_indices[i, 2] == BORDER:
                self.line_color[2 * i] = [0, 0, 0]
                self.line_color[2 * i + 1] = [0, 0, 0]
            elif line_indices[i, 2] == VALLEY:
                self.line_color[2 * i] = [0, 0.17, 0.83]
                self.line_color[2 * i + 1] = [0, 0.17, 0.83]
            else:
                self.line_color[2 * i] = [0.75, 0.2, 0.05]
                self.line_color[2 * i + 1] = [0.75, 0.2, 0.05]

        # 计算折纸最大能量
        self.total_energy_maximum[0] = 0.0
        for i in ti.ndrange(self.crease_pairs_num):
            index1 = crease_pairs[i, 0]
            index2 = crease_pairs[i, 1]
            # length = (self.original_vertices[index2] - self.original_vertices[index1]).norm()
            self.total_energy_maximum[0] += 100.
            
        # 初始化TSA信息
        self.string_dis.fill(-1)
        self.constraint_angle.fill(0.)
        self.constraint_angle_enable.fill(False)
        self.first_calculated.fill(True)

        # 暂不考虑ABAB, 初始化控制单元和穿孔方向
        for i in ti.ndrange(self.constraint_number):
            index = 0
            for j in ti.ndrange(string_number_each[i] - 1):
                if string_total_information[i, j + 1, 0] != 0:
                    self.unit_control[i, index] = string_total_information[i, j + 1, 1]
                    self.hole_dir[i, index] = string_total_information[i, j + 1, 2]
                    index += 1

        # 计算末端点信息
        for i in ti.ndrange(self.completed_constraint_number):
            if tsa_end[i] != -1:
                id = tsa_end[i]
                self.constraint_end_point[i] = [
                    tm.cos(id / self.panel_resolution * 2 * tm.pi) * self.panel_size, 
                    tm.sin(id / self.panel_resolution * 2 * tm.pi) * self.panel_size, 
                    self.origami_z_bias
                ]
                self.constraint_end_point_existence[i] = True
            else:
                self.constraint_end_point[i] = [0.0, 0.0, 0.0]
                self.constraint_end_point_existence[i] = False
   
        # 计算重复点信息
        for i in ti.ndrange(self.actuation_number):
            start_id = tsa_root[i, 0]
            self.constraint_start_point_duplicate[i] = [
                tm.cos(start_id / self.panel_resolution * 2 * tm.pi) * self.panel_size, 
                tm.sin(start_id / self.panel_resolution * 2 * tm.pi) * self.panel_size, 
                self.origami_z_bias
            ]
        for i in ti.ndrange(self.total_root_number - self.actuation_number):
            start_id = string_root[i, 0]
            self.constraint_start_point_duplicate[i + self.actuation_number] = [
                tm.cos(start_id / self.panel_resolution * 2 * tm.pi) * self.panel_size, 
                tm.sin(start_id / self.panel_resolution * 2 * tm.pi) * self.panel_size, 
                self.origami_z_bias
            ]
        
        # for i in range(self.total_root_number):
        #     print(self.constraint_start_point_duplicate[i])

        # 计算旋转方向
        for i in ti.ndrange(self.total_root_number):
            ele = self.constraint_start_point_duplicate[i]
            self.direction[i] = [
                -ele[Y] / self.panel_size * self.d1 / 2.0, 
                ele[X] / self.panel_size * self.d1 / 2.0, 
                0.0
            ]
            
        # 计算分点信息
        ti.loop_config(serialize=True)
        accumulate_index = 0
        for i in ti.ndrange(self.actuation_number):
            dup_time = tsa_root[i, 1]
            for j in ti.ndrange(dup_time):
                self.constraint_start_point[j + accumulate_index] = [
                    self.constraint_start_point_duplicate[i][X] + self.direction[i][X] * tm.cos(j / dup_time * 2 * tm.pi), 
                    self.constraint_start_point_duplicate[i][Y] + self.direction[i][Y] * tm.cos(j / dup_time * 2 * tm.pi), 
                    self.origami_z_bias + tm.sqrt(self.direction[i][X] ** 2 + self.direction[i][Y] ** 2) * tm.sin(j / dup_time * 2 * tm.pi)
                ]
            self.constraint_point_number[i] = dup_time
            accumulate_index += dup_time
        ti.loop_config(serialize=False)

        for i in ti.ndrange(self.total_root_number - self.actuation_number):
            self.constraint_start_point[i + accumulate_index] = [
                self.constraint_start_point_duplicate[i + self.actuation_number][X], 
                self.constraint_start_point_duplicate[i + self.actuation_number][Y], 
                self.origami_z_bias
            ]
            self.constraint_point_number[i + self.actuation_number] = 1

        # calculate initial center point
        for i in ti.ndrange(self.unit_indices_num):
            unit_indice = self.unit_indices[i]
            total_points = tm.vec3([0., 0., 0.])
            real_point_number = 0
            for j in ti.ndrange(self.unit_edge_max):
                if unit_indice[j] != -1:
                    total_points += self.original_vertices[unit_indice[j]]
                    real_point_number += 1
                else:
                    break
            center_point = total_points / real_point_number
            self.unit_center_initial_point[i] = center_point
            self.unit_center[i] = center_point

        # calculate initial length
        for i in ti.ndrange(self.constraint_number):
            self.constraint_initial_length[i] = 0.0
            start_point = self.constraint_start_point[i]
            for j in ti.ndrange(self.max_control_length):
                if self.unit_control[i, j] != -1:
                    self.constraint_initial_length[i] += (self.unit_center_initial_point[self.unit_control[i, j]] - start_point).norm()
                    start_point = self.unit_center_initial_point[self.unit_control[i, j]]
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

        # 初始化线的信息
        self.string_params[0] = self.string_k
        self.string_params[1] = self.shearing_k

        # 初始化边缘信息
        for i in ti.ndrange(30):
            self.border_vertex[2 * i] = [tm.cos(i / 15. * tm.pi) * self.panel_size, tm.sin(i / 15. * tm.pi) * self.panel_size, self.origami_z_bias]
            self.border_vertex[2 * i + 1] = [tm.cos((i + 1) / 15. * tm.pi) * self.panel_size, tm.sin((i + 1) / 15. * tm.pi) * self.panel_size, self.origami_z_bias]

        # 拉梅常数增益
        self.lames_bonus[0] = self.mu
        self.lames_bonus[1] = self.landa

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
        for i in range(self.line_total_indice_num):
            indice1 = self.line_pairs[i, 0]
            indice2 = self.line_pairs[i, 1]
            self.line_vertex[2 * i] = self.vertices[indice1]
            self.line_vertex[2 * i + 1] = self.vertices[indice2]

    @ti.func
    def getBendingForce(self, cs, ce, p1, p2, k, theta, crease_type, debug=False, enable_dynamic_change=False, index=-1):
        # 求折痕的信息
        xc = ce - cs
        xc_norm = xc.norm()

        # 求单元法向量
        f11 = p1 - cs
        f22 = p2 - cs
        n1 = xc.cross(f11)
        n2 = f22.cross(xc)
        self.n1[0] = n1
        self.n2[0] = n2

        # 求2倍单元面积
        a1 = n1.norm()
        a2 = n2.norm()

        # 法向量归一化
        un1 = tm.normalize(n1)
        un2 = tm.normalize(n2)
        uxc = tm.normalize(xc)

        dir = un1.cross(un2).dot(uxc)

        val = un1.dot(un2)

        if val > 1.0:
            val = 1.0
        elif val < -1.0:
            val = -1.0

        n_value = 0.
        # 求折叠角
        if not enable_dynamic_change:
            if index == -1:
                if crease_type == MOUNTAIN:
                    if dir >= 0.:
                        n_value = tm.cos(theta) - val
                    else:
                        if val <= -0.5: # 180~270
                            n_value = tm.cos(theta) + val + 2.
                        else:
                            n_value = tm.cos(theta) + val - 2.
                else:
                    if dir <= 0.:
                        n_value = val - tm.cos(theta)
                    else:
                        if val <= -0.5:
                            n_value = -2. - val - tm.cos(theta)
                        else:
                            n_value = 2. - val - tm.cos(theta)                 
        else:
            # if index == -1: # facet crease
            if dir >= 0.:
                n_value = tm.cos(theta) - val
            else:
                n_value = val - tm.cos(theta)
        # if index != -1:
        #     self.crease_folding_angle[index] = current_folding_angle
        
        # if index == 2:
        #     print(f11, f12, n2, val, dir, current_folding_angle)

        # 求折叠角与目标之差
        if abs(val + 1.) <= 0.015:
            self.folding_angle_reach_pi[0] = True

        # delta_folding_angle = current_folding_angle - theta
        # if debug:
        #     print(x0, p1, dir, current_folding_angle)

        # 计算折痕等效弯曲系数
        # k_crease = k * xc_norm if abs(current_folding_angle) < tm.pi else k * xc_norm * tm.exp(current_folding_angle)
        k_crease = k * xc_norm

        # 计算力矩，谷折痕折叠角大于0
        h1 = a1 / xc_norm
        h2 = a2 / xc_norm
        rpf1 = k_crease * n_value / h1 * un1
        rpf2 = k_crease * n_value / h2 * un2
        
        t1 = f11.dot(xc) / (xc_norm ** 2)
        t2 = f22.dot(xc) / (xc_norm ** 2)

        csf = -((1 - t1) * rpf1 + (1 - t2) * rpf2)
        cef = -(     t1  * rpf1 +      t2  * rpf2)

        self.bending_params[0] = k_crease / (a1 ** 2 * a2) / h1
        self.bending_params[1] = k_crease / (a1 * a2 ** 2) / h2
        self.bending_params[2] = t1
        self.bending_params[3] = t2
        self.bending_params[4] = a1
        self.bending_params[5] = a2
        self.bending_params[6] = dir

        #计算能量
        # energy = 0.5 * k_crease * delta_folding_angle ** 2 if abs(current_folding_angle) < tm.pi else 0.5 * k_crease * delta_folding_angle ** 2 * tm.exp(current_folding_angle)
        energy = 0.5 * k_crease * n_value ** 2

        return csf, cef, rpf1, rpf2, energy, n_value

    @ti.func
    def getViscousity(self, posvel: tm.vec3, other_posvel: tm.vec3, i, j) -> tm.vec3:
        ret = tm.vec3([0., 0., 0.])
        if abs(self.connection_matrix[i, j] - self.ori_sim.spring_k) < 1e-5:
            direction = other_posvel - posvel
            ret = self.viscousity * direction
        return ret

    @ti.func
    def getConstraintForce():
        pass

    @ti.func
    def get_position_with_index(self, index: int) -> tm.vec3:
        return self.x[index]

    @ti.func
    def get_velocity_with_index(self, index: int) -> tm.vec3:
        return self.v[index]

    @ti.func
    def calculateCenterPoint3DWithUnitId(self, unit_kps):
        kp_len = 0
        for i in range(len(unit_kps)):
            if unit_kps[i] != -1:
                self.points[i] = self.x[unit_kps[i]]
                kp_len += 1
        xx = 0.0
        y = 0.0
        z = 0.0
        for i in range(kp_len):
            xx += self.points[i][X]
            y += self.points[i][Y]
            z += self.points[i][Z]
        xx /= kp_len
        y /= kp_len
        z /= kp_len
        return tm.vec3([xx, y, z])

    @ti.func
    def calculateKpNumWithUnitId(self, unit_kps):
        kp_len = 0
        for i in range(len(unit_kps)):
            if unit_kps[i] != -1:
                kp_len += 1
        return kp_len
    
    @ti.func
    def getAxisForce(self, force_dir, i):
        delta_length = self.constraint_length[i] - self.constraint_initial_length[i]
        force = tm.vec3([.0, .0, .0])
        if delta_length < self.max_stretch_length:
            force = tm.normalize(force_dir) * self.string_params[0] * delta_length
        else:
            force = tm.normalize(force_dir) * self.string_params[0] * ((delta_length - 1 / 3) ** 3 + 19. / 27.)
        return force
    
    @ti.func
    def calculateNormalVectorWithUnitId(self, unit_kps):
        n = tm.vec3([0.0, 0.0, 0.0])
        kp_len = 0
        for i in range(len(unit_kps)):
            if unit_kps[i] != -1:
                kp_len += 1
        for i in range(1, kp_len - 1):
            v1 = self.x[unit_kps[i]] - self.x[unit_kps[0]]
            v2 = self.x[unit_kps[i + 1]] - self.x[unit_kps[0]]
            n += v1.cross(v2)
        return tm.normalize(n)

    @ti.func
    def getEnergy(self, sim_mode: bool, theta: data_type, gravitational_acc: tm.vec3) -> data_type:
        total_energy = 0.0
        for i in range(self.div_indices_num):
            x0 = self.x[self.indices[3 * i]]
            x1 = self.x[self.indices[3 * i + 1]]
            x2 = self.x[self.indices[3 * i + 2]]
            n = tm.normalize((x1 - x0).cross(x2 - x0))

            ds = ti.Matrix.cols([x1 - x0, x2 - x0, n])

            f = ds @ self.dm[i]

            #stvk model
            I = ti.Matrix.identity(data_type, 3)
            green_tensor = 0.5 * (f.transpose() @ f - I)
            # energy tensor
            energy_tensor = 0.0
            for j, k in ti.ndrange(3, 3):
                energy_tensor += green_tensor[j, k] ** 2
            #stress energy
            psi = self.lames_bonus[0] * energy_tensor + self.lames_bonus[1] / 2.0 * ti.Matrix.trace(green_tensor) ** 2
            total_energy += psi * self.A[i]
        # print(total_energy)
        
        #2 bending force for each crease
        # Second we calculate k_bending force
        if sim_mode == self.FOLD_SIM:
            for i in range(self.crease_pairs_num):
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
                    target_folding_angle = tm.pi * 0.99
                else:
                    coeff = self.crease_coeff[i]
                    target_folding_angle = (percent_theta - percent_low) / (percent_high - percent_low) * tm.pi
                    target_folding_angle = 2. * tm.atan2(coeff * tm.tan(target_folding_angle / 2.), 1.)

                if self.crease_type[i] == MOUNTAIN:
                    target_folding_angle = -target_folding_angle
                
                # 计算弯曲力
                csf, cef, rpf1, rpf2, energy, current_folding_angle = self.getBendingForce(
                    self.get_position_with_index(crease_start_index), 
                    self.get_position_with_index(crease_end_index), 
                    self.get_position_with_index(related_p1), 
                    self.get_position_with_index(related_p2),
                    self.ori_sim.bending_k, target_folding_angle, self.crease_type[i], False, False, -1)
            
                # self.crease_angle[i] = current_folding_angle
   
                total_energy += energy
        else:
            for i in range(self.crease_pairs_num):
                crease_start_index = self.crease_pairs[i, 0]
                crease_end_index = self.crease_pairs[i, 1]
                related_p1 = self.bending_pairs[i, 0]
                related_p2 = self.bending_pairs[i, 1]
                # 计算弯曲力
                
                csf, cef, rpf1, rpf2, energy, current_folding_angle = self.getBendingForce(
                    self.get_position_with_index(crease_start_index), 
                    self.get_position_with_index(crease_end_index), 
                    self.get_position_with_index(related_p1), 
                    self.get_position_with_index(related_p2),
                    self.ori_sim.bending_k, 0.0, 0, False, True, -1)

                # self.crease_angle[i] = current_folding_angle
                total_energy += energy
        # print(total_energy)
        # Third we calculate facet crease
        facet_k = 0.0
        if sim_mode == self.FOLD_SIM:
            facet_k = self.facet_k * self.lames_bonus[0]
        else:
            facet_k = self.facet_k * self.lames_bonus[0]

        if self.facet_bending_pairs_num > 0:
            for i in range(self.facet_crease_pairs_num):
                crease_start_index = self.facet_crease_pairs[i, 0]
                crease_end_index = self.facet_crease_pairs[i, 1]
                related_p1 = self.facet_bending_pairs[i, 0]
                related_p2 = self.facet_bending_pairs[i, 1]
                # 计算弯曲力
                csf, cef, rpf1, rpf2, energy, _ = self.getBendingForce(
                    self.get_position_with_index(crease_start_index), 
                    self.get_position_with_index(crease_end_index), 
                    self.get_position_with_index(related_p1), 
                    self.get_position_with_index(related_p2), 
                    facet_k, 0.0, 0, False, True, -1)

                total_energy += energy
        # print(total_energy)
        for i in range(self.kp_num):
            total_energy += 0.5 * self.masses[i] * self.v[i].norm() ** 2 - self.masses[i] * gravitational_acc[Z] * self.x[i][Z]
        # print(total_energy)
        return total_energy

    @ti.kernel
    def F(self, theta: data_type, sim_mode: bool, gravitational_acc: tm.vec3, enable_ground: bool, tsa_turning_angle: data_type):
        self.total_energy[0] = 0.0
        self.max_force[0] = 0.0

        for i in range(self.kp_num):
            self.force[i] = tm.vec3([0., 0., 0.])
        
        # print('LOGGING')
        #1 Elastic force for each triangle mesh

        #1.1 get mesh point of a triangle
        for i in range(self.div_indices_num):
            x0 = self.x[self.indices[3 * i]]
            x1 = self.x[self.indices[3 * i + 1]]
            x2 = self.x[self.indices[3 * i + 2]]

            n = tm.normalize((x1 - x0).cross(x2 - x0))

            ds = ti.Matrix.cols([x1 - x0, x2 - x0, n])

            f = ds @ self.dm[i]

            #stvk model
            identity = ti.Matrix.identity(data_type, 3)
            green_tensor = 0.5 * (f.transpose() @ f - identity)
            # energy tensor
            energy_tensor = 0.0
            for j, k in ti.ndrange(3, 3):
                energy_tensor += green_tensor[j, k] ** 2
            #stress energy
            psi = self.lames_bonus[0] * energy_tensor + self.lames_bonus[1] / 2.0 * ti.Matrix.trace(green_tensor) ** 2

            new_identity = tm.mat3([1., 0., 0.], [0., 1., 0.], [0., 0., 0.])
            piola_temp = 2.0 * self.lames_bonus[0] * green_tensor + self.lames_bonus[1] * ti.Matrix.trace(green_tensor) * new_identity
            piola = f @ piola_temp 
            H = -self.A[i] * piola @ self.dm[i].transpose()

            f1 = tm.vec3([H[0, 0], H[1, 0], H[2, 0]])
            f2 = tm.vec3([H[0, 1], H[1, 1], H[2, 1]])
            f0 = -f1 - f2

            self.force[self.indices[3 * i]] += f0
            self.force[self.indices[3 * i + 1]] += f1
            self.force[self.indices[3 * i + 2]] += f2

            if f0.norm() > self.max_force[0]:
                self.max_force[0] = f0.norm()
            if f1.norm() > self.max_force[0]:
                self.max_force[0] = f1.norm()
            if f2.norm() > self.max_force[0]:
                self.max_force[0] = f2.norm()

            self.total_energy[0] += psi * self.A[i]

            indice_x0 = 3 * self.indices[3 * i]
            indice_x1 = 3 * self.indices[3 * i + 1]
            indice_x2 = 3 * self.indices[3 * i + 2]
            # print(indice_x0, 3 * self.indices[3 * i])
            # print(indice_x1, 3 * self.indices[3 * i + 1])
            # print(indice_x2, 3 * self.indices[3 * i + 2])
            # derivative
            for j, k in ti.ndrange(3, 3):
                dF = self.dDs[j, k] @ self.dm[i]
                dE = 0.5 * (dF.transpose() @ f + f.transpose() @ dF)
                dP = dF @ piola_temp + f @ (2 * self.lames_bonus[0] * dE + self.lames_bonus[1] * ti.Matrix.trace(dE) * new_identity)
                dH = -self.A[i] * dP @ self.dm[i].transpose()
                df1 = tm.vec3([dH[0, 0], dH[1, 0], dH[2, 0]])
                df2 = tm.vec3([dH[0, 1], dH[1, 1], dH[2, 1]])
                df0 = -df1 - df2
    
                for l in ti.static(range(3)):
                    self.K_element[i][0 + l, 3 * j + k] = df0[l]
                    self.K_element[i][3 + l, 3 * j + k] = df1[l]
                    self.K_element[i][6 + l, 3 * j + k] = df2[l]
                
            self.triplets[i] = [indice_x0, indice_x1, indice_x2]
            # print("triplet: ")
            # print(self.triplets[i])
            # print("position: ")
            # print(x0, x1, x2)
            # print("K: ")
            # print(self.K_element[i])
            # print('\n')
            # sum = 0.
            # for j in range(9):
            #     for k in range(9):
            #         sum += self.K_element[i][j, k]
            # print(i, sum)
        # print("\nafter 1")
        # # # for i in range(self.kp_num):
        # # #     self.print_force[i] = self.force[i]
        # print(self.total_energy[0])
        # total_energy_store = self.total_energy[0]

        for i in range(self.kp_num):
            self.record_force[i] = ti.Vector([0.0, 0.0, 0.0])
            self.print_force[i] = self.force[i]


        # print("triplet: ")
        # print(self.triplets[0])
        # print("position: ")
        # print(self.x[self.triplets[0][0] / 3], self.x[self.triplets[0][1] / 3], self.x[self.triplets[0][2] / 3])
        # print(self.K_element[0])
        # print('\n')
        # print("triplet: ")
        # print(self.triplets[1])
        # print("position: ")
        # print(self.x[self.triplets[1][0] / 3], self.x[self.triplets[1][1] / 3], self.x[self.triplets[1][2] / 3])
        # print(self.K_element[1])
        # print('\n')
        #2 bending force for each crease
        # Second we calculate k_bending force
        if sim_mode == self.FOLD_SIM:
            for i in range(self.crease_pairs_num):
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
                    target_folding_angle = tm.pi * 0.99
                else:
                    coeff = self.crease_coeff[i]
                    target_folding_angle = (percent_theta - percent_low) / (percent_high - percent_low) * tm.pi
                    target_folding_angle = 2. * tm.atan2(coeff * tm.tan(target_folding_angle / 2.), 1.)

                if self.crease_type[i]:
                    target_folding_angle = -target_folding_angle
                
                x1 = self.get_position_with_index(crease_start_index)
                x2 = self.get_position_with_index(related_p2)
                x3 = self.get_position_with_index(crease_end_index)
                x4 = self.get_position_with_index(related_p1)
                # print(x1)
                # print(x2)
                # print(x3)
                # print(x4)
                # 计算弯曲力
                csf, cef, rpf1, rpf2, energy, n_value = self.getBendingForce(x1, x3, x4, x2,
                    self.ori_sim.bending_k, target_folding_angle, self.crease_type[i], False, False, -1)
                # 增加至force
                self.record_force[crease_start_index] += csf
                self.record_force[crease_end_index] += cef
                self.record_force[related_p1] += rpf1
                self.record_force[related_p2] += rpf2

                self.triplets_bending[i] = [3 * crease_start_index, 3 * related_p2, 3 * crease_end_index, 3 * related_p1]
                # print("triplet_bending: ")
                # print(self.triplets_bending[0])
                # print("position: ")
                # print(x1, x2, x3, x4)
                C1 = self.bending_params[0]
                C2 = self.bending_params[1]
                t1 = self.bending_params[2]
                t2 = self.bending_params[3]
                A = self.bending_params[4] * self.bending_params[5]
                dir = self.bending_params[6]
                n1 = self.n1[0]
                n2 = self.n2[0]
                df1dn1 = ti.Matrix.zero(data_type, 3, 3)
                df1dn2 = ti.Matrix.zero(data_type, 3, 3)
                df2dn1 = ti.Matrix.zero(data_type, 3, 3)
                df2dn2 = ti.Matrix.zero(data_type, 3, 3)
                if (self.crease_type[i] == VALLEY and dir <= 0) or (self.crease_type[i] == MOUNTAIN and dir >= 0):
                    df1dn1 = C1 * (ti.Matrix.cols([n1[X] * n2, n1[Y] * n2, n1[Z] * n2]) + A * n_value * ti.Matrix.identity(data_type, 3))
                    df1dn2 = C1 * (ti.Matrix.cols([n1[X] * n1, n1[Y] * n1, n1[Z] * n1]))
                    df2dn1 = C2 * (ti.Matrix.cols([n2[X] * n2, n2[Y] * n2, n2[Z] * n2]))
                    df2dn2 = C2 * (ti.Matrix.cols([n2[X] * n1, n2[Y] * n1, n2[Z] * n1]) + A * n_value * ti.Matrix.identity(data_type, 3))
                    # print(df2dn1)
                else:
                    df1dn1 = C1 * (-ti.Matrix.cols([n1[X] * n2, n1[Y] * n2, n1[Z] * n2]) + A * n_value * ti.Matrix.identity(data_type, 3))
                    df1dn2 = C1 * (-ti.Matrix.cols([n1[X] * n1, n1[Y] * n1, n1[Z] * n1]))
                    df2dn1 = C2 * (-ti.Matrix.cols([n2[X] * n2, n2[Y] * n2, n2[Z] * n2]))
                    df2dn2 = C2 * (-ti.Matrix.cols([n2[X] * n1, n2[Y] * n1, n2[Z] * n1]) + A * n_value * ti.Matrix.identity(data_type, 3))
                dn1dx1 = ti.Matrix.cols([[0., x4[Z] - x3[Z], x3[Y] - x4[Y]], [x3[Z] - x4[Z], 0., x4[X] - x3[X]], [x4[Y] - x3[Y], x3[X] - x4[X], 0.]])
                # dn1dx2 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                dn1dx3 = ti.Matrix.cols([[0., x1[Z] - x4[Z], x4[Y] - x1[Y]], [x4[Z] - x1[Z], 0., x1[X] - x4[X]], [x1[Y] - x4[Y], x4[X] - x1[X], 0.]])
                dn1dx4 = ti.Matrix.cols([[0., x3[Z] - x1[Z], x1[Y] - x3[Y]], [x1[Z] - x3[Z], 0., x3[X] - x1[X]], [x3[Y] - x1[Y], x1[X] - x3[X], 0.]])
                dn2dx1 = ti.Matrix.cols([[0., x3[Z] - x2[Z], x2[Y] - x3[Y]], [x2[Z] - x3[Z], 0., x3[X] - x2[X]], [x3[Y] - x2[Y], x2[X] - x3[X], 0.]])
                # print(dn2dx1)
                dn2dx2 = ti.Matrix.cols([[0., x1[Z] - x3[Z], x3[Y] - x1[Y]], [x3[Z] - x1[Z], 0., x1[X] - x3[X]], [x1[Y] - x3[Y], x3[X] - x1[X], 0.]])
                dn2dx3 = ti.Matrix.cols([[0., x2[Z] - x1[Z], x1[Y] - x2[Y]], [x1[Z] - x2[Z], 0., x2[X] - x1[X]], [x2[Y] - x1[Y], x1[X] - x2[X], 0.]])
                # dn2dx4 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                block_rp1_1 = dn1dx1.transpose() @ df1dn1 + dn2dx1.transpose() @ df1dn2
                block_rp1_2 = dn2dx2.transpose() @ df1dn2
                block_rp1_3 = dn1dx3.transpose() @ df1dn1 + dn2dx3.transpose() @ df1dn2
                block_rp1_4 = dn1dx4.transpose() @ df1dn1
                block_rp2_1 = dn1dx1.transpose() @ df2dn1 + dn2dx1.transpose() @ df2dn2
                block_rp2_2 = dn2dx2.transpose() @ df2dn2
                block_rp2_3 = dn1dx3.transpose() @ df2dn1 + dn2dx3.transpose() @ df2dn2
                block_rp2_4 = dn1dx4.transpose() @ df2dn1
                for k in range(3, 6):
                    for j in range(0, 3):
                        self.K_element_bending[i][j, k] = block_rp2_1[j, k - 3]
                    for j in range(3, 6):
                        self.K_element_bending[i][j, k] = block_rp2_2[j - 3, k - 3]
                    for j in range(6, 9):
                        self.K_element_bending[i][j, k] = block_rp2_3[j - 6, k - 3]
                    for j in range(9, 12):
                        self.K_element_bending[i][j, k] = block_rp2_4[j - 9, k - 3]
                for k in range(9, 12):
                    for j in range(0, 3):
                        self.K_element_bending[i][j, k] = block_rp1_1[j, k - 9]
                    for j in range(3, 6):
                        self.K_element_bending[i][j, k] = block_rp1_2[j - 3, k - 9]
                    for j in range(6, 9):
                        self.K_element_bending[i][j, k] = block_rp1_3[j - 6, k - 9]
                    for j in range(9, 12):
                        self.K_element_bending[i][j, k] = block_rp1_4[j - 9, k - 9]
                for k in range(0, 3):
                    for j in range(12):
                        self.K_element_bending[i][j, k    ] = -((1 - t1) * self.K_element_bending[i][j, k + 9] + (1 - t2) * self.K_element_bending[i][j, k + 3])
                        self.K_element_bending[i][j, k + 6] = -(     t1  * self.K_element_bending[i][j, k + 9] +      t2  * self.K_element_bending[i][j, k + 3])
                self.K_element_bending[i] = self.K_element_bending[i].transpose()
                self.crease_angle[i] = n_value
                self.total_energy[0] += energy
        else:
            for i in range(self.crease_pairs_num):
                crease_start_index = self.crease_pairs[i, 0]
                crease_end_index = self.crease_pairs[i, 1]
                related_p1 = self.bending_pairs[i, 0]
                related_p2 = self.bending_pairs[i, 1]
                # 计算弯曲力
                
                x1 = self.get_position_with_index(crease_start_index)
                x2 = self.get_position_with_index(related_p2)
                x3 = self.get_position_with_index(crease_end_index)
                x4 = self.get_position_with_index(related_p1)
                # 计算弯曲力
                csf, cef, rpf1, rpf2, energy, n_value = self.getBendingForce(x1, x3, x4, x2,
                    self.ori_sim.bending_k, 0.0, 0, False, True, -1)
                # 增加至force
                self.record_force[crease_start_index] += csf
                self.record_force[crease_end_index] += cef
                self.record_force[related_p1] += rpf1
                self.record_force[related_p2] += rpf2
            
                self.triplets_bending[i] = [3 * crease_start_index, 3 * related_p2, 3 * crease_end_index, 3 * related_p1]
                C1 = self.bending_params[0]
                C2 = self.bending_params[1]
                t1 = self.bending_params[2]
                t2 = self.bending_params[3]
                A = self.bending_params[4] * self.bending_params[5]
                dir = self.bending_params[6]
                n1 = self.n1[0]
                n2 = self.n2[0]
                df1dn1 = ti.Matrix.zero(data_type, 3, 3)
                df1dn2 = ti.Matrix.zero(data_type, 3, 3)
                df2dn1 = ti.Matrix.zero(data_type, 3, 3)
                df2dn2 = ti.Matrix.zero(data_type, 3, 3)
                if dir < 0:
                    df1dn1 = C1 * (ti.Matrix.cols([n1[X] * n2, n1[Y] * n2, n1[Z] * n2]) + A * n_value * ti.Matrix.identity(data_type, 3))
                    df1dn2 = C1 * (ti.Matrix.cols([n1[X] * n1, n1[Y] * n1, n1[Z] * n1]))
                    df2dn1 = C2 * (ti.Matrix.cols([n2[X] * n2, n2[Y] * n2, n2[Z] * n2]))
                    df2dn2 = C2 * (ti.Matrix.cols([n2[X] * n1, n2[Y] * n1, n2[Z] * n1]) + A * n_value * ti.Matrix.identity(data_type, 3))
                else:
                    df1dn1 = C1 * (-ti.Matrix.cols([n1[X] * n2, n1[Y] * n2, n1[Z] * n2]) + A * n_value * ti.Matrix.identity(data_type, 3))
                    df1dn2 = C1 * (-ti.Matrix.cols([n1[X] * n1, n1[Y] * n1, n1[Z] * n1]))
                    df2dn1 = C2 * (-ti.Matrix.cols([n2[X] * n2, n2[Y] * n2, n2[Z] * n2]))
                    df2dn2 = C2 * (-ti.Matrix.cols([n2[X] * n1, n2[Y] * n1, n2[Z] * n1]) + A * n_value * ti.Matrix.identity(data_type, 3))
                dn1dx1 = ti.Matrix.cols([[0., x4[Z] - x3[Z], x3[Y] - x4[Y]], [x3[Z] - x4[Z], 0., x4[X] - x3[X]], [x4[Y] - x3[Y], x3[X] - x4[X], 0.]])
                # dn1dx2 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                dn1dx3 = ti.Matrix.cols([[0., x1[Z] - x4[Z], x4[Y] - x1[Y]], [x4[Z] - x1[Z], 0., x1[X] - x4[X]], [x1[Y] - x4[Y], x4[X] - x1[X], 0.]])
                dn1dx4 = ti.Matrix.cols([[0., x3[Z] - x1[Z], x1[Y] - x3[Y]], [x1[Z] - x3[Z], 0., x3[X] - x1[X]], [x3[Y] - x1[Y], x1[X] - x3[X], 0.]])
                dn2dx1 = ti.Matrix.cols([[0., x3[Z] - x2[Z], x2[Y] - x3[Y]], [x2[Z] - x3[Z], 0., x3[X] - x2[X]], [x3[Y] - x2[Y], x2[X] - x3[X], 0.]])
                dn2dx2 = ti.Matrix.cols([[0., x1[Z] - x3[Z], x3[Y] - x1[Y]], [x3[Z] - x1[Z], 0., x1[X] - x3[X]], [x1[Y] - x3[Y], x3[X] - x1[X], 0.]])
                dn2dx3 = ti.Matrix.cols([[0., x2[Z] - x1[Z], x1[Y] - x2[Y]], [x1[Z] - x2[Z], 0., x2[X] - x1[X]], [x2[Y] - x1[Y], x1[X] - x2[X], 0.]])
                # dn2dx4 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                block_rp1_1 = dn1dx1.transpose() @ df1dn1 + dn2dx1.transpose() @ df1dn2
                block_rp1_2 = dn2dx2.transpose() @ df1dn2
                block_rp1_3 = dn1dx3.transpose() @ df1dn1 + dn2dx3.transpose() @ df1dn2
                block_rp1_4 = dn1dx4.transpose() @ df1dn1
                block_rp2_1 = dn1dx1.transpose() @ df2dn1 + dn2dx1.transpose() @ df2dn2
                block_rp2_2 = dn2dx2.transpose() @ df2dn2
                block_rp2_3 = dn1dx3.transpose() @ df2dn1 + dn2dx3.transpose() @ df2dn2
                block_rp2_4 = dn1dx4.transpose() @ df2dn1
                for k in range(3, 6):
                    for j in range(0, 3):
                        self.K_element_bending[i][j, k] = block_rp2_1[j, k - 3]
                    for j in range(3, 6):
                        self.K_element_bending[i][j, k] = block_rp2_2[j - 3, k - 3]
                    for j in range(6, 9):
                        self.K_element_bending[i][j, k] = block_rp2_3[j - 6, k - 3]
                    for j in range(9, 12):
                        self.K_element_bending[i][j, k] = block_rp2_4[j - 9, k - 3]
                for k in range(9, 12):
                    for j in range(0, 3):
                        self.K_element_bending[i][j, k] = block_rp1_1[j, k - 9]
                    for j in range(3, 6):
                        self.K_element_bending[i][j, k] = block_rp1_2[j - 3, k - 9]
                    for j in range(6, 9):
                        self.K_element_bending[i][j, k] = block_rp1_3[j - 6, k - 9]
                    for j in range(9, 12):
                        self.K_element_bending[i][j, k] = block_rp1_4[j - 9, k - 9]
                for k in range(0, 3):
                    for j in range(12):
                        self.K_element_bending[i][j, k    ] = -((1 - t1) * self.K_element_bending[i][j, k + 9] + (1 - t2) * self.K_element_bending[i][j, k + 3])
                        self.K_element_bending[i][j, k + 6] = -(     t1  * self.K_element_bending[i][j, k + 9] +      t2  * self.K_element_bending[i][j, k + 3])
                self.K_element_bending[i] = self.K_element_bending[i].transpose()
                self.crease_angle[i] = n_value
                self.total_energy[0] += energy
            
        # print(self.K_element_bending[0])
        # print('\nLOGGING END\n\n')
        # print("\nafter 2")
        for i in range(self.kp_num):
           self.print_force[i] = self.record_force[i]
        # print(self.total_energy[0] - total_energy_store)
        
        for i in range(self.kp_num):
            force_value = self.record_force[i].norm()
            if force_value > self.max_force[0]:
                self.max_force[0] = force_value
            self.force[i] += self.record_force[i]
            self.record_force[i] = ti.Vector([0.0, 0.0, 0.0])

        # Third we calculate facet crease
        facet_k = 0.0
        if sim_mode == self.FOLD_SIM:
            facet_k = self.facet_k * self.lames_bonus[0]
        else:
            facet_k = self.facet_k * self.lames_bonus[0]
        if self.facet_bending_pairs_num > 0:
            for i in range(self.facet_crease_pairs_num):
                crease_start_index = self.facet_crease_pairs[i, 0]
                crease_end_index = self.facet_crease_pairs[i, 1]
                related_p1 = self.facet_bending_pairs[i, 0]
                related_p2 = self.facet_bending_pairs[i, 1]
                # 计算弯曲力
                x1 = self.get_position_with_index(crease_start_index)
                x2 = self.get_position_with_index(related_p2)
                x3 = self.get_position_with_index(crease_end_index)
                x4 = self.get_position_with_index(related_p1)
                # 计算弯曲力
                csf, cef, rpf1, rpf2, energy, n_value = self.getBendingForce(x1, x3, x4, x2,
                    facet_k, 0.0, 0, False, True, -1)
                # 增加至force
                self.record_force[crease_start_index] += csf
                self.record_force[crease_end_index] += cef
                self.record_force[related_p1] += rpf1
                self.record_force[related_p2] += rpf2

                index = i + self.crease_pairs_num
                self.triplets_bending[index] = [3 * crease_start_index, 3 * related_p2, 3 * crease_end_index, 3 * related_p1]
                C1 = self.bending_params[0]
                C2 = self.bending_params[1]
                t1 = self.bending_params[2]
                t2 = self.bending_params[3]
                A = self.bending_params[4] * self.bending_params[5]
                dir = self.bending_params[6]
                n1 = self.n1[0]
                n2 = self.n2[0]
                # print(n1)
                # print(n2)
                df1dn1 = ti.Matrix.zero(data_type, 3, 3)
                df1dn2 = ti.Matrix.zero(data_type, 3, 3)
                df2dn1 = ti.Matrix.zero(data_type, 3, 3)
                df2dn2 = ti.Matrix.zero(data_type, 3, 3)
                if dir < 0:
                    df1dn1 = C1 * (ti.Matrix.cols([n1[X] * n2, n1[Y] * n2, n1[Z] * n2]) + A * n_value * ti.Matrix.identity(data_type, 3))
                    df1dn2 = C1 * (ti.Matrix.cols([n1[X] * n1, n1[Y] * n1, n1[Z] * n1]))
                    df2dn1 = C2 * (ti.Matrix.cols([n2[X] * n2, n2[Y] * n2, n2[Z] * n2]))
                    df2dn2 = C2 * (ti.Matrix.cols([n2[X] * n1, n2[Y] * n1, n2[Z] * n1]) + A * n_value * ti.Matrix.identity(data_type, 3))
                else:
                    df1dn1 = C1 * (-ti.Matrix.cols([n1[X] * n2, n1[Y] * n2, n1[Z] * n2]) + A * n_value * ti.Matrix.identity(data_type, 3))
                    df1dn2 = C1 * (-ti.Matrix.cols([n1[X] * n1, n1[Y] * n1, n1[Z] * n1]))
                    df2dn1 = C2 * (-ti.Matrix.cols([n2[X] * n2, n2[Y] * n2, n2[Z] * n2]))
                    df2dn2 = C2 * (-ti.Matrix.cols([n2[X] * n1, n2[Y] * n1, n2[Z] * n1]) + A * n_value * ti.Matrix.identity(data_type, 3))
                dn1dx1 = ti.Matrix.cols([[0., x4[Z] - x3[Z], x3[Y] - x4[Y]], [x3[Z] - x4[Z], 0., x4[X] - x3[X]], [x4[Y] - x3[Y], x3[X] - x4[X], 0.]])
                # dn1dx2 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                dn1dx3 = ti.Matrix.cols([[0., x1[Z] - x4[Z], x4[Y] - x1[Y]], [x4[Z] - x1[Z], 0., x1[X] - x4[X]], [x1[Y] - x4[Y], x4[X] - x1[X], 0.]])
                dn1dx4 = ti.Matrix.cols([[0., x3[Z] - x1[Z], x1[Y] - x3[Y]], [x1[Z] - x3[Z], 0., x3[X] - x1[X]], [x3[Y] - x1[Y], x1[X] - x3[X], 0.]])
                dn2dx1 = ti.Matrix.cols([[0., x3[Z] - x2[Z], x2[Y] - x3[Y]], [x2[Z] - x3[Z], 0., x3[X] - x2[X]], [x3[Y] - x2[Y], x2[X] - x3[X], 0.]])
                dn2dx2 = ti.Matrix.cols([[0., x1[Z] - x3[Z], x3[Y] - x1[Y]], [x3[Z] - x1[Z], 0., x1[X] - x3[X]], [x1[Y] - x3[Y], x3[X] - x1[X], 0.]])
                dn2dx3 = ti.Matrix.cols([[0., x2[Z] - x1[Z], x1[Y] - x2[Y]], [x1[Z] - x2[Z], 0., x2[X] - x1[X]], [x2[Y] - x1[Y], x1[X] - x2[X], 0.]])
                # dn2dx4 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                block_rp1_1 = dn1dx1.transpose() @ df1dn1 + dn2dx1.transpose() @ df1dn2
                block_rp1_2 = dn2dx2.transpose() @ df1dn2
                block_rp1_3 = dn1dx3.transpose() @ df1dn1 + dn2dx3.transpose() @ df1dn2
                block_rp1_4 = dn1dx4.transpose() @ df1dn1
                block_rp2_1 = dn1dx1.transpose() @ df2dn1 + dn2dx1.transpose() @ df2dn2
                block_rp2_2 = dn2dx2.transpose() @ df2dn2
                block_rp2_3 = dn1dx3.transpose() @ df2dn1 + dn2dx3.transpose() @ df2dn2
                block_rp2_4 = dn1dx4.transpose() @ df2dn1
                for k in range(3, 6):
                    for j in range(0, 3):
                        self.K_element_bending[index][j, k] = block_rp2_1[j, k - 3]
                    for j in range(3, 6):
                        self.K_element_bending[index][j, k] = block_rp2_2[j - 3, k - 3]
                    for j in range(6, 9):
                        self.K_element_bending[index][j, k] = block_rp2_3[j - 6, k - 3]
                    for j in range(9, 12):
                        self.K_element_bending[index][j, k] = block_rp2_4[j - 9, k - 3]
                for k in range(9, 12):
                    for j in range(0, 3):
                        self.K_element_bending[index][j, k] = block_rp1_1[j, k - 9]
                    for j in range(3, 6):
                        self.K_element_bending[index][j, k] = block_rp1_2[j - 3, k - 9]
                    for j in range(6, 9):
                        self.K_element_bending[index][j, k] = block_rp1_3[j - 6, k - 9]
                    for j in range(9, 12):
                        self.K_element_bending[index][j, k] = block_rp1_4[j - 9, k - 9]
                for k in range(0, 3):
                    for j in range(12):
                        self.K_element_bending[index][j, k    ] = -((1 - t1) * self.K_element_bending[index][j, k + 9] + (1 - t2) * self.K_element_bending[index][j, k + 3])
                        self.K_element_bending[index][j, k + 6] = -(     t1  * self.K_element_bending[index][j, k + 9] +      t2  * self.K_element_bending[index][j, k + 3])
                self.K_element_bending[index] = self.K_element_bending[index].transpose()
                self.total_energy[0] += energy
                # print(energy)

        # print("\nafter 3") 
        # # for i in range(self.kp_num):
        # #     print(self.record_force[i])
        # print(self.total_energy[0])

        for i in range(self.kp_num):
            force_value = self.record_force[i].norm()
            if force_value > self.max_force[0]:
                self.max_force[0] = force_value
            self.force[i] += self.record_force[i]
            self.record_force[i] = ti.Vector([0.0, 0.0, 0.0])

        # # Fourth
        # for i in range(self.kp_num):
        #     point_v = self.get_velocity_with_index(i)
        #     for j in range(self.kp_num):
        #         f = self.getViscousity(point_v, self.get_velocity_with_index(j), i, j)
        #         self.record_force[i] += f

        # for i in range(self.kp_num):
        #     force_value = self.record_force[i].norm()
        #     if force_value > self.max_force[0]:
        #         self.max_force[0] = force_value
        #     self.force[i] += self.record_force[i]
        #     self.record_force[i] = ti.Vector([0.0, 0.0, 0.0])

        # # Fifth we calculate total viscousity
        # for i in range(self.kp_num):
        #     point_v = self.get_velocity_with_index(i)
        #     self.force[i] += -.01 * self.viscousity * point_v

        for i in range(self.kp_num):
            self.total_energy[0] += 0.5 * self.masses[i] * self.v[i].norm() ** 2 - self.masses[i] * gravitational_acc[Z] * self.x[i][Z]

        # print(self.total_energy[0])
            
        # Sixth TSA_SIM PART
        
        # begin
        # Seventh we calculate constraint force
        if sim_mode == self.TSA_SIM:
            for i in range(self.constraint_number):
                # self.constraint_length[i] = 0.0
                # start_point = self.constraint_start_point[i]
                for j in range(self.max_control_length):
                    unit_id = self.unit_control[i, j]
                    if unit_id != -1:
                        center = self.calculateCenterPoint3DWithUnitId(self.unit_indices[unit_id])
                        # force_dir = center - start_point
                        self.unit_center[unit_id] = center
                        # self.constraint_length[i] += force_dir.norm()
                        # start_point = center            
                    else:
                        break
            accumulate_index = 0
            self.calculateStringDistanceParallel(tsa_turning_angle)
            for i in range(self.total_root_number):
                dup_time = self.constraint_point_number[i]
                if dup_time > 1:
                    # if abs(self.tsa_turning_angle) <= tm.pi * 2 / dup_time:
                    if not self.constraint_angle_enable[i] or (self.constraint_angle_enable[i] and abs(tsa_turning_angle) < abs(self.constraint_angle[i])):
                        for j in range(dup_time):
                            self.constraint_start_point[j + accumulate_index] = [
                                self.constraint_start_point_duplicate[i][X] + self.direction[i][X] * tm.cos(tsa_turning_angle + math.pi * 2 * j / dup_time), 
                                self.constraint_start_point_duplicate[i][Y] + self.direction[i][Y] * tm.cos(tsa_turning_angle + math.pi * 2 * j / dup_time), 
                                self.origami_z_bias + self.r1 * tm.sin(tsa_turning_angle + math.pi * 2 * j / dup_time)
                            ]
                            self.constraint_length[j + accumulate_index] = 0.0
                            start_point = self.constraint_start_point[j + accumulate_index]
                            for k in range(self.max_control_length):
                                if self.unit_control[j + accumulate_index, k] != -1:
                                    end_point = self.unit_center[self.unit_control[j + accumulate_index, k]]
                                    self.constraint_length[j + accumulate_index] += (end_point - start_point).norm()
                                    start_point = end_point
                                else:
                                    break
                            if self.constraint_end_point_existence[j + accumulate_index]:
                                self.constraint_length[j + accumulate_index] += (self.constraint_end_point[j + accumulate_index] - start_point).norm()
                            # self.constraint_initial_length[j + accumulate_index] = self.backup_constraint_length[j + accumulate_index]
                        accumulate_index += dup_time
                    else:
                        true_angle = tsa_turning_angle - self.constraint_angle[i]
                        for j in range(dup_time):
                            self.visual_constraint_start_point[j + accumulate_index] = [
                                self.constraint_start_point_duplicate[i][X] + self.direction[i][X] * tm.cos(tsa_turning_angle + math.pi * 2 * j / dup_time), 
                                self.constraint_start_point_duplicate[i][Y] + self.direction[i][Y] * tm.cos(tsa_turning_angle + math.pi * 2 * j / dup_time), 
                                self.origami_z_bias + self.r1 * tm.sin(tsa_turning_angle + math.pi * 2 * j / dup_time)
                            ]

                        x0 = self.constraint_start_point_duplicate[i]

                        x1 = tm.vec3([.0, .0, .0])
                        for j in range(accumulate_index, accumulate_index + dup_time):
                            x1 += self.unit_center[self.unit_control[j, 0]]
                        
                        x1 /= dup_time

                        # print(x1)

                        d2 = .0
                        for j in range(accumulate_index, accumulate_index + dup_time):
                            d2 += (self.unit_center[self.unit_control[i, 0]] - x1).norm()
                        d2 = d2 / dup_time * 2.

                        eff_1 = self.d1 / (d2 + self.d1 + abs(true_angle) * self.ds)
                        eff_2 = d2 / (d2 + self.d1 + abs(true_angle) * self.ds)

                        new_cp1 = eff_1 * (x1 - x0) + x0
                        new_cp2 = x1 - eff_2 * (x1 - x0)
                        
                        self.intersection_point[i] = new_cp1

                        for j in range(accumulate_index, accumulate_index + dup_time):
                            self.constraint_start_point[j] = new_cp2
                            # self.constraint_length[j] = tm.sqrt((x1 - x0).norm() ** 2 + (d2 + self.d1 + abs(true_angle) * self.ds) ** 2 / 4.)
                            self.constraint_length[j] = (self.unit_center[self.unit_control[j, 0]] - new_cp2).norm() + (new_cp1 - self.visual_constraint_start_point[j]).norm() + tm.sqrt((new_cp2 - new_cp1).norm() ** 2 + (true_angle * self.ds) ** 2)
                            
                            start_point = self.unit_center[self.unit_control[j, 0]]
                            for k in range(1, self.max_control_length):
                                if self.unit_control[j, k] != -1:
                                    end_point = self.unit_center[self.unit_control[j, k]]
                                    self.constraint_length[j] += (end_point - start_point).norm()
                                    start_point = end_point
                                else:
                                    break
                            if self.constraint_end_point_existence[j]:
                                self.constraint_length[j] += (self.constraint_end_point[j] - start_point).norm()

                        accumulate_index += dup_time
                else:
                    self.constraint_start_point[accumulate_index] = [
                        self.constraint_start_point_duplicate[i][X], 
                        self.constraint_start_point_duplicate[i][Y], 
                        self.origami_z_bias
                    ]
                    self.constraint_length[accumulate_index] = 0.0
                    start_point = self.constraint_start_point[accumulate_index]
                    for k in range(self.max_control_length):
                        if self.unit_control[accumulate_index, k] != -1:
                            end_point = self.unit_center[self.unit_control[accumulate_index, k]]
                            self.constraint_length[accumulate_index] += (end_point - start_point).norm()
                            start_point = end_point
                        else:
                            break
                    if self.constraint_end_point_existence[accumulate_index]:
                        self.constraint_length[accumulate_index] += (self.constraint_end_point[accumulate_index] - start_point).norm()
                    accumulate_index += 1  
            
            for i in range(self.constraint_number):
                # self.constraint_length[i] += (self.constraint_end_point[i] - start_point).norm()
                if self.constraint_length[i] > self.constraint_initial_length[i]: # add force
                    self.total_energy[0] += self.string_params[0] * tm.exp(self.constraint_length[i] - self.constraint_initial_length[i])
                    start_point = self.constraint_start_point[i]
                    before_force_dir = tm.vec3([0.0, 0.0, 0.0])
                    before_tight_indice = 0.0
                    for j in range(self.max_control_length):
                        if self.unit_control[i, j] != -1:
                            kp_id = self.unit_indices[self.unit_control[i, j]]
                            unit_kp_num = self.calculateKpNumWithUnitId(kp_id)
                            end_point = self.calculateCenterPoint3DWithUnitId(kp_id)
                            force_dir = end_point - start_point
                            tight_indice = 0.0
                            axis_force = tm.vec3([0., 0., 0.])
                                
                            if j == 0: # 1 point
                                #引导力
                                hole_direction = self.hole_dir[i, j]
                                nm = self.calculateNormalVectorWithUnitId(self.unit_indices[self.unit_control[i, j]])
                                penetration = tm.normalize(force_dir).dot(nm) * hole_direction
                                # print(penetration)
                                if penetration >= -self.beta: #penetration
                                    tight_indice = (penetration + self.beta) / self.beta
                                    if penetration >= 0:
                                        force_dir = force_dir - (force_dir.dot(nm)) * nm - nm * hole_direction
                                axis_force = self.getAxisForce(force_dir, i)
                                # axis_force = tm.normalize(force_dir) * self.string_params[0] * (self.constraint_length[i] - self.constraint_initial_length[i])
                                for k in range(unit_kp_num):
                                    # self.force[kp_id[k]] += -axis_force
                                    self.record_force[kp_id[k]] += -axis_force
                            else:
                                #引导力, 只算系数
                                true_tight_indice = 0.0
                                hole_direction = self.hole_dir[i, j - 1]
                                nm = self.calculateNormalVectorWithUnitId(self.unit_indices[self.unit_control[i, j - 1]])
                                penetration = tm.normalize(force_dir).dot(nm) * hole_direction
                                # print(penetration)
                                if penetration >= -self.beta: #penetration
                                    true_tight_indice = (penetration + self.beta) / self.beta
                                #引导力, 用于存储
                                tight_indice = 0.0
                                hole_direction = self.hole_dir[i, j]
                                nm = self.calculateNormalVectorWithUnitId(self.unit_indices[self.unit_control[i, j]])
                                # print(nm)
                                penetration = tm.normalize(force_dir).dot(nm) * hole_direction
                                if penetration >= -self.beta: #penetration
                                    tight_indice = (penetration + self.beta) / self.beta
                                    if penetration >= 0:
                                        force_dir = force_dir - (force_dir.dot(nm)) * nm - nm * hole_direction

                                before_unit_id = self.unit_control[i, j - 1]
                                before_kp_id = self.unit_indices[before_unit_id]
                                before_kp_num = self.calculateKpNumWithUnitId(before_kp_id)
                                before_n = self.calculateNormalVectorWithUnitId(self.unit_indices[before_unit_id])

                                axis_force = self.getAxisForce(force_dir, i)
                                for k in range(before_kp_num):
                                    # self.force[before_kp_id[k]] += axis_force
                                    self.record_force[before_kp_id[k]] += axis_force
                                for k in range(unit_kp_num):
                                    # self.force[kp_id[k]] += -axis_force
                                    self.record_force[kp_id[k]] += -axis_force
                                
                                # print(tm.normalize(force_dir))
                                # print(tm.normalize(before_force_dir))
                                # shearing_dir = tm.normalize(force_dir) - tm.normalize(before_force_dir)
                                # for k in range(len(before_kp_id)):
                                #     force[before_kp_id[k]] += shearing_dir * shearing_k * (20.0 * shearing_dir.norm())
                                #shearing
                                n_string = before_force_dir.cross(force_dir)
                                if abs(n_string.norm()) > 1e-5:
                                    n_string = tm.normalize(n_string)
                                    val = before_force_dir.dot(force_dir) / before_force_dir.norm() / force_dir.norm()
                                    if val > 1.0:
                                        val = 1.0
                                    elif val < -1.0:
                                        val = -1.0
                                    angle = tm.acos(val)
                                    # print(angle)
                                    # if i == 0:
                                    #     print(angle)
                                    avg_vel = tm.vec3([.0, .0, .0])
                                    for k in range(before_kp_num):
                                        avg_vel += self.get_velocity_with_index(before_kp_id[k])
                                    if abs(avg_vel.norm()) > 1e-5 :
                                        avg_vel = tm.normalize(avg_vel)
                                    #     angle1 = avg_vel.dot(-before_force_dir / before_force_dir.norm())
                                    #     angle2 = avg_vel.dot(force_dir / force_dir.norm())
                                    # else:
                                    #     dir = [.0, .0, .0]
                                    # print(avg_vel)
                                    shearing_force = self.string_params[1] * angle * n_string.cross(before_force_dir) / (before_force_dir.norm() ** 2) + self.string_params[1] * angle * n_string.cross(force_dir) / (force_dir.norm() ** 2) + 2.0 * self.string_params[1] * angle * tm.normalize(-before_force_dir + force_dir) / (before_force_dir.norm() + force_dir.norm())
                                    if self.constraint_length[i] - self.constraint_initial_length[i] > 0:
                                        shearing_force *= (4.0 * (self.constraint_length[i] - self.constraint_initial_length[i]) / self.max_stretch_length + 1.0) / before_kp_num
                                    # print(true_tight_indice, before_tight_indice)
                                    if before_tight_indice <= 0.0:
                                        force_val = true_tight_indice * (shearing_force - self.miu * shearing_force.norm() * avg_vel)
                                        for k in range(before_kp_num):
                                            # force[before_kp_id[k]] += shearing_force - miu * (shearing_force.norm()) * avg_vel
                                            self.record_force[before_kp_id[k]] += force_val
                                    else:
                                        if true_tight_indice <= 0.0:
                                            force_val = before_tight_indice * (shearing_force - self.miu * shearing_force.norm() * avg_vel)
                                            for k in range(before_kp_num):
                                                # force[before_kp_id[k]] += shearing_force - miu * (shearing_force.norm()) * avg_vel
                                                self.record_force[before_kp_id[k]] += force_val
                                        else:
                                            force_val = abs(true_tight_indice - before_tight_indice) * (shearing_force - self.miu * shearing_force.norm() * avg_vel)
                                            for k in range(before_kp_num):
                                                # force[before_kp_id[k]] += shearing_force - miu * (shearing_force.norm()) * avg_vel
                                                self.record_force[before_kp_id[k]] += force_val

                                    # for k in range(before_kp_num):
                                    #     # force[before_kp_id[k]] += shearing_force - miu * (shearing_force.norm()) * avg_vel
                                    #     force[before_kp_id[k]] += tight_indice * shearing_force - miu * shearing_force.norm() * avg_vel - before_tight_indice * shearing_force
                                    
                                    #bonus momentum
                                    if before_tight_indice > 0.0 and true_tight_indice > 0.0:
                                        vp = tm.normalize(before_force_dir - before_force_dir.dot(before_n) * before_n)
                                        vq = tm.normalize(force_dir - force_dir.dot(before_n) * before_n)
                                        vt = tm.normalize(vp + vq)
                                        # print("vt: ", vt)
                                        # print("before_n: ", before_n)
                                        bonus_val = shearing_force.norm() * self.d_hole * (before_tight_indice + true_tight_indice) / 2.0
                                        total_force = tm.vec3([.0, .0, .0])
                                        # print("bonus_val: ", bonus_val)

                                        center = self.unit_center[before_unit_id] 
    
                                        hole_direction = self.hole_dir[i, j - 1]

                                        share_bonus = bonus_val * before_n * hole_direction

                                        for k in range(before_kp_num):
                                            vl = center - self.x[before_kp_id[k]]
                                            length = vl.dot(vt)
                                            temp_force = -bonus_val / length * before_n * hole_direction
                                            if bonus_val / length >= shearing_force.norm() * self.d_hole:
                                                temp_force = -shearing_force.norm() * self.d_hole * before_n * hole_direction
                                            elif bonus_val / length <= -shearing_force.norm() * self.d_hole:
                                                temp_force = shearing_force.norm() * self.d_hole * before_n * hole_direction
                                            total_force += temp_force
                                            # self.force[before_kp_id[k]] += temp_force
                                            self.record_force[before_kp_id[k]] += temp_force
                                        # for k in range(before_kp_num):
                                        #     vl = center - self.x[before_kp_id[k]]
                                        #     length = vl.dot(vt)
                                        #     temp_force = -share_bonus / length
                                        #     if bonus_val / length >= share_bonus.norm():
                                        #         temp_force = -share_bonus
                                        #     elif bonus_val / length <= -share_bonus.norm():
                                        #         temp_force = share_bonus
                                        #     total_force += temp_force
                                        #     self.force[before_kp_id[k]] += temp_force
                                            # self.record_force[before_kp_id[k]] += temp_force
                                            # if before_kp_id[k] == 3:
                                            #     print(temp_force)
                                        
                                        #补偿
                                        add_force = -total_force / before_kp_num
                                        for k in range(before_kp_num):
                                            self.record_force[before_kp_id[k]] += add_force
                                            # self.record_force[before_kp_id[k]] += add_force
                                            # if before_kp_id[k] == 3:
                                            #     print(add_force)
                                        
                                        # print("checkpoint")
                                        # for k in range(kp_num):
                                        #     print(k, force[k] / point_mass + gravitational_acc)
                                        # #补偿
                                        # add_force = -total_force / len(before_kp_id)
                                        # for k in range(len(before_kp_id)):
                                        #     force[before_kp_id[k]] += add_force
                                        #     record_force[before_kp_id[k]] += add_force
                                        #     if before_kp_id[k] == 3:
                                        #         print(add_force)
                                            # print("unit_id: ", before_kp_id[k])
                                            # print("force: ", -bonus_val / length * before_n)

                                        # for k in range(kp_num):
                                        #     print(k, force[k])
                            # print(j)
                            # print(axis_force)
                            start_point = end_point
                            before_force_dir = force_dir
                            before_tight_indice = tight_indice
                        else: #end
                            if self.constraint_end_point_existence[i]:
                                before_unit_id = self.unit_control[i, j - 1]
                                force_dir = self.constraint_end_point[i] - start_point
                                
                                #引导力
                                true_tight_indice = 0.0
                                hole_direction = self.hole_dir[i, j - 1]
                                nm = self.calculateNormalVectorWithUnitId(self.unit_indices[before_unit_id])

                                penetration = tm.normalize(force_dir).dot(nm) * hole_direction
                                if penetration >= -self.beta: #penetration
                                    true_tight_indice = (penetration + self.beta) / self.beta
                                    # force_dir = force_dir - (force_dir.dot(nm)) * nm - nm * hole_direction

                                before_kp_id = self.unit_indices[before_unit_id]
                                before_kp_num = self.calculateKpNumWithUnitId(before_kp_id)
                                before_n = self.calculateNormalVectorWithUnitId(self.unit_indices[before_unit_id])

                                axis_force = axis_force = self.getAxisForce(force_dir, i)
                                self.end_force[0] = axis_force.norm()

                                for k in range(before_kp_num):
                                    # self.force[before_kp_id[k]] += axis_force
                                    self.record_force[before_kp_id[k]] += axis_force
                                #shearing
                                n_string = before_force_dir.cross(force_dir)
                                if abs(n_string.norm()) > 1e-5:
                                    n_string = tm.normalize(n_string)
                                    val = before_force_dir.dot(force_dir) / before_force_dir.norm() / force_dir.norm()
                                    if val > 1.0:
                                        val = 1.0
                                    elif val < -1.0:
                                        val = -1.0
                                    angle = tm.acos(val)

                                    # avg_vel = tm.vec3([.0, .0, .0])
                                    # for k in range(before_kp_num):
                                    #     avg_vel += get_velocity_with_index(before_kp_id[k])
                                    
                                    # if abs(avg_vel.norm()) > 1e-5 :
                                    #     avg_vel = tm.normalize(avg_vel)
                                    #     angle1 = tm.acos(avg_vel.dot(-before_force_dir) / before_force_dir.norm())
                                    #     print(avg_vel)
                                    avg_vel = tm.vec3([.0, .0, .0])
                                    for k in range(before_kp_num):
                                        avg_vel += self.get_velocity_with_index(before_kp_id[k])
                                    if abs(avg_vel.norm()) > 1e-5 :
                                        avg_vel = tm.normalize(avg_vel)
                                    # dir = -(before_force_dir) / before_force_dir.norm()
                                    # else:
                                    #     print("0")
                                    shearing_force = self.string_params[1] * angle * n_string.cross(before_force_dir) / (before_force_dir.norm() ** 2) + self.string_params[1] * angle * n_string.cross(force_dir) / (force_dir.norm() ** 2) + 2.0 * self.string_params[1] * angle * tm.normalize(-before_force_dir + force_dir) / (before_force_dir.norm() + force_dir.norm())
                                    if self.constraint_length[i] - self.constraint_initial_length[i] > 0:
                                        shearing_force *= (4.0 * (self.constraint_length[i] - self.constraint_initial_length[i]) / self.max_stretch_length + 1.0) / before_kp_num
                                    # print(true_tight_indice, before_tight_indice)
                                    if before_tight_indice <= 0.0:
                                        force_val = true_tight_indice * shearing_force - self.miu * shearing_force.norm() * avg_vel
                                        for k in range(before_kp_num):
                                            # force[before_kp_id[k]] += shearing_force - miu * (shearing_force.norm()) * avg_vel
                                            self.record_force[before_kp_id[k]] += force_val
                                    else:
                                        if true_tight_indice <= 0.0:
                                            force_val = before_tight_indice * shearing_force - self.miu * shearing_force.norm() * avg_vel
                                            for k in range(before_kp_num):
                                                # force[before_kp_id[k]] += shearing_force - miu * (shearing_force.norm()) * avg_vel
                                                self.record_force[before_kp_id[k]] += force_val
                                        else:
                                            force_val = abs(true_tight_indice - before_tight_indice) * shearing_force - self.miu * shearing_force.norm() * avg_vel
                                            for k in range(before_kp_num):
                                                # force[before_kp_id[k]] += shearing_force - miu * (shearing_force.norm()) * avg_vel
                                                self.record_force[before_kp_id[k]] += force_val
                                    # for k in range(before_kp_num):
                                    #     force[before_kp_id[k]] += -tight_indice * shearing_force - miu * shearing_force.norm() * avg_vel + before_tight_indice * shearing_force
                                        # miu * (shearing_force.norm() - point_mass * gravitational_acc[Z]) * avg_vel
                                    #bonus momentum
                                    if before_tight_indice > 0.0 and true_tight_indice > 0.0:
                                        vp = tm.normalize(before_force_dir - before_force_dir.dot(before_n) * before_n)
                                        vq = tm.normalize(force_dir - force_dir.dot(before_n) * before_n)
                                        vt = tm.normalize(vp + vq)
                                        bonus_val = shearing_force.norm() * self.d_hole * (before_tight_indice + true_tight_indice) / 2.0
                                        total_force = tm.vec3([.0, .0, .0])

                                        center = self.unit_center[before_unit_id]

                                        hole_direction = self.hole_dir[i, j - 1]

                                        share_bonus = bonus_val * before_n * hole_direction

                                        for k in range(before_kp_num):
                                            vl = center - self.x[before_kp_id[k]]
                                            length = vl.dot(vt)
                                            temp_force = -bonus_val / length * before_n * hole_direction
                                            if bonus_val / length >= shearing_force.norm() * self.d_hole:
                                                temp_force = -shearing_force.norm() * self.d_hole * before_n * hole_direction
                                            elif bonus_val / length <= -shearing_force.norm() * self.d_hole:
                                                temp_force = shearing_force.norm() * self.d_hole * before_n * hole_direction
                                            total_force += temp_force
                                            # self.force[before_kp_id[k]] += temp_force
                                            self.record_force[before_kp_id[k]] += temp_force
                                        # for k in range(before_kp_num):
                                        #     vl = center - self.x[before_kp_id[k]]
                                        #     length = vl.dot(vt)
                                        #     temp_force = -share_bonus / length
                                        #     if bonus_val / length >= share_bonus.norm():
                                        #         temp_force = -share_bonus
                                        #     elif bonus_val / length <= -share_bonus.norm():
                                        #         temp_force = share_bonus
                                        #     total_force += temp_force
                                        #     self.force[before_kp_id[k]] += temp_force
                                            # self.record_force[before_kp_id[k]] += temp_force
                                            # if before_kp_id[k] == 3:
                                            #     print(temp_force)
                                        
                                        #补偿
                                        add_force = -total_force / before_kp_num
                                        for k in range(before_kp_num):
                                            self.record_force[before_kp_id[k]] += add_force
                                            # self.record_force[before_kp_id[k]] += add_force
                                            # if before_kp_id[k] == 3:
                                            #     print(add_force)
                                        # for k in range(kp_num):
                                        #     print(k, force[k] / point_mass + gravitational_acc)
                            break

        for i in range(self.kp_num):
            force_value = self.record_force[i].norm()
            if force_value > self.max_force[0]:
                self.max_force[0] = force_value
            self.force[i] += self.record_force[i] 
        # end
            
        # Seventh
        for i in range(self.kp_num):
            f = gravitational_acc * self.masses[i]
            self.record_force[i] += f

        for i in range(self.kp_num):
            force_value = self.record_force[i].norm()
            if force_value > self.max_force[0]:
                self.max_force[0] = force_value
            self.force[i] += self.record_force[i]
            self.record_force[i] = ti.Vector([0.0, 0.0, 0.0])

        # sum = tm.vec3([0., 0., 0.])
        # for i in range(self.kp_num):
        #     # print(self.force[i])
        #     sum += self.force[i]
        #     self.dv[i] = self.force[i] / self.masses[i] + gravitational_acc  
        # print(sum)

    @ti.kernel
    def fill_K(self, h: data_type, builder: ti.types.sparse_matrix_builder()):
        for index in range(3 * self.kp_num):
            builder[index, index] += self.masses[index // 3]

        for i in range(self.div_indices_num):
            indice_list = self.triplets[i]
            K = self.K_element[i]
            for j, k in ti.ndrange(3, 3):
                for jj, kk in ti.ndrange(3, 3):
                    builder[indice_list[j] + jj, indice_list[k] + kk] += -(h ** 2) * K[3 * j + jj, 3 * k + kk]
        
        for ii in range(self.bending_pairs_num + self.facet_bending_pairs_num):
            indice_list_bending = self.triplets_bending[ii]
            K_bending = self.K_element_bending[ii]
            # print(ii, "/", self.bending_pairs_num + self.facet_bending_pairs_num)
            for ij, ik in ti.ndrange(4, 4):
                for ijj, ikk in ti.ndrange(3, 3):
                    builder[indice_list_bending[ij] + ijj, indice_list_bending[ik] + ikk] += -(h ** 2) * K_bending[3 * ij + ijj, 3 * ik + ikk]
    
    @ti.kernel
    def fill_b(self, h: data_type):
        for i in range(3 * self.kp_num):
            self.b[i] = self.masses[i // 3] * self.v[i // 3][i % 3] + h * self.force[i // 3][i % 3]

    @ti.kernel
    def step_xv(self, time_step: data_type, enable_ground: bool, sim_mode: bool, theta: data_type, gravitational_acc: tm.vec3) -> int:
        can_step = 0
        
        # print("u0, ", self.u0)

        # linear search
        self.backup_x()
        self.backup_v()
        # self.backup_angle()

        now_energy = self.total_energy[0]

        # print(now_energy)
        # for i in range(self.crease_pairs_num):
        #     print(self.crease_angle[i])

        # min_energy = now_energy

        # beta = 0.
        beta = 1.

        min_energy_beta = 0.

        max_dim = self.linear_search_step

        max_scale = 1.0

        # sum of u0
        # ux = 0.0
        # uy = 0.0
        # uz = 0.0
        # for i in range(self.kp_num):
        #     ux += self.u0[3 * i]
        #     uy += self.u0[3 * i + 1]
        #     uz += self.u0[3 * i + 2]
        # print(ux, uy, uz)
        # small_step = 1. / 2 ** (max_dim - 1)
        small_step = 1. / max_dim * self.linear_search_start[0]

        for time in range(0, max_dim):
            # beta = beta + direction / (2 ** time)
            beta = self.linear_search_start[0] - time * small_step
            # print("info: ", time, beta)
            for i in range(3 * self.kp_num):
                self.v[i // 3][i % 3] = beta * self.u0[i] + (1 - beta) * self.back_up_v[i // 3][i % 3]
            for i in range(self.kp_num):
                self.x[i] = self.back_up_x[i] + time_step * self.v[i]
            new_energy = self.getEnergy(sim_mode, theta, gravitational_acc)
            self.energy_buffer[time] = new_energy

            # print(beta, new_energy)
            # if new_energy - now_energy > 25:
            #     print(new_energy, beta)
            #     for i in range(self.crease_pairs_num):
            #         print(self.crease_angle[i] - self.backup_crease_angle[i])

            # print("new_energy: ", new_energy, "/", now_energy)
            if tm.isnan(new_energy):
                can_step = 0
                self.energy_buffer[time] = now_energy * 100.
            # else:
            #     if new_energy < now_energy:
            #         can_step = 1
            #         # min_energy = new_energy
            #         if beta > min_energy_beta:
            #             min_energy_beta = beta
            #         # #upper
            #         # for i in range(3 * self.kp_num):
            #         #     self.v[i // 3][i % 3] = (beta + small_step) * self.u0[i] + (1 - beta - small_step) * self.back_up_v[i // 3][i % 3]
            #         # for i in range(self.kp_num):
            #         #     self.x[i] = self.back_up_x[i] + time_step * self.v[i]
            #         # new_energy_upper = self.getEnergy(sim_mode, theta)
            #         # #lower
            #         # for i in range(3 * self.kp_num):
            #         #     self.v[i // 3][i % 3] = (beta - small_step) * self.u0[i] + (1 - beta + small_step) * self.back_up_v[i // 3][i % 3]
            #         # for i in range(self.kp_num):
            #         #     self.x[i] = self.back_up_x[i] + time_step * self.v[i]
            #         # new_energy_lower = self.getEnergy(sim_mode, theta)
            #         # if new_energy_lower < new_energy_upper:
            #         #     direction = -1.
            #         # else:
            #         #     if time == 0:
            #         #         direction = -1.
            #         #     else:
            #         #         direction = 1.
            #     else:
            #         max_scale = new_energy / now_energy
        
        # print(min_energy_beta)
        # if can_step == 0:
        #     for i in range(self.kp_num):
        #         self.v[i] = self.back_up_v[i]
        #         self.x[i] = self.back_up_x[i] + time_step * self.v[i]
        # print("\nbeta: ", beta)
        min_energy_beta = 1.0
        min_energy = self.energy_buffer[0]

        for time in range(1, max_dim):
            if self.energy_buffer[time] < min_energy:
                min_energy = self.energy_buffer[time]
                min_energy_beta = self.linear_search_start[0] - time * small_step

        max_scale = (min_energy + 1.0) / (now_energy + 1.0)

        if max_scale > 5.:
            self.linear_search_start[0] /= 2.
            for i in range(self.kp_num):
                self.v[i] = tm.vec3([0., 0., 0.])
                self.x[i] = self.back_up_x[i]
        else:
            for i in range(3 * self.kp_num):
                self.v[i // 3][i % 3] = min_energy_beta * self.u0[i] + (1 - min_energy_beta) * self.back_up_v[i // 3][i % 3]
            for i in range(self.kp_num):
                self.x[i] = self.back_up_x[i] + time_step * self.v[i]

        if min_energy - now_energy < 1e-5:
            self.linear_search_start[0] *= 2.
            if self.linear_search_start[0] > 1.:
                self.linear_search_start[0] = 1.

        return can_step
            
    @ti.func
    def backup_v(self):
        for i in range(self.kp_num):
            self.back_up_v[i] = self.v[i]

    @ti.func
    def backup_angle(self):
        for i in range(self.crease_pairs_num):
            self.backup_crease_angle[i] = self.crease_angle[i]

    @ti.func
    def backup_x(self):
        for i in range(self.kp_num):
            self.back_up_x[i] = self.x[i]

    @ti.kernel
    def backup_xv(self):
        self.backup_v()
        self.backup_x()

    @ti.kernel
    def update_vertices(self):
        for i in range(self.kp_num):
            self.vertices[i] = self.x[i]

    def initializeRunning(self):
        # turn to numpy structures to initialize the kernel
        numpy_indices                       = np.array(self.ori_sim.indices)
        numpy_kps                           = np.array(self.kps) - np.array(self.total_bias + [-self.origami_z_bias])
        numpy_mass_list                     = np.array(self.mass_list)
        numpy_tri_indices                   = np.array(self.tri_indices)
        numpy_connection_matrix             = np.array(self.ori_sim.connection_matrix)
        numpy_bending_pairs                 = np.array(self.ori_sim.bending_pairs)
        numpy_crease_pairs                  = np.array(self.ori_sim.crease_pairs)
        numpy_line_indices                  = np.array(self.ori_sim.line_indices)
        if len(self.ori_sim.facet_bending_pairs) == 0:
            numpy_facet_bending_pairs           = np.array([[0, 0]])
            numpy_facet_crease_pairs            = np.array([[0, 0]])
        else:
            numpy_facet_bending_pairs           = np.array(self.ori_sim.facet_bending_pairs)
            numpy_facet_crease_pairs            = np.array(self.ori_sim.facet_crease_pairs)
        # construct tb_line information which contains start_index, end_index, level and coeff
        tb_line = []
        for line in self.tb.lines:
            tb_line.append([line.start_index, line.end_index, line.level, line.coeff])
        numpy_tb_line                       = np.array(tb_line)
        # construct string information including string number in each completed constraint and every id and dir
        numpy_string_number                 = np.array([len(self.string_total_information[i]) for i in range(self.constraint_number)])
        if numpy_string_number.size != 0:
            max_string_number = max(numpy_string_number)
        else:
            max_string_number = 0
            numpy_string_number = np.array([1])
        parsed_string_information = []
        for i in range(self.constraint_number):
            parsed_string = []
            for j in range(numpy_string_number[i]):
                parsed_string.append([0 if self.string_total_information[i][j].point_type == 'A' else 1, self.string_total_information[i][j].id, self.string_total_information[i][j].dir])
            for j in range(numpy_string_number[i], max_string_number):
                parsed_string.append([0, -1, 0])
            parsed_string_information.append(parsed_string)
        numpy_parsed_string_information     = np.array(parsed_string_information) if len(parsed_string_information) != 0 else np.array([[[0, -1, 0]]])
        numpy_tsa_end                       = np.array([ele if ele != None else -1 for ele in self.tsa_end])
        numpy_tsa_root                      = np.array(self.tsa_root) if self.actuation_number > 0 else np.array([[-1, 0]])
        numpy_string_root                   = np.array(self.string_root) if self.total_root_number - self.actuation_number > 0 else np.array([[-1, 0]])
        # initialize!
        self.initialize(
            numpy_indices, numpy_kps, numpy_mass_list, numpy_tri_indices, numpy_connection_matrix, numpy_bending_pairs, numpy_crease_pairs, 
            numpy_line_indices, numpy_facet_bending_pairs, numpy_facet_crease_pairs, numpy_tb_line, numpy_string_number, 
            numpy_parsed_string_information, numpy_tsa_end, self.sequence_level_initial, numpy_tsa_root, numpy_string_root
        )
        # parameters reset
        self.dead_count = 0
        self.recorded_turning_angle = []
        self.recorded_dead_count = []
        self.recorded_energy = []
        self.stable_state = 0
        self.past_move_indice = 0.0

        self.can_rotate = False
        self.tsa_turning_angle = 0.0

    def step(self):
        max_delta_length = 0.0
        for i in range(self.constraint_number):
            length = self.constraint_length[i]
            origin = self.constraint_initial_length[i]
            if length - origin > max_delta_length:
                max_delta_length = length - origin
                    
        if self.can_rotate:
            self.tsa_turning_angle += self.enable_tsa_rotate

        self.F(self.folding_angle, self.sim_mode, self.gravitational_acc, self.enable_ground, self.tsa_turning_angle)
        self.step_xv(self.dt, self.enable_ground, self.NO_QUASI, self.folding_angle)
        self.current_t += self.dt

        if not self.can_rotate:
            move_indice = 0.0
            for i in range(self.kp_num):
                move_indice += self.v[i].norm()
            
            # print(self.current_t)
            move_indice /= self.kp_num
            if self.debug_mode:
                print("Max force: " + str(round(self.max_force[0], 1)) + ", Move indice: " + str(round(move_indice, 1)) + ", Current time: " + str(round(self.current_t, 3)) + ", Stable state: " + str(self.stable_state) + ", Parameters: " + str(self.string_params[0]) + ", " + str(self.string_params[1]) + ", " + str(self.miu) + ", " + str(self.n) + ", " + str(self.substeps))
            if abs(move_indice - self.past_move_indice) < 0.1:
                self.stable_state += 1
            else:
                self.stable_state = 0

            if self.stable_state >= 100: #0.4s
                self.can_rotate = True
                print("Actuation ok")

            self.past_move_indice = move_indice

        if self.can_rotate:
            self.recorded_turning_angle.append(self.tsa_turning_angle)
            self.recorded_energy.append(self.total_energy[0] / self.total_energy_maximum[0])
            self.recorded_dead_count.append(self.max_force[0])

    def deal_with_key(self, key):
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
                self.enable_ground = False
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
                self.enable_ground = False
                self.folding_angle = 0.0
                self.enable_add_folding_angle = 0.0
            else:
                self.gravitational_acc[Z] = 0.
                self.enable_ground = False
                self.tsa_turning_angle = 0.0
                self.enable_tsa_rotate = 0.0
                self.folding_angle = 0.0
        else:
            if self.sim_mode == self.FOLD_SIM:
                if key == 'w': 
                    self.folding_angle += self.folding_step
                    if self.folding_angle >= self.folding_max:
                        self.folding_angle = self.folding_max
                
                elif key == 's': 
                    self.folding_angle -= self.folding_step
                    if self.folding_angle <= 0:
                        self.folding_angle = 0

                elif key == 'q': 
                    self.enable_add_folding_angle = self.folding_micro_step[0]
                
                elif key == 'a': 
                    self.enable_add_folding_angle = 0.0
                
                elif key == 'z': 
                    self.enable_add_folding_angle = -self.folding_micro_step[0]
            
            else:
                if key == 'i': 
                    self.enable_tsa_rotate = self.rotation_step

                elif key == 'k': 
                    self.enable_tsa_rotate = 0.0
            
                elif key == 'm': 
                    self.enable_tsa_rotate = -self.rotation_step
    
    def update_folding_target(self):
        if self.sim_mode == self.FOLD_SIM:
            self.folding_angle += self.enable_add_folding_angle
            if self.folding_angle >= self.folding_max:
                self.folding_angle = self.folding_max
            if self.folding_angle <= 0.0:
                self.folding_angle = 0.0

        else:
            self.can_rotate = True
            max_delta_length = 0.0
            for i in range(self.constraint_number):
                length = self.constraint_length[i]
                origin = self.constraint_initial_length[i]
                if length - origin > max_delta_length:
                    max_delta_length = length - origin

            if self.can_rotate:
                self.tsa_turning_angle += self.enable_tsa_rotate
            
    def run(self):
        self.initializeRunning()
        while self.window.running:
            if self.window.get_event(ti.ui.PRESS):
                self.deal_with_key(self.window.event.key)

            self.update_folding_target()
            if self.folding_angle_reach_pi[0] and self.sim_mode == self.TSA_SIM:
                break

            for i in range(1):

                self.F(self.folding_angle, self.sim_mode, self.gravitational_acc, self.enable_ground, self.tsa_turning_angle)

                self.fill_K(self.dt, self.AK)

                self.fill_b(self.dt)

                sparse_solver = ti.linalg.SparseSolver(data_type, "LU")
                A = self.AK.build()
                sparse_solver.analyze_pattern(A)
                sparse_solver.factorize(A)
                self.u0.from_numpy(sparse_solver.solve(self.b))
                # print(self.u0)

                flag = self.step_xv(self.dt, self.enable_ground, self.sim_mode, self.folding_angle, self.gravitational_acc)

                self.current_t += self.dt

                if flag == -1:
                    self.n += 100
                    for j in range(self.kp_num):
                        self.v[j] = [0., 0., 0.]

            self.update_vertices() 
            # for i in range(self.crease_pairs_num):
            #     print("[", self.crease_pairs[i, 0], self.crease_pairs[i, 1], self.crease_type[i], "]")

            if self.sim_mode == self.TSA_SIM:
                if self.debug_mode:
                    move_indice = 0.0
                    for i in range(self.kp_num):
                        move_indice += self.v[i].norm()
                    
                    # print(self.current_t)
                    move_indice /= self.kp_num

                    print("Max force: " + str(round(self.max_force[0], 1)) + ", Move indice: " + str(round(move_indice, 1)) + ", Current time: " + str(round(self.current_t, 3)) + ", Stable state: " + str(self.stable_state))
                    
                    if abs(move_indice - self.past_move_indice) < 0.1:
                        self.stable_state += 1
                    else:
                        self.stable_state = 0

                    if self.stable_state >= 100: #0.5s
                        self.can_rotate = True
                        print("Actuation ok")

                    self.past_move_indice = move_indice
                    
                self.camera.position(0, 2. * -self.panel_size, 2. * self.panel_size)
                # camera.position(-panel_size * 2.2, half_max_size, 10.0)
                
                self.camera.lookat(0, 0, 10.0)
            else:
                self.camera.position(0., -1.3 * self.max_size, self.max_size)
                self.camera.up(0.2, 0.4, 0.9)
                self.camera.lookat(0, 0, 0)
            self.scene.set_camera(self.camera)

            self.scene.point_light(pos=(0., 1.2 * self.max_size, 3. * self.max_size), color=(1, 1, 1))
            self.scene.ambient_light((0.5, 0.5, 0.5))
            self.scene.mesh(self.vertices,
                    indices=self.indices,
                    color=(1., 0.93, 0.97),
                    two_sided=True)
            
            self.fill_line_vertex()
            self.scene.lines(vertices=self.line_vertex,
                        width=2,
                        per_vertex_color=self.line_color)
            
            if self.sim_mode == self.TSA_SIM:
                for i in range(self.constraint_number):
                    self.gui.text(text=f"delta_length[{i}]: " + str(round(self.constraint_length[i] - self.constraint_initial_length[i], 4)))
                # self.gui.text(text=f"string_dis[{i}]: " + str(round(self.string_dis[0, 1], 4)))
                self.gui.text(text="TSA rotate angle: " + str(round(self.tsa_turning_angle, 4)))
                self.gui.slider_float('Total folding energy', round(self.total_energy[0], 4), 0.0, round(self.total_energy_maximum[0], 4))

                accumulate_index = 0
                current_string_index = 0
                endpoint_index = 0
                for i in range(self.total_root_number):
                    dup_time = self.constraint_point_number[i]
                    if dup_time > 1:
                        if not self.constraint_angle_enable[i] or (self.constraint_angle_enable[i] and abs(self.tsa_turning_angle) < abs(self.constraint_angle[i])):
                            for j in range(accumulate_index, accumulate_index + dup_time):
                                self.string_vertex[current_string_index] = self.constraint_start_point[j]
                                current_string_index += 1
                                for k in range(self.max_control_length):
                                    if self.unit_control[j, k] != -1:
                                        self.string_vertex[current_string_index] = self.unit_center[self.unit_control[j, k]]
                                        if not (self.unit_control[j, k + 1] == -1 and not self.constraint_end_point_existence[j]):
                                            self.string_vertex[current_string_index + 1] = self.unit_center[self.unit_control[j, k]]
                                            current_string_index += 1
                                        # self.string_vertex[current_string_index + 1] = self.unit_center[self.unit_control[j, k]]
                                        current_string_index += 1
                                    else:
                                        break
                                if self.constraint_end_point_existence[j]:
                                    self.string_vertex[current_string_index] = self.constraint_end_point[j]
                                    current_string_index += 1
                                    self.endpoint_vertex[endpoint_index] = self.constraint_end_point[j]
                                    endpoint_index += 1
                                else:
                                    self.endpoint_vertex[endpoint_index] = self.unit_center[self.unit_control[j, k - 1]]
                                    endpoint_index += 1
                        else: # twisted
                            for j in range(accumulate_index, accumulate_index + dup_time):
                                self.string_vertex[current_string_index] = self.visual_constraint_start_point[j]
                                current_string_index += 1
                                self.string_vertex[current_string_index] = self.intersection_point[i]
                                self.string_vertex[current_string_index + 1] = self.intersection_point[i]
                                current_string_index += 2
                                self.string_vertex[current_string_index] = self.constraint_start_point[j]
                                self.string_vertex[current_string_index + 1] = self.constraint_start_point[j]
                                current_string_index += 2
                                for k in range(self.max_control_length):
                                    if self.unit_control[j, k] != -1:
                                        self.string_vertex[current_string_index] = self.unit_center[self.unit_control[j, k]]
                                        if not (self.unit_control[j, k + 1] == -1 and not self.constraint_end_point_existence[j]):
                                            self.string_vertex[current_string_index + 1] = self.unit_center[self.unit_control[j, k]]
                                            current_string_index += 1
                                        current_string_index += 1
                                    else:
                                        break
                                if self.constraint_end_point_existence[j]:
                                    self.string_vertex[current_string_index] = self.constraint_end_point[j]
                                    current_string_index += 1
                                    self.endpoint_vertex[endpoint_index] = self.constraint_end_point[j]
                                    endpoint_index += 1
                                else:
                                    self.endpoint_vertex[endpoint_index] = self.unit_center[self.unit_control[j, k - 1]]
                                    endpoint_index += 1
                    else:
                        self.string_vertex[current_string_index] = self.constraint_start_point[accumulate_index]
                        current_string_index += 1
                        for k in range(self.max_control_length):
                            if self.unit_control[accumulate_index, k] != -1:
                                self.string_vertex[current_string_index] = self.unit_center[self.unit_control[accumulate_index, k]]
                                if self.unit_control[accumulate_index, k + 1] != -1 or self.constraint_end_point_existence[accumulate_index]:
                                    self.string_vertex[current_string_index + 1] = self.unit_center[self.unit_control[accumulate_index, k]]
                                    current_string_index += 1
                                current_string_index += 1
                            else:
                                break
                        if self.constraint_end_point_existence[accumulate_index]:
                            self.string_vertex[current_string_index] = self.constraint_end_point[accumulate_index]
                            current_string_index += 1
                            self.endpoint_vertex[endpoint_index] = self.constraint_end_point[accumulate_index]
                            endpoint_index += 1
                        else:
                            self.endpoint_vertex[endpoint_index] = self.unit_center[self.unit_control[accumulate_index, k - 1]]
                            endpoint_index += 1
                    accumulate_index += dup_time

                
                # public code
                while current_string_index < int(self.tsa_visual_string_number * 2):
                    self.string_vertex[current_string_index] = tm.vec3([0., 0., 0.])
                    self.string_vertex[current_string_index + 1] = tm.vec3([0., 0., 0.])
                    current_string_index += 2

                if self.constraint_number:
                    self.scene.lines(vertices=self.string_vertex, width=1, color=(.6, .03, .8))

                self.scene.lines(vertices=self.border_vertex,
                        width=3,
                        color=(.00, .00, .00))
            
                self.scene.particles(centers=self.endpoint_vertex, radius=1.5, color=(.6, .03, .8))

                # overtwisted
                # if abs(self.tsa_turning_angle) > tm.pi and (self.constraint_start_point[0] - self.intersection_point).norm() + 1.0 < (abs(self.tsa_turning_angle) - tm.pi) * self.ds / tm.pi:
                #     self.enable_tsa_rotate = 0.0
            else:
                self.folding_angle = self.gui.slider_float('Folding angle', self.folding_angle, 0.0, self.folding_max)
                self.gui.slider_float('Total folding energy', round(self.total_energy[0], 4), 0.0, round(self.total_energy_maximum[0], 4))

            self.string_params[0] = self.gui.slider_float('String_k', self.string_params[0], 0.0, 10.0)
            self.string_params[1] = self.gui.slider_float('Shearing_k', self.string_params[1], 0.0, 100.0)
            self.n = self.gui.slider_int('Step number', self.n, 10, 6000)
            self.dt = self.gui.slider_float('Dt', 0.1 / self.n, 0.0, 0.01)
            self.substeps = self.gui.slider_int('Substeps', round(1. / 250. // self.dt), 1, 10000)
            self.lame_k = self.gui.slider_float('Lame_k', self.lame_k, 1., 1000.)
            self.lames_bonus[0] = self.lame_k * self.mu
            self.lames_bonus[1] = self.lame_k * self.landa

            # for i in range(self.kp_num):
            #     self.force_vertex[2 * i] = self.x[i]
            #     self.force_vertex[2 * i + 1] = self.x[i] + self.print_force[i] / self.masses[i] / 5000.0

            # self.scene.lines(vertices=self.force_vertex,
            #                 width=2,
            #                 color=(.9, .03, .9))

            # Draw a smaller ball to avoid visual penetration
            # scene.particles(ball_center, radius=ball_radius * 0.98, color=(0.5, 0.42, 0.8))
            self.canvas.scene(self.scene)
            self.window.show()

# if __name__ == "__main__":
#     ori = OrigamiSimulator()
#     ori.start("sd", 4, ori.FOLD_SIM)
#     ori.run()
#     ori.window.destroy()

if __name__ == "__main__":
    dxfg = DxfDirectGrabber()
    dxfg.readFile("./dxfResult/phys_sim.dxf")
    unit_parser = UnitPackParserReverse(
                    tsp=[0.0, 0.0],
                    kps=dxfg.kps,
                    lines=dxfg.lines,
                    lines_type=dxfg.lines_type
                )

    lines = unit_parser.getLine()

    unit_parser.setMaximumNumberOfEdgeInAllUnit(4) #For every units, there exists at most 4 edges
    units = unit_parser.getUnits()

    ori_sim = OrigamiSimulationSystem()
    for ele in units:
        ori_sim.addUnit(ele)
    ori_sim.mesh() #构造三角剖分

    tb = TreeBasedOrigamiGraph(ori_sim.kps, ori_sim.getNewLines())
    tb.calculateTreeBasedGraph()

    mcts = MCTS(units, tb.lines, tb.kps, 100., 72)
    

    for i in range(10000):
        methods = mcts.ask(4)
        reward_list = []
        a_number_list = []
        for method in methods:
            value = 0.0
            a_number = 0
            for string in method:
                for tsa_point in string:
                    if tsa_point[0] == 'A':
                        a_number += 1
                    value += data_type(tsa_point[1])
            if a_number >= 3:
                reward_list.append(1. / (1. + math.exp(value / 50.)))
            else:
                reward_list.append(0.0)
            a_number_list.append(a_number)

        maximum_reward = max(reward_list)
        print("Epoch: " + str(i) + ", max_value: " + str(maximum_reward) + ", length: " + str(len(reward_list)) + ", a_number_list: " + str(a_number_list))

        mcts.tell(reward_list)

        # id倾向性

    # gm = GraphModel(units, tb.lines, tb.kps)
    # gm.calculateConnectionMatrix()

    # gm.aggregation()

    # gm.calculateP()
    
    # id_list = gm.generateIndexList()

    # a = gm.getMaximumLikelihood(id_list)

    # a = 1

    
