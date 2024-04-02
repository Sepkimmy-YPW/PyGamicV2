import taichi as ti
import taichi.math as tm

from ori_sim_sys import *
from desc import *
from units import *

from cdftool import GraphModel, MCTS

#----折纸信息初始化开始----#
# def initTI():
ti.init(arch=ti.cpu, default_fp=ti.f32, fast_math=False)

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

        self.data_type = ti.f32 #数据类型

        self.dxfg = DxfDirectGrabber()
        self.tsa_strings = []

        self.FOLD_SIM = 0
        self.TSA_SIM = 1

        self.strings = []
        self.string_total_information = []
        self.pref_pack = None
    
    def set_tsa_strings(self, strings: list):
        self.tsa_strings = strings

    def pointInUnit(self, point):
        length = len(self.units)
        for unit_id in range(length):
            unit = self.units[unit_id]
            kps = unit.getSeqPoint()
            if pointInPolygon(point, kps):
                return unit_id
        return None
    
    def startOnlyTSA(self, units, max_size, total_bias, unit_edge_max):
        self.units = units
        self.max_size = max_size
        self.total_bias = total_bias

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

        self.sequence_level_max = 0
        self.sequence_level_min = 0

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
        self.cm = ori_sim.connection_matrix                              # connection matrix of kps
        self.odm = ori_sim.origin_distance_matrix                        # original distance of kps

        self.point_mass = 5e-6 #节点质量
        self.gravitational_acc = ti.Vector([0., 0., -9810.]) #全局重力场
        self.face_k = ori_sim.face_k #折纸三角面抗剪强度
        self.string_k = .5 #绳的轴向弹性模量
        self.shearing_k = 4. #绳的抗剪强度
        self.miu = 6. #摩擦系数
        self.rotation_step = tm.pi / 36.0
        self.max_stretch_length = 1.

        self.enable_add_folding_angle = 0 #启用折角增加的仿真模式
        self.enable_tsa_rotate = 0 #启用TSA驱动的仿真模式

        self.n = 500 #仿真的时间间隔
        self.dt = 0.1 / self.n #仿真的时间间隔
        self.substeps = int(1 / 250 // self.dt) #子步长，用于渲染

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
        self.total_energy_maximum = 0.0
        self.total_energy = ti.field(self.data_type, shape=1)
        self.max_force = ti.field(self.data_type, shape=1)

        #最大末端作用力
        self.end_force = ti.field(self.data_type, shape=1)

        #----define parameters for taichi----#
        self.string_params = ti.field(dtype=float, shape=2)
        self.original_vertices = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) # 原始点坐标
        self.vertices = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) # 点坐标
        self.unit_indices = ti.Vector.field(unit_edge_max, dtype=int, shape=self.unit_indices_num) # 每个单元的索引信息

        self.unit_center_initial_point = ti.Vector.field(3, dtype=self.data_type, shape=self.unit_indices_num) # 每个单元的初始中心点位置
        self.unit_center = ti.Vector.field(3, dtype=self.data_type, shape=self.unit_indices_num) # 每个单元的中心点位置
        self.points = ti.Vector.field(3, dtype=self.data_type, shape=unit_edge_max) #计算中心点坐标的暂存数
        self.tri_unit_initial_angle = ti.Vector.field(3, dtype=self.data_type, shape=self.div_indices_num) #每个三角形的初始角度信息
        self.indices = ti.field(int, shape=self.indices_num) #三角面索引信息

        self.bending_pairs = ti.field(dtype=int, shape=(self.bending_pairs_num, 2)) #弯曲对索引信息
        self.crease_pairs = ti.field(dtype=int, shape=(self.crease_pairs_num, 2)) #折痕对索引信息
        self.line_pairs = ti.field(dtype=int, shape=(self.line_total_indice_num, 2)) #线段索引信息，用于初始化渲染

        self.crease_folding_angle = ti.field(dtype=self.data_type, shape=self.crease_pairs_num) #折痕折角
        self.crease_folding_accumulate = ti.field(dtype=self.data_type, shape=self.crease_pairs_num) #补偿折角

        self.crease_type = ti.field(dtype=int, shape=self.crease_pairs_num) #折痕类型信息，与折痕对一一对应
        self.crease_level = ti.field(dtype=int, shape=self.crease_pairs_num)
        self.crease_coeff = ti.field(dtype=self.data_type, shape=self.crease_pairs_num)

        self.connection_matrix = ti.field(dtype=self.data_type, shape=(self.kp_num, self.kp_num)) #关键点之间的连接矩阵
        self.original_distance_matrix = ti.field(dtype=self.data_type, shape=(self.kp_num, self.kp_num)) #关键点之间的原始距离矩阵

        self.line_color = ti.Vector.field(3, dtype=self.data_type, shape=self.line_total_indice_num*2) #线段颜色，用于渲染
        self.line_vertex = ti.Vector.field(3, dtype=self.data_type, shape=self.line_total_indice_num*2) #线段顶点位置，用于渲染

        self.border_vertex = ti.Vector.field(3, dtype=self.data_type, shape=60)
        #----TSA constraints----#

        self.constraint_number = len(self.string_total_information) #约束的数量
        self.actuation_number = len(self.tsa_root)
        self.total_root_number = len(self.tsa_root) + len(self.string_root)

        #根据约束数量，确定约束起始点和终止点位置， 若初始点一致，则识别为TSA
        if self.constraint_number == 0:
            self.constraint_start_point = ti.Vector.field(3, dtype=self.data_type, shape=1)
            self.constraint_point_number = ti.field(dtype=int, shape=1)
            self.string_dis = ti.field(dtype=float, shape=(1, 1))
            self.string_dis.fill(-1)
            self.first_calculated = ti.field(dtype=bool, shape=(1, 1))
            self.string_dis.fill(True)

            self.constraint_angle = ti.field(dtype=float, shape=1)
            self.constraint_angle_enable = ti.field(dtype=bool, shape=1)
            self.constraint_angle_enable.fill(False)

            self.constraint_start_point_duplicate = ti.Vector.field(3, dtype=self.data_type, shape=1)

            self.direction = ti.Vector.field(3, dtype=self.data_type, shape=1)

            self.visual_constraint_start_point = ti.Vector.field(3, dtype=self.data_type, shape=1)

            self.constraint_end_point_existence = ti.field(dtype=bool, shape=1)
            self.constraint_end_point = ti.Vector.field(3, dtype=self.data_type, shape=1)

            # TSA交叉点
            self.intersection_point = ti.Vector.field(3, dtype=self.data_type, shape=1)

            # 视觉上的绳信息
            # self.visual_string_vertex = ti.Vector.field(3, dtype=self.data_type, shape=1)

            # 绳信息
            self.string_vertex = ti.Vector.field(3, dtype=self.data_type, shape=1)
            self.max_control_length = 1
            self.endpoint_vertex = ti.Vector.field(3, dtype=self.data_type, shape=1)
            #---#
            self.unit_control = ti.field(dtype=int, shape=(1, 1))
            #穿线方向信息，-1表示从下往上穿，1表示从上往下穿
            self.hole_dir = ti.field(dtype=self.data_type, shape=(1, 1))
            # 约束绳的长度信息
            self.constraint_initial_length = ti.field(self.data_type, shape=1)
            self.constraint_length = ti.field(self.data_type, shape=1)
            self.backup_constraint_length = ti.field(self.data_type, shape=1)
        else:
            self.constraint_start_point = ti.Vector.field(3, dtype=self.data_type, shape=self.constraint_number)
            self.constraint_point_number = ti.field(dtype=int, shape=self.total_root_number)
            self.string_dis = ti.field(dtype=float, shape=(self.constraint_number, self.constraint_number))
            self.string_dis.fill(-1)
            self.first_calculated = ti.field(dtype=bool, shape=(self.constraint_number, self.constraint_number))
            self.string_dis.fill(True)

            self.constraint_angle = ti.field(dtype=float, shape=self.total_root_number)
            self.constraint_angle_enable = ti.field(dtype=bool, shape=self.total_root_number)
            self.constraint_angle_enable.fill(False)

            self.constraint_start_point_duplicate = ti.Vector.field(3, dtype=self.data_type, shape=self.total_root_number)

            self.direction = ti.Vector.field(3, dtype=self.data_type, shape=self.total_root_number)

            self.constraint_end_point_existence = ti.field(dtype=bool, shape=self.constraint_number)
            self.constraint_end_point = ti.Vector.field(3, dtype=self.data_type, shape=self.constraint_number)

            # 要控制的点的索引信息，至多控制单元数目个数的点, TYPE B的点
            self.max_control_length = max([len(ele) for ele in self.string_total_information])
            self.unit_control = ti.field(dtype=int, shape=(self.constraint_number, self.max_control_length))
            self.unit_control.fill(-1)

            #穿线方向信息，-1表示从下往上穿，1表示从上往下穿
            self.hole_dir = ti.field(dtype=self.data_type, shape=(self.constraint_number, self.max_control_length))
            self.hole_dir.fill(0.)

            #TSA旋转打结后，新的渲染点位
            self.visual_constraint_start_point = ti.Vector.field(3, dtype=self.data_type, shape=self.constraint_number)

            # TSA交叉点
            self.intersection_point = ti.Vector.field(3, dtype=self.data_type, shape=max(1, sum([self.tsa_root[i][1] - 1 for i in range(len(self.tsa_root))])))

            # TSA是否交叉
            self.have_intersection = ti.field(dtype=bool, shape=max(1, len(self.tsa_root)))

            # 绳信息
            self.string_vertex = ti.Vector.field(3, dtype=self.data_type, shape=int(self.tsa_visual_string_number * 2))
            
            self.endpoint_vertex = ti.Vector.field(3, dtype=self.data_type, shape=self.constraint_number)
            #---#
            
            # 约束绳的长度信息
            self.constraint_initial_length = ti.field(self.data_type, shape=self.constraint_number)
            self.constraint_length = ti.field(self.data_type, shape=self.constraint_number)
            self.backup_constraint_length = ti.field(self.data_type, shape=self.constraint_number)

        # 受力顶点，渲染时用
        self.force_vertex = ti.Vector.field(3, dtype=self.data_type, shape=2*self.kp_num)

        # 有可能所有单元都是三角形，故没有面折痕，根据特定条件初始化面折痕信息
        if self.facet_bending_pairs_num > 0:
            self.facet_bending_pairs = ti.field(dtype=int, shape=(self.facet_bending_pairs_num, 2))
            self.facet_crease_pairs = ti.field(dtype=int, shape=(self.facet_crease_pairs_num, 2))
        else:
            self.facet_bending_pairs = ti.field(dtype=int, shape=(1, 2))
            self.facet_crease_pairs = ti.field(dtype=int, shape=(1, 2))
        
        #----simulator information----#
        self.x = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的位置
        self.v = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的速度
        self.dv = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的加速度
        self.force = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的力
        self.record_force = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #跟踪的某一类型的力
        self.old_x = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的位置
        self.viscousity = .8e-3
        self.enable_ground = False

        self.next_x = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的位置
        self.next_v = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的速度
        self.answer_x = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num)
        self.answer_v = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num)

        self.back_up_x = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num)
        self.back_up_v = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num)

        self.sim_mode = self.TSA_SIM

        self.enable_ground = False
        self.gravitational_acc = [0., 0., -9810.]     
            
        self.current_t = 0.0
        self.half_dt = self.dt / 2.0

        self.folding_step = tm.pi * 0.99
        self.folding_max = tm.pi * 0.99

        self.half_max_size = self.max_size / 2.0
        self.r1 = self.d1 / 2.0

        self.folding_angle_reach_pi = ti.field(dtype=bool, shape=1)

        self.recorded_recover_k = self.string_k

        self.x0 = tm.vec3([.0, .0, .0])

        self.past_move_indice = 0.0

        self.stable_state = 0

    
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

        self.tb = TreeBasedOrigamiGraph(self.ori_sim.kps, self.ori_sim.getNewLines())
        self.tb.calculateTreeBasedGraph()

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
        self.cm = ori_sim.connection_matrix                              # connection matrix of kps
        self.odm = ori_sim.origin_distance_matrix                        # original distance of kps
        self.sequence_level_max = int(self.tb.lines[0].level)
        self.sequence_level_min = int(self.tb.lines[0].level)

        self.point_mass = 5e-6 #节点质量
        self.gravitational_acc = ti.Vector([0., 0., -9810.]) #全局重力场
        self.face_k = ori_sim.face_k #折纸三角面抗剪强度
        self.string_k = .5 #绳的轴向弹性模量
        self.shearing_k = 4. #绳的抗剪强度
        self.miu = 6. #摩擦系数
        self.rotation_step = tm.pi / 36.0
        self.max_stretch_length = 1.

        self.enable_add_folding_angle = 0 #启用折角增加的仿真模式
        self.enable_tsa_rotate = 0 #启用TSA驱动的仿真模式

        self.n = 500 #仿真的时间间隔
        self.dt = 0.1 / self.n #仿真的时间间隔
        self.substeps = 40 #子步长，用于渲染

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
        self.total_energy_maximum = 0.0
        self.total_energy = ti.field(self.data_type, shape=1)
        self.max_force = ti.field(self.data_type, shape=1)

        #最大末端作用力
        self.end_force = ti.field(self.data_type, shape=1)

        #----define parameters for taichi----#
        self.string_params = ti.field(dtype=float, shape=2)
        self.original_vertices = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) # 原始点坐标
        self.vertices = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) # 点坐标
        self.unit_indices = ti.Vector.field(unit_edge_max, dtype=int, shape=self.unit_indices_num) # 每个单元的索引信息

        self.unit_center_initial_point = ti.Vector.field(3, dtype=self.data_type, shape=self.unit_indices_num) # 每个单元的初始中心点位置
        self.unit_center = ti.Vector.field(3, dtype=self.data_type, shape=self.unit_indices_num) # 每个单元的中心点位置
        self.points = ti.Vector.field(3, dtype=self.data_type, shape=unit_edge_max) #计算中心点坐标的暂存数
        self.tri_unit_initial_angle = ti.Vector.field(3, dtype=self.data_type, shape=self.div_indices_num) #每个三角形的初始角度信息
        self.indices = ti.field(int, shape=self.indices_num) #三角面索引信息

        self.bending_pairs = ti.field(dtype=int, shape=(self.bending_pairs_num, 2)) #弯曲对索引信息
        self.crease_pairs = ti.field(dtype=int, shape=(self.crease_pairs_num, 2)) #折痕对索引信息
        self.line_pairs = ti.field(dtype=int, shape=(self.line_total_indice_num, 2)) #线段索引信息，用于初始化渲染

        self.crease_folding_angle = ti.field(dtype=self.data_type, shape=self.crease_pairs_num) #折痕折角
        self.crease_folding_accumulate = ti.field(dtype=self.data_type, shape=self.crease_pairs_num) #补偿折角

        self.crease_type = ti.field(dtype=int, shape=self.crease_pairs_num) #折痕类型信息，与折痕对一一对应
        self.crease_level = ti.field(dtype=int, shape=self.crease_pairs_num)
        self.crease_coeff = ti.field(dtype=self.data_type, shape=self.crease_pairs_num)

        self.connection_matrix = ti.field(dtype=self.data_type, shape=(self.kp_num, self.kp_num)) #关键点之间的连接矩阵
        self.original_distance_matrix = ti.field(dtype=self.data_type, shape=(self.kp_num, self.kp_num)) #关键点之间的原始距离矩阵

        self.line_color = ti.Vector.field(3, dtype=self.data_type, shape=self.line_total_indice_num*2) #线段颜色，用于渲染
        self.line_vertex = ti.Vector.field(3, dtype=self.data_type, shape=self.line_total_indice_num*2) #线段顶点位置，用于渲染

        self.border_vertex = ti.Vector.field(3, dtype=self.data_type, shape=60)
        #----TSA constraints----#

        self.constraint_number = len(self.string_total_information) #约束的数量
        self.actuation_number = len(self.tsa_root)
        self.total_root_number = len(self.tsa_root) + len(self.string_root)

        #根据约束数量，确定约束起始点和终止点位置， 若初始点一致，则识别为TSA
        if self.constraint_number == 0:
            self.constraint_start_point = ti.Vector.field(3, dtype=self.data_type, shape=1)
            self.constraint_point_number = ti.field(dtype=int, shape=1)
            self.string_dis = ti.field(dtype=float, shape=(1, 1))
            self.string_dis.fill(-1)
            self.first_calculated = ti.field(dtype=bool, shape=(1, 1))
            self.string_dis.fill(True)

            self.constraint_angle = ti.field(dtype=float, shape=1)
            self.constraint_angle_enable = ti.field(dtype=bool, shape=1)
            self.constraint_angle_enable.fill(False)

            self.constraint_start_point_duplicate = ti.Vector.field(3, dtype=self.data_type, shape=1)
            self.visual_constraint_start_point = ti.Vector.field(3, dtype=self.data_type, shape=1)

            self.direction = ti.Vector.field(3, dtype=self.data_type, shape=1)

            self.constraint_end_point_existence = ti.field(dtype=bool, shape=1)
            self.constraint_end_point = ti.Vector.field(3, dtype=self.data_type, shape=1)

            # TSA交叉点
            self.intersection_point = ti.Vector.field(3, dtype=self.data_type, shape=1)

            # 视觉上的绳信息
            # self.visual_string_vertex = ti.Vector.field(3, dtype=self.data_type, shape=1)

            # 绳信息
            self.string_vertex = ti.Vector.field(3, dtype=self.data_type, shape=1)
            self.endpoint_vertex = ti.Vector.field(3, dtype=self.data_type, shape=1)
            self.max_control_length = 1
            #---#
            self.unit_control = ti.field(dtype=int, shape=(1, 1))
            #穿线方向信息，-1表示从下往上穿，1表示从上往下穿
            self.hole_dir = ti.field(dtype=self.data_type, shape=(1, 1))
            # 约束绳的长度信息
            self.constraint_initial_length = ti.field(self.data_type, shape=1)
            self.constraint_length = ti.field(self.data_type, shape=1)
            self.backup_constraint_length = ti.field(self.data_type, shape=1)
        else:
            self.constraint_start_point = ti.Vector.field(3, dtype=self.data_type, shape=self.constraint_number)
            self.constraint_point_number = ti.field(dtype=int, shape=self.total_root_number)
            self.string_dis = ti.field(dtype=float, shape=(self.constraint_number, self.constraint_number))
            self.string_dis.fill(-1)
            self.first_calculated = ti.field(dtype=bool, shape=(self.constraint_number, self.constraint_number))
            self.string_dis.fill(True)

            self.constraint_angle = ti.field(dtype=float, shape=self.total_root_number)
            self.constraint_angle_enable = ti.field(dtype=bool, shape=self.total_root_number)
            self.constraint_angle_enable.fill(False)

            self.constraint_start_point_duplicate = ti.Vector.field(3, dtype=self.data_type, shape=self.total_root_number)

            self.direction = ti.Vector.field(3, dtype=self.data_type, shape=self.total_root_number)

            self.constraint_end_point_existence = ti.field(dtype=bool, shape=self.constraint_number)
            self.constraint_end_point = ti.Vector.field(3, dtype=self.data_type, shape=self.constraint_number)

            # 要控制的点的索引信息，至多控制单元数目个数的点, TYPE B的点
            self.max_control_length = max([len(ele) for ele in self.string_total_information])
            self.unit_control = ti.field(dtype=int, shape=(self.constraint_number, self.max_control_length))
            self.unit_control.fill(-1)

            #穿线方向信息，-1表示从下往上穿，1表示从上往下穿
            self.hole_dir = ti.field(dtype=self.data_type, shape=(self.constraint_number, self.max_control_length))
            self.hole_dir.fill(0.)

            #TSA旋转打结后，新的渲染点位
            self.visual_constraint_start_point = ti.Vector.field(3, dtype=self.data_type, shape=self.constraint_number)

            # TSA交叉点
            self.intersection_point = ti.Vector.field(3, dtype=self.data_type, shape=max(1, sum([self.tsa_root[i][1] - 1 for i in range(len(self.tsa_root))])))

            # TSA是否交叉
            self.have_intersection = ti.field(dtype=bool, shape=max(1, len(self.tsa_root)))

            # 绳信息
            self.string_vertex = ti.Vector.field(3, dtype=self.data_type, shape=int(self.tsa_visual_string_number * 2))

            self.endpoint_vertex = ti.Vector.field(3, dtype=self.data_type, shape=self.constraint_number)
            
            #---#
            
            # 约束绳的长度信息
            self.constraint_initial_length = ti.field(self.data_type, shape=self.constraint_number)
            self.constraint_length = ti.field(self.data_type, shape=self.constraint_number)
            self.backup_constraint_length = ti.field(self.data_type, shape=self.constraint_number)

        # 受力顶点，渲染时用
        self.force_vertex = ti.Vector.field(3, dtype=self.data_type, shape=2*self.kp_num)

        # 有可能所有单元都是三角形，故没有面折痕，根据特定条件初始化面折痕信息
        if self.facet_bending_pairs_num > 0:
            self.facet_bending_pairs = ti.field(dtype=int, shape=(self.facet_bending_pairs_num, 2))
            self.facet_crease_pairs = ti.field(dtype=int, shape=(self.facet_crease_pairs_num, 2))
        else:
            self.facet_bending_pairs = ti.field(dtype=int, shape=(1, 2))
            self.facet_crease_pairs = ti.field(dtype=int, shape=(1, 2))
        
        #----simulator information----#
        self.x = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的位置
        self.v = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的速度
        self.dv = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的加速度
        self.force = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的力
        self.record_force = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #跟踪的某一类型的力
        self.old_x = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的位置
        self.viscousity = .8e-3
        self.enable_ground = False

        self.next_x = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的位置
        self.next_v = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num) #点的速度
        self.answer_x = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num)
        self.answer_v = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num)

        self.back_up_x = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num)
        self.back_up_v = ti.Vector.field(3, dtype=self.data_type, shape=self.kp_num)

        self.sim_mode = sim_type
        if sim_type == self.FOLD_SIM:
            self.enable_ground = False
            self.gravitational_acc = [0., 0., 0.]
        else:
            self.enable_ground = False
            self.gravitational_acc = [0., 0., -9810.]     
            
        self.current_t = 0.0
        self.half_dt = self.dt / 2.0

        self.folding_step = tm.pi * 0.99
        self.folding_max = tm.pi * 0.99
        self.folding_micro_step = tm.pi / 180.0 / (1 + self.sequence_level_max - self.sequence_level_min)

        self.half_max_size = self.max_size / 2.0
        self.r1 = self.d1 / 2.0

        self.folding_angle_reach_pi = ti.field(dtype=bool, shape=1)

        self.recorded_recover_k = self.string_k

        self.x0 = tm.vec3([.0, .0, .0])

        self.past_move_indice = 0.0

        self.stable_state = 0

        for i in range(self.crease_pairs_num):
            index1 = ori_sim.crease_pairs[i][0]
            index2 = ori_sim.crease_pairs[i][1]
            for true_index in range(self.line_total_indice_num):
                index_11 = self.tb.lines[true_index].start_index
                index_22 = self.tb.lines[true_index].end_index
                if (index1 == index_11 and index2 == index_22) or (index2 == index_11 and index1 == index_22):
                    break
            self.crease_level[i] = int(self.tb.lines[true_index].level)
            self.crease_coeff[i] = self.tb.lines[true_index].coeff
            if self.crease_level[i] > self.sequence_level_max:
                self.sequence_level_max = self.crease_level[i]
            if self.crease_level[i] < self.sequence_level_min:
                self.sequence_level_min = self.crease_level[i]
    
    @ti.func
    def calculateStringDistanceParallel(self, cur_angle: float):
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
    def calculateStringDistance(self, cur_angle: float):
        self.calculateStringDistanceParallel(cur_angle)

    def initialize(self):
        # 初始化单元索引
        self.unit_indices.fill(-1)
        ori_sim = self.ori_sim
        for i in range(self.unit_indices_num):
            for j in range(len(ori_sim.indices[i])):
                self.unit_indices[i][j] = ori_sim.indices[i][j]
        # 初始化节点位置
        for i in range(self.kp_num):
            self.original_vertices[i] = [self.kps[i][X] - self.total_bias[X], self.kps[i][Y] - self.total_bias[Y], self.kps[i][Z]]
        # 初始化三角面索引
        for i in range(self.indices_num):
            self.indices[i] = self.tri_indices[i]
        # 初始化连接矩阵
        for i in range(self.kp_num):
            for j in range(self.kp_num):
                self.connection_matrix[i, j] = self.cm[i][j]
                self.original_distance_matrix[i, j] = self.odm[i][j]
        # 初始化弯曲对和折痕对
        for i in range(self.bending_pairs_num):
            for j in range(2):
                self.bending_pairs[i, j] = ori_sim.bending_pairs[i][j]
                self.crease_pairs[i, j] = ori_sim.crease_pairs[i][j]
        # 初始化线段对
        for i in range(self.line_total_indice_num):
            for j in range(2):
                self.line_pairs[i, j] = ori_sim.line_indices[i][0][j]
        # 初始化面折痕对
        for i in range(self.facet_bending_pairs_num):
            for j in range(2):
                self.facet_bending_pairs[i, j] = ori_sim.facet_bending_pairs[i][j]
                self.facet_crease_pairs[i, j] = ori_sim.facet_crease_pairs[i][j]
        #初始化折痕折角
        for i in range(self.crease_pairs_num):
            self.crease_folding_angle[i] = 0.0
            self.crease_folding_accumulate[i] = 0.0
        # 初始化折痕类型
        for i in range(self.crease_pairs_num):
            self.crease_pair = ori_sim.crease_pairs[i]
            for j in range(len(ori_sim.line_indices)):
                if self.crease_pair[0] == ori_sim.line_indices[j][0][0] and self.crease_pair[1] == ori_sim.line_indices[j][0][1]:
                    self.crease_type[i] = ori_sim.line_indices[j][1]
                    break
        if self.sim_mode == self.FOLD_SIM:
            # 初始化折叠等级和系数
            for i in range(self.crease_pairs_num):
                index1 = ori_sim.crease_pairs[i][0]
                index2 = ori_sim.crease_pairs[i][1]
                for true_index in range(self.line_total_indice_num):
                    index_11 = self.tb.lines[true_index].start_index
                    index_22 = self.tb.lines[true_index].end_index
                    if (index1 == index_11 and index2 == index_22) or (index2 == index_11 and index1 == index_22):
                        break
                self.crease_level[i] = int(self.tb.lines[true_index].level)
                self.crease_coeff[i] = self.tb.lines[true_index].coeff
                if self.crease_level[i] > self.sequence_level_max:
                    self.sequence_level_max = self.crease_level[i]
                if self.crease_level[i] < self.sequence_level_min:
                    self.sequence_level_min = self.crease_level[i]
            self.folding_micro_step = tm.pi / 180.0 / (1 + self.sequence_level_max - self.sequence_level_min)
        # 初始化渲染的线的颜色信息
        for i in range(self.line_total_indice_num):
            if ori_sim.line_indices[i][1] == BORDER:
                self.line_color[2*i] = [0, 0, 0]
                self.line_color[2*i+1] = [0, 0, 0]
            elif ori_sim.line_indices[i][1] == VALLEY:
                self.line_color[2*i] = [0, 0.17, 0.83]
                self.line_color[2*i+1] = [0, 0.17, 0.83]
            else:
                self.line_color[2*i] = [0.75, 0.2, 0.05]
                self.line_color[2*i+1] = [0.75, 0.2, 0.05]
        # 计算折纸最大能量
        self.total_energy_maximum = 0.0
        for i in range(self.crease_pairs_num):
            index1 = self.crease_pairs[i, 0]
            index2 = self.crease_pairs[i, 1]
            # length = (self.original_vertices[index2] - self.original_vertices[index1]).norm()
            self.total_energy_maximum += tm.pi
            
        # 初始化TSA信息
        self.string_dis.fill(-1)
        self.constraint_angle.fill(0.)
        self.constraint_angle_enable.fill(False)
        self.first_calculated.fill(True)

        # 暂不考虑ABAB, 初始化控制单元和穿孔方向
        for i in range(self.constraint_number):
            index = 0
            for j in range(1, len(self.string_total_information[i])):
                if self.string_total_information[i][j].point_type != 'A':
                    self.unit_control[i, index] = self.string_total_information[i][j].id
                    self.hole_dir[i, index] = self.string_total_information[i][j].dir
                    index += 1

        for i in range(self.constraint_number):
            print(len(self.string_total_information[i]))
            for j in range(len(self.string_total_information[i]) - 1):
                print(self.unit_control[i, j])

        # 计算末端点信息
        for i in range(len(self.tsa_end)):
            if self.tsa_end[i] != None:
                id = self.tsa_end[i]
                self.constraint_end_point[i] = [
                    math.cos(id / self.panel_resolution * 2 * math.pi) * self.panel_size, 
                    math.sin(id / self.panel_resolution * 2 * math.pi) * self.panel_size, 
                    self.origami_z_bias
                ]
                self.constraint_end_point_existence[i] = True
            else:
                self.constraint_end_point[i] = [0.0, 0.0, 0.0]
                self.constraint_end_point_existence[i] = False

        # 计算重复点信息
        for i in range(len(self.tsa_root)):
            start_id = self.tsa_root[i]
            self.constraint_start_point_duplicate[i] = [
                math.cos(start_id[0] / self.panel_resolution * 2 * math.pi) * self.panel_size, 
                math.sin(start_id[0] / self.panel_resolution * 2 * math.pi) * self.panel_size, 
                self.origami_z_bias
            ]
        for i in range(len(self.string_root)):
            start_id = self.string_root[i]
            self.constraint_start_point_duplicate[i + len(self.tsa_root)] = [
                math.cos(start_id[0] / self.panel_resolution * 2 * math.pi) * self.panel_size, 
                math.sin(start_id[0] / self.panel_resolution * 2 * math.pi) * self.panel_size, 
                self.origami_z_bias
            ]
        
        # 计算旋转方向
        for i in range(self.total_root_number):
            ele = self.constraint_start_point_duplicate[i]
            self.direction[i] = [
                -ele[Y] / self.panel_size * self.d1 / 2.0, 
                ele[X] / self.panel_size * self.d1 / 2.0, 
                0.0
            ]

        # 计算分点信息
        accumulate_index = 0
        for i in range(len(self.tsa_root)):
            dup_time = self.tsa_root[i][1]
            for j in range(dup_time):
                self.constraint_start_point[j + accumulate_index] = [
                    self.constraint_start_point_duplicate[i][X] + self.direction[i][X] * math.cos(j / dup_time * 2 * math.pi), 
                    self.constraint_start_point_duplicate[i][Y] + self.direction[i][Y] * math.cos(j / dup_time * 2 * math.pi), 
                    self.origami_z_bias + math.sqrt(self.direction[i][X] ** 2 + self.direction[i][Y] ** 2) * math.sin(j / dup_time * 2 * math.pi)
                ]
            self.constraint_point_number[i] = dup_time
            accumulate_index += dup_time
        for i in range(len(self.string_root)):
            self.constraint_start_point[i + accumulate_index] = [
                self.constraint_start_point_duplicate[i + len(self.tsa_root)][X], 
                self.constraint_start_point_duplicate[i + len(self.tsa_root)][Y], 
                self.origami_z_bias
            ]
            self.constraint_point_number[i + len(self.tsa_root)] = 1

        # calculate initial center point
        for i in range(self.unit_indices_num):
            unit_indice = ori_sim.indices[i]
            kp_len = len(unit_indice)
            total_points = []
            for j in range(kp_len):
                total_points.append([self.kps[unit_indice[j]][X] - self.total_bias[X], self.kps[unit_indice[j]][Y] - self.total_bias[Y], self.kps[unit_indice[j]][Z]])
            center_point = calculateCenterPoint(total_points) + [self.origami_z_bias]
            self.unit_center_initial_point[i] = center_point
            self.unit_center[i] = center_point

        # calculate initial length
        for i in range(self.constraint_number):
            self.constraint_initial_length[i] = 0.0
            start_point = self.constraint_start_point[i]
            for j in range(self.max_control_length):
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
        self.stable_state = 0
        self.past_move_indice = 0.0

        self.can_rotate = False
        self.tsa_turning_angle = 0.0

        # 初始化线的信息
        self.string_params[0] = self.string_k
        self.string_params[1] = self.shearing_k

        # 初始化边缘信息
        for i in range(30):
            self.border_vertex[2 * i] = [tm.cos(i / 15. * tm.pi) * self.panel_size, tm.sin(i / 15. * tm.pi) * self.panel_size, self.origami_z_bias]
            self.border_vertex[2 * i + 1] = [tm.cos((i + 1) / 15. * tm.pi) * self.panel_size, tm.sin((i + 1) / 15. * tm.pi) * self.panel_size, self.origami_z_bias]

    @ti.kernel
    def initialize_angle(self):
        for i in range(self.div_indices_num):
            indice = [self.indices[3 * i], self.indices[3 * i + 1], self.indices[3 * i + 2]]
            delta_p1 = tm.normalize(self.original_vertices[indice[1]] - self.original_vertices[indice[0]])
            delta_p2 = tm.normalize(self.original_vertices[indice[2]] - self.original_vertices[indice[1]])
            delta_p3 = tm.normalize(self.original_vertices[indice[0]] - self.original_vertices[indice[2]])
            self.tri_unit_initial_angle[i][0] = tm.acos(delta_p1.dot(-delta_p3))
            self.tri_unit_initial_angle[i][1] = tm.acos(delta_p2.dot(-delta_p1))
            self.tri_unit_initial_angle[i][2] = tm.acos(delta_p3.dot(-delta_p2))

    @ti.kernel
    def fill_line_vertex(self):
        for i in range(self.line_total_indice_num):
            indice1 = self.line_pairs[i, 0]
            indice2 = self.line_pairs[i, 1]
            self.line_vertex[2 * i] = self.vertices[indice1]
            self.line_vertex[2 * i + 1] = self.vertices[indice2]

    @ti.kernel
    def initialize_mass_points(self):
        z_bias = self.origami_z_bias
        for i in range(self.kp_num):
            self.x[i] = [self.original_vertices[i][0], self.original_vertices[i][1], self.original_vertices[i][2] + z_bias]
            self.old_x[i] = self.x[i]
            self.v[i] = [0., 0., 0.]
            self.dv[i] = [0., 0., 0.]

    @ti.func
    def getSpringForce(self, pos: tm.vec3, other_pos: tm.vec3, i, j) -> tm.vec3:
        ret = tm.vec3([0., 0., 0.])
        if abs(self.connection_matrix[i, j] - self.ori_sim.spring_k) < 1e-5:
            original_distance = self.original_distance_matrix[i, j]
            direction = other_pos - pos
            distance = direction.norm()
            delta_distance = distance - original_distance
            ret = tm.normalize(direction) * delta_distance * self.connection_matrix[i, j]
        return ret

    @ti.func
    def getPushForce(self, pos: tm.vec3, other_pos: tm.vec3, i, j) -> tm.vec3:
        ret = tm.vec3([0., 0., 0.])
        original_distance = self.original_distance_matrix[i, j]
        direction = other_pos - pos
        distance = direction.norm()
        delta_distance = distance - original_distance
        ret = tm.normalize(direction) * delta_distance * self.ori_sim.bending_k
        return ret

    @ti.func
    def getBendingForce(self, cs, ce, p1, p2, k, theta, crease_type, debug=False, enable_dynamic_change=False, index=-1):
        # 求折痕的信息
        x0 = cs
        x1 = ce
        xc = x1 - x0
        xc_norm = xc.norm()

        # 求单元法向量
        f11 = x0 - p1
        f12 = x1 - p1
        f21 = x1 - p2
        f22 = x0 - p2
        n1 = f11.cross(f12)
        n2 = f21.cross(f22)

        # 求2倍单元面积
        a1 = n1.norm()
        a2 = n2.norm()

        # 法向量归一化
        un1 = tm.normalize(n1)
        un2 = tm.normalize(n2)
        uxc = tm.normalize(xc)

        dir = un1.cross(un2).dot(uxc)

        un1_dot_un2 = un1.dot(un2)

        val = un1_dot_un2

        if val > 1.0:
            val = 1.0
        elif val < -1.0:
            val = -1.0

        current_folding_angle = 0.0
        # 求折叠角
        if not enable_dynamic_change:
            if index == -1:
                if crease_type:
                    if dir >= 1e-5:
                        current_folding_angle = -tm.acos(val)
                    elif dir <= -1e-5:
                        if un1_dot_un2 <= -0.5: # 180~270
                            current_folding_angle = -2 * tm.pi + tm.acos(val)   
                        else:
                            current_folding_angle = tm.acos(val)
                else:
                    if dir <= -1e-5:
                        current_folding_angle = tm.acos(val)
                    elif dir >= 1e-5:
                        if un1_dot_un2 <= -0.5:
                            current_folding_angle = 2 * tm.pi - tm.acos(val)
                        else:
                            current_folding_angle = -tm.acos(val)
            else:
                # if not crease_type:
                #     if dir >= 1e-5:
                #         current_folding_angle = -tm.acos(val) + crease_folding_accumulate[index]
                #         before_folding_angle = crease_folding_angle[index]
                #         if before_folding_angle - current_folding_angle > 1.5 * tm.pi:
                #             print(index, before_folding_angle, current_folding_angle, crease_folding_accumulate[index])
                #             crease_folding_accumulate[index] += tm.pi * 2.0
                #             current_folding_angle += tm.pi * 2.0
                #     elif dir <= -1e-5:
                #         current_folding_angle = tm.acos(val) + crease_folding_accumulate[index]
                #         before_folding_angle = crease_folding_angle[index]
                #         if before_folding_angle - current_folding_angle < -1.5 * tm.pi:
                #             print(index, before_folding_angle, current_folding_angle)
                #             crease_folding_accumulate[index] -= tm.pi * 2.0
                #             current_folding_angle -= tm.pi * 2.0
                # else:
                #     if dir <= -1e-5:
                #         current_folding_angle = tm.acos(val) + crease_folding_accumulate[index]
                #         before_folding_angle = crease_folding_angle[index]
                #         if before_folding_angle - current_folding_angle < -1.5 * tm.pi:
                #             print(index, before_folding_angle, current_folding_angle)
                #             crease_folding_accumulate[index] -= tm.pi * 2.0
                #             current_folding_angle -= tm.pi * 2.0
                #     elif dir >= 1e-5:
                #         current_folding_angle = -tm.acos(val) + crease_folding_accumulate[index]
                #         before_folding_angle = crease_folding_angle[index]
                #         if before_folding_angle - current_folding_angle > 1.5 * tm.pi:
                #             print(index, before_folding_angle, current_folding_angle)
                #             crease_folding_accumulate[index] += tm.pi * 2.0
                #             current_folding_angle += tm.pi * 2.
                if crease_type:
                    if dir >= 1e-5:
                        current_folding_angle = -tm.acos(val)
                    elif dir <= -1e-5:
                        if un1_dot_un2 <= -0.5: # 180~270
                            current_folding_angle = -2 * tm.pi + tm.acos(val)   
                        else:
                            current_folding_angle = tm.acos(val)
                else:
                    if dir <= -1e-5:
                        current_folding_angle = tm.acos(val)
                    elif dir >= 1e-5:
                        if un1_dot_un2 <= -0.5:
                            current_folding_angle = 2 * tm.pi - tm.acos(val)
                        else:
                            current_folding_angle = -tm.acos(val)
                # if dir >= 1e-5:
                #     current_folding_angle = -tm.acos(val) + crease_folding_accumulate[index]
                # else:
                #     current_folding_angle = tm.acos(val) + crease_folding_accumulate[index]
                # if not crease_type:
                #     before_folding_angle = crease_folding_angle[index]
                #     if before_folding_angle - current_folding_angle < -tm.pi:
                #         crease_folding_accumulate[index] -= tm.pi * 2.0
                #         current_folding_angle -= tm.pi * 2.0
                #     elif before_folding_angle - current_folding_angle > tm.pi:
                #         crease_folding_accumulate[index] += tm.pi * 2.0
                #         current_folding_angle += tm.pi * 2.0
                # else:
                #     before_folding_angle = crease_folding_angle[index]
                #     if before_folding_angle - current_folding_angle < -tm.pi:
                #         crease_folding_accumulate[index] -= tm.pi * 2.0
                #         current_folding_angle -= tm.pi * 2.0
                #     elif before_folding_angle - current_folding_angle > tm.pi:
                #         crease_folding_accumulate[index] += tm.pi * 2.0
                #         current_folding_angle += tm.pi * 2.0
                    
        else:
            # if index == -1: # facet crease
            if dir >= 1e-5:
                current_folding_angle = -tm.acos(val)
            else:
                current_folding_angle = tm.acos(val)
            # else:
            #     if dir >= 1e-5:
            #         current_folding_angle = -tm.acos(val) + crease_folding_accumulate[index]
            #         before_folding_angle = crease_folding_angle[index]
            #         if before_folding_angle - current_folding_angle > tm.pi:
            #             crease_folding_accumulate[index] += tm.pi * 2.0
            #             current_folding_angle += tm.pi * 2.0
            #     elif dir <= -1e-5:
            #         current_folding_angle = tm.acos(val) + crease_folding_accumulate[index]
            #         before_folding_angle = crease_folding_angle[index]
            #         if before_folding_angle - current_folding_angle < -tm.pi:
            #             crease_folding_accumulate[index] -= tm.pi * 2.0
            #             current_folding_angle -= tm.pi * 2.0
        if index != -1:
            self.crease_folding_angle[index] = current_folding_angle
        
        # if index == 2:
        #     print(f11, f12, n2, val, dir, current_folding_angle)

        # 求折叠角与目标之差
        if abs(current_folding_angle) >= tm.pi / 1.06:
            self.folding_angle_reach_pi[0] = True

        delta_folding_angle = current_folding_angle - theta
        if debug:
            print(x0, p1, dir, current_folding_angle)

        # 计算折痕等效弯曲系数
        k_crease = k * xc_norm

        # 计算力矩，谷折痕折叠角大于0
        h1 = a1 / xc_norm
        h2 = a2 / xc_norm
        rpf1 = -k_crease * delta_folding_angle / h1 * un1
        rpf2 = -k_crease * delta_folding_angle / h2 * un2
        
        t1 = -f11.dot(xc) / (xc_norm ** 2)
        t2 = -f22.dot(xc) / (xc_norm ** 2)

        csf = k_crease * delta_folding_angle * (1 - t1) / h1 * un1 + k_crease * delta_folding_angle * (1 - t2) / h2 * un2
        cef = k_crease * delta_folding_angle * (t1) / h1 * un1 + k_crease * delta_folding_angle * (t2) / h2 * un2

        #计算能量
        energy = abs(current_folding_angle)

        return csf, cef, rpf1, rpf2, energy

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
    def calculateKpNumWithUnitId(self, unit_kps):
        kp_len = 0
        for i in range(len(unit_kps)):
            if unit_kps[i] != -1:
                kp_len += 1
        return kp_len
    
    def implicit_xv(self, theta: float, sim_mode: bool, gravitational_acc: tm.vec3, enable_ground: bool):
        for i in range(self.kp_num):
            self.next_x[i] = self.x[i]
            self.next_v[i] = self.v[i]
        
        for _ in range(100):
            total_error = 0.0
            for i in range(self.kp_num):
                self.x[i] = self.next_x[i]
                self.v[i] = self.next_v[i]
            self.F(theta, sim_mode, gravitational_acc, enable_ground)
            for i in range(self.kp_num):
                self.answer_v[i] = self.next_v[i] - self.dt * self.dv[i]
                self.answer_x[i] = self.next_x[i] - self.dt * self.answer_v[i]
                error_dir_x = self.answer_x[i] - self.x[i]
                error_dir_v = self.answer_v[i] - self.v[i]
                total_error += error_dir_x.norm() + error_dir_v.norm()
                self.next_x[i] -= 1. * error_dir_x
                self.next_v[i] -= 1. * error_dir_v
            if total_error < 1.:
                break
        
        for i in range(self.kp_num):
            self.x[i] = self.next_x[i]
            self.v[i] = self.next_v[i]

    @ti.func
    def getAxisForce(self, force_dir, i):
        delta_length = self.constraint_length[i] - self.constraint_initial_length[i]
        force = tm.vec3([.0, .0, .0])
        if delta_length < self.max_stretch_length:
            force = tm.normalize(force_dir) * self.string_params[0] * delta_length
        else:
            force = tm.normalize(force_dir) * self.string_params[0] * ((delta_length - 1 / 3) ** 3 + 19. / 27.)
        return force

    @ti.kernel
    def F(self, theta: float, sim_mode: bool, gravitational_acc: tm.vec3, enable_ground: bool, tsa_turning_angle: float):
        self.total_energy[0] = 0.0
        self.max_force[0] = 0.0
        kp_num = self.kp_num
        # First we calculate k_spring force
        for i in range(kp_num):
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])
            point = self.get_position_with_index(i)
            for j in range(kp_num):
                f = self.getSpringForce(point, self.get_position_with_index(j), i, j)
                self.force[i] += f
                force_value = f.norm()
                if force_value > self.max_force[0]:
                    self.max_force[0] = force_value
        # print("after 1")
        # print(force[7])
        for i in range(kp_num):
            self.record_force[i] = ti.Vector([0.0, 0.0, 0.0])

        # Second we calculate k_bending force
        if sim_mode == self.FOLD_SIM:
            for i in range(self.crease_pairs_num):
                crease_start_index = self.crease_pairs[i, 0]
                crease_end_index = self.crease_pairs[i, 1]
                related_p1 = self.bending_pairs[i, 0]
                related_p2 = self.bending_pairs[i, 1]

                target_folding_angle = 0.0
                percent_low = (self.sequence_level_max - self.crease_level[i]) / (self.sequence_level_max - self.sequence_level_min + 1.)
                percent_high = (self.sequence_level_max - self.crease_level[i] + 1.) / (self.sequence_level_max - self.sequence_level_min + 1.)
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
                
                # 计算弯曲力
                csf, cef, rpf1, rpf2, energy = self.getBendingForce(
                    self.get_position_with_index(crease_start_index), 
                    self.get_position_with_index(crease_end_index), 
                    self.get_position_with_index(related_p1), 
                    self.get_position_with_index(related_p2),
                    self.ori_sim.bending_k, target_folding_angle, self.crease_type[i], False, False, -1)
                # 增加至force
                self.record_force[crease_start_index] += csf
                self.record_force[crease_end_index] += cef
                self.record_force[related_p1] += rpf1
                self.record_force[related_p2] += rpf2
                self.total_energy[0] += energy
        else:
            for i in range(self.crease_pairs_num):
                crease_start_index = self.crease_pairs[i, 0]
                crease_end_index = self.crease_pairs[i, 1]
                related_p1 = self.bending_pairs[i, 0]
                related_p2 = self.bending_pairs[i, 1]
                # 计算弯曲力
                
                csf, cef, rpf1, rpf2, energy = self.getBendingForce(
                    self.get_position_with_index(crease_start_index), 
                    self.get_position_with_index(crease_end_index), 
                    self.get_position_with_index(related_p1), 
                    self.get_position_with_index(related_p2),
                    self.ori_sim.bending_k, 0.0, 0, False, True, -1)
                # 增加至force
                self.record_force[crease_start_index] += csf
                self.record_force[crease_end_index] += cef
                self.record_force[related_p1] += rpf1
                self.record_force[related_p2] += rpf2
                self.total_energy[0] += energy
        # print("after 2")
        # print(force[7])
        
        for i in range(kp_num):
            force_value = self.record_force[i].norm()
            if force_value > self.max_force[0]:
                self.max_force[0] = force_value
            self.force[i] += self.record_force[i]
            self.record_force[i] = ti.Vector([0.0, 0.0, 0.0])

        # Third we calculate facet crease
        facet_k = 0.0
        if sim_mode == self.FOLD_SIM:
            facet_k = 8. * self.ori_sim.bending_k
        else:
            facet_k = 12. * self.ori_sim.bending_k
        if self.facet_bending_pairs_num > 0:
            for i in range(self.facet_crease_pairs_num):
                crease_start_index = self.facet_crease_pairs[i, 0]
                crease_end_index = self.facet_crease_pairs[i, 1]
                related_p1 = self.facet_bending_pairs[i, 0]
                related_p2 = self.facet_bending_pairs[i, 1]
                # 计算弯曲力
                csf, cef, rpf1, rpf2, energy = self.getBendingForce(
                    self.get_position_with_index(crease_start_index), 
                    self.get_position_with_index(crease_end_index), 
                    self.get_position_with_index(related_p1), 
                    self.get_position_with_index(related_p2), 
                    facet_k, 0.0, 0, False, True, -1)
                # 增加至force
                self.record_force[crease_start_index] += csf
                self.record_force[crease_end_index] += cef
                self.record_force[related_p1] += rpf1
                self.record_force[related_p2] += rpf2
                # self.total_energy[0] += energy

        for i in range(kp_num):
            force_value = self.record_force[i].norm()
            if force_value > self.max_force[0]:
                self.max_force[0] = force_value
            self.force[i] += self.record_force[i]
            self.record_force[i] = ti.Vector([0.0, 0.0, 0.0])
            
        # print("after 3")
        # print(force[7])

        # Fourth we calculate viscousity
        for i in range(kp_num):
            point_v = self.get_velocity_with_index(i)
            for j in range(kp_num):
                f = self.getViscousity(point_v, self.get_velocity_with_index(j), i, j)
                self.force[i] += f
        
        # print("after 4")
        # print(force[7])

        #Fifth we calculate total viscousity
        for i in range(kp_num):
            point_v = self.get_velocity_with_index(i)
            self.force[i] += -.01 * self.viscousity * point_v
        if enable_ground:
            for i in range(kp_num):
                if self.x[i][Z] == 0.0:
                    vel = tm.vec3([.0, .0, .0])
                    vel[X] = self.v[i][X]
                    vel[Y] = self.v[i][Y]
                    vel = tm.normalize(vel)
                    self.force[i] += self.miu * self.point_mass * self.gravitational_acc[Z] * vel

        # print("after 5")
        # print(force[7])
            
        # Sixth we calculate face shearing force
        current_index = -1
        before_index = -1
        next_index = -1
        for i in range(self.div_indices_num):
            x0 = self.x[self.indices[3 * i]]
            x1 = self.x[self.indices[3 * i + 1]]
            x2 = self.x[self.indices[3 * i + 2]]
            n = tm.normalize((x1 - x0).cross(x2 - x0))
            for j in range(3):
                if j == 0:
                    current_index = self.indices[3 * i]
                    before_index = self.indices[3 * i + 2]
                    next_index = self.indices[3 * i + 1]
                elif j == 1:
                    current_index = self.indices[3 * i + 1]
                    before_index = self.indices[3 * i]
                    next_index = self.indices[3 * i + 2]
                else:
                    current_index = self.indices[3 * i + 2]
                    before_index = self.indices[3 * i + 1]
                    next_index = self.indices[3 * i]
                pos_vec1 = self.x[before_index] - self.x[current_index]
                pos_vec2 = self.x[next_index] - self.x[current_index]

                val = tm.normalize(pos_vec2).dot(tm.normalize(pos_vec1))
                if val > 1.0:
                    val = 1.0
                elif val < -1.0:
                    val = -1.0
                current_shape_angle = tm.acos(val)
                delta_angle = (current_shape_angle - self.tri_unit_initial_angle[i][j])
    
                shearing_force_value_1 = self.face_k * delta_angle * n.cross(pos_vec1) / (pos_vec1.norm() ** 2)
                shearing_force_value_2 = self.face_k * delta_angle * n.cross(pos_vec2) / (pos_vec2.norm() ** 2)

                self.record_force[before_index] += -shearing_force_value_1
                self.record_force[current_index] += shearing_force_value_1 - shearing_force_value_2
                self.record_force[next_index] += shearing_force_value_2
        
        # print("after 6")
        # print(self.force[0])

        for i in range(kp_num):
            force_value = self.record_force[i].norm()
            if force_value > self.max_force[0]:
                self.max_force[0] = force_value
            self.force[i] += self.record_force[i]
            self.record_force[i] = ti.Vector([0.0, 0.0, 0.0])

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

        for i in range(kp_num):
            force_value = self.record_force[i].norm()
            if force_value > self.max_force[0]:
                self.max_force[0] = force_value
            self.force[i] += self.record_force[i]
            # self.record_force[i] = ti.Vector([0.0, 0.0, 0.0])

        # print("total")
        for i in range(kp_num):
            self.dv[i] = self.force[i] / self.point_mass + gravitational_acc    
            # print(self.force[i])
            # self.record_force[i] = self.force[i]
            # if i == 7:
            #     print(self.force[i])
            # total_energy += -point_mass * gravitational_acc[Z] * x[i][Z]
        
        # print(gravitational_acc[Z])
        
        # print_force = tm.vec3([.0, .0, .0])
        # for k in range(kp_num):
        #     print_force += record_force[k]
        # print(total_energy)

    @ti.func
    def explicit_step_x(self, time_step: float, enable_ground: bool):
        h = time_step
        # -- YOUR CODE BEGINS HERE --
        for i in range(0, self.kp_num):
            self.x[i] = self.x[i] + h * self.v[i]
            if enable_ground and self.x[i][Z] < 0.0:
                self.x[i][Z] = 0.0
        # print(x[0][Z])

        # for i in range(constraint_number):
        #     new_constraint_length[i] = 0.0
        #     start_point = constraint_start_point[i]
        #     for j in range(unit_indices_num):
        #         if unit_control[i][j] != -1:
        #             center = calculateCenterPoint3DWithUnitId(unit_indices[unit_control[i][j]])
        #             new_constraint_length[i] += (center - start_point).norm()
        #             start_point = unit_center_initial_point[unit_control[i][j]]
        #         else:
        #             break
        #     new_constraint_length[i] += (constraint_end_point[i] - start_point).norm()
        #     if new_constraint_length[i] > constraint_length[i] and new_constraint_length[i] > constraint_initial_length[i]: # can't update
        #         can_update = False

        # -- THE END OF YOUR CODE --


    @ti.func
    def explicit_step_v(self, time_step: float, mode: bool):
        h = time_step
        # -- YOUR CODE BEGINS HERE --
        for i in range(0, self.kp_num):
            if mode:
                self.v[i] = self.v[i] + h * self.dv[i]
            else:
                self.v[i] = h * self.dv[i]
        # -- THE END OF YOUR CODE --

    @ti.kernel
    def explicit_step_xv(self, time_step: float, enable_ground: bool, mode: bool):
        self.explicit_step_v(time_step, mode)
        self.explicit_step_x(time_step, enable_ground)

    @ti.func
    def verlet_x(self, time_step: float, enable_ground: bool):
        h = time_step
        for i in range(self.kp_num):
            temp = self.x[i]
            self.x[i] = self.x[i] + (self.x[i] - self.old_x[i]) + self.dv[i] * h * h
            self.old_x[i] = temp
            if enable_ground and self.x[i][Z] < 0.0:
                self.x[i][Z] = 0.0

    @ti.kernel
    def verlet(self, time_step: float, enable_ground: bool):
        self.verlet_x(time_step, enable_ground)

    @ti.func
    def backup_v(self):
        for i in range(self.kp_num):
            self.back_up_v[i] = self.v[i]

    @ti.func
    def backup_x(self):
        for i in range(self.kp_num):
            self.back_up_x[i] = self.x[i]

    @ti.kernel
    def backup_xv(self):
        self.backup_v()
        self.backup_x()

    @ti.func
    def rk_step_x(self, time_step: float, enable_ground: bool):
        h = time_step
        # -- YOUR CODE BEGINS HERE --
        for i in range(0, self.kp_num):
            self.x[i] = self.back_up_x[i] + h * self.v[i]
            if enable_ground and self.x[i][Z] < 0.0:
                self.x[i][Z] = 0.0
        # -- THE END OF YOUR CODE --


    @ti.func
    def rk_step_v(self, time_step: float, mode: bool):
        h = time_step
        # -- YOUR CODE BEGINS HERE --
        for i in range(0, self.kp_num):
            if mode:
                self.v[i] = self.back_up_v[i] + h * self.dv[i]
            else:
                self.v[i] = h * self.dv[i]
        # -- THE END OF YOUR CODE --

    @ti.kernel
    def rk_step_xv(self, time_step: float, enable_ground: bool, mode: bool):
        self.rk_step_v(time_step, mode)
        self.rk_step_x(time_step, enable_ground)
    
    @ti.kernel
    def clear_v(self):
        self.explicit_step_v(0, self.QUASI)

    def RK2(self, time_step: float, enable_ground: bool, folding_angle: float, sim_mode: bool, gravitational_acc: tm.vec3, mode: bool):
        self.backup_xv()
        self.F(folding_angle, sim_mode, gravitational_acc, enable_ground)
        self.explicit_step_xv(time_step / 2.0, enable_ground, mode)
        self.F(folding_angle, sim_mode, gravitational_acc, enable_ground)
        self.rk_step_xv(time_step, enable_ground, mode)

    @ti.kernel
    def update_vertices(self):
        for i in range(self.kp_num):
            self.vertices[i] = self.x[i]

    def initializeRunning(self):
        self.initialize()
        self.initialize_mass_points()
        self.initialize_angle()
        self.dead_count = 0
        self.recorded_turning_angle = []
        self.recorded_dead_count = []
        self.recorded_energy = []
        self.can_rotate = False

    def step(self):
        max_delta_length = 0.0
        for i in range(self.constraint_number):
            length = self.constraint_length[i]
            origin = self.constraint_initial_length[i]
            if length - origin > max_delta_length:
                max_delta_length = length - origin
                    
        if self.can_rotate:
            self.tsa_turning_angle += self.enable_tsa_rotate

        for i in range(self.substeps):
            self.F(self.folding_angle, self.sim_mode, self.gravitational_acc, self.enable_ground, self.tsa_turning_angle)
            self.explicit_step_xv(self.dt, self.enable_ground, self.NO_QUASI)
            self.current_t += self.dt
        self.update_vertices()

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
            self.recorded_energy.append(self.total_energy[0] / self.total_energy_maximum)
            self.recorded_dead_count.append(self.max_force[0])
        
        # if self.debug_mode:
        #     if self.sim_mode == self.TSA_SIM:   
        #         self.camera.position(0, 2. * -self.panel_size, 2. * self.panel_size)
        #         self.camera.lookat(0, 0, 10.0)
        #     self.scene.set_camera(self.camera)
        #     self.scene.point_light(pos=(0., 1.2 * self.max_size, 3. * self.max_size), color=(1, 1, 1))
        #     self.scene.ambient_light((0.5, 0.5, 0.5))
        #     self.scene.mesh(self.vertices,
        #             indices=self.indices,
        #             color=(0.88, 0.88, 0.88),
        #             two_sided=True)
            
        #     self.fill_line_vertex()
        #     self.scene.lines(vertices=self.line_vertex,
        #                 width=5,
        #                 per_vertex_color=self.line_color)
            
        #     if self.sim_mode == self.TSA_SIM:
        #         accumulate_index = 0
        #         current_string_index = 0
        #         for i in range(self.total_root_number):
        #             dup_time = self.constraint_point_number[i]
        #             if dup_time > 1:
        #                 if not self.constraint_angle_enable[i] or (self.constraint_angle_enable[i] and abs(self.tsa_turning_angle) < abs(self.constraint_angle[i])):
        #                     for j in range(accumulate_index, accumulate_index + dup_time):
        #                         self.string_vertex[current_string_index] = self.constraint_start_point[j]
        #                         current_string_index += 1
        #                         for k in range(self.max_control_length):
        #                             if self.unit_control[j, k] != -1:
        #                                 self.string_vertex[current_string_index] = self.unit_center[self.unit_control[j, k]]
        #                                 self.string_vertex[current_string_index + 1] = self.unit_center[self.unit_control[j, k]]
        #                                 current_string_index += 2
        #                             else:
        #                                 break
        #                         if self.constraint_end_point_existence[j]:
        #                             self.string_vertex[current_string_index] = self.constraint_end_point[j]
        #                             current_string_index += 1
        #                         else:
        #                             current_string_index -= 1
        #                 else: # twisted
        #                     for j in range(accumulate_index, accumulate_index + dup_time):
        #                         self.string_vertex[current_string_index] = self.visual_constraint_start_point[j]
        #                         current_string_index += 1
        #                         self.string_vertex[current_string_index] = self.intersection_point[i]
        #                         self.string_vertex[current_string_index + 1] = self.intersection_point[i]
        #                         current_string_index += 2
        #                         self.string_vertex[current_string_index] = self.constraint_start_point[j]
        #                         self.string_vertex[current_string_index + 1] = self.constraint_start_point[j]
        #                         current_string_index += 2
        #                         for k in range(self.max_control_length):
        #                             if self.unit_control[j, k] != -1:
        #                                 self.string_vertex[current_string_index] = self.unit_center[self.unit_control[j, k]]
        #                                 if not (self.unit_control[j, k + 1] == -1 and not self.constraint_end_point_existence[j]):
        #                                     self.string_vertex[current_string_index + 1] = self.unit_center[self.unit_control[j, k]]
        #                                     current_string_index += 1
        #                                 current_string_index += 1
        #                             else:
        #                                 break
        #                         if self.constraint_end_point_existence[j]:
        #                             self.string_vertex[current_string_index] = self.constraint_end_point[j]
        #                             current_string_index += 1
        #             else:
        #                 self.string_vertex[current_string_index] = self.constraint_start_point[accumulate_index]
        #                 current_string_index += 1
        #                 for k in range(self.max_control_length):
        #                     if self.unit_control[accumulate_index, k] != -1:
        #                         self.string_vertex[current_string_index] = self.unit_center[self.unit_control[accumulate_index, k]]
        #                         if self.unit_control[accumulate_index, k + 1] != -1 or self.constraint_end_point_existence[accumulate_index]:
        #                             self.string_vertex[current_string_index + 1] = self.unit_center[self.unit_control[accumulate_index, k]]
        #                             current_string_index += 2
        #                         else:
        #                             current_string_index += 1
        #                     else:
        #                         break
        #                 if self.constraint_end_point_existence[accumulate_index]:
        #                     self.string_vertex[current_string_index] = self.constraint_end_point[accumulate_index]
        #                     current_string_index += 1
        #             accumulate_index += dup_time

        #         # public code
        #         while current_string_index < int(self.tsa_visual_string_number * 2):
        #             self.string_vertex[current_string_index] = tm.vec3([0., 0., 0.])
        #             self.string_vertex[current_string_index + 1] = tm.vec3([0., 0., 0.])
        #             current_string_index += 2

        #         if self.constraint_number:
        #             self.scene.lines(vertices=self.string_vertex, width=2, color=(.6, .03, .8))

        #         self.scene.lines(vertices=self.border_vertex,
        #                 width=4,
        #                 color=(.00, .00, .00))
            
        #     self.string_params[0] = self.gui.slider_float('String_k', self.string_params[0], 0.0, 10.0)
        #     self.string_params[1] = self.gui.slider_float('Shearing_k', self.string_params[1], 0.0, 100.0)
        #     self.n = self.gui.slider_int('Step number', self.n, 100, 2000)
        #     self.dt = self.gui.slider_float('Dt', 0.1 / self.n, 0.0, 0.001)
        #     self.substeps = self.gui.slider_int('Substeps', int(1 / 250 // self.dt), 1, 100)

        #     self.canvas.scene(self.scene)
        #     self.window.show()

    def run(self):
        self.initializeRunning()
        while self.window.running:
            if self.window.get_event(ti.ui.PRESS):
                if self.window.event.key == 'r':
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

                elif self.window.event.key == 'c':
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
                        if self.window.event.key == 'w': 
                            self.folding_angle += self.folding_step
                            if self.folding_angle >= self.folding_max:
                                self.folding_angle = self.folding_max
                        
                        elif self.window.event.key == 's': 
                            self.folding_angle -= self.folding_step
                            if self.folding_angle <= 0:
                                self.folding_angle = 0

                        elif self.window.event.key == 'q': 
                            self.enable_add_folding_angle = self.folding_micro_step
                        
                        elif self.window.event.key == 'a': 
                            self.enable_add_folding_angle = 0.0
                        
                        elif self.window.event.key == 'z': 
                            self.enable_add_folding_angle = -self.folding_micro_step
                    
                    else:
                        if self.window.event.key == 'i': 
                            self.enable_tsa_rotate = self.rotation_step

                        elif self.window.event.key == 'k': 
                            self.enable_tsa_rotate = 0.0
                    
                        elif self.window.event.key == 'm': 
                            self.enable_tsa_rotate = -self.rotation_step

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
                
            if self.folding_angle_reach_pi[0] and self.sim_mode == self.TSA_SIM:
                break

            for i in range(self.substeps):
                # rk
                # if self.sim_mode == self.FOLD_SIM or (self.sim_mode == self.TSA_SIM and not can_rotate):
                # self.RK2(self.dt, self.enable_ground, self.folding_angle, self.sim_mode, self.gravitational_acc, self.NO_QUASI if self.sim_mode == self.FOLD_SIM else self.NO_QUASI)
                # explicit
                self.F(self.folding_angle, self.sim_mode, self.gravitational_acc, self.enable_ground, self.tsa_turning_angle)

                # if self.sim_mode == self.FOLD_SIM:
                self.explicit_step_xv(self.dt, self.enable_ground, self.NO_QUASI)
                # else:
                #     self.verlet(self.dt, self.enable_ground)

                # implicit
                # self.implicit_xv(self.folding_angle, self.sim_mode, self.gravitational_acc, self.enable_ground)
                self.current_t += self.dt
            self.update_vertices() 

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
                self.gui.slider_float('Total folding energy', round(self.total_energy[0], 4), 0.0, round(self.total_energy_maximum, 4))

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
                self.gui.slider_float('Total folding energy', round(self.total_energy[0], 4), 0.0, round(self.total_energy_maximum, 4))

            self.string_params[0] = self.gui.slider_float('String_k', self.string_params[0], 0.0, 10.0)
            self.string_params[1] = self.gui.slider_float('Shearing_k', self.string_params[1], 0.0, 100.0)
            self.n = self.gui.slider_int('Step number', self.n, 100, 2000)
            self.dt = self.gui.slider_float('Dt', 0.1 / self.n, 0.0, 0.001)
            self.substeps = self.gui.slider_int('Substeps', int(1 / 250 // self.dt), 1, 100)

            # for i in range(self.kp_num):
            #     self.force_vertex[2 * i] = self.x[i]
            #     self.force_vertex[2 * i + 1] = self.x[i] - self.record_force[i] / self.point_mass / 500.0

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

    ori_sim = OrigamiSimulationSystem(4)
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
                    value += float(tsa_point[1])
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

    
