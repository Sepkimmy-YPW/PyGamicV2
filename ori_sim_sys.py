from utils import *
# import dxfgrabber

# 定义折纸系统，包含各刚度和单元信息
class OrigamiSimulationSystem:
    def __init__(self, unit_edge_max, spring_k=20., bending_k=.8, face_k=.5, material_density=1e-9) -> None:
        self.unit_edge_max = unit_edge_max
        self.unit_list = []
        self.kps = []
        self.line_indices = []
        self.indices = []
        self.tri_indices = []
        self.connection_matrix = None
        self.spring_k = spring_k
        self.bending_k = bending_k
        self.face_k = face_k
        self.material_density = material_density
        self.bending_pairs = []
        self.crease_pairs = []
        self.facet_bending_pairs = []
        self.facet_crease_pairs = []
        self.mass_list = []

    def getNewLines(self):
        new_lines = []
        for ele in self.line_indices:
            new_lines.append(Crease(self.kps[ele[0][START]], self.kps[ele[0][END]], ele[1]))
        return new_lines
    
    def addToLineIndices(self, pair, linetype):
        duplicated = False
        for ele in self.line_indices:
            if ele[0][0] == pair[0] and ele[0][1] == pair[1]:
                duplicated = True
                break
            if ele[0][1] == pair[0] and ele[0][0] == pair[1]:
                duplicated = True
                break
        if not duplicated:
            self.line_indices.append([pair, linetype])

    def distance(self, kp1, kp2):
        return ((kp1[0] - kp2[0]) ** 2 + (kp1[1] - kp2[1]) ** 2 + (kp1[2] - kp2[2]) ** 2) ** 0.5
        
    def pointInList(self, kp, tolerance=1e-5):
        for i in range(len(self.kps)):
            if self.distance(kp, self.kps[i]) < tolerance:
                return i
        return -1
    
    def addUnit(self, unit: Unit):    
        temp_indice = []
        kps = unit.getSeqPoint()
        kp_num = len(kps)
        # invalid unit
        if kp_num > self.unit_edge_max:
            return
        area = unit.calculateArea()
        mass = area * self.material_density / kp_num
        for i in range(kp_num):
            kp = kps[i]
            if len(kp) == 2:
                kp += [0.0]
            exist_indice = self.pointInList(kp)
            if exist_indice < 0:
                temp_indice.append(len(self.kps))
                self.kps.append(kp)
                self.mass_list.append(mass)
            else:
                temp_indice.append(exist_indice)
                self.mass_list[exist_indice] += mass
        
        for i in range(len(kps)):
            next_i = (i + 1) % len(kps)
            linetype = unit.crease[i].getType()
            indice1 = temp_indice[i]
            indice2 = temp_indice[next_i]
            self.addToLineIndices([indice1, indice2], linetype)

        self.unit_list.append(unit)
        self.indices.append(temp_indice)

    def fillBlankIndices(self):
        for indice in self.indices:
            # fill the blank
            indice_num = len(indice)
            for i in range(indice_num, self.unit_edge_max):
                indice.append(-1)

    def calculateElementK(self, n):
        k_element = [[0] * n for _ in range(n)]
        k1 = self.spring_k
        k_element = [[k1 if i != j else 0 for i in range(n)] for j in range(n)]
        for i in range(1, n - 2):
            k_element[i][i + 2] = 1000.0 * k1
            k_element[i + 2][i] = 1000.0 * k1
        return k_element

    def mesh(self):
        self.tri_indices.clear()
        kp_len = len(self.kps)
        self.connection_matrix = [[0] * kp_len for _ in range(kp_len)]
        self.origin_distance_matrix = [[0] * kp_len for _ in range(kp_len)]

        #origin distance
        for i in range(kp_len):
            for j in range(kp_len):
                self.origin_distance_matrix[i][j] = self.distance(self.kps[i], self.kps[j])

        #spring force
        for i in range(len(self.unit_list)):
            indices = self.indices[i]
            unit_kp_len = len(indices)

            k_element = self.calculateElementK(unit_kp_len)
            
            start_indice = indices[0]
            for j in range(1, unit_kp_len - 1):
                cur_indice = indices[j]
                next_indice = indices[(j + 1) % unit_kp_len]
                self.tri_indices += [start_indice, cur_indice, next_indice]
            for k in range(unit_kp_len):
                indice_k = indices[k]
                for l in range(unit_kp_len):
                    indice_l = indices[l]
                    self.connection_matrix[indice_k][indice_l] = k_element[k][l]
        
        #bending force
        for i in range(len(self.unit_list)):
            indices = self.indices[i]
            unit_kp_len = len(indices)
            for j in range(unit_kp_len):
                line_start_indice = indices[j]
                line_end_indice = indices[(j + 1) % unit_kp_len]
                start_row = self.connection_matrix[line_start_indice]
                end_row = self.connection_matrix[line_end_indice]

                relevant_kp = []
                for k in range(kp_len):
                    if abs(start_row[k] - end_row[k]) < 1e-5 and start_row[k] > 0:
                        relevant_kp.append(k)
                
                if len(relevant_kp) == 2 and [line_end_indice, line_start_indice] not in self.crease_pairs:
                    crease_pair = [line_start_indice, line_end_indice]
                    # 检测正则
                    vec1xy = [self.kps[line_start_indice][0] - self.kps[relevant_kp[0]][0], self.kps[line_start_indice][1] - self.kps[relevant_kp[0]][1]]
                    vec2xy = [self.kps[line_end_indice][0] - self.kps[line_start_indice][0], self.kps[line_end_indice][1] - self.kps[line_start_indice][1]]
                    result = vec1xy[0] * vec2xy[1] - vec1xy[1] * vec2xy[0]
                    if result >= 0:
                        self.bending_pairs.append([relevant_kp[0], relevant_kp[1]])
                    else:
                        self.bending_pairs.append([relevant_kp[1], relevant_kp[0]])
                    self.crease_pairs.append(crease_pair)
        
        # facet
        for i in range(len(self.unit_list)):
            indices = self.indices[i]
            unit_kp_len = len(indices)
            
            for start_id in range(0, unit_kp_len - 2):
                line_start_indice = indices[start_id]
                for j in range(start_id + 2, unit_kp_len):
                    line_end_indice = indices[j]
                    start_row = self.connection_matrix[line_start_indice]
                    end_row = self.connection_matrix[line_end_indice]

                    relevant_kp = []
                    for k in range(kp_len):
                        if abs(start_row[k] - end_row[k]) < 1e-5 and start_row[k] > 0:
                            relevant_kp.append(k)
                    
                    if len(relevant_kp) == 2 and [line_end_indice, line_start_indice] not in self.facet_crease_pairs and relevant_kp[0] in indices and relevant_kp[1] in indices:
                        facet_crease_pair = [line_start_indice, line_end_indice]
                        vec1xy = [self.kps[line_start_indice][0] - self.kps[relevant_kp[0]][0], self.kps[line_start_indice][1] - self.kps[relevant_kp[0]][1]]
                        vec2xy = [self.kps[relevant_kp[1]][0] - self.kps[line_start_indice][0], self.kps[relevant_kp[1]][1] - self.kps[line_start_indice][1]]
                        result = vec1xy[0] * vec2xy[1] - vec1xy[1] * vec2xy[0]
                        if result >= 0:
                            self.facet_bending_pairs.append([relevant_kp[0], relevant_kp[1]])
                        else:
                            self.facet_bending_pairs.append([relevant_kp[1], relevant_kp[0]])
                        self.facet_crease_pairs.append(facet_crease_pair)