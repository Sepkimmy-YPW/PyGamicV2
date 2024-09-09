from utils import *
# import dxfgrabber

# 定义折纸系统，包含各刚度和单元信息
class OrigamiSimulationSystem:
    def __init__(self, unit_edge_max, spring_k=20., bending_k=0.3, face_k=7.5, material_density=1e-9) -> None:
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
        self.dup_time_list = []

    def getNewLines(self):
        new_lines = []
        for ele in self.line_indices:
            if ele[1] != 3:
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
        
    def pointInList(self, kp, tolerance=2):
        for i in range(len(self.kps)):
            if self.distance(kp, self.kps[i]) < tolerance:
                return i
        return -1
    
    def addUnit(self, unit: Unit):    
        temp_indice = []
        unit.repair()
        kps = unit.getSeqPoint()
        kp_num = len(kps)
        # invalid unit
        if kp_num > self.unit_edge_max:
            return
        # area = unit.calculateArea()
        # mass = area * self.material_density / kp_num
        unit.setupMass(self.material_density)

        for i in range(kp_num):
            kp = kps[i]
            if len(kp) == 2:
                kp += [0.0]
            exist_indice = self.pointInList(kp)
            if exist_indice < 0:
                temp_indice.append(len(self.kps))
                self.kps.append(kp)
                self.mass_list.append(unit.mass[i])
                self.dup_time_list.append(1.)
            else:
                self.kps[exist_indice] = [
                    (self.kps[exist_indice][X] * self.dup_time_list[exist_indice] + kp[X]) / (self.dup_time_list[exist_indice] + 1.),
                    (self.kps[exist_indice][Y] * self.dup_time_list[exist_indice] + kp[Y]) / (self.dup_time_list[exist_indice] + 1.),
                    (self.kps[exist_indice][Z] * self.dup_time_list[exist_indice] + kp[Z]) / (self.dup_time_list[exist_indice] + 1.)
                ]
                self.dup_time_list[exist_indice] += 1.
                temp_indice.append(exist_indice)
                self.mass_list[exist_indice] += unit.mass[i]
        
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

    def calculateElementK(self, tri_indices):
        k1 = self.spring_k
        # for i in range(0, n):
        #     k_element[i][(i + 1) % n] = k1
        #     k_element[(i + 1) % n][i] = k1
        for index in tri_indices:
            self.connection_matrix[index[0]][index[1]] = k1
            self.connection_matrix[index[1]][index[0]] = k1
            self.connection_matrix[index[0]][index[2]] = k1
            self.connection_matrix[index[2]][index[0]] = k1
            self.connection_matrix[index[1]][index[2]] = k1
            self.connection_matrix[index[2]][index[1]] = k1
        # return k_element

    def calculateMaximumDeltaAngle(self, x0, x1, x2):
        x0x1 = distance(x0, x1)
        x1x2 = distance(x1, x2)
        x2x0 = distance(x2, x0)
        alpha0 = math.acos((x0x1**2+x2x0**2-x1x2**2)/(2*x0x1*x2x0))
        alpha1 = math.acos((x1x2**2+x0x1**2-x2x0**2)/(2*x1x2*x0x1))
        alpha2 = math.acos((x2x0**2+x1x2**2-x0x1**2)/(2*x2x0*x1x2))
        return max([abs(alpha0 - alpha1), abs(alpha0 - alpha2), abs(alpha1 - alpha2)])
    
    def mesh(self):
        self.tri_indices.clear()
        kp_len = len(self.kps)
        self.connection_matrix = [[0] * kp_len for _ in range(kp_len)]
        self.origin_distance_matrix = [[0] * kp_len for _ in range(kp_len)]

        #origin distance
        for i in range(kp_len):
            for j in range(kp_len):
                self.origin_distance_matrix[i][j] = self.distance(self.kps[i], self.kps[j])
        
        # for i in range(kp_len):
        #     self.mass_list[i] = 5e-7

        facet_pair = []
        #spring force
        for i in range(len(self.unit_list)):
            indices = self.indices[i]
            unit = self.unit_list[i].getSeqPoint()
            unit_kp_len = len(indices)
            
            complete_id = []
            tri_indices = []

            while unit_kp_len - len(complete_id) > 3:
                delta_angle_max = 3.15
                temp_tri_indices = None
                pointer = 0
                
                while pointer < unit_kp_len - len(complete_id):
                    while indices[pointer] in complete_id:
                        pointer += 1
                    next_pointer = (pointer + 1) % unit_kp_len
                    while indices[next_pointer] in complete_id:
                        next_pointer = (next_pointer + 1) % unit_kp_len
                    next_next_pointer = (next_pointer + 1) % unit_kp_len
                    while indices[next_next_pointer] in complete_id:
                        next_next_pointer = (next_next_pointer + 1) % unit_kp_len
                    
                    current_delta_angle = self.calculateMaximumDeltaAngle(unit[pointer], unit[next_pointer], unit[next_next_pointer])
                    if current_delta_angle < delta_angle_max:
                        temp_tri_indices = [indices[pointer], indices[next_pointer], indices[next_next_pointer]]
                        delta_angle_max = current_delta_angle
                    pointer += 1
                tri_indices.append(temp_tri_indices)
                complete_id.append(temp_tri_indices[1])
                facet_pair.append([temp_tri_indices[0], temp_tri_indices[2]])
                # self.addToLineIndices([temp_tri_indices[0], temp_tri_indices[2]], 3)
            temp_tri_indices = []
            for pointer in range(unit_kp_len):
                if indices[pointer] not in complete_id:
                    temp_tri_indices.append(indices[pointer])
            tri_indices.append(temp_tri_indices)
            
            self.calculateElementK(tri_indices)

            for j in range(len(tri_indices)):
                tri_index = tri_indices[j]
                self.tri_indices += [tri_index[0], tri_index[1], tri_index[2]]
            # for k in range(unit_kp_len):
            #     indice_k = indices[k]
            #     for l in range(unit_kp_len):
            #         indice_l = indices[l]
            #         self.connection_matrix[indice_k][indice_l] = k_element[k][l]
        
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
        for i in range(len(facet_pair)):
            indices = self.indices[i]
            line_start_indice = facet_pair[i][0]
            line_end_indice = facet_pair[i][1]
            start_row = self.connection_matrix[line_start_indice]
            end_row = self.connection_matrix[line_end_indice]

            relevant_kp = []
            for k in range(kp_len):
                if abs(start_row[k] - end_row[k]) < 1e-5 and start_row[k] > 0:
                    relevant_kp.append(k)
            
            if len(relevant_kp) == 2 and [line_end_indice, line_start_indice] not in self.facet_crease_pairs:
                facet_crease_pair = [line_start_indice, line_end_indice]
                vec1xy = [self.kps[line_start_indice][0] - self.kps[relevant_kp[0]][0], self.kps[line_start_indice][1] - self.kps[relevant_kp[0]][1]]
                vec2xy = [self.kps[relevant_kp[1]][0] - self.kps[line_start_indice][0], self.kps[relevant_kp[1]][1] - self.kps[line_start_indice][1]]
                result = vec1xy[0] * vec2xy[1] - vec1xy[1] * vec2xy[0]
                if result >= 0:
                    self.facet_bending_pairs.append([relevant_kp[0], relevant_kp[1]])
                else:
                    self.facet_bending_pairs.append([relevant_kp[1], relevant_kp[0]])
                self.facet_crease_pairs.append(facet_crease_pair)
            
        total_mass = sum(self.mass_list)
        avg_mass = total_mass / len(self.mass_list)
        for i in range(len(self.mass_list)):
            self.mass_list[i] = avg_mass