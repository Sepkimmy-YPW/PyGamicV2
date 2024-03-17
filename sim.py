# import numpy as np

# class Object3D:
#     def __init__(self, vertices, faces):
#         self.vertices = np.array(vertices, dtype=float)
#         self.faces = np.array(faces)

#     def translate(self, vector):
#         self.vertices += np.array(vector)

#     def rotate(self, axis, angle):
#         angle = np.radians(angle)
#         axis = np.array(axis) / np.linalg.norm(axis)
#         R = np.array([[np.cos(angle) + axis[0]**2*(1-np.cos(angle)), 
#                        axis[0]*axis[1]*(1-np.cos(angle)) - axis[2]*np.sin(angle), 
#                        axis[0]*axis[2]*(1-np.cos(angle)) + axis[1]*np.sin(angle)],
#                       [axis[1]*axis[0]*(1-np.cos(angle)) + axis[2]*np.sin(angle), 
#                        np.cos(angle) + axis[1]**2*(1-np.cos(angle)), 
#                        axis[1]*axis[2]*(1-np.cos(angle)) - axis[0]*np.sin(angle)],
#                       [axis[2]*axis[0]*(1-np.cos(angle)) - axis[1]*np.sin(angle), 
#                        axis[2]*axis[1]*(1-np.cos(angle)) + axis[0]*np.sin(angle), 
#                        np.cos(angle) + axis[2]**2*(1-np.cos(angle))]])
#         self.vertices = np.dot(self.vertices, R.T)

#     def plot(self, ax):
#         ax.plot(self.vertices[:,0], self.vertices[:,1], self.vertices[:,2])
#         for face in self.faces:
#             ax.plot(self.vertices[face,0], self.vertices[face,1], self.vertices[face,2], 'k--')

# class PaperFolding:
#     def __init__(self, object3d):
#         self.object = object3d
#         self.fold_lines = []

#     def add_fold_line(self, start, end):
#         self.fold_lines.append((start, end))

#     def fold(self, axis, angle):
#         for i in range(len(self.fold_lines)):
#             start, end = self.fold_lines[i]
#             midpoint = np.array([(start[x] + end[x]) / 2 for x in range(3)])
#             self.object.translate(-midpoint)
#             self.object.rotate(axis, angle)
#             self.object.translate(midpoint)

#     def plot(self, ax):
#         self.object.plot(ax)
#         for line in self.fold_lines:
#             ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]], 'r')

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 创建一个三维图形窗口
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 创建一个Object3D对象和一个PaperFolding对象
# vertices = [(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)]
# faces = [(0,1,2,3), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7), (4,5,6,7)]
# obj = Object3D(vertices, faces)
# paper = PaperFolding(obj)

# # 添加折痕并折叠
# paper.add_fold_line((0,0,0), (1,1,1))
# paper.add_fold_line((1,0,0), (0,1,1))
# paper.fold((0,0,1), 45)

# # 绘制折纸
# paper.plot(ax)

# # 设置图像的范围和标签
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# # 显示图像
# plt.show()

# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider

# # 创建一个Figure对象和一个Axes对象
# fig, ax = plt.subplots()

# # 定义初始斜率和线段
# slope = 1
# start = (0, 0)
# end = (1, slope)

# # 绘制初始线段
# line, = ax.plot([start[0], end[0]], [start[1], end[1]])

# # 定义一个函数，用于更新线段
# def update_line(val):
#     global slope, start, end
#     slope = val
#     end = (start[0] + 1, start[1] + slope)
#     line.set_xdata([start[0], end[0]])
#     line.set_ydata([start[1], end[1]])
#     fig.canvas.draw_idle()

# # 创建一个Slider对象，用于控制斜率
# slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
# slider = Slider(slider_ax, "斜率", -5, 5, valinit=slope)
# slider.on_changed(update_line)

# # 设置图像的范围和标签
# ax.set_xlim(-1, 2)
# ax.set_ylim(-5, 5)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")

# # 显示图像
# plt.show()

# import math

# def cosine_similarity(x, y):
#     numerator = sum(a * b for a, b in zip(x, y))
#     denominator = math.sqrt(sum(a * a for a in x)) * math.sqrt(sum(b * b for b in y))
#     if denominator == 0:
#         return 0
#     else:
#         return numerator / denominator

# def similarity_curve(curve1, curve2):
#     # 对两条曲线进行插值，使得点数一致
#     n = max(len(curve1), len(curve2))
#     curve1 = [curve1[int(i * (len(curve1) - 1) / (n - 1))] for i in range(n)]
#     curve2 = [curve2[int(i * (len(curve2) - 1) / (n - 1))] for i in range(n)]
#     # 计算余弦相似度
#     return cosine_similarity([x[1] for x in curve1], [y[1] for y in curve2])
    
# if __name__ == '__main__':
#     curve1 = [(0, 0), (1, 1), (2, 2), (3, 3)]
#     curve2 = [(0, 0), (1, 0), (2, 0), (3, 0)]
#     similarity = similarity_curve(curve1, curve2)
#     print('The similarity between curve1 and curve2 is:', similarity)

# from shapesimilarity import shape_similarity
# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(-1, 1, num=200)
# x1 = np.linspace(-1, 0, num=150)
# x2 = np.linspace(0, 1, num=50)

# # 方法一 通过函数构建曲线
# y1 = x**2 + 1
# y2 = np.array((x1**2 + 1).tolist() + (x2**2 + 1).tolist())
# shape1 = np.column_stack((x, y1))
# shape2 = np.column_stack((np.array(x1.tolist() + x2.tolist()), y2))

# # 方法二 直接输入节点坐标(x, y)
# shape3 = [(1, 1),(2.2, 2),(3.1, 3),(4, 4)]
# shape4 = [(1, 1),(1, 2),(1.1, 3),(1.1, 4)]
# shape3 = np.column_stack(shape3)
# shape4 = np.column_stack(shape4)

# # 调用库计算相似度
# similarity1 = shape_similarity(shape1, shape2)
# similarity2 = shape_similarity(shape3, shape4)

# # 方法一和方法二的相似度输出
# print("similarity1:{}".format(similarity1))
# print("similarity2:{}".format(similarity2))

# # 图形展示部分
# # 方法一：
# plt.plot(shape1[:, 0], shape1[:, 1], linewidth=2.0)
# plt.plot(shape2[:, 0], shape2[:, 1], linewidth=2.0)

# plt.title(f'Shape similarity is: {similarity1}', fontsize=14, fontweight='bold')
# plt.show()

# # 方法二
# plt.plot(shape3[:, 0], shape3[:, 1], linewidth=2.0)
# plt.plot(shape4[:, 0], shape4[:, 1], linewidth=2.0)

# plt.title(f'Shape similarity is: {similarity2}', fontsize=14, fontweight='bold')
# plt.show()

# lis = [0, 1]

# print(str(lis))

# import numpy as np

# def is_closed(polygon):
#     """判断一个多边形是否封闭"""
#     if len(polygon) < 3:
#         return False
#     # 计算多边形首尾两个点的距离
#     dist = np.linalg.norm(polygon[0] - polygon[-1])
#     return dist < 1e-6

# def is_intersected(segment1, segment2):
#     """判断两条线段是否相交"""
#     # 将线段表示成向量形式
#     v1 = segment1[1] - segment1[0]
#     v2 = segment2[1] - segment2[0]
#     # 计算向量的叉积
#     cross1 = np.cross(v1, segment2[0] - segment1[0])
#     cross2 = np.cross(v1, segment2[1] - segment1[0])
#     cross3 = np.cross(v2, segment1[0] - segment2[0])
#     cross4 = np.cross(v2, segment1[1] - segment2[0])
#     # 判断两个向量是否同向
#     if np.dot(v1, v2) > 0:
#         return False
#     # 判断两个向量是否有一个为零向量
#     if np.allclose(v1, 0) or np.allclose(v2, 0):
#         return False
#     # 判断两条线段是否有交点
#     if np.sign(cross1) != np.sign(cross2) and np.sign(cross3) != np.sign(cross4):
#         return True
#     return False

# def get_polygons(segments):
#     """从线段列表中获取所有封闭图形"""
#     polygons = []
#     while len(segments) > 0:
#         polygon = []
#         segment = segments.pop(0)
#         polygon.append(segment[0])
#         polygon.append(segment[1])
#         while True:
#             flag = False
#             for i in range(len(segments)):
#                 if is_intersected(segment, segments[i]):
#                     segment = segments.pop(i)
#                     polygon.append(segment[1])
#                     flag = True
#                     break
#             if not flag:
#                 break
#         if is_closed(polygon):
#             polygons.append(polygon)
#     return polygons

# # 示例输入数据
# segments = [np.array([[0, 0], [1, 1]]),
#             np.array([[1, 1], [2, 1]]),
#             np.array([[2, 1], [2, 0]]),
#             np.array([[2, 0], [1, 0]]),
#             np.array([[1, 0], [0, 0]]),
#             np.array([[1, 1], [1, 0]])]

# # 获取所有封闭图形
# polygons = get_polygons(segments)

# # 输出结果
# for polygon in polygons:
#     print(polygon)

print([1, 1] == [1, 1])