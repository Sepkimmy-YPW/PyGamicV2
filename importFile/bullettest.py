# import pybullet as p
# import time
# import pybullet_data

# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw = 3.6,cameraPitch=-27, cameraTargetPosition=[0.19,1.21,-0.44])

# p.loadURDF("plane.urdf", 0, 0, -2)
# wheelA = p.loadURDF("differential/diff_ring.urdf", [0, 0, 0])
# for i in range(p.getNumJoints(wheelA)):
#   print(p.getJointInfo(wheelA, i))
#   p.setJointMotorControl2(wheelA, i, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

# c = p.createConstraint(wheelA,
#                        1,
#                        wheelA,
#                        3,
#                        jointType=p.JOINT_GEAR,
#                        jointAxis=[0, 1, 0],
#                        parentFramePosition=[0, 0, 0],
#                        childFramePosition=[0, 0, 0])
# p.changeConstraint(c, gearRatio=1, maxForce=10000,erp=0.2)

# c = p.createConstraint(wheelA,
#                        2,
#                        wheelA,
#                        4,
#                        jointType=p.JOINT_GEAR,
#                        jointAxis=[0, 1, 0],
#                        parentFramePosition=[0, 0, 0],
#                        childFramePosition=[0, 0, 0])
# p.changeConstraint(c, gearRatio=-1, maxForce=10000,erp=0.2)

# c = p.createConstraint(wheelA,
#                        1,
#                        wheelA,
#                        4,
#                        jointType=p.JOINT_GEAR,
#                        jointAxis=[0, 1, 0],
#                        parentFramePosition=[0, 0, 0],
#                        childFramePosition=[0, 0, 0])
# p.changeConstraint(c, gearRatio=-1, maxForce=10000,erp=0.2)

# p.setRealTimeSimulation(1)
# while (1):
#   p.setGravity(0, 0, -10)
#   time.sleep(0.01)
# #p.removeConstraint(c)


# from itertools import combinations

# def unique_combinations(nums, m):
#     return set(combinations(nums, m))

# n = int(input("Enter the value of n: "))
# m = int(input("Enter the value of m: "))

# numbers = [i for i in range(1, n+1)]
# result = unique_combinations(numbers, m)

# print("Unique combinations:")
# for combo in result:
#     print(combo)

import taichi as ti
import taichi.math as tm

ti.init()
N_param = 3
N_loss = 3
x = ti.field(dtype=ti.f32, shape=N_param, needs_dual=True)
y = ti.field(dtype=ti.f32, shape=N_loss, needs_dual=True)
z = ti.Vector.field(2, ti.f32, shape=N_param)

@ti.func
def fun(x):
    a = x[0] + 1
    return a
    
@ti.kernel
def compute_y():
    y[0] += x[2]
    for i in range(N_loss - 1):
        y[i] += x[i] * x[2]
        y[0] += 1
        y[2] += fun([x[0], x[1], x[2]])
    y[2] += x[1]

for i in range(N_param):
  with ti.ad.FwdMode(loss=y, param=x, seed=[1 if j == i else 0 for j in range(N_param)]):
      compute_y()
  print('dy/dx =', y.dual, y, ' at x =', x[i])

# @ti.func
# def compute_x(x):
#     x[0] = x[0] + 1

# @ti.kernel
# def compute():
#     compute_x(z[0])

# print(z)
# compute()
# print(z)

# Compute derivatives with respect to x_0
# `seed` is required if `param` is not a scalar field

# compute_y()
# print('dy/dx =', y.dual, y, ' at x =', x[i])