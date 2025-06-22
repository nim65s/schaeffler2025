"""
Variation of the floating.py example, using a humanoid robot.
This test exhibits that the quaternion should be kept of norm=1
when computing the robot configuration. Otherwise, a nonunit quaternion
does not correspond to any meaningful rotation. Here the viewer tends
to interpret this nonunit quaternion as the composition of a rotation
and a nonhomogeneous scaling, leading to absurd 3d display.
When the quaternion is explicitly kept of norm 1, everything works
fine and similarly to the unconstrained case.
"""

# %jupyter_snippet import
import time
import unittest

import example_robot_data as robex
import numpy as np
from numpy.linalg import norm
from scipy.optimize import fmin_bfgs
import pinocchio as pin

from schaeffler2025.meshcat_viewer_wrapper import MeshcatVisualizer

# --- Load robot model
robot = robex.load("talos")
NQ = robot.model.nq
NV = robot.model.nv

# Open the viewer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
# %end_jupyter_snippet

# %jupyter_snippet 1
robot.feetIndexes = [
    robot.model.getFrameId(frameName)
    for frameName in [
        "gripper_right_fingertip_3_link",
        "gripper_left_fingertip_3_link",
        "right_sole_link",
        "left_sole_link",
    ]
]
# %end_jupyter_snippet

# Assert all frame names are valid.
assert len(robot.model.frames) not in robot.feetIndexes

# --- Add box to represent target
# %jupyter_snippet 2
# We define 4 targets, one for each leg.
colors = ["red", "blue", "green", "magenta"]
for color in colors:
    viz.addSphere("world/%s" % color, 0.05, color)
    viz.addSphere("world/%s_des" % color, 0.05, color)

targets = [
    np.array([-0.7, -0.2, 1.2]),
    np.array([-0.3, 0.5, 0.8]),
    np.array([0.3, 0.1, -1.1]),
    np.array([0.9, 0.9, 0.5]),
]
# Make the targets reachable
targets = [t * 0.6 for t in targets]
# %end_jupyter_snippet


# %jupyter_snippet cost
def cost(q):
    """
    Compute score from a configuration: sum of the 4 reaching
    tasks, one for each leg.
    """
    q[3:7] = q[3:7] / norm(q[3:7])
    cost = 0.0
    pin.framesForwardKinematics(robot.model,robot.data,q)
    for i in range(4):
        p_i = robot.data.oMf[robot.feetIndexes[i]].translation
        cost += norm(p_i - targets[i]) ** 2
    cost += (norm(q[3:7]) ** 2 - 1) ** 2
    return cost
# %end_jupyter_snippet

# %jupyter_snippet callback
def callback(q):
    """
    Diplay the robot, postion a ball of different color for
    each foot, and the same ball of same color for the location
    of the corresponding target.
    """
    q[3:7] = q[3:7] / norm(q[3:7])
    pin.framesForwardKinematics(robot.model,robot.data,q)
    for i in range(4):
        p_i = robot.data.oMf[robot.feetIndexes[i]]
        viz.applyConfiguration("world/%s" % colors[i], p_i)
        viz.applyConfiguration(
            "world/%s_des" % colors[i], list(targets[i]) + [1, 0, 0, 0]
        )

    viz.display(q)
    time.sleep(1e-2)
# %end_jupyter_snippet

# %jupyter_snippet optim
qopt = fmin_bfgs(cost, robot.q0, callback=callback)
# %end_jupyter_snippet


### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class FloatingTest(unittest.TestCase):
    def test_cost(self):
        self.assertLess(cost(qopt), 1e-10)


FloatingTest().test_cost()
