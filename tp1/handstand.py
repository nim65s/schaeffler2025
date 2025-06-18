"""
Variation of the program simple_pick_and_place.py

Example of forward geometry: the robot is moved along an arbitrary trajectory
(computed with ad-hoc sinus waves) and at each time step, the end effector
position is computed; a 3d object is then placed at this position, displaying
a rigid constraints between the robot effector and the object. Then the same
is done with the placement of the end effector, to which a purple brick is
attached.
"""

import math
import time
import unittest

import example_robot_data as robex
import numpy as np
import pinocchio as pin

from schaeffler2025.meshcat_viewer_wrapper import MeshcatVisualizer, colors

# %jupyter_snippet 1
robot = robex.load("talos")
# %end_jupyter_snippet
NQ = robot.model.nq
NV = robot.model.nv

# Open the viewer
viz = MeshcatVisualizer(robot)

# %jupyter_snippet 2
# Add a red box in the viewer
ballID = "world/ball"
viz.addSphere(ballID, 0.1, colors.red)

# Place the ball at the position ( 0.5, 0.1, 0.2 )
# The viewer expect position and rotation, apppend the identity quaternion
q_ball = [0.5, 0.1, 1.2, 1, 0, 0, 0]
viz.applyConfiguration(ballID, q_ball)
# %end_jupyter_snippet

#
# PICK #############################################################
#
# First choose a robot configuration where the end-effector is
# touching the sphere object. The angles of the robot are manually
# tuned to obtain the contact.

# Configuration for picking the box
# %jupyter_snippet 3
q0 = robot.q0.copy()
q0[29:36] = [.9, -0.1, 3.2, 1.7, 0, 0.3, 0.1 ]
viz.display(q0)
# %end_jupyter_snippet
print("The robot is display with end effector on the red ball.")

#
# MOVE 3D #############################################################
#
# Then move the robot along an arbitrary trajectory and position (3d)
# the external spherical object so that it keeps the same rigid position
# with respect to the end effector, hence displaying a rigid 3d constraint.


print("Let's start the movement ...")

# %jupyter_snippet 4

idx = robot.model.getFrameId("gripper_right_fingertip_3_link")
pin.framesForwardKinematics(robot.model,robot.data,q0)
# Position of end-eff wrt world at current configuration
o_eff = robot.data.oMf[idx].translation
o_ball = q_ball[:3]  # Position of ball wrt world
eff_ball = o_ball - o_eff  # Position of ball wrt eff

delta = np.random.rand(robot.model.nq-7)*.2 - .1
delta[:12] = 0 # Do not move the legs
for t in range(50):
    # Chose new configuration of the robot
    q = q0.copy()
    q[7:] = q0[7:] + np.sin(3.14*t/100.0)*delta

    # Gets the new position of the ball
    pin.framesForwardKinematics(robot.model,robot.data,q)
    o_ball = robot.data.oMf[idx] * eff_ball

    # Display new configuration for robot and ball
    viz.applyConfiguration(ballID, o_ball.tolist() + [1, 0, 0, 0])
    viz.display(q)
    time.sleep(1e-2)
# %end_jupyter_snippet

#
# MOVE 6D #############################################################
#
# Finally, compute the placement of the end effector and properly place
# a non-spherical object, hence displaying a rigid 6d constraint when
# the robot moves.

# Choose the reference posture of the robot
q0 = robot.q0.copy()
q0[20:22] = [-.5,0.2]
q0[29:35] = [ .3, -1.5, -0.7, -1.6, -1.2, -0.8 ]

delta = np.random.rand(robot.model.nq-7)*.2 - .1
for t in range(50):
    q = q0.copy()
    q[7:] = q0[7:] + np.sin(3.14*t/100.0)*delta

    pin.framesForwardKinematics(robot.model,robot.data,q)
    oMbasis = robot.data.oMf[1]
    oMeff = robot.data.oMf[idx]
    effMbasis = oMeff.inverse()*oMbasis

    q[:3] = effMbasis.translation
    q[3:7] = pin.Quaternion(effMbasis.rotation).coeffs()
    viz.display(q)
    time.sleep(.01)


### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class SimplePickAndPlaceTest(unittest.TestCase):
    def test_hand_placement(self):
        pin.framesForwardKinematics(robot.model,robot.data,q)
        self.assertTrue(
            (
                np.abs(
                    pin.log(robot.data.oMf[idx]).vector)
                < 1e-5
            ).all()
        )


SimplePickAndPlaceTest().test_hand_placement()
