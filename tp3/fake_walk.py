"""
Kinematics excercice with 3 tasks:
- COM at constant velocity
- right foot and left foot alternating at constant velocity

"""

import time
import unittest

import example_robot_data as robex
import pinocchio as pin
import numpy as np
from numpy.linalg import norm,pinv
from collections import namedtuple

from schaeffler2025.meshcat_viewer_wrapper import MeshcatVisualizer

# --- Load robot model
robot = robex.load("talos")
model = robot.model
data = model.createData()
NQ = model.nq
NV = model.nv

# Open the viewer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)

# %jupyter_snippet 1
feetIndexes = {
    "right": robot.model.getFrameId("right_sole_link"),
    "left": robot.model.getFrameId("left_sole_link"),
}

# Assert all frame names are valid.
assert len(robot.model.frames) not in feetIndexes

# Structure to define a phase of locomotion
Phase = namedtuple('Phase', [ 'duration','delta','side' ])

# Definition of the task
comVelocity = np.array([0.7,0,0])
stepDuration = .8
step = comVelocity[:2]*stepDuration*2
phases = [ 
    Phase(stepDuration/2,step/2,'right'),
    Phase(stepDuration,  step,  'left' ),
    Phase(stepDuration,  step,  'right'),
    Phase(stepDuration,  step,  'left' ),
    Phase(stepDuration,  step,  'right'),
    Phase(stepDuration/2,step/2,'left' ),
]
DT = 1e-2

q = robot.q0.copy()
for phase in phases:
    for t in np.arange(0,phase.duration,DT):

        # Pre-compute kinematics, jacobians, com etc
        pin.computeAllTerms(model,data,q,np.zeros(NV))

        # Right foot
        edot_right = phase.delta/phase.duration if phase.side=='right' else np.zeros(2)
        edot_right = np.r_[edot_right,[0,0,0,0]] # Make it 6D
        J_right = pin.getFrameJacobian(model,data,feetIndexes['right'],pin.LOCAL)
        # Left foot
        edot_left = phase.delta/phase.duration if phase.side=='left' else np.zeros(2)
        J_left = pin.getFrameJacobian(model,data,feetIndexes['left'],pin.LOCAL)
        edot_left = np.r_[edot_left,[0,0,0,0]] # Make it 6D
        # COM
        edot_com = comVelocity
        J_com = pin.jacobianCenterOfMass(model,data,q)

        edot = np.r_[edot_right,edot_left,edot_com]
        J = np.r_[J_right,J_left,J_com]

        vq = pinv(J)@edot
        q = pin.integrate(model,q,vq*DT)
        viz.display(q)
        time.sleep(DT)
                                       
### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class FloatingTest(unittest.TestCase):
    def test_finish(self):
        com1 = pin.centerOfMass(model,data,q).copy()
        com0 = pin.centerOfMass(model,data,robot.q0).copy()
        dcom = com1-com0
        dcom_ref = comVelocity*sum([ p.duration for p in phases])

        self.assertTrue(
            np.allclose(dcom,dcom_ref,atol=1e-2)
        )

FloatingTest().test_finish()
