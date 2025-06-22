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
import matplotlib.pyplot as plt; plt.ion()
import proxsuite

from schaeffler2025.meshcat_viewer_wrapper import MeshcatVisualizer

np.set_printoptions(precision=3, linewidth=350, suppress=True,threshold=1e6)

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
Phase = namedtuple('Phase', [ 'duration','delta','support' ])

# Definition of the task
comVelocity = np.array([.25,0,0])
stepDuration = 0.8
step = comVelocity[:2]*stepDuration*2
phases = [ 
    Phase(stepDuration*2,np.zeros(2),'double'),
    Phase(stepDuration/2,step/2,'right'),
    Phase(stepDuration,  step,  'left' ),
    Phase(stepDuration,  step,  'right'),
    Phase(stepDuration,  step,  'left' ),
    Phase(stepDuration,  step,  'right'),
    Phase(stepDuration/2,step/2,'left' ),
    Phase(stepDuration*2,np.zeros(2),'double'),
]
DT = 6e-2

# ### PRE-COMPUTE TRAJECTORIES ###
# ### PRE-COMPUTE TRAJECTORIES ###
# ### PRE-COMPUTE TRAJECTORIES ###
q = robot.q0.copy()
pin.computeAllTerms(model,data,q,np.zeros(NV))
pin.framesForwardKinematics(model,data,q)
com0 = data.com[0].copy()[:2]
right0 = data.oMf[feetIndexes['right']].translation.copy()[:2]
left0 = data.oMf[feetIndexes['left']].translation.copy()[:2]
omega2 = -model.gravity.linear[2]/data.com[0][2]

right0[0] = 0 # debug
left0[0] = 0 # debug

# ### FEET AND COP REF ###
'''
Although not easy to read because compact, this algorithm
simply compute the right and left feet trajectory as piecewise 
linear interpolation following the delta of each phase.
The COP trajectory is then just selected to be stationary at the
position of the support foot.
'''

# Start phase should be double
assert( phases[0].support == 'double' )

# times is just for plotting nice plots
times = [ 0 ]
right_refs = [ right0.copy() ]
left_refs = [ left0.copy() ]
cop_refs = [ np.mean([right0,left0],0) ]
for phase in phases:
    times.extend( [ times[-1]+t           for t in np.arange(DT,phase.duration+DT,DT) ] )
    delta = phase.delta/phase.duration
    right_refs.extend( [ right_refs[-1]+(delta*t if phase.support=='left' else 0)
                             for t in np.arange(DT,phase.duration+DT,DT) ] )
    left_refs.extend( [ left_refs[-1]+(delta*t if phase.support=='right' else 0)
                             for t in np.arange(DT,phase.duration+DT,DT) ] )    
    cop_refs.extend( [ right_refs[-1] if phase.support == 'right'
                       else left_refs[-1] if phase.support == 'left'
                       else np.mean([right_refs[-1],left_refs[-1]],0)
                        for t in np.arange(0,phase.duration,DT) ] )

plt.figure(1)
plt.subplot(211)
plt.plot(times,[ r[0] for r in right_refs ])
plt.plot(times,[ r[0] for r in left_refs ])
plt.plot(times,[ r[0] for r in cop_refs ])
plt.legend(['right', 'left', 'cop' ])
plt.subplot(212)
plt.plot(times,[ r[1] for r in right_refs ])
plt.plot(times,[ r[1] for r in left_refs ])
plt.plot(times,[ r[1] for r in cop_refs ])
plt.legend(['right', 'left', 'cop' ])

# ### OCP ###
'''
LIPM OPTIMAL CONTROL PROBLEM

decide: c[0..T], cdot[0..T], cop[0..T]

minimizing sum_T EPS || cddot_t ||**2 + || cop-cop_ref ||**2

respecting
   cdot_{t+1} = cdot_{t} + cddot{t}*DT
   c_{t+1} = c_{t} + cdot{t}*DT

with 
   cddot_t := w2 (c_t - cop_t)

'''
T = len(times)
WEIGHT_LEAST_MOTION = 5e-3 # best jerk
WITH_JERK = False

# xu = [ c cdot cop ]
# cddot = [ w2 0 -w2 ] xu
# (cop-cop_ref)**2 = xu [ 0,0,0;0,0,0;0,0,1 ] xu - 2*[0,0,cop_ref]*xu + cop_ref**2
# cddot**2 = xu [ w4 0 -w4 ; 0 0 0 ; -w4 0 w4 ] xu
# [cdot';c']  = [1,0,0;0,1,0]*xu'
# [ cdot+cddot*DT; c+cdot*DT+cddot*DT**2 ] = [ w2*DT,1,-w2*DT; 1+w2*DT**2,DT,-w2*DT**2 ]*xu

w2 = omega2
w4 = w2**2
Lcom = WEIGHT_LEAST_MOTION*np.array([ [w4,0,-w4], [0,0,0], [-w4,0,w4] ])
Lcop = np.array([ [0,0,0], [0,0,0], [0,0,1] ])
Fy = np.array([ [1,DT,0],[w2*DT,1,-w2*DT] ])
# Implicit integration in option Fy += np.array([ [w2*DT**2,0,-w2*DT**2].[0,0,0] ])
Hessian = [ np.zeros([T*3,T*3]) for _ in range(2) ]
gradient = [ np.zeros(T*3) for _ in range(2) ]
Jacobian = [ np.zeros([T*2,T*3]) for _ in range(2) ]
gap = [ np.zeros(T*2) for _ in range(2) ]
for k,cop in enumerate(cop_refs):
    # For each direction forward(x) / sideway(y)
    for d in range(2):
        # Hessian of OCP
        Hessian[d][3*k:3*k+3,3*k:3*k+3] = Lcom+Lcop
        # Minimum acceleration or minimum jerk?
        if WITH_JERK and k>0:
            Hessian[d][3*k-3:3*k,3*k:3*k+3] = - Lcom
            Hessian[d][3*k:3*k+3,3*k-3:3*k] = - Lcom
        # Gradient of OCP
        gradient[d][3*k+2] = -cop[d]

        Jacobian[d][2*k:2*k+2,3*k:3*k+3] = np.eye(3)[:2]
        if k>0: Jacobian[d][2*k:2*k+2,3*k-3:3*k] = -Fy
        else: gap[d][:2] = [com0[d],0]
 

# Solve OCP
qp = [ proxsuite.proxqp.dense.QP(T*3, T*2, 0, False) for d in range(2) ]    
for d in range(2):
    qp[d].settings.eps_abs = 1e-12
    qp[d].init(Hessian[d],gradient[d],Jacobian[d],gap[d])
    qp[d].solve()

for d in range(2):
    plt.figure(2+d)
    if d==0:
        plt.title('Results in forward direction (x)')
    else:
        plt.title('Results in sideway direction (y)')
              
    plt.subplot(511)
    plt.plot(times,qp[d].results.x[::3])
    plt.ylabel('com')
    plt.subplot(512)
    plt.plot(times,qp[d].results.x[1::3])
    plt.ylabel('vel com')
    plt.subplot(513)
    acc = omega2*(qp[d].results.x[::3]-qp[d].results.x[2::3])
    plt.plot(times,acc)
    plt.ylabel('acc com')
    plt.subplot(514)
    jerk = (acc[1:]-acc[:-1])/DT
    plt.plot(np.array(times[1:])-DT/2,jerk)
    plt.ylabel('j.com')
    plt.subplot(515)
    plt.plot(times,qp[d].results.x[2::3])
    plt.plot(times,[ c[d] for c in cop_refs ])
    plt.ylabel('cop')

plt.figure(4)
plt.plot(qp[0].results.x[::3], qp[1].results.x[::3])
plt.plot(qp[0].results.x[2::3], qp[1].results.x[2::3], 'r')
plt.plot([ c[0] for c in cop_refs ], [c[1] for c in cop_refs], 'r--')
plt.legend(['COM', 'COP', 'COP ref'])
plt.xlabel('forward (x)')
plt.ylabel('sideway (y)')


# ### CONTROL ###
# ### CONTROL ###
# ### CONTROL ###
for k,t in enumerate(times):

        # Pre-compute kinematics, jacobians, com etc
        pin.computeAllTerms(model,data,q,np.zeros(NV))

        # Right foot
        edot_right = right_refs[k]-data.oMf[feetIndexes['right']].translation[:2]
        edot_right = np.r_[edot_right,[0,0,0,0]] # Make it 6D
        J_right = pin.getFrameJacobian(model,data,feetIndexes['right'],pin.LOCAL)
        # Left foot
        edot_left = left_refs[k]-data.oMf[feetIndexes['left']].translation[:2]
        J_left = pin.getFrameJacobian(model,data,feetIndexes['left'],pin.LOCAL)
        edot_left = np.r_[edot_left,[0,0,0,0]] # Make it 6D
        # COM
        com_ref = np.r_[qp[0].results.x[k*3], qp[1].results.x[k*3] ]
        edot_com = com_ref - data.com[0][:2]
        J_com = pin.jacobianCenterOfMass(model,data,q)[:2]

        edot = 5*np.r_[edot_right,edot_left,edot_com]
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
