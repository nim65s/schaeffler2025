import unittest

import hppfcl
import numpy as np
import pinocchio as pin
from numpy import pi, sin
from pinocchio.utils import rotate


def Capsule(name, joint, radius, length, placement, color=[0.7, 0.7, 0.98, 1]):
    """Create a Pinocchio::FCL::Capsule to be added in the Geom-Model."""
    ### They should be capsules ... but hppfcl current version is buggy with Capsules...
    # hppgeom = hppfcl.Capsule(radius,length)
    hppgeom = hppfcl.Cylinder(radius, length)
    geom = pin.GeometryObject(name, joint, joint, placement, hppgeom)
    geom.meshColor = np.array(color)
    return geom


def Sphere(name, joint, radius, placement, color=[0.9, 0.9, 0.98, 1]):
    """Create a Pinocchio::FCL::Capsule to be added in the Geom-Model."""
    hppgeom = hppfcl.Sphere(radius)
    geom = pin.GeometryObject(name, joint, joint, placement, hppgeom)
    geom.meshColor = np.array(color)
    return geom


class RobotHand:
    """
    Define a class Robot with 7DOF (shoulder=3 + elbow=1 + wrist=3).
    The configuration is nq=7. The velocity is the same.
    The members of the class are:
    * viewer: a display encapsulating a gepetto viewer client to create 3D objects
    and place them.
    * model: the kinematic tree of the robot.
    * data: the temporary variables to be used by the kinematic algorithms.
    * visuals: the list of all the 'visual' 3D objects to render the robot, each
    element of the list being
    an object Visual (see above).

    CollisionPairs is a list of visual indexes.
    Reference to the collision pair is used in the collision test and jacobian
    of the collision
    (which are simply proxy method to methods of the visual class).
    """

    def __init__(self):
        self.viewer = None
        self.model = pin.Model()
        self.gmodel = pin.GeometryModel()

        self.createHand()
        self.addCollisionPairs()

        self.data = self.model.createData()
        self.gdata = pin.GeometryData(self.gmodel)
        self.gdata.collisionRequests.enable_contact = True

        self.q0 = np.zeros(self.model.nq)
        self.q0[-2] = np.pi / 3
        self.q0[2:-4] = np.pi / 6
        self.q0[11:] = np.pi / 4
        self.v0 = np.zeros(self.model.nv)
        self.model.referenceConfigurations["default"] = self.q0.copy()
        self.collisionPairs = []

        self.collision_model = self.gmodel
        self.collision_data = self.gmodel.createData()
        self.visual_model = self.gmodel
        self.visual_data = self.gmodel.createData()

    def addCollisionPairs(self):
        pairs = [
            ["finger12", "wrist"],
            ["finger12", "palm_left"],
            ["finger12", "palm_right"],
            ["finger12", "palm_front"],
            ["finger13", "wrist"],
            ["finger13", "palm_left"],
            ["finger13", "palm_right"],
            ["finger13", "palm_front"],
            ["finger13", "palm2"],
            ["finger22", "wrist"],
            ["finger22", "palm_left"],
            ["finger22", "palm_right"],
            ["finger22", "palm_front"],
            ["finger23", "wrist"],
            ["finger23", "palm_left"],
            ["finger23", "palm_right"],
            ["finger23", "palm_front"],
            ["finger23", "palm2"],
            ["finger32", "wrist"],
            ["finger32", "palm_left"],
            ["finger32", "palm_right"],
            ["finger32", "palm_front"],
            ["finger33", "wrist"],
            ["finger33", "palm_left"],
            ["finger33", "palm_right"],
            ["finger33", "palm_front"],
            ["finger33", "palm2"],
            ["thumb1", "wrist"],
            ["thumb1", "palm_left"],
            ["thumb1", "palm_front"],
            ["thumb1", "palm2"],
            ["thumb1", "finger11"],
            ["thumb1", "finger12"],
            ["thumb1", "finger13"],
            ["thumb2", "wrist"],
            ["thumb2", "palm_left"],
            ["thumb2", "palm_right"],
            ["thumb2", "palm_front"],
            ["thumb2", "palm2"],
            ["thumb2", "finger11"],
            ["thumb2", "finger12"],
            ["thumb2", "finger13"],
            ["thumb2", "finger21"],
            ["thumb2", "finger22"],
            ["thumb2", "finger23"],
        ]
        for n1, n2 in pairs:
            self.gmodel.addCollisionPair(
                pin.CollisionPair(
                    self.gmodel.getGeometryId("world/" + n1),
                    self.gmodel.getGeometryId("world/" + n2),
                )
            )

    def addCapsule(self, name, joint, placement, radius, length, color=[1, 1, 0.78, 1]):
        caps = Capsule(name, joint, radius * 0.99, length, placement)
        caps.meshColor = np.array([1.0] * 4)
        self.gmodel.addGeometryObject(caps)

    def createHand(self, rootId=0, jointPlacement=None):
        [_red, _green, _blue, _transparency] = [1, 1, 0.78, 1.0]

        jointId = rootId

        cm = 1e-2

        def trans(x, y, z):
            return pin.SE3(np.eye(3), np.array([x, y, z]))

        def inertia(m, c):
            return pin.Inertia(m, np.array(c, np.double), np.eye(3) * m**2)

        name = "wrist"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        # jointPlacement= jointPlacement if jointPlacement!=None else pin.SE3.Identity()
        jointPlacement = (
            jointPlacement
            if jointPlacement is not None
            else pin.SE3(pin.utils.rotate("y", np.pi), np.zeros(3))
        )
        jointId = self.model.addJoint(
            jointId, pin.JointModelRY(), jointPlacement, jointName
        )
        self.model.appendBodyToJoint(jointId, inertia(3, [0, 0, 0]), pin.SE3.Identity())

        ## Hand dimsensions: length, width, height(=depth), finger-length
        L = 3 * cm
        W = 5 * cm
        H = 1 * cm
        FL = 4 * cm
        self.gmodel.addGeometryObject(
            Sphere(
                "world/wrist",
                jointId,
                0.02,
                pin.SE3(rotate("x", pi / 2), np.array([0, 0, 0])),
            )
        )

        self.gmodel.addGeometryObject(
            Capsule(
                "world/palm_right",
                jointId,
                H,
                L,
                pin.SE3(
                    rotate("z", -0.3) @ rotate("y", pi / 2),
                    np.array([L / 2, -W / 2.6, 0]),
                ),
            )
        )
        self.gmodel.addGeometryObject(
            Capsule(
                "world/palm_left",
                jointId,
                H,
                L,
                pin.SE3(rotate("y", pi / 2), np.array([L / 2, W / 2, 0])),
            )
        )
        self.gmodel.addGeometryObject(
            Capsule(
                "world/palm_front",
                jointId,
                H,
                W,
                pin.SE3(rotate("x", pi / 2), np.array([L, 0, 0])),
            )
        )
        self.gmodel.addGeometryObject(
            Sphere(
                "world/palm_frontleft",
                jointId,
                H,
                pin.SE3(rotate("x", pi / 2), np.array([L, W / 2, 0])),
                color=[0.7, 0.7, 0.98, 1],
            )
        )
        self.gmodel.addGeometryObject(
            Sphere(
                "world/palm_frontright",
                jointId,
                H,
                pin.SE3(rotate("x", pi / 2), np.array([L, -W / 2, 0])),
                color=[0.7, 0.7, 0.98, 1],
            )
        )

        name = "palm"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(np.eye(3), np.array([5 * cm, 0, 0]))
        jointId = self.model.addJoint(
            jointId, pin.JointModelRY(), jointPlacement, jointName
        )
        self.model.appendBodyToJoint(jointId, inertia(2, [0, 0, 0]), pin.SE3.Identity())
        self.gmodel.addGeometryObject(
            Capsule(
                "world/palm2",
                jointId,
                1 * cm,
                W,
                pin.SE3(rotate("x", pi / 2), np.zeros(3)),
            )
        )
        palmIdx = jointId

        name = "finger11"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(np.eye(3), np.array([2 * cm, W / 2, 0]))
        jointId = self.model.addJoint(
            palmIdx, pin.JointModelRY(), jointPlacement, jointName
        )
        self.model.appendBodyToJoint(
            jointId, inertia(0.5, [0, 0, 0]), pin.SE3.Identity()
        )
        self.gmodel.addGeometryObject(
            Capsule(
                "world/finger11",
                jointId,
                H,
                FL - 0 * H,
                pin.SE3(rotate("y", pi / 2), np.array([FL / 2 - H, 0, 0])),
            )
        )

        name = "finger12"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(np.eye(3), np.array([FL, 0, 0]))
        jointId = self.model.addJoint(
            jointId, pin.JointModelRY(), jointPlacement, jointName
        )
        self.model.appendBodyToJoint(
            jointId, inertia(0.5, [0, 0, 0]), pin.SE3.Identity()
        )
        self.gmodel.addGeometryObject(
            Capsule(
                "world/finger12",
                jointId,
                H,
                FL - 0 * H,
                pin.SE3(rotate("y", pi / 2), np.array([FL / 2 - H, 0, 0])),
            )
        )

        name = "finger13"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(np.eye(3), np.array([FL - H, 0, 0]))
        jointId = self.model.addJoint(
            jointId, pin.JointModelRY(), jointPlacement, jointName
        )
        self.model.appendBodyToJoint(
            jointId, inertia(0.3, [0, 0, 0]), pin.SE3.Identity()
        )
        self.gmodel.addGeometryObject(
            Sphere("world/finger13", jointId, H, trans(H, 0, 0))
        )

        name = "finger21"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(np.eye(3), np.array([2 * cm, 0, 0]))
        jointId = self.model.addJoint(
            palmIdx, pin.JointModelRY(), jointPlacement, jointName
        )
        self.model.appendBodyToJoint(
            jointId, inertia(0.5, [0, 0, 0]), pin.SE3.Identity()
        )
        self.gmodel.addGeometryObject(
            Capsule(
                "world/finger21",
                jointId,
                H,
                FL - 0 * H,
                pin.SE3(rotate("y", pi / 2), np.array([FL / 2 - H, 0, 0])),
            )
        )

        name = "finger22"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(np.eye(3), np.array([FL, 0, 0]))
        jointId = self.model.addJoint(
            jointId, pin.JointModelRY(), jointPlacement, jointName
        )
        self.model.appendBodyToJoint(
            jointId, inertia(0.5, [0, 0, 0]), pin.SE3.Identity()
        )
        self.gmodel.addGeometryObject(
            Capsule(
                "world/finger22",
                jointId,
                H,
                FL - 0 * H,
                pin.SE3(rotate("y", pi / 2), np.array([FL / 2 - H, 0, 0])),
            )
        )

        name = "finger23"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(np.eye(3), np.array([FL - H, 0, 0]))
        jointId = self.model.addJoint(
            jointId, pin.JointModelRY(), jointPlacement, jointName
        )
        self.model.appendBodyToJoint(
            jointId, inertia(0.3, [0, 0, 0]), pin.SE3.Identity()
        )
        self.gmodel.addGeometryObject(
            Sphere("world/finger23", jointId, H, trans(H, 0, 0))
        )

        name = "finger31"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(np.eye(3), np.array([2 * cm, -W / 2, 0]))
        jointId = self.model.addJoint(
            palmIdx, pin.JointModelRY(), jointPlacement, jointName
        )
        self.model.appendBodyToJoint(
            jointId, inertia(0.5, [0, 0, 0]), pin.SE3.Identity()
        )
        self.gmodel.addGeometryObject(
            Capsule(
                "world/finger31",
                jointId,
                H,
                FL - 0 * H,
                pin.SE3(rotate("y", pi / 2), np.array([FL / 2 - H, 0, 0])),
            )
        )

        name = "finger32"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(np.eye(3), np.array([FL, 0, 0]))
        jointId = self.model.addJoint(
            jointId, pin.JointModelRY(), jointPlacement, jointName
        )
        self.model.appendBodyToJoint(
            jointId, inertia(0.5, [0, 0, 0]), pin.SE3.Identity()
        )
        self.gmodel.addGeometryObject(
            Capsule(
                "world/finger32",
                jointId,
                H,
                FL - 0 * H,
                pin.SE3(rotate("y", pi / 2), np.array([FL / 2 - H, 0, 0])),
            )
        )

        name = "finger33"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(np.eye(3), np.array([FL - H, 0, 0]))
        jointId = self.model.addJoint(
            jointId, pin.JointModelRY(), jointPlacement, jointName
        )
        self.model.appendBodyToJoint(
            jointId, inertia(0.3, [0, 0, 0]), pin.SE3.Identity()
        )
        self.gmodel.addGeometryObject(
            Sphere("world/finger33", jointId, H, trans(H, 0, 0))
        )

        name = "thumb1"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(
            rotate("z", -1.5), np.array([1 * cm, -W / 2 - H * 1.3, 0])
        )
        jointId = self.model.addJoint(1, pin.JointModelRY(), jointPlacement, jointName)
        self.model.appendBodyToJoint(
            jointId, inertia(0.5, [0, 0, 0]), pin.SE3.Identity()
        )
        # self.gmodel.addGeometryObject( Capsule('world/thumb1_mid',jointId,H,2*cm,
        #   pin.SE3(rotate('z',pi/3)@rotate('x',pi/2),np.array([1*cm,-1*cm,0])) ))

        name = "thumb1a"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(np.eye(3), np.zeros(3))
        jointId = self.model.addJoint(
            jointId, pin.JointModelRX(), jointPlacement, jointName
        )
        # self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject(
            Capsule(
                "world/thumb1",
                jointId,
                H,
                2 * cm,
                pin.SE3(
                    rotate("z", pi / 3) @ rotate("x", pi / 2),
                    np.array([0.3 * cm, -1.0 * cm, 0]),
                ),
            )
        )

        name = "thumb2"
        jointName, _bodyName = [name + "_joint", name + "_body"]
        jointPlacement = pin.SE3(
            rotate("z", pi / 3) @ rotate("x", pi), np.array([3.4 * cm, -1.99 * cm, 0])
        )
        jointId = self.model.addJoint(
            jointId, pin.JointModelRZ(), jointPlacement, jointName
        )
        self.model.appendBodyToJoint(
            jointId, inertia(0.4, [0, 0, 0]), pin.SE3.Identity()
        )
        self.gmodel.addGeometryObject(
            Capsule(
                "world/thumb2",
                jointId,
                H,
                FL - 0 * H,
                pin.SE3(
                    rotate("x", 2 * pi / 3),
                    np.array([-0.007 * cm, 0.008 * cm, -0.5 * cm]),
                ),
            )
        )

        self.model.lowerPositionLimit = np.ones(self.model.nq) * -2
        self.model.upperPositionLimit = np.ones(self.model.nq) * 2


### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class RobotHandTest(unittest.TestCase):
    def test_logs(self):
        print(self.__class__.__name__)
        robot = RobotHand()

        self.assertTrue(robot.model.nq == 14)  # Check NDOF
        self.assertTrue(robot.model.nv == 14)  # Check N tangent space NV


if __name__ == "__main__":
    RobotHandTest().test_logs()

### EXAMPLE ################################################################
if __name__ == "__main__":
    import time

    from schaeffler2025.meshcat_viewer_wrapper import MeshcatVisualizer

    robot = RobotHand()

    viz = MeshcatVisualizer(robot)
    viz.display(robot.q0)

    q = robot.q0.copy()
    T = 10
    for t in range(T):
        x = sin(t / 30)
        q[12:13] = x + 0.5
        viz.display(q)
        time.sleep(5e-2)
