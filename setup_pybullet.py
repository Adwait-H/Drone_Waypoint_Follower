import pybullet as p
import pybullet_data

def setup_pybullet(gui: bool = True, dt: float = 1/240):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.80665)
    p.setTimeStep(dt)
    p.loadURDF("plane.urdf")
    return cid
