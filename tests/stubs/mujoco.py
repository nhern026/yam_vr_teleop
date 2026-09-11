# Test stub: only the mujoco calls quest_teleop's pure helpers use.
import numpy as np
class mjtObj: mjOBJ_JOINT = 0; mjOBJ_SITE = 1
def mju_mat2Quat(q, m):
  m = np.asarray(m).reshape(3, 3); t = np.trace(m)
  if t > 0:
    s = 2 * np.sqrt(t + 1); q[:] = [0.25 * s, (m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s]
  else:
    i = int(np.argmax(np.diag(m)))
    if i == 0:
      s = 2*np.sqrt(1+m[0,0]-m[1,1]-m[2,2]); q[:] = [(m[2,1]-m[1,2])/s, 0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s]
    elif i == 1:
      s = 2*np.sqrt(1+m[1,1]-m[0,0]-m[2,2]); q[:] = [(m[0,2]-m[2,0])/s, (m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s]
    else:
      s = 2*np.sqrt(1+m[2,2]-m[0,0]-m[1,1]); q[:] = [(m[1,0]-m[0,1])/s, (m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s]
