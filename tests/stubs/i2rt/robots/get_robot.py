import numpy as np
class FakeYam:
  def __init__(self, n): self.q = np.linspace(0.1, 0.6, n); self.sent = []; self.closed = False
  def get_observations(self): return {"joint_vel": np.zeros(6)}
  def get_joint_pos(self): return self.q.copy()
  def command_joint_pos(self, q): self.sent.append(np.asarray(q).copy())
  def close(self): self.closed = True
LAST = {}
def get_yam_robot(channel, arm_type, gripper_type, zero_gravity_mode=True):
  n = 6 if gripper_type.name == "NO_GRIPPER" else 7
  LAST["robot"] = FakeYam(n); LAST["channel"] = channel
  return LAST["robot"]
