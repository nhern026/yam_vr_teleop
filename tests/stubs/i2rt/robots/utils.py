import enum
class ArmType(enum.Enum): YAM = 1
class GripperType(enum.Enum): NO_GRIPPER = 0; LINEAR_4310 = 1
def combine_arm_and_gripper_xml(*a): raise NotImplementedError
