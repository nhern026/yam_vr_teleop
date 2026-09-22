import importlib
import importlib.metadata
for name in ('numpy', 'cv2', 'h5py', 'yaml', 'mujoco', 'mink', 'i2rt', 'pyzed.sl'):
    module = importlib.import_module(name)
    print(name, getattr(module, '__version__', 'import OK'))
import pyzed.sl as sl
assert hasattr(sl, 'CameraOne'), 'PyZED does not provide CameraOne'
print('SDK:', sl.Camera.get_sdk_version())
