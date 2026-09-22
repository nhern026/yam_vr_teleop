"""Build the pinned i2rt wheel with Raspberry-Pi-only GPIO made optional.

i2rt 1.1.2's platform marker installs rpi-lgpio on every aarch64 Linux host,
including Jetsons. lgpio's generated wrapper fails on Python 3.13. This changes
only wheel dependency metadata, never robot code; the original and adjusted
wheels and the exact modification are retained for inspection.
"""
import base64
import csv
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import zipfile

revision = '120c3c81400171174604e503943f8d1ebc891058'
root = Path(__file__).resolve().parents[1] / '.vendor-wheels'
original, adjusted = root / 'original', root / 'jetson'
original.mkdir(parents=True, exist_ok=True)
adjusted.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--wheel-dir', str(original),
                f'i2rt @ git+https://github.com/i2rt-robotics/i2rt@{revision}'], check=True)
source = next(original.glob('i2rt-1.1.2-*.whl'))
with zipfile.ZipFile(source) as archive:
    files = {name: archive.read(name) for name in archive.namelist()}
metadata = next(name for name in files if name.endswith('.dist-info/METADATA'))
record = next(name for name in files if name.endswith('.dist-info/RECORD'))
lines = files[metadata].decode().splitlines()
removed = [line for line in lines if line.startswith('Requires-Dist: rpi-lgpio')]
if len(removed) != 1:
    raise RuntimeError('pinned wheel metadata changed; inspect before adapting')
lines = [line for line in lines if line not in removed]
# Retain a documented optional dependency for actual Raspberry Pi deployments.
position = lines.index('')
lines[position:position] = ['Provides-Extra: raspberry-pi', 'Requires-Dist: rpi-lgpio>=0.6; extra == "raspberry-pi"']
files[metadata] = ('\n'.join(lines) + '\n').encode()

# i2rt 1.1.2 closes SocketCAN before its background control worker exits. The
# worker can then call select() on fd -1 during normal Ctrl-C shutdown. Retain
# the worker and join it before closing the bus.
driver = next(name for name in files if name.endswith('i2rt/motor_drivers/dm_driver.py'))
source_text = files[driver].decode()
old_start = '''        thread = threading.Thread(target=self._set_torques_and_update_state)\n        thread.start()\n'''
new_start = '''        self._control_thread = threading.Thread(target=self._set_torques_and_update_state)\n        self._control_thread.start()\n'''
old_close = '''    def close(self) -> None:\n        self.running = False\n        self.motor_interface.close()\n'''
new_close = '''    def close(self) -> None:\n        if getattr(self, "_interface_closed", False):\n            return\n        self.running = False\n        thread = getattr(self, "_control_thread", None)\n        if thread is not None:\n            if thread is threading.current_thread():\n                raise RuntimeError("CAN worker cannot close its own interface")\n            thread.join(timeout=5.0)\n            if thread.is_alive():\n                raise RuntimeError("CAN worker did not stop; refusing to close an active socket")\n        self.motor_interface.close()\n        self._interface_closed = True\n'''
old_initial_command = '''        starting_command = []\n        for motor_state in self.state:\n            starting_command.append(MotorCmd(torque=motor_state.torque))\n'''
new_initial_command = '''        starting_command = []\n        for motor_state, (motor_id, _motor_type) in zip(self.state, self.motor_list):\n            # Do not replay measured feedback torque open-loop on the gripper.\n            # Preserve existing arm bring-up behavior; motor 7 starts neutral.\n            torque = 0.0 if motor_id == 0x07 else motor_state.torque\n            starting_command.append(MotorCmd(torque=torque))\n'''
if (source_text.count(old_start) != 1 or source_text.count(old_close) != 1
        or source_text.count(old_initial_command) != 1):
    raise RuntimeError('pinned i2rt driver code changed; inspect before adapting')
files[driver] = (source_text.replace(old_start, new_start)
                 .replace(old_close, new_close)
                 .replace(old_initial_command, new_initial_command).encode())
output = io.StringIO(newline='')
writer = csv.writer(output)
for name, data in files.items():
    if name != record:
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        writer.writerow([name, 'sha256=' + digest, len(data)])
writer.writerow([record, '', ''])
files[record] = output.getvalue().encode()
with zipfile.ZipFile(adjusted / source.name, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, data in files.items():
        archive.writestr(name, data)
(root / 'METADATA_CHANGE.txt').write_text(
    f'i2rt source commit: {revision}\nMade optional on Jetson:\n'
    + '\n'.join(removed)
    + '\nPatched DMChainCanInterface.close to join its control worker before closing SocketCAN.\n'
    + 'Patched CAN bring-up so gripper motor 7 starts neutral instead of replaying feedback torque.\n'
)
print(adjusted / source.name)
