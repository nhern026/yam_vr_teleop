"""Real dual-arm IK/simulation + recorder, optionally with real ZED cameras.

Never opens CAN. No Quest required. Output is marked synthetic in metadata.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.config import load_deployment_config
from deployment.quest_teleop import ControllerSample, TeleopSession, IDLE, ENGAGED
from deployment.recording import EpisodeRecorder
from deployment.zed_capture import CaptureEvent, FramePacket, ZedPairCapture
from deployment.export_dataset import export

class SimulatedQuest:
    def sample(self, hand):
        return ControllerSample(hand, np.array([0., 0., -.3]), np.array([1., 0., 0., 0.]),
                                .3 if hand == 'left' else .7, 1., True, False, False,
                                (0., 0.), time.monotonic())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zed', action='store_true')
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load((Path(__file__).resolve().parents[1] / 'deployment' / 'recording.yaml').read_text())
    camera_sides = ('left', 'right', 'overhead')
    configs = []
    for name in ('config.yaml', 'config_left.yaml'):
        cfg = load_deployment_config(Path(__file__).resolve().parents[1] / 'deployment' / name)
        cfg['robot'].update(backend='sim', gripper_type='linear_4310')
        cfg['teleop'].update(gripper=True, print_hz=0)
        configs.append(cfg)
    session = TeleopSession(configs[0], SimulatedQuest(), backend='sim', extra=[(configs[1], None)])
    recorder = EpisodeRecorder(args.output, task='synthetic smoke test', min_free_gb=.2,
                               camera_sides=camera_sides,
                               metadata={'synthetic_robot': True, 'synthetic_camera': not args.zed,
                                         'capture': settings})
    session.recorder = recorder
    capture = None
    try:
        if args.zed:
            capture = ZedPairCapture(callback=recorder.add_capture, error_callback=recorder.fail,
                                     **settings['cameras'])
            capture.start()
        session.start()
        deadline = time.monotonic() + 10
        while session.status()['mode'] not in (IDLE, ENGAGED):
            if not session.alive() or time.monotonic() > deadline:
                raise RuntimeError(session.status())
            time.sleep(.01)
        if args.zed:
            time.sleep(args.seconds)
        else:
            start = time.monotonic()
            for i in range(int(args.seconds * 30)):
                time.sleep(max(0, start + i/30 - time.monotonic()))
                stamp, epoch = time.monotonic_ns(), time.time_ns()
                frames = {side: FramePacket(side, serial, i, epoch, epoch, stamp, stamp,
                          np.full((600, 960, 3), i % 255, dtype=np.uint8))
                          for side, serial in zip(camera_sides, (301058360, 306353224, 41925345))}
                recorder.add_capture(CaptureEvent(i, 'pair', frames['left'], frames['right'],
                                                  overhead=frames['overhead']))
    except Exception as exc:
        recorder.fail(str(exc))
        raise
    finally:
        recorder.metadata['control_status'] = session.status()
        session.close()
        recorder.finish_controls()
        if capture:
            capture.close()
            recorder.metadata['camera_status'] = capture.status()
        path = recorder.close()
        print(path)
    outputs = export(path, args.output / 'exports')
    print(json.dumps({'episode': str(path), 'trajectories': list(map(str, outputs)),
                      'counts': recorder.metadata['counts'], 'control_hz': recorder.metadata['control_status']['hz']}, indent=2))
    if recorder.metadata['counts']['valid'] < args.seconds * 30 * .8:
        raise RuntimeError('less than 80% of expected training rows were valid')

if __name__ == '__main__':
    main()
