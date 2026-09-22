"""Offline regression tests. Factories are simulated or mocked; never opens CAN."""
import copy
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from deployment import quest_teleop as qt
from deployment.config import load_deployment_config, validate_config
from deployment.robot import ArmState, I2rtArm
from deployment.preflight import Report, check_joint_box

ROOT = Path(__file__).resolve().parents[1]

def config(hand="right"):
    c = load_deployment_config(ROOT / "deployment" / ("config.yaml" if hand == "right" else "config_left.yaml"))
    c["robot"]["backend"] = "sim"
    c["teleop"]["print_hz"] = 0
    return c

def sample(hand="right", **kw):
    args = dict(hand=hand, position=np.array([0., 0., -.3]), quat=np.array([1.,0.,0.,0.]),
                trigger=0., grip=0., clutch=False, home=False, stop=False,
                joystick=(0.,0.), received_s=time.monotonic())
    args.update(kw)
    return qt.ControllerSample(**args)

class Reader:
    def sample(self, hand): return sample(hand)
    def alive(self): return True

class Regressions(unittest.TestCase):
    def session(self, dual=False):
        s = qt.TeleopSession(config(), Reader(), extra=[(config("left"), None)] if dual else None)
        self.addCleanup(s.close)
        return s

    def test_startup_holds_measured_pose(self):
        s = self.session(True)
        self.assertEqual(s.status()["mode"], qt.IDLE)
        for c in s.channels:
            np.testing.assert_allclose(c.hold, c.read_state().joint_position)
        s.start(); time.sleep(.08)
        self.assertTrue(s.alive(), s.status())
        self.assertEqual(s.status()["mode"], qt.IDLE)

    def test_model_includes_actual_gripper(self):
        a = qt.TeleopIK(config()["safety"], {"gripper_type":"no_gripper"})
        b = qt.TeleopIK(config()["safety"], {"gripper_type":"linear_4310"})
        state = ArmState(np.array([0.,.9,1.2,0.,.6,0.]), np.zeros(6))
        self.assertGreater(np.linalg.norm(a.ee_pose(state)[0] - b.ee_pose(state)[0]), .01)
        goal, rot = b.ee_pose(state); goal = goal + [.01,0,0]
        for _ in range(100):
            state = ArmState(b.joint_target_from_pose(state, goal, rot, .01), np.zeros(6))
        self.assertLess(np.linalg.norm(b.ee_pose(state)[0]-goal), .001)

    def test_stop_edges_are_per_hand(self):
        s=self.session(True)
        s._apply_buttons(sample(stop=True)); s._apply_buttons(sample("left"))
        s._note="unchanged"
        s._apply_buttons(sample(stop=True))
        self.assertEqual(s._note,"unchanged")
        self.assertEqual(s.status()["mode"],qt.PARKED)

    def test_stale_buttons_cannot_pause_or_resume(self):
        s=self.session()
        s._apply_buttons(sample(stop=True,received_s=time.monotonic()-1))
        self.assertEqual(s.status()["mode"],qt.IDLE)

    def test_clutch_must_release_after_start_and_loss(self):
        s=self.session(); c=s.channels[0]; state=c.read_state()
        held=sample(clutch=True,trigger=1.)
        _, engaged, _=c.drive(state,held,time.monotonic(),.15)
        self.assertFalse(engaged)
        c.drive(state,sample(),time.monotonic(),.15)
        _,engaged,_=c.drive(state,sample(clutch=True,trigger=1.),time.monotonic(),.15)
        self.assertTrue(engaged)
        c.drive(state,None,time.monotonic(),.15)
        _,engaged,_=c.drive(state,sample(clutch=True,trigger=1.),time.monotonic(),.15)
        self.assertFalse(engaged)

    def test_one_missing_controller_pauses_both(self):
        s=self.session(True)
        states=[c.read_state() for c in s.channels]
        s._step(states,{"right":sample()},time.monotonic())
        self.assertEqual(s.status()["mode"],qt.PARKED)
        self.assertTrue(all(not c._engaged for c in s.channels))

    def test_runaway_trip_parks_both_arms(self):
        s=self.session(True); a,b=s.channels
        state=a.read_state(); state.joint_velocity=np.ones(6)*99
        targets=s._step([state,b.read_state()],{"right":sample(),"left":sample("left")},time.monotonic())
        self.assertIsNotNone(s.status()["fault"])
        self.assertEqual(s.status()["mode"],qt.RETURNING)
        self.assertEqual(len(targets),2)

    def test_joint_command_rate_bound(self):
        s=self.session(); c=s.channels[0]; prev=c._last_command.copy()
        c.command(prev+.5)
        self.assertLessEqual(np.max(np.abs(c._last_command-prev)), .01500001)
        np.testing.assert_allclose(c.hold,c._last_command)

    def test_duplicate_bus_rejected_before_opening(self):
        a,b=config(),config("left");b["robot"]["channel"]=a["robot"]["channel"]
        with patch.object(qt,"ArmChannel") as create:
            with self.assertRaises(ValueError): qt.TeleopSession(a,Reader(),backend="i2rt",extra=[(b,None)])
            create.assert_not_called()

    def test_partial_startup_closes_first_arm(self):
        first=Mock()
        with patch.object(qt,"ArmChannel",side_effect=[first,RuntimeError("second failed")]):
            with self.assertRaises(RuntimeError): qt.TeleopSession(config(),Reader(),extra=[(config("left"),None)])
        first.close.assert_called_once()

    def test_shutdown_attempts_every_arm(self):
        s=self.session(True); a,b=s.channels
        with patch.object(a,"close",side_effect=RuntimeError("failed")),patch.object(b,"close") as close:
            with self.assertRaises(RuntimeError):s.close()
            close.assert_called_once()

    def test_loop_exception_is_reported(self):
        s=self.session()
        with patch.object(s.channels[0],"read_state",side_effect=RuntimeError("lost feedback")):
            s.start(); s._thread.join(timeout=2)
        self.assertFalse(s.alive())
        self.assertIn("lost feedback",s.status()["fault"])

    def test_invalid_config(self):
        for section,key,value in [("teleop","position_scale",float("nan")),("teleop","control_hz",0),
                                  ("safety","joint_position_min_rad",[float("nan")]*6)]:
            c=config();c[section][key]=value
            with self.assertRaises(ValueError):validate_config(c)

    def test_rotation_and_button_parser(self):
        def payload(m,buttons="rightTrig 0.2,rightGrip 0.1"):
            return "r:"+" ".join(map(str,m.reshape(-1)))+"&"+buttons
        self.assertIn("right",qt.parse_log_payload(payload(np.eye(4)),0))
        for m in [np.diag([2,.5,1,1]),np.diag([1,1,1,0]),np.full((4,4),float("nan"))]:
            self.assertEqual(qt.parse_log_payload(payload(m),0),{})
        self.assertEqual(qt.parse_log_payload(payload(np.eye(4),"rightTrig 0.2 0.3"),0),{})
        self.assertEqual(qt.parse_log_payload(payload(np.eye(4),"rightJS 0.2"),0),{})

    def test_serial_selection(self):
        with patch.object(qt.shutil,"which",return_value="adb"):
            reader=qt.QuestReader()
        with patch.object(reader,"_run",return_value="List of devices attached\na\tdevice\nb\tdevice\n"):
            with self.assertRaises(RuntimeError):reader.device_serial()
            reader._serial="b";self.assertEqual(reader.device_serial(),"b")
            reader._serial="c"
            with self.assertRaises(RuntimeError):reader.device_serial()

    def test_joint_box_checks_mechanical_limits(self):
        c=config();ik=qt.TeleopIK(c["safety"],c["robot"])
        c["safety"]["joint_position_min_rad"][0] = -.5
        report=Report();check_joint_box(report,"arm",c,ik)
        self.assertEqual(report.rows[0][0],"PASS")

    def test_hardware_factory_not_retried(self):
        factory=Mock(side_effect=TypeError("constructor failed"))
        lock=Mock()
        with patch("deployment.robot._claim_bus",return_value=lock),patch("deployment.robot._import_get_yam_robot",return_value=factory):
            with self.assertRaises(TypeError):I2rtArm({"gripper_type":"linear_4310"})
        factory.assert_called_once();lock.close.assert_called_once()

    def test_hardware_feedback_failure_is_not_hidden(self):
        robot=Mock();robot.get_joint_pos.return_value=np.zeros(7)
        robot.motor_chain.last_feedback_monotonic=time.monotonic();robot.motor_chain.running=True;robot._server_thread.is_alive.return_value=True
        robot.get_observations.side_effect=RuntimeError("feedback unavailable")
        with patch("deployment.robot._claim_bus",return_value=Mock()),patch("deployment.robot._import_get_yam_robot",return_value=Mock(return_value=robot)):
            arm=I2rtArm({"gripper_type":"linear_4310"})
            with self.assertRaises(RuntimeError):arm.read_state()
            with self.assertRaises(ValueError):arm.command_gripper(float("nan"))
            arm.close()

if __name__=="__main__":unittest.main(verbosity=2)
