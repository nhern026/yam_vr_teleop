"""Tests for the local i2rt shutdown patch. No real CAN interfaces are opened."""
import unittest
from unittest.mock import Mock, patch, call
import numpy as np
from i2rt.motor_drivers import dm_driver
from i2rt.robots import get_robot

class DriverStartup(unittest.TestCase):
    def test_partial_motor_enable_closes_every_motor(self):
        interface=Mock()
        interface.motor_on.side_effect=[Mock(),RuntimeError("second motor missing")]
        with patch.object(dm_driver,"run_startup_checks"), patch.object(dm_driver,"DMSingleMotorCanInterface",return_value=interface):
            with self.assertRaises(RuntimeError):
                dm_driver.DMChainCanInterface([[1,"DM4340"],[2,"DM4340"]],[0.,0.],[1,1],"can0",start_thread=False)
        self.assertEqual(interface.motor_off.call_args_list,[call(1),call(2)])
        interface.close.assert_called_once()

    def test_robot_construction_failure_closes_chain(self):
        chain=Mock()
        chain.read_states.return_value=[Mock(pos=0.) for _ in range(7)]
        chain.motor_offset=np.zeros(7)
        with patch.object(get_robot,"DMChainCanInterface",return_value=chain),patch.object(get_robot,"MotorChainRobot",side_effect=RuntimeError("calibration failed")):
            with self.assertRaises(RuntimeError):get_robot.get_yam_robot(channel="can0")
        chain.close.assert_called_once()

if __name__=="__main__":unittest.main(verbosity=2)
