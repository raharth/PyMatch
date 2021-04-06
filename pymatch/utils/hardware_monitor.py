"""
Inspired by the monitoring solution from https://stackoverflow.com/questions/3262603/accessing-cpu-temperature-in-python
To run this you need to download the OpenHardwareMonitor from https://openhardwaremonitor.org/downloads/ and reference
the `OpenHardwareMonitorLib.dll` using clr
"""
import clr  # package pythonnet, not clr
import time
import pandas as pd
import threading
import json
import warnings

with open("pymatch_config", "r") as f:
    config = json.load(f)

if config.get("hardware_monitor", None) is not None:
    clr.AddReference(f"{config['hardware_monitor']['dll_path']}")
    from OpenHardwareMonitor import Hardware
else:
    warnings.warn('No .dll for Hardware Monitor provided. Therefore, monitoring the hardware will raise errors.\n'
                  'To enable monitoring you should provide the path to the .dll in the pymatch_config file,'
                  'which has a json structure with the arguments `hardware_monitor`-> `dll_path` ')


class HardwareMonitor:
    def __init__(self, path, sleep):
        """
        Small module to monitor the local hardware.

        Args:
            path:   path to dump the measurements to
            sleep:  time between measurements
        """
        handle = Hardware.Computer()
        handle.MainboardEnabled = True
        handle.CPUEnabled = True
        handle.RAMEnabled = True
        handle.GPUEnabled = True
        handle.HDDEnabled = True
        handle.Open()
        self.handle = handle

        self.hwtypes = ['Mainboard', 'SuperIO', 'CPU', 'RAM', 'GpuNvidia', 'GpuAti', 'TBalancer', 'Heatmaster', 'HDD']
        self.sensortypes = ['Voltage', 'Clock', 'Temperature', 'Load', 'Fan', 'Flow', 'Control', 'Level',
                            'Factor', 'Power', 'Data', 'SmallData']
        self.measurements = None
        self.path = path
        self.sleep = sleep
        self.terminate_flag = False

    def fetch_stats(self):
        """
        Collects also measurements from the sensors.

        Returns:
            dictionary of measurements

        """
        measurements = {'time': time.strftime("%m/%d/%Y, %H:%M:%S")}
        for i in self.handle.Hardware:
            i.Update()
            for sensor in i.Sensors:
                measurements = self.parse_sensor(sensor, measurements)
            for j in i.SubHardware:
                j.Update()
                for subsensor in j.Sensors:
                    measurements = self.parse_sensor(subsensor, measurements)
        return measurements

    def parse_sensor(self, sensor, measurements):
        """
        Get info from a single sensor.

        Args:
            sensor:         sensor to collect data measurements from
            measurements:   dictionary containing all already collected measurements from other sensors

        Returns:
            dictionary, containing measurements from sensors expanded by the new sensor
        """
        if sensor.Value is not None:
            key = f'{self.hwtypes[sensor.Hardware.HardwareType]}({sensor.Hardware.Name}-{sensor.Index}): ' \
                  f'{self.sensortypes[sensor.SensorType]}({sensor.Name})'
            measurements[key] = sensor.Value
        return measurements

    def monitor(self):
        """
        Starts monitoring the hardware.
        Automatically dumps the recordings to the file system.

        Returns:

        """
        self.measurements = pd.DataFrame(data=self.fetch_stats(), index=[0])
        while not self.terminate_flag:
            self.measurements = pd.concat([self.measurements,
                                           pd.DataFrame(data=self.fetch_stats(), index=[0])], ignore_index=True)
            self.measurements.to_csv(self.path)
            time.sleep(self.sleep)
        print('Monitor stopped measuring')

    def terminate(self):
        """
        Terminates the monitoring.

        Returns:

        """
        self.terminate_flag = True


if __name__ == '__main__':
    monitor = HardwareMonitor(path='exploring_stuff/hardware_monitoring/test.csv', sleep=2)
    x = threading.Thread(target=monitor.monitor, args=())
    x.start()
    time.sleep(10)
    print('termiating')
    monitor.terminate()