# import wmi
#
# w = wmi.WMI(namespace=r"root\OpenHardwareMonitor")
# temperature_infos = w.Sensor()
#
# for sensor in temperature_infos:
#     if sensor.SensorType == u'Temperature':
#         print(sensor.Name)
#         print(sensor.Value)
#
# import wmi
#
# w = wmi.WMI(namespace="root\OpenHardwareMonitor")
# temperature_infos = w.Sensor()
# for sensor in temperature_infos:
#     if sensor.SensorType == u'Temperature':
#         if sensor.Name == 'CPU Package':
#             print(sensor.Name)
#             print(sensor.Value)


# import wmi
#
# w = wmi.WMI(namespace=r'root\wmi')
#
# temp = w.MSAcpi_ThermalZoneTemperature()[0].CurrentTemperature
#
# kelvin = temp / 10
# celsius = kelvin - 273.15
# fahrenheit = (9/5) * celsius + 32
#
# print(f'Kalvin:{kelvin:^10.2f}\tCelsius:{celsius:^10.2f}\tFahrenheit:{fahrenheit:^10.2f}')


# hwmon = wmi.WMI(namespace="root\OpenHardwareMonitor")
# sensors = hwmon.Sensor(["Name", "Parent", "Value", "Identifier"], SensorType="Temperature")

import clr  # package pythonnet, not clr
import time
import pandas as pd
import matplotlib.pyplot as plt


openhardwaremonitor_hwtypes = ['Mainboard', 'SuperIO', 'CPU', 'RAM', 'GpuNvidia', 'GpuAti', 'TBalancer', 'Heatmaster',
                               'HDD']
cputhermometer_hwtypes = ['Mainboard', 'SuperIO', 'CPU', 'GpuNvidia', 'GpuAti', 'TBalancer', 'Heatmaster', 'HDD']
openhardwaremonitor_sensortypes = ['Voltage', 'Clock', 'Temperature', 'Load', 'Fan', 'Flow', 'Control', 'Level',
                                   'Factor', 'Power', 'Data', 'SmallData']
cputhermometer_sensortypes = ['Voltage', 'Clock', 'Temperature', 'Load', 'Fan', 'Flow', 'Control', 'Level']


def initialize_openhardwaremonitor():
    file = r'C:\Program Files\OpenHardwareMonitor\OpenHardwareMonitorLib.dll'
    clr.AddReference(file)

    from OpenHardwareMonitor import Hardware

    handle = Hardware.Computer()
    handle.MainboardEnabled = True
    handle.CPUEnabled = True
    handle.RAMEnabled = True
    handle.GPUEnabled = True
    handle.HDDEnabled = True
    handle.Open()
    return handle


def write_total_CPU(handle, out_path):
    for i in handle.Hardware:
        if i.Name == 'AMD Ryzen 9 3900X':
            i.Update()
            for sensor in i.Sensors:
                if sensor.Name == 'CPU Total':
                    with open(out_path, 'a') as f:
                        f.write(f'{time.strftime("%m/%d/%Y, %H:%M:%S")}:\t{sensor.Value}\n')


if __name__ == "__main__":
    print("OpenHardwareMonitor:")
    HardwareHandle = initialize_openhardwaremonitor()
    while True:
        write_total_CPU(HardwareHandle, out_path='research_master/DQN/CartPole/boosting/exp_73/hardware_monitor.csv')
        time.sleep(10)
        mon = pd.read_csv('research_master/DQN/CartPole/boosting/exp_73/hardware_monitor.csv', sep='\t', header=None)
        plt.title('CPU usage %')
        plt.plot([0, len(mon)], [100, 100], alpha=.5)
        plt.plot(mon.values[:, 1])
        plt.savefig('research_master/DQN/CartPole/boosting/exp_73/cpu_monitor.png')
        plt.close()





