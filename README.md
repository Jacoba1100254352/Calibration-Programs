# Calibration-Programs

### Install Main Dependencies

```bash
pip install seaborn pyserial pandas matplotlib scikit-learn scipy numpy time
```

### Install RTC Dependencies

```bash
pip install pynput smbus2
```
(Another optional package if using test2.py is to install keyboard with pip)

### Install NN Dependencies

```bash
pip install tensorflow scikeras torch brevitas tensorflow_model_optimization
```

## Main Programs
The main files that should be used are:
- [Export_Sensor_Data.py](./Workflow_Programs/Export_Sensor_Data.py): Use this file to collect data from the arduino serial output during calibration.
- [Configuration_Variables.py](./Workflow_Programs/Configuration_Variables.py): Use this file to set the test number, sensor set number, and sensor ranges (or single sensor values when applicable).
- [Project_Manager.py](./Workflow_Programs/Project_Manager.py): This file is the main file that should be used to run the calibration program. Use it to convert from the original data, to aligned and calibrated data, as well as to graph.

