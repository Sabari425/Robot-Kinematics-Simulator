# Robot Kinematics Simulator

A professional-grade Python application for robotic manipulator simulation and analysis using Denavit–Hartenberg (DH) parameters. Includes real-time 3D visualization, interactive joint control, and complete kinematic analysis.


![Python]([https://img.shields.io/badge/Python-3.8%2B-blue](https://www.python.org/downloads/))
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)
![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-orange)
![Robotics](https://img.shields.io/badge/Field-Robotics-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

![Python](https://www.python.org/downloads/)
![PyQt6](https://www.pythonguis.com/pyqt6-tutorial/)
![Matplotlib](https://matplotlib.org/)
![Robotics](https://www.tm-robot.com/en/robotic-arms/)

---

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Mathematical Foundation](#mathematical-foundation)
- [Code Structure](#code-structure)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)
- [Acknowledgments](#acknowledgments)

---

## Features

### Robot Modeling
- Full DH parameter support: θ, d, a, α  
- Revolute and prismatic joint types  
- SCARA, Articulated, 6-DOF industrial presets  
- Build any custom serial-link robot  

### Real-time Visualization
- Fully interactive 3D viewport  
- XY / XZ / YZ / isometric views  
- Joint coordinate frame display  
- Clean dark theme  

### Interactive Controls
- Real-time joint sliders  
- Joint limit configuration  
- Instant end-effector feedback  

### Advanced Kinematic Analysis
- Forward kinematics  
- Full transformation chains  
- RPY / Euler orientation  
- Workspace visualization  

---

## Quick Start

### Prerequisites
- Python 3.8+
- pip
- Git

### Install & Run
```bash
git clone https://github.com/Sabari425/Robot-Kinematics-Simulator.git
cd Robot-Kinematics-Simulator
pip install numpy sympy pyqt6 matplotlib
python robot_kinematic_simulator.py
```

### Platform-Specific Installation

#### Windows
```bash
python -m pip install numpy sympy pyqt6 matplotlib
```

#### macOS
```bash
pip3 install numpy sympy pyqt6 matplotlib
```

#### Linux
```bash
sudo apt-get update
sudo apt-get install python3-pip
pip3 install numpy sympy pyqt6 matplotlib
```


### Verify Installation
```bash
python -c "import numpy, sympy, PyQt6, matplotlib; print('All dependencies installed successfully!')"
```


## Run
```bash
python robot_kinematic_simulator.py
```
