# Robot Kinematics Simulator

A professional-grade Python application for robotic manipulator simulation and analysis using Denavit-Hartenberg (DH) parameters. Features real-time 3D visualization, interactive joint control, and comprehensive kinematic analysis.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)
![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-orange)
![Robotics](https://img.shields.io/badge/Field-Robotics-red)

## Features

### Robot Modeling
- **Complete DH Parameter Support**: Configure θ, d, a, α parameters for each joint
- **Multiple Joint Types**: Support for both revolute and prismatic joints
- **Pre-built Configurations**: Includes SCARA, Articulated, 6-DOF Industrial, and more
- **Custom Robot Design**: Build any serial-link manipulator from scratch

### Real-time Visualization
- **Interactive 3D Viewport**: Rotate, zoom, and pan the robot model
- **Multiple View Perspectives**: XY, XZ, YZ planes and isometric views
- **Coordinate Frame Display**: Visualize coordinate systems at each joint
- **Professional Styling**: Dark theme with clear visual hierarchy

### Interactive Controls
- **Dynamic Joint Sliders**: Real-time control of variable joints
- **Custom Joint Limits**: Set min/max constraints for each joint
- **Instant Feedback**: Live updates of end-effector position and orientation

### Advanced Analysis
- **Transformation Matrices**: Complete chain from base to end-effector
- **DH Final Analysis**: Detailed breakdown of pose and orientation
- **RPY & Euler Analysis**: Multiple orientation representation methods
- **Workspace Analysis**: Reachability and volume calculations

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Sabari425/Robot-Kinematics-Simulator.git
cd Robot-Kinematics-Simulator
