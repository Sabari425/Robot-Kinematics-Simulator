import sys
import numpy as np
import math as m
import sympy as sp
from math import radians, atan2, degrees, cos, sin, pi, sqrt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QSlider, QGroupBox, QFormLayout, QLineEdit, QTextEdit, QSizePolicy,
    QMessageBox, QComboBox, QDoubleSpinBox, QCheckBox, QSplitter, QFrame,
    QScrollArea, QGridLayout, QProgressBar, QHeaderView, QToolBar, QMenu,
    QFileDialog, QDialog, QDialogButtonBox, QMainWindow, QStatusBar,
    QRadioButton, QButtonGroup, QTreeWidget, QTreeWidgetItem, QProgressDialog
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QFont, QAction, QIcon, QColor, QPalette
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D
import matplotlib.animation as animation
from scipy.optimize import minimize, fsolve, least_squares
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from collections import defaultdict
import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable, Union, Any
from enum import Enum

# =============================================================================
# ENHANCED MATHEMATICAL OPERATIONS
# =============================================================================

class JointType(Enum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"

@dataclass
class DHParameter:
    """Data class for DH parameters with enhanced properties"""
    theta_deg: float = 0.0
    d: float = 0.0
    a: float = 1.0
    alpha_deg: float = 0.0
    variable: bool = True
    joint_type: str = 'revolute'
    theta_min: float = -180.0
    theta_max: float = 180.0
    d_min: float = -0.5
    d_max: float = 0.5
    
    @property
    def joint_type_enum(self) -> JointType:
        return JointType.REVOLUTE if self.joint_type == 'revolute' else JointType.PRISMATIC
    
    def to_dict(self) -> dict:
        return {
            'theta_deg': self.theta_deg,
            'd': self.d,
            'a': self.a,
            'alpha_deg': self.alpha_deg,
            'variable': self.variable,
            'joint_type': self.joint_type,
            'theta_min': self.theta_min,
            'theta_max': self.theta_max,
            'd_min': self.d_min,
            'd_max': self.d_max
        }

@dataclass
class RobotState:
    """Data class for robot state including position, velocity, acceleration"""
    position: np.ndarray
    orientation: np.ndarray
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None
    timestamp: float = 0.0

class EnhancedRobotKinematics:
    """Enhanced kinematics class with advanced features"""
    
    def __init__(self):
        self.use_numerical = True
        self.epsilon = 1e-10
        self.max_iterations = 1000
        self.convergence_threshold = 1e-8
        
    def dh_transform(self, theta_rad: float, d: float, a: float, alpha_rad: float) -> np.ndarray:
        """Enhanced DH transform with numerical stability checks"""
        ct = np.cos(theta_rad)
        st = np.sin(theta_rad)
        ca = np.cos(alpha_rad)
        sa = np.sin(alpha_rad)
        
        # Handle numerical precision
        if abs(ct) < self.epsilon:
            ct = 0.0
        if abs(st) < self.epsilon:
            st = 0.0
            
        T = np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return T
    
    def compute_forward_kinematics(self, dh_params: List[DHParameter]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute forward kinematics with enhanced error checking"""
        Ts = []
        positions = [np.array([0.0, 0.0, 0.0])]
        T_current = np.eye(4)
        Ts.append(T_current.copy())
        
        for param in dh_params:
            if param.joint_type == 'revolute':
                theta = radians(param.theta_deg)
                d = param.d
            else:  # prismatic
                theta = radians(param.theta_deg)
                d = param.a + param.d  # Account for prismatic displacement
            
            T_i = self.dh_transform(theta, d, param.a, radians(param.alpha_deg))
            T_current = T_current @ T_i
            Ts.append(T_current.copy())
            positions.append(T_current[:3, 3].copy())
            
        return Ts, positions
    
    def compute_velocities(self, dh_params: List[DHParameter], 
                          joint_velocities: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector velocity using geometric Jacobian"""
        Ts, positions = self.compute_forward_kinematics(dh_params)
        n = len(dh_params)
        
        J = np.zeros((6, n))  # 6 DOF Jacobian (3 linear, 3 angular)
        
        for i in range(n):
            z_axis = Ts[i][:3, 2]
            o_i = Ts[i][:3, 3]
            o_n = Ts[-1][:3, 3]
            
            if dh_params[i].joint_type == 'revolute':
                J[:3, i] = np.cross(z_axis, o_n - o_i)
                J[3:, i] = z_axis
            else:  # prismatic
                J[:3, i] = z_axis
                J[3:, i] = np.zeros(3)
        
        q_dot = np.array(joint_velocities)
        x_dot = J @ q_dot
        
        return x_dot, J
    
    def compute_jacobian(self, dh_params: List[DHParameter], 
                         positions: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Compute the analytical Jacobian matrix"""
        if positions is None:
            _, positions = self.compute_forward_kinematics(dh_params)
            
        Ts, _ = self.compute_forward_kinematics(dh_params)
        n = len(dh_params)
        
        J = np.zeros((6, n))
        
        for i in range(n):
            z_axis = Ts[i][:3, 2]
            o_i = Ts[i][:3, 3]
            o_n = Ts[-1][:3, 3]
            
            if dh_params[i].joint_type == 'revolute':
                J[:3, i] = np.cross(z_axis, o_n - o_i)
                J[3:, i] = z_axis
            else:
                J[:3, i] = z_axis
                
        return J
    
    def compute_inverse_kinematics_analytical(self, 
                                            dh_params: List[DHParameter],
                                            target_position: np.ndarray,
                                            target_orientation: Optional[np.ndarray] = None) -> Optional[List[float]]:
        """Analytical inverse kinematics for common robot configurations"""
        n = len(dh_params)
        
        if n == 3:
            return self._ik_3dof(dh_params, target_position)
        elif n == 6:
            return self._ik_6dof(dh_params, target_position, target_orientation)
        else:
            return None  # Fall back to numerical method
    
    def compute_inverse_kinematics_numerical(self,
                                            dh_params: List[DHParameter],
                                            target_pose: np.ndarray,
                                            initial_guess: Optional[List[float]] = None) -> Tuple[Optional[List[float]], Dict]:
        """Numerical inverse kinematics using iterative methods"""
        n = len(dh_params)
        
        if initial_guess is None:
            initial_guess = [0.0] * n
            if dh_params[0].joint_type == 'revolute':
                initial_guess[0] = 45.0
        
        info = {'iterations': 0, 'error_history': [], 'converged': False}
        
        def cost_function(joint_values):
            temp_params = []
            for i, val in enumerate(joint_values):
                temp_param = DHParameter(**dh_params[i].to_dict())
                if temp_param.joint_type == 'revolute':
                    temp_param.theta_deg = val
                else:
                    temp_param.d = val
                temp_params.append(temp_param)
            
            _, positions = self.compute_forward_kinematics(temp_params)
            current_position = positions[-1]
            
            # Optimize for position
            pos_error = np.linalg.norm(current_position - target_pose[:3, 3])
            
            # Add orientation error if target orientation is specified
            ori_error = 0.0
            if target_pose.shape == (4, 4):
                Ts, _ = self.compute_forward_kinematics(temp_params)
                current_ori = Ts[-1][:3, :3]
                target_ori = target_pose[:3, :3]
                ori_error = np.linalg.norm(np.eye(3) - current_ori @ target_ori.T)
                
            return pos_error + 0.5 * ori_error
        
        try:
            result = minimize(
                cost_function,
                initial_guess,
                method='L-BFGS-B',
                bounds=[(-180, 180) if p.joint_type == 'revolute' else (-0.5, 0.5) 
                       for p in dh_params],
                options={'maxiter': self.max_iterations, 'ftol': self.convergence_threshold}
            )
            
            info['iterations'] = result.nit
            info['converged'] = result.success
            info['error_history'] = [result.fun]
            
            if result.success:
                return result.x.tolist(), info
            
        except Exception as e:
            info['error'] = str(e)
            
        return None, info
    
    def _ik_3dof(self, dh_params: List[DHParameter], 
                 target_position: np.ndarray) -> Optional[List[float]]:
        """Analytical IK for 3-DOF planar robot"""
        x, y, z = target_position
        
        # Calculate joint angles
        try:
            l1 = dh_params[1].a
            l2 = dh_params[2].a
            
            # Check reachability
            r = sqrt(x**2 + y**2)
            if r > (l1 + l2) or r < abs(l1 - l2):
                return None
            
            # Calculate elbow-up and elbow-down solutions
            cos_theta3 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
            cos_theta3 = np.clip(cos_theta3, -1, 1)
            sin_theta3 = sqrt(1 - cos_theta3**2)
            
            theta3 = degrees(atan2(sin_theta3, cos_theta3))
            
            # Calculate theta2
            phi = atan2(y, x)
            beta = atan2(l2 * sin_theta3, l1 + l2 * cos_theta3)
            theta2 = degrees(phi - beta)
            
            # Calculate theta1
            theta1 = degrees(atan2(z, sqrt(x**2 + y**2)))
            
            return [theta1, theta2, theta3]
            
        except:
            return None
    
    def _ik_6dof(self, dh_params: List[DHParameter],
                 target_position: np.ndarray,
                 target_orientation: np.ndarray) -> Optional[List[float]]:
        """Analytical IK for 6-DOF articulated robot"""
        # Implementation for standard 6-DOF articulated arm
        # This would incorporate wrist center calculation, etc.
        return None  # Complex implementation required
    
    def compute_workspace(self, dh_params: List[DHParameter], 
                         num_points: int = 10000) -> np.ndarray:
        """Generate workspace points using Monte Carlo method"""
        workspace_points = []
        
        for _ in range(num_points):
            temp_params = []
            for param in dh_params:
                temp_param = DHParameter(**param.to_dict())
                if temp_param.joint_type == 'revolute':
                    temp_param.theta_deg = np.random.uniform(param.theta_min, param.theta_max)
                else:
                    temp_param.d = np.random.uniform(param.d_min, param.d_max)
                temp_params.append(temp_param)
            
            _, positions = self.compute_forward_kinematics(temp_params)
            workspace_points.append(positions[-1])
            
        return np.array(workspace_points)
    
    def compute_dynamics(self, dh_params: List[DHParameter],
                        masses: List[float],
                        inertias: List[np.ndarray],
                        joint_positions: List[float],
                        joint_velocities: List[float]) -> Dict:
        """Compute robot dynamics using Lagrangian formulation"""
        n = len(dh_params)
        
        # Compute mass matrix
        M = np.zeros((n, n))
        
        # Compute Coriolis and centrifugal terms
        C = np.zeros((n, n))
        
        # Compute gravity vector
        G = np.zeros(n)
        
        # Simplified dynamics - full implementation would require:
        # 1. Compute kinetic energy
        # 2. Compute potential energy
        # 3. Form Lagrangian
        # 4. Derive equations of motion
        
        return {
            'mass_matrix': M,
            'coriolis': C,
            'gravity': G,
            'tau_required': G  # For static case
        }
    
    def detect_singularities(self, dh_params: List[DHParameter]) -> Dict:
        """Detect kinematic singularities of the manipulator"""
        J = self.compute_jacobian(dh_params)
        
        # Compute manipulability measures
        manipulability = sqrt(np.linalg.det(J @ J.T))
        
        # Compute condition number
        U, S, Vt = np.linalg.svd(J)
        condition_number = S[0] / S[-1] if S[-1] > 0 else float('inf')
        
        # Compute dexterity
        dexterity = 1.0 / condition_number if condition_number > 0 else 0.0
        
        # Detect boundary singularities
        is_singular = condition_number > 1000
        
        return {
            'manipulability': manipulability,
            'condition_number': condition_number,
            'dexterity': dexterity,
            'singular_values': S.tolist(),
            'is_singular': is_singular,
            'null_space': Vt[-1].tolist() if S[-1] < 1e-6 else None
        }
    
    def plan_trajectory(self, 
                       dh_params: List[DHParameter],
                       start_point: np.ndarray,
                       end_point: np.ndarray,
                       num_waypoints: int = 50,
                       trajectory_type: str = 'cubic') -> Dict:
        """Plan a trajectory between two points"""
        
        # Generate linear path in Cartesian space
        waypoints = np.linspace(start_point, end_point, num_waypoints)
        
        joint_angles_list = []
        velocities_list = []
        accelerations_list = []
        
        current_config = [p.theta_deg if p.joint_type == 'revolute' else p.d 
                         for p in dh_params]
        
        for i, point in enumerate(waypoints):
            # Compute IK for each waypoint
            T_target = np.eye(4)
            T_target[:3, 3] = point
            
            joint_config, info = self.compute_inverse_kinematics_numerical(
                dh_params, T_target, current_config
            )
            
            if joint_config:
                joint_angles_list.append(joint_config)
                current_config = joint_config
        
        if not joint_angles_list:
            return {'success': False}
        
        joint_angles_array = np.array(joint_angles_list)
        
        # Apply smoothing with cubic spline
        t = np.linspace(0, 1, len(joint_angles_array))
        smoothed_trajectories = []
        
        for j in range(len(dh_params)):
            cs = CubicSpline(t, joint_angles_array[:, j])
            smoothed_trajectories.append(cs(t))
            
            # Calculate velocities and accelerations
            if trajectory_type == 'cubic':
                velocities_list.append(cs(t, 1))
                accelerations_list.append(cs(t, 2))
        
        smoothed_array = np.array(smoothed_trajectories).T
        
        return {
            'success': True,
            'joint_trajectories': smoothed_array,
            'velocities': np.array(velocities_list).T if velocities_list else None,
            'accelerations': np.array(accelerations_list).T if accelerations_list else None,
            'time': t,
            'waypoints': waypoints
        }
    
    def compute_dh_error(self, dh_params: List[DHParameter]) -> Dict:
        """Analyze DH parameter consistency and common errors"""
        errors = []
        warnings = []
        
        # Check DH parameter conventions
        for i, param in enumerate(dh_params):
            # Check alpha angle range
            if abs(param.alpha_deg) > 180:
                warnings.append(f"Joint {i+1}: Alpha angle outside [-180, 180]")
            
            # Check link length
            if param.a < 0:
                errors.append(f"Joint {i+1}: Negative link length")
            
            # Check joint limits
            if param.joint_type == 'revolute':
                if param.theta_max <= param.theta_min:
                    errors.append(f"Joint {i+1}: Invalid joint limits")
        
        # Check for parallel/consecutive axes
        for i in range(len(dh_params)-1):
            z1 = np.array([0, 0, 1])
            # Simplified check - would need actual transformation matrices
            if abs(dh_params[i+1].alpha_deg) < 1e-6 and abs(dh_params[i].alpha_deg) < 1e-6:
                warnings.append(f"Joints {i+1} and {i+2}: Parallel axes may cause singularities")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

# =============================================================================
# ENHANCED GUI CLASS
# =============================================================================

class RobotControlWorker(QThread):
    """Worker thread for heavy computations"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, task_type, *args, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        try:
            if self.task_type == 'forward_kinematics':
                result = self.compute_fk()
            elif self.task_type == 'inverse_kinematics':
                result = self.compute_ik()
            elif self.task_type == 'workspace':
                result = self.compute_workspace()
            elif self.task_type == 'trajectory':
                result = self.compute_trajectory()
            else:
                result = {}
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def compute_fk(self):
        kinematics, dh_params = self.args
        Ts, positions = kinematics.compute_forward_kinematics(dh_params)
        return {'Ts': Ts, 'positions': positions}
    
    def compute_ik(self):
        kinematics, dh_params, target = self.args
        solution, info = kinematics.compute_inverse_kinematics_numerical(dh_params, target)
        return {'solution': solution, 'info': info}
    
    def compute_workspace(self):
        kinematics, dh_params, num_points = self.args
        points = kinematics.compute_workspace(dh_params, num_points)
        return {'workspace_points': points}
    
    def compute_trajectory(self):
        kinematics, dh_params, start, end, waypoints, traj_type = self.args
        result = kinematics.plan_trajectory(dh_params, start, end, waypoints, traj_type)
        return result

class DHManipulatorGUI(QMainWindow):
    """Enhanced main GUI class"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Robot Kinematics Simulator Pro - v3.0")
        self.setMinimumSize(1920, 1080)
        
        # Initialize core components
        self.kinematics = EnhancedRobotKinematics()
        self.dh_params = []
        self.sliders = []
        self.slider_map = {}
        self.current_kinematics_results = ""
        self.animation_timer = None
        self.trajectory_data = None
        self.trajectory_index = 0
        self.workspace_points = None
        
        # Settings
        self.settings = QSettings('RoboticsSim', 'KinematicsPro')
        self.load_settings()
        
        # Initialize default configuration
        self.init_default_parameters(n=6)
        
        # Setup UI
        self.init_ui()
        self.apply_professional_theme()
        self.setup_status_bar()
        self.setup_toolbar()
        
        # Perform initial computation
        QTimer.singleShot(200, self.on_update)
        
    def load_settings(self):
        """Load user settings"""
        self.show_grid = self.settings.value('show_grid', True, type=bool)
        self.show_frames = self.settings.value('show_frames', True, type=bool)
        self.animation_speed = self.settings.value('animation_speed', 50, type=int)
        self.color_scheme = self.settings.value('color_scheme', 'dark', type=str)
        
    def save_settings(self):
        """Save user settings"""
        self.settings.setValue('show_grid', self.show_grid)
        self.settings.setValue('show_frames', self.show_frames)
        self.settings.setValue('animation_speed', self.animation_speed)
        self.settings.setValue('color_scheme', self.color_scheme)
        
    def init_default_parameters(self, n=6):
        """Initialize default DH parameters"""
        self.dh_params = []
        default_configs = [
            DHParameter(theta_deg=0, d=0.3, a=0, alpha_deg=90),
            DHParameter(theta_deg=-45, d=0, a=0.8, alpha_deg=0),
            DHParameter(theta_deg=0, d=0, a=0.6, alpha_deg=90),
            DHParameter(theta_deg=0, d=0.4, a=0, alpha_deg=-90),
            DHParameter(theta_deg=0, d=0, a=0, alpha_deg=90),
            DHParameter(theta_deg=0, d=0.1, a=0, alpha_deg=0)
        ]
        
        self.dh_params = default_configs[:n]
        for param in self.dh_params:
            param.variable = True
            
    def apply_professional_theme(self):
        """Apply professional dark theme"""
        if self.color_scheme == 'dark':
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1e1e2e;
                }
                QWidget {
                    background-color: #2b2b3c;
                    color: #cdd6f4;
                    font-family: 'Segoe UI', 'Cascadia Code', monospace;
                    font-size: 10pt;
                }
                QPushButton {
                    background-color: #45475a;
                    border: 2px solid #585b70;
                    padding: 8px 16px;
                    border-radius: 8px;
                    font-weight: bold;
                    min-width: 100px;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    background-color: #585b70;
                    border: 2px solid #6c7086;
                }
                QPushButton:pressed {
                    background-color: #6c7086;
                }
                QPushButton:disabled {
                    background-color: #313244;
                    color: #6c7086;
                }
                QTableWidget {
                    background-color: #313244;
                    gridline-color: #45475a;
                    border: 2px solid #45475a;
                    border-radius: 8px;
                    font-size: 9pt;
                }
                QTableWidget::item {
                    padding: 8px;
                    border-bottom: 1px solid #45475a;
                }
                QTableWidget::item:selected {
                    background-color: #585b70;
                }
                QHeaderView::section {
                    background-color: #45475a;
                    padding: 10px;
                    border: 1px solid #585b70;
                    font-weight: bold;
                    font-size: 9pt;
                }
                QTabWidget::pane {
                    border: 2px solid #45475a;
                    border-radius: 12px;
                    background-color: #313244;
                }
                QTabBar::tab {
                    background-color: #45475a;
                    padding: 12px 20px;
                    border: 2px solid #585b70;
                    border-bottom: none;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    margin-right: 4px;
                    font-weight: bold;
                    font-size: 10pt;
                }
                QTabBar::tab:selected {
                    background-color: #585b70;
                    border-color: #6c7086;
                }
                QTabBar::tab:hover:!selected {
                    background-color: #4a4a5a;
                }
                QGroupBox {
                    border: 3px solid #585b70;
                    border-radius: 10px;
                    margin-top: 15px;
                    padding-top: 20px;
                    font-weight: bold;
                    background-color: #313244;
                    font-size: 10pt;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 15px;
                    padding: 0 10px 0 10px;
                    color: #cdd6f4;
                    font-size: 11pt;
                }
                QSlider::groove:horizontal {
                    border: 2px solid #585b70;
                    height: 12px;
                    background: #45475a;
                    margin: 3px 0;
                    border-radius: 6px;
                }
                QSlider::handle:horizontal {
                    background: #89b4fa;
                    border: 2px solid #b4befe;
                    width: 24px;
                    margin: -10px 0;
                    border-radius: 8px;
                }
                QSlider::handle:horizontal:hover {
                    background: #b4befe;
                }
                QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                    background-color: #45475a;
                    border: 2px solid #585b70;
                    padding: 8px;
                    border-radius: 8px;
                    color: #cdd6f4;
                    min-height: 25px;
                    font-size: 10pt;
                }
                QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
                    border: 2px solid #89b4fa;
                }
                QTextEdit {
                    background-color: #313244;
                    border: 2px solid #45475a;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 9pt;
                    font-family: 'Cascadia Code', 'Consolas', monospace;
                }
                QLabel {
                    color: #cdd6f4;
                    padding: 3px;
                }
                QStatusBar {
                    background-color: #45475a;
                    color: #cdd6f4;
                    font-size: 10pt;
                }
                QProgressBar {
                    border: 2px solid #45475a;
                    border-radius: 6px;
                    text-align: center;
                    background-color: #313244;
                    font-size: 9pt;
                }
                QProgressBar::chunk {
                    background-color: #89b4fa;
                    border-radius: 4px;
                }
                QMenu {
                    background-color: #313244;
                    border: 2px solid #45475a;
                }
                QMenu::item {
                    padding: 8px 32px;
                }
                QMenu::item:selected {
                    background-color: #585b70;
                }
            """)
        else:
            # Light theme
            self.setStyleSheet("")
            
    def setup_status_bar(self):
        """Setup enhanced status bar"""
        self.statusBar().showMessage("Ready | Current Configuration: 6-DOF Industrial Robot")
        
        # Add permanent widgets
        self.coord_label = QLabel("X: 0.000 | Y: 0.000 | Z: 0.000")
        self.coord_label.setStyleSheet("padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.coord_label)
        
        self.joint_count_label = QLabel("Joints: 6")
        self.statusBar().addPermanentWidget(self.joint_count_label)
        
        self.singularity_label = QLabel("Status: ✓ Non-singular")
        self.statusBar().addPermanentWidget(self.singularity_label)
        
    def setup_toolbar(self):
        """Setup main toolbar"""
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("QToolBar { spacing: 5px; padding: 5px; }")
        
        # File operations
        save_action = QAction("↓ Save", self)
        save_action.setToolTip("Save current configuration")
        save_action.triggered.connect(self.save_configuration)
        toolbar.addAction(save_action)
        
        load_action = QAction("↑ Load", self)
        load_action.setToolTip("Load configuration")
        load_action.triggered.connect(self.load_configuration)
        toolbar.addAction(load_action)
        
        toolbar.addSeparator()
        
        # Export options
        export_action = QAction(" Export Data", self)
        export_action.setToolTip("Export kinematics data")
        export_action.triggered.connect(self.export_data)
        toolbar.addAction(export_action)
        
        export_plot_action = QAction(" Export Plot", self)
        export_plot_action.setToolTip("Export current plot")
        export_plot_action.triggered.connect(self.export_plot)
        toolbar.addAction(export_plot_action)
        
        toolbar.addSeparator()
        
        # View options
        view_menu = QMenu("View Options", self)
        
        grid_action = QAction("Show Grid", self, checkable=True)
        grid_action.setChecked(self.show_grid)
        grid_action.triggered.connect(lambda checked: setattr(self, 'show_grid', checked))
        view_menu.addAction(grid_action)
        
        frames_action = QAction("Show Coordinate Frames", self, checkable=True)
        frames_action.setChecked(self.show_frames)
        frames_action.triggered.connect(lambda checked: setattr(self, 'show_frames', checked))
        view_menu.addAction(frames_action)
        
        view_button = QPushButton(" View")
        view_button.setMenu(view_menu)
        toolbar.addWidget(view_button)
        
        toolbar.addSeparator()
        
        # Animation controls
        play_action = QAction("▶ Play Trajectory", self)
        play_action.triggered.connect(self.play_trajectory)
        toolbar.addAction(play_action)
        
        stop_action = QAction("◼ Stop", self)
        stop_action.triggered.connect(self.stop_trajectory)
        toolbar.addAction(stop_action)
        
        # Speed control
        speed_label = QLabel("Speed:")
        toolbar.addWidget(speed_label)
        speed_spin = QSpinBox()
        speed_spin.setRange(1, 200)
        speed_spin.setValue(self.animation_speed)
        speed_spin.setSuffix(" ms")
        speed_spin.valueChanged.connect(self.update_animation_speed)
        toolbar.addWidget(speed_spin)
        
        toolbar.addSeparator()
        
        # Theme toggle
        theme_action = QAction("◑ Toggle Theme", self)
        theme_action.triggered.connect(self.toggle_theme)
        toolbar.addAction(theme_action)
        
    def init_ui(self):
        """Initialize the enhanced UI"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        
        # Right panel - Visualization and Analysis
        right_panel = self.create_visualization_panel()
        
        # Add to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 1300])
        
        main_layout.addWidget(splitter)
        
    def create_control_panel(self) -> QWidget:
        """Create the control panel with all widgets"""
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(550)
        left_scroll.setMaximumWidth(750)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title with version
        title_layout = QVBoxLayout()
        title = QLabel("🤖 Advanced Robot Kinematics Simulator")
        title.setProperty("title", True)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16pt; font-weight: bold; padding: 10px;")
        title_layout.addWidget(title)
        
        version_label = QLabel("Version 3.0 | Professional Edition")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("font-size: 9pt; color: #6c7086;")
        title_layout.addWidget(version_label)
        left_layout.addLayout(title_layout)
        
        # Robot Configuration
        config_group = QGroupBox("⌘ Robot Configuration")
        config_layout = QGridLayout()
        
        # Robot type selection
        config_layout.addWidget(QLabel("Robot Type:"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "3-DOF Planar",
            "4-DOF SCARA", 
            "6-DOF Industrial (PUMA 560)",
            "6-DOF Stanford",
            "7-DOF Redundant",
            "Cylindrical",
            "Spherical",
            "Cartesian/Gantry",
            "Delta Parallel"
        ])
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        config_layout.addWidget(self.preset_combo, 0, 1)
        
        # Joint count
        config_layout.addWidget(QLabel("Number of Joints:"), 1, 0)
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 12)
        self.spin_n.setValue(len(self.dh_params))
        self.spin_n.valueChanged.connect(self.on_n_changed)
        config_layout.addWidget(self.spin_n, 1, 1)
        
        # IK Method selection
        config_layout.addWidget(QLabel("IK Method:"), 2, 0)
        self.ik_method_combo = QComboBox()
        self.ik_method_combo.addItems([
            "Numerical (L-BFGS-B)",
            "Numerical (Newton-Raphson)",
            "Analytical (if available)",
            "Hybrid"
        ])
        config_layout.addWidget(self.ik_method_combo, 2, 1)
        
        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)
        
        # DH Parameters Table
        table_group = QGroupBox("DH Parameters")
        table_layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "θ (deg)", "d (m)", "a (m)", "α (deg)", 
            "Variable", "Jt Type", "θ Min", "θ Max", "d Min", "d Max"
        ])
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(35)
        
        # Set font
        font = QFont("Cascadia Code", 11)
        font.setBold(False)
        self.table.setFont(font)
        
        # Set column widths
        min_widths = [80, 65, 65, 80, 65, 90, 70, 70, 70, 70]
        for i, width in enumerate(min_widths):
            self.table.setColumnWidth(i, width)
        
        self.populate_table()
        table_layout.addWidget(self.table)
        
        # Advanced DH options
        options_layout = QHBoxLayout()
        
        self.auto_update_check = QCheckBox("Auto-Update")
        self.auto_update_check.setChecked(True)
        self.auto_update_check.toggled.connect(self.toggle_auto_update)
        options_layout.addWidget(self.auto_update_check)
        
        self.show_advanced_check = QCheckBox("Advanced Mode")
        self.show_advanced_check.toggled.connect(self.toggle_advanced_mode)
        options_layout.addWidget(self.show_advanced_check)
        
        options_layout.addStretch()
        table_layout.addLayout(options_layout)
        
        table_group.setLayout(table_layout)
        left_layout.addWidget(table_group)
        
        # Control Buttons
        control_group = QGroupBox("Controls")
        control_layout = QGridLayout()
        
        self.btn_update = QPushButton("▶ Compute Forward Kinematics")
        self.btn_update.clicked.connect(self.on_update)
        self.btn_update.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; font-size: 11pt;")
        control_layout.addWidget(self.btn_update, 0, 0, 1, 2)
        
        self.btn_inverse = QPushButton("⟳ Inverse Kinematics")
        self.btn_inverse.clicked.connect(self.on_inverse_kinematics)
        self.btn_inverse.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e;")
        control_layout.addWidget(self.btn_inverse, 1, 0)
        
        self.btn_workspace = QPushButton("◉ Compute Workspace")
        self.btn_workspace.clicked.connect(self.on_compute_workspace)
        control_layout.addWidget(self.btn_workspace, 1, 1)
        
        self.btn_singularity = QPushButton("⚠ Check Singularities")
        self.btn_singularity.clicked.connect(self.on_check_singularities)
        control_layout.addWidget(self.btn_singularity, 2, 0)
        
        self.btn_trajectory = QPushButton("▲ Plan Trajectory")
        self.btn_trajectory.clicked.connect(self.on_plan_trajectory)
        control_layout.addWidget(self.btn_trajectory, 2, 1)
        
        self.btn_reset = QPushButton("↻ Reset")
        self.btn_reset.clicked.connect(self.on_reset)
        control_layout.addWidget(self.btn_reset, 3, 0)
        
        self.btn_clear = QPushButton("☒ Clear")
        self.btn_clear.clicked.connect(self.clear_results)
        control_layout.addWidget(self.btn_clear, 3, 1)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        # Joint Control Sliders
        self.sliders_group = QGroupBox("—●— Real-time Joint Control")
        self.sliders_layout = QVBoxLayout()
        self.sliders_group.setLayout(self.sliders_layout)
        left_layout.addWidget(self.sliders_group)
        
        # Status and Information
        info_group = QGroupBox("ℹ Robot Information")
        info_layout = QVBoxLayout()
        
        self.ee_label = QLabel("End Effector: X=0.000, Y=0.000, Z=0.000")
        self.ee_label.setWordWrap(True)
        self.ee_label.setStyleSheet("font-family: 'Cascadia Code', monospace; padding: 5px;")
        info_layout.addWidget(self.ee_label)
        
        self.orientation_label = QLabel("Orientation: Roll=0.0°, Pitch=0.0°, Yaw=0.0°")
        self.orientation_label.setStyleSheet("font-family: 'Cascadia Code', monospace; padding: 5px;")
        info_layout.addWidget(self.orientation_label)
        
        self.workspace_info = QLabel("Workspace: Not calculated")
        self.workspace_info.setWordWrap(True)
        info_layout.addWidget(self.workspace_info)
        
        self.singularity_info = QLabel("Singularity Status: Not checked")
        info_layout.addWidget(self.singularity_info)
        
        self.performance_label = QLabel("Performance: --")
        info_layout.addWidget(self.performance_label)
        
        # Progress indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        info_layout.addWidget(self.progress_bar)
        
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)
        
        left_layout.addStretch()
        left_scroll.setWidget(left_widget)
        
        return left_scroll
        
    def create_visualization_panel(self) -> QWidget:
        """Create the visualization and analysis panel"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Tab 1: 3D Visualization
        self.create_visualization_tab()
        
        # Tab 2: Workspace View
        self.create_workspace_tab()
        
        # Tab 3: Joint Space Analysis
        self.create_joint_space_tab()
        
        # Tab 4: Dynamics
        self.create_dynamics_tab()
        
        # Tab 5: Code/Export
        self.create_export_tab()
        
        right_layout.addWidget(self.tabs)
        
        return right_widget
        
    def create_visualization_tab(self):
        """Create the 3D visualization tab"""
        tab_vis = QWidget()
        tv_layout = QVBoxLayout(tab_vis)
        
        # View controls
        view_controls = QHBoxLayout()
        view_label = QLabel("View:")
        view_label.setStyleSheet("font-weight: bold;")
        view_controls.addWidget(view_label)
        
        views = [
            ("XY Top", 0, 0),
            ("XZ Front", 0, 90),
            ("YZ Side", 90, 0),
            ("Isometric", 30, 45),
            ("Default", 25, 45),
        ]
        
        for name, elev, azim in views:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, e=elev, a=azim: self.set_view(e, a))
            view_controls.addWidget(btn)
        
        view_controls.addStretch()
        tv_layout.addLayout(view_controls)
        
        # Plot canvas
        plot_frame = QFrame()
        plot_frame.setFrameStyle(QFrame.Shape.Box)
        plot_frame.setStyleSheet("background-color: #1e1e2e; border: 2px solid #45475a; border-radius: 10px;")
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(10, 10, 10, 10)
        
        self.fig = Figure(figsize=(12, 9), facecolor='#1e1e2e')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1e1e2e')
        self.ax.set_proj_type('ortho')
        
        # Style the 3D axes
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#45475a')
        self.ax.yaxis.pane.set_edgecolor('#45475a')
        self.ax.zaxis.pane.set_edgecolor('#45475a')
        
        self.ax.grid(True, linestyle='--', alpha=0.3, color='#585b70')
        self.ax.set_xlabel('X (m)', fontsize=11, color='#cdd6f4', labelpad=10)
        self.ax.set_ylabel('Y (m)', fontsize=11, color='#cdd6f4', labelpad=10)
        self.ax.set_zlabel('Z (m)', fontsize=11, color='#cdd6f4', labelpad=10)
        
        # Set initial view
        self.ax.view_init(elev=25, azim=45)
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([0, 3])
        
        self.canvas.mpl_connect("motion_notify_event", self.on_plot_hover)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        plot_layout.addWidget(self.canvas)
        
        tv_layout.addWidget(plot_frame)
        self.tabs.addTab(tab_vis, " 3D Visualization ")
        
    def create_workspace_tab(self):
        """Create workspace analysis tab"""
        tab_ws = QWidget()
        ws_layout = QVBoxLayout(tab_ws)
        
        # Workspace controls
        ws_controls = QHBoxLayout()
        ws_controls.addWidget(QLabel("Sample Points:"))
        self.workspace_points_spin = QSpinBox()
        self.workspace_points_spin.setRange(1000, 100000)
        self.workspace_points_spin.setValue(10000)
        self.workspace_points_spin.setSingleStep(1000)
        ws_controls.addWidget(self.workspace_points_spin)
        
        self.btn_generate_ws = QPushButton("Generate Workspace")
        self.btn_generate_ws.clicked.connect(self.on_compute_workspace)
        ws_controls.addWidget(self.btn_generate_ws)
        
        ws_controls.addStretch()
        ws_layout.addLayout(ws_controls)
        
        # Workspace plot
        ws_frame = QFrame()
        ws_frame.setStyleSheet("background-color: #1e1e2e; border: 2px solid #45475a; border-radius: 10px;")
        ws_plot_layout = QVBoxLayout(ws_frame)
        
        self.ws_fig = Figure(figsize=(10, 8), facecolor='#1e1e2e')
        self.ws_canvas = FigureCanvas(self.ws_fig)
        self.ws_ax = self.ws_fig.add_subplot(111, projection='3d')
        self.ws_ax.set_facecolor('#1e1e2e')
        
        ws_plot_layout.addWidget(self.ws_canvas)
        ws_layout.addWidget(ws_frame)
        
        # Workspace statistics
        self.ws_stats_text = QTextEdit()
        self.ws_stats_text.setReadOnly(True)
        self.ws_stats_text.setStyleSheet("font-family: 'Cascadia Code', monospace;")
        ws_layout.addWidget(self.ws_stats_text)
        
        self.tabs.addTab(tab_ws, " Workspace ")
        
    def create_joint_space_tab(self):
        """Create joint space analysis tab"""
        tab_js = QWidget()
        js_layout = QVBoxLayout(tab_js)
        
        # Joint space plot
        js_frame = QFrame()
        js_frame.setStyleSheet("background-color: #1e1e2e; border: 2px solid #45475a; border-radius: 10px;")
        js_plot_layout = QVBoxLayout(js_frame)
        
        self.js_fig = Figure(figsize=(10, 8), facecolor='#1e1e2e')
        self.js_canvas = FigureCanvas(self.js_fig)
        
        # Create subplots for each joint
        n_joints = max(len(self.dh_params), 1)
        if n_joints <= 3:
            self.js_axes = [self.js_fig.add_subplot(n_joints, 1, i+1) for i in range(n_joints)]
        else:
            n_rows = (n_joints + 1) // 2
            self.js_axes = [self.js_fig.add_subplot(n_rows, 2, i+1) for i in range(n_joints)]
        
        for ax in self.js_axes:
            ax.set_facecolor('#1e1e2e')
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors='#cdd6f4')
            
        js_plot_layout.addWidget(self.js_canvas)
        js_layout.addWidget(js_frame)
        
        # Joint limits display
        self.joint_limits_text = QTextEdit()
        self.joint_limits_text.setReadOnly(True)
        self.joint_limits_text.setStyleSheet("font-family: 'Cascadia Code', monospace; max-height: 100px;")
        js_layout.addWidget(self.joint_limits_text)
        
        self.tabs.addTab(tab_js, " Joint Space ")
        
    def create_dynamics_tab(self):
        """Create dynamics analysis tab"""
        tab_dyn = QWidget()
        dyn_layout = QVBoxLayout(tab_dyn)
        
        # Mass and inertia inputs
        dyn_inputs = QGroupBox("Dynamic Parameters")
        dyn_input_layout = QGridLayout()
        
        # Will be populated dynamically based on joint count
        self.dyn_mass_spins = []
        self.dyn_inertia_spins = []
        
        dyn_inputs.setLayout(dyn_input_layout)
        dyn_layout.addWidget(dyn_inputs)
        
        # Dynamics results
        self.dyn_results_text = QTextEdit()
        self.dyn_results_text.setReadOnly(True)
        self.dyn_results_text.setStyleSheet("font-family: 'Cascadia Code', monospace;")
        dyn_layout.addWidget(self.dyn_results_text)
        
        self.tabs.addTab(tab_dyn, " Dynamics ")
        
    def create_export_tab(self):
        """Create export and code generation tab"""
        tab_exp = QWidget()
        exp_layout = QVBoxLayout(tab_exp)
        
        # Code generation options
        code_group = QGroupBox("Code Generation")
        code_layout = QVBoxLayout()
        
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Language:"))
        self.code_lang_combo = QComboBox()
        self.code_lang_combo.addItems([
            "Python (NumPy)",
            "Python (SymPy)",
            "C++ (Eigen)",
            "MATLAB",
            "URDF"
        ])
        lang_layout.addWidget(self.code_lang_combo)
        lang_layout.addStretch()
        code_layout.addLayout(lang_layout)
        
        self.btn_generate_code = QPushButton("Generate Kinematics Code")
        self.btn_generate_code.clicked.connect(self.generate_code)
        code_layout.addWidget(self.btn_generate_code)
        
        code_group.setLayout(code_layout)
        exp_layout.addWidget(code_group)
        
        # Generated code output
        self.code_output = QTextEdit()
        self.code_output.setReadOnly(True)
        self.code_output.setStyleSheet("font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 9pt;")
        exp_layout.addWidget(self.code_output)
        
        self.tabs.addTab(tab_exp, " Code ")
        
    def populate_table(self):
        """Populate the DH parameters table"""
        self.table.setRowCount(len(self.dh_params))
        for i, param in enumerate(self.dh_params):
            items = [
                QTableWidgetItem(f"{param.theta_deg:.4f}"),
                QTableWidgetItem(f"{param.d:.4f}"),
                QTableWidgetItem(f"{param.a:.4f}"),
                QTableWidgetItem(f"{param.alpha_deg:.4f}"),
                QTableWidgetItem("1" if param.variable else "0"),
            ]
            
            for j, item in enumerate(items):
                self.table.setItem(i, j, item)
            
            # Joint type combo
            joint_combo = QComboBox()
            joint_combo.addItems(['revolute', 'prismatic'])
            joint_combo.setCurrentText(param.joint_type)
            joint_combo.currentTextChanged.connect(lambda text, row=i: self.on_joint_type_changed(text, row))
            self.table.setCellWidget(i, 5, joint_combo)
            
            # Joint limits
            self.table.setItem(i, 6, QTableWidgetItem(f"{param.theta_min:.1f}"))
            self.table.setItem(i, 7, QTableWidgetItem(f"{param.theta_max:.1f}"))
            self.table.setItem(i, 8, QTableWidgetItem(f"{param.d_min:.3f}"))
            self.table.setItem(i, 9, QTableWidgetItem(f"{param.d_max:.3f}"))
            
    def read_table(self) -> bool:
        """Read DH parameters from table"""
        try:
            new_params = []
            for i in range(self.table.rowCount()):
                theta = float(self.table.item(i, 0).text())
                d = float(self.table.item(i, 1).text())
                a = float(self.table.item(i, 2).text())
                alpha = float(self.table.item(i, 3).text())
                variable = int(float(self.table.item(i, 4).text())) != 0
                joint_type = self.table.cellWidget(i, 5).currentText()
                theta_min = float(self.table.item(i, 6).text())
                theta_max = float(self.table.item(i, 7).text())
                d_min = float(self.table.item(i, 8).text())
                d_max = float(self.table.item(i, 9).text())
                
                param = DHParameter(
                    theta_deg=theta,
                    d=d,
                    a=a,
                    alpha_deg=alpha,
                    variable=variable,
                    joint_type=joint_type,
                    theta_min=theta_min,
                    theta_max=theta_max,
                    d_min=d_min,
                    d_max=d_max
                )
                new_params.append(param)
            
            self.dh_params = new_params
            return True
        except (ValueError, AttributeError) as e:
            QMessageBox.warning(self, "Invalid Input", f"Error in row {i+1}: {str(e)}")
            return False
    
    def on_update(self):
        """Handle compute forward kinematics button click"""
        if not self.read_table():
            return
            
        self.show_progress()
        start_time = time.time()
        
        try:
            Ts, positions = self.kinematics.compute_forward_kinematics(self.dh_params)
            
            # Update all displays
            self.update_robot_display(Ts, positions)
            self.update_status_info(positions[-1], Ts[-1])
            self.update_joint_space(Ts, positions)
            self.create_sliders_for_variables()
            
            # Check performance
            elapsed = time.time() - start_time
            self.performance_label.setText(f"Performance: {elapsed*1000:.1f}ms | {len(self.dh_params)} joints")
            
            self.statusBar().showMessage(f"✓ Forward kinematics computed in {elapsed*1000:.1f}ms")
            
        except Exception as e:
            self.statusBar().showMessage(f"✗ Error: {str(e)}")
            traceback.print_exc()
        finally:
            self.hide_progress()
    
    def update_robot_display(self, Ts, positions):
        """Update the 3D visualization"""
        self.ax.clear()
        
        # Extract coordinates
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        
        # Calculate limits
        all_positions = np.array(positions)
        max_range = np.max(np.abs(all_positions)) * 1.3
        limit = max(2.0, max_range)
        
        self.ax.set_xlim3d(-limit, limit)
        self.ax.set_ylim3d(-limit, limit)
        self.ax.set_zlim3d(0, max(1.0, limit * 1.5))
        
        # Draw robot links with gradient coloring
        n = len(xs)
        for i in range(n - 1):
            # Color gradient from blue (base) to red (end effector)
            t = i / max(n - 2, 1)
            r = int(255 * t)
            g = int(64 * (1 - abs(2*t - 1)))
            b = int(255 * (1 - t))
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            self.ax.plot(
                [xs[i], xs[i+1]],
                [ys[i], ys[i+1]],
                [zs[i], zs[i+1]],
                '-',
                linewidth=5.0,
                color=color,
                solid_capstyle='round',
                zorder=5
            )
        
        # Draw joints as spheres
        for i, (x, y, z) in enumerate(positions):
            marker_size = 120 if i == len(positions) - 1 else 80
            marker_color = '#ff5555' if i == len(positions) - 1 else '#89b4fa'
            
            self.ax.scatter(
                [x], [y], [z],
                s=marker_size,
                color=marker_color,
                edgecolors='white',
                linewidth=2.0,
                zorder=10,
                alpha=1.0
            )
        
        # Draw coordinate frames if enabled
        if self.show_frames:
            frame_scale = limit * 0.15
            for i, T in enumerate(Ts):
                origin = T[:3, 3]
                colors = ['#ff5555', '#55ff55', '#5555ff']
                for j in range(3):
                    axis = T[:3, j] * frame_scale
                    self.ax.quiver(
                        *origin, *axis,
                        color=colors[j],
                        linewidth=2.5,
                        arrow_length_ratio=0.15,
                        alpha=0.9,
                        zorder=10
                    )
        
        # Draw base
        self.ax.scatter([0], [0], [0], s=200, color='#ffaa00',
                       marker='s', edgecolors='white', linewidth=2.5, zorder=15)
        
        # Add ground plane
        xx, yy = np.meshgrid(
            np.linspace(-limit*1.2, limit*1.2, 20),
            np.linspace(-limit*1.2, limit*1.2, 20)
        )
        zz = np.zeros_like(xx)
        self.ax.plot_surface(xx, yy, zz, alpha=0.1, color='#585b70')
        
        # Update plot style
        self.ax.grid(self.show_grid, linestyle='--', alpha=0.3, color='#585b70')
        self.ax.set_title('Advanced Robot Kinematics - Real-time 3D View',
                         fontsize=13, fontweight='bold', color='#cdd6f4', pad=20)
        
        self.canvas.draw_idle()
    
    def update_status_info(self, ee_position, T_final):
        """Update status bar and information displays"""
        # Update coordinates
        x, y, z = ee_position
        self.coord_label.setText(f"X: {x:.4f} | Y: {y:.4f} | Z: {z:.4f}")
        self.ee_label.setText(f"End Effector Position:\nX = {x:.6f} m\nY = {y:.6f} m\nZ = {z:.6f} m")
        
        # Calculate orientation
        roll = degrees(atan2(T_final[2, 1], T_final[2, 2]))
        pitch = degrees(atan2(-T_final[2, 0], np.sqrt(T_final[2, 1]**2 + T_final[2, 2]**2)))
        yaw = degrees(atan2(T_final[1, 0], T_final[0, 0]))
        
        self.orientation_label.setText(
            f"Orientation (RPY):\nRoll = {roll:.2f}°\nPitch = {pitch:.2f}°\nYaw = {yaw:.2f}°"
        )
        
        self.joint_count_label.setText(f"Joints: {len(self.dh_params)}")
        
        # Check singularity
        singular = self.kinematics.detect_singularities(self.dh_params)
        if singular['is_singular']:
            self.singularity_label.setText("⚠ Singular Configuration!")
            self.singularity_label.setStyleSheet("color: #ff5555; font-weight: bold;")
        else:
            self.singularity_label.setText("✓ Non-singular")
            self.singularity_label.setStyleSheet("color: #55ff55;")
    
    def update_joint_space(self, Ts, positions):
        """Update joint space analysis"""
        if not hasattr(self, 'js_axes'):
            return
            
        # Clear axes
        for ax in self.js_axes:
            ax.clear()
            ax.set_facecolor('#1e1e2e')
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors='#cdd6f4')
        
        # Plot joint configuration
        n = len(self.dh_params)
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        
        for i in range(min(n, len(self.js_axes))):
            ax = self.js_axes[i]
            
            # Plot current joint position
            value = self.dh_params[i].theta_deg if self.dh_params[i].joint_type == 'revolute' else self.dh_params[i].d
            
            ax.axvline(x=value, color=colors[i], linewidth=3, label=f'Current: {value:.1f}')
            
            # Plot joint limits
            if self.dh_params[i].joint_type == 'revolute':
                min_val = self.dh_params[i].theta_min
                max_val = self.dh_params[i].theta_max
                ax.axvspan(min_val, max_val, alpha=0.2, color='#89b4fa')
            else:
                min_val = self.dh_params[i].d_min
                max_val = self.dh_params[i].d_max
            
            ax.set_xlim([min_val * 1.2, max_val * 1.2])
            ax.set_ylabel(f'Joint {i+1}', color='#cdd6f4')
            ax.legend(loc='upper right')
        
        self.js_canvas.draw_idle()
        
        # Update joint limits text
        limits_text = "Joint Limits:\n"
        for i, param in enumerate(self.dh_params):
            if param.joint_type == 'revolute':
                limits_text += f"Joint {i+1}: [{param.theta_min:.1f}°, {param.theta_max:.1f}°]\n"
            else:
                limits_text += f"Joint {i+1}: [{param.d_min:.3f}m, {param.d_max:.3f}m]\n"
        
        self.joint_limits_text.setPlainText(limits_text)
    
    def create_sliders_for_variables(self):
        """Create interactive sliders for joint control"""
        self.clear_sliders()
        
        for i, param in enumerate(self.dh_params):
            if param.variable:
                slider_container = QWidget()
                slider_layout = QHBoxLayout(slider_container)
                slider_layout.setContentsMargins(0, 3, 0, 3)
                
                if param.joint_type == 'revolute':
                    label = QLabel(f"J{i+1} θ:")
                    min_val = int(param.theta_min)
                    max_val = int(param.theta_max)
                    current_val = int(param.theta_deg)
                    suffix = '°'
                else:
                    label = QLabel(f"J{i+1} d:")
                    # Convert to mm for slider precision
                    min_val = int(param.d_min * 1000)
                    max_val = int(param.d_max * 1000)
                    current_val = int(param.d * 1000)
                    suffix = 'mm'
                
                label.setMinimumWidth(60)
                label.setStyleSheet("font-weight: bold;")
                slider_layout.addWidget(label)
                
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(min_val, max_val)
                slider.setValue(current_val)
                slider.setStyleSheet("QSlider::groove:horizontal { height: 8px; }")
                
                value_label = QLabel(f"{current_val:6.1f}{suffix}")
                value_label.setMinimumWidth(80)
                value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
                value_label.setStyleSheet("font-family: 'Cascadia Code', monospace;")
                
                slider.valueChanged.connect(
                    self.make_slider_handler(i, value_label)
                )
                
                slider_layout.addWidget(slider, 1)
                slider_layout.addWidget(value_label)
                
                self.sliders_layout.addWidget(slider_container)
                self.slider_map[i] = (slider, value_label)
    
    def clear_sliders(self):
        """Clear all sliders"""
        for i in reversed(range(self.sliders_layout.count())):
            widget = self.sliders_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.slider_map.clear()
    
    def make_slider_handler(self, idx: int, label_widget: QLabel) -> Callable:
        """Create slider value changed handler"""
        def handler(val):
            param = self.dh_params[idx]
            if param.joint_type == 'revolute':
                param.theta_deg = float(val)
                label_widget.setText(f"{val:6.1f}°")
                self.table.item(idx, 0).setText(f"{val:.4f}")
            else:
                d_value = val / 1000.0
                param.d = d_value
                label_widget.setText(f"{val:6.1f}mm")
                self.table.item(idx, 1).setText(f"{d_value:.4f}")
            
            if self.auto_update_check.isChecked():
                self.on_update()
        
        return handler
    
    def on_inverse_kinematics(self):
        """Handle inverse kinematics computation"""
        if not self.read_table():
            return
        
        # Get target position from dialog
        dialog = IKTargetDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        target = dialog.get_target_matrix()
        
        self.show_progress()
        start_time = time.time()
        
        try:
            # Try analytical first if selected
            method = self.ik_method_combo.currentText()
            solution = None
            
            if "Analytical" in method:
                solution = self.kinematics.compute_inverse_kinematics_analytical(
                    self.dh_params,
                    target[:3, 3],
                    target[:3, :3] if target.shape == (4, 4) else None
                )
            
            if solution is None:
                # Use numerical method
                initial_guess = [p.theta_deg if p.joint_type == 'revolute' else p.d 
                               for p in self.dh_params]
                
                solution, info = self.kinematics.compute_inverse_kinematics_numerical(
                    self.dh_params,
                    target,
                    initial_guess
                )
            
            if solution:
                # Apply solution
                for i, val in enumerate(solution):
                    if self.dh_params[i].joint_type == 'revolute':
                        self.dh_params[i].theta_deg = val
                    else:
                        self.dh_params[i].d = val
                
                self.populate_table()
                self.on_update()
                
                elapsed = time.time() - start_time
                self.statusBar().showMessage(
                    f"✓ IK solved in {elapsed*1000:.1f}ms | {info.get('iterations', 0)} iterations"
                )
            else:
                QMessageBox.warning(self, "IK Failed", 
                                  "Could not find valid inverse kinematics solution.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"IK computation failed: {str(e)}")
        finally:
            self.hide_progress()
    
    def on_compute_workspace(self):
        """Compute and display workspace"""
        if not self.read_table():
            return
        
        self.show_progress()
        num_points = self.workspace_points_spin.value()
        
        try:
            self.statusBar().showMessage(f"Computing workspace with {num_points} points...")
            
            # Compute workspace
            points = self.kinematics.compute_workspace(self.dh_params, num_points)
            self.workspace_points = points
            
            # Update workspace visualization
            self.ws_ax.clear()
            self.ws_ax.set_facecolor('#1e1e2e')
            
            # Plot points with transparency
            self.ws_ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=points[:, 2],
                cmap='viridis',
                alpha=0.1,
                s=1,
                marker='.'
            )
            
            # Plot robot
            Ts, positions = self.kinematics.compute_forward_kinematics(self.dh_params)
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            zs = [p[2] for p in positions]
            self.ws_ax.plot(xs, ys, zs, 'r-', linewidth=3, label='Current Config')
            
            self.ws_ax.set_xlabel('X (m)')
            self.ws_ax.set_ylabel('Y (m)')
            self.ws_ax.set_zlabel('Z (m)')
            self.ws_ax.set_title(f'Workspace ({num_points} points)')
            self.ws_canvas.draw_idle()
            
            # Calculate workspace statistics
            ranges = np.ptp(points, axis=0)
            volume = np.prod(ranges)
            
            stats = "WORKSPACE STATISTICS:\n"
            stats += "=" * 60 + "\n"
            stats += f"Points sampled: {num_points}\n"
            stats += f"X range: [{points[:,0].min():.2f}, {points[:,0].max():.2f}] m\n"
            stats += f"Y range: [{points[:,1].min():.2f}, {points[:,1].max():.2f}] m\n"
            stats += f"Z range: [{points[:,2].min():.2f}, {points[:,2].max():.2f}] m\n"
            stats += f"Bounding box volume: {volume:.2f} m³\n"
            stats += f"Reach: {np.max(np.linalg.norm(points, axis=1)):.2f} m\n"
            
            self.ws_stats_text.setPlainText(stats)
            self.workspace_info.setText(
                f"Workspace: X∈[{ranges[0]:.1f},{ranges[0]+ranges[0]:.1f}] "
                f"Y∈[{ranges[1]:.1f},{ranges[1]+ranges[1]:.1f}] "
                f"Z∈[{ranges[2]:.1f},{ranges[2]+ranges[2]:.1f}]"
            )
            
            self.statusBar().showMessage("✓ Workspace computation complete")
            
        except Exception as e:
            self.statusBar().showMessage(f"✗ Workspace error: {str(e)}")
        finally:
            self.hide_progress()
    
    def on_check_singularities(self):
        """Check for kinematic singularities"""
        if not self.read_table():
            return
        
        try:
            result = self.kinematics.detect_singularities(self.dh_params)
            
            info_text = "SINGULARITY ANALYSIS:\n"
            info_text += "=" * 60 + "\n"
            info_text += f"Manipulability: {result['manipulability']:.6e}\n"
            info_text += f"Condition number: {result['condition_number']:.2f}\n"
            info_text += f"Dexterity: {result['dexterity']:.6f}\n"
            info_text += f"Singular: {'YES ⚠' if result['is_singular'] else 'NO ✓'}\n"
            info_text += f"Singular values: {result['singular_values']}\n"
            
            if result['null_space']:
                info_text += f"Null space direction: {result['null_space']}\n"
            
            self.singularity_info.setText(info_text)
            
            if result['is_singular']:
                QMessageBox.warning(self, "Singularity Detected",
                                  "Robot is in or near a singular configuration!\n"
                                  f"Condition number: {result['condition_number']:.1f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Singularity check failed: {str(e)}")
    
    def on_plan_trajectory(self):
        """Plan and execute trajectory"""
        if not self.read_table():
            return
        
        # Get trajectory parameters
        dialog = TrajectoryDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        start, end, waypoints, traj_type = dialog.get_parameters()
        
        self.show_progress()
        
        try:
            result = self.kinematics.plan_trajectory(
                self.dh_params,
                start,
                end,
                waypoints,
                traj_type
            )
            
            if result['success']:
                self.trajectory_data = result
                self.trajectory_index = 0
                
                # Start animation
                self.play_trajectory()
                
                self.statusBar().showMessage("✓ Trajectory planned successfully")
            else:
                QMessageBox.warning(self, "Planning Failed",
                                  "Could not plan trajectory to target point")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Trajectory planning failed: {str(e)}")
        finally:
            self.hide_progress()
    
    def play_trajectory(self):
        """Play the planned trajectory"""
        if self.trajectory_data is None:
            return
        
        if self.animation_timer:
            self.animation_timer.stop()
        
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_trajectory)
        self.animation_timer.start(self.animation_speed)
        
        self.statusBar().showMessage("▶ Playing trajectory...")
    
    def animate_trajectory(self):
        """Animate one step of the trajectory"""
        if self.trajectory_data is None:
            return
        
        joint_trajectories = self.trajectory_data['joint_trajectories']
        
        if self.trajectory_index >= len(joint_trajectories):
            self.stop_trajectory()
            return
        
        # Apply joint positions
        joint_values = joint_trajectories[self.trajectory_index]
        for i, val in enumerate(joint_values):
            if self.dh_params[i].joint_type == 'revolute':
                self.dh_params[i].theta_deg = val
            else:
                self.dh_params[i].d = val
        
        self.trajectory_index += 1
        self.populate_table()
        self.on_update()
    
    def stop_trajectory(self):
        """Stop trajectory animation"""
        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer = None
        
        self.trajectory_data = None
        self.trajectory_index = 0
        
        self.statusBar().showMessage("⏹ Trajectory stopped")
    
    def update_animation_speed(self, speed):
        """Update animation speed"""
        self.animation_speed = speed
        if self.animation_timer:
            self.animation_timer.setInterval(speed)
    
    def set_view(self, elev, azim):
        """Set 3D view angle"""
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()
    
    def load_preset(self, preset_name):
        """Load robot preset configuration"""
        presets = {
            "3-DOF Planar": [
                DHParameter(45, 0, 1.0, 0),
                DHParameter(-30, 0, 1.0, 0),
                DHParameter(15, 0, 0.8, 0),
            ],
            "4-DOF SCARA": [
                DHParameter(30, 0.3, 1.0, 0),
                DHParameter(-45, 0, 0.8, 0),
                DHParameter(0, 0, 0, 0, joint_type='prismatic'),
                DHParameter(0, 0, 0, 0),
            ],
            "6-DOF Industrial (PUMA 560)": [
                DHParameter(0, 0.67, 0, 90),
                DHParameter(-45, 0, 0.432, 0),
                DHParameter(0, 0.149, -0.02, -90),
                DHParameter(0, 0.432, 0, 90),
                DHParameter(0, 0, 0, -90),
                DHParameter(0, 0.056, 0, 0),
            ],
            "6-DOF Stanford": [
                DHParameter(0, 0.24, 0, -90),
                DHParameter(-90, 0.2, 0, 90),
                DHParameter(90, 0, 0, 0, joint_type='prismatic'),
                DHParameter(0, 0, 0, -90),
                DHParameter(0, 0, 0, 90),
                DHParameter(0, 0.1, 0, 0),
            ],
            "7-DOF Redundant": [
                DHParameter(0, 0.3, 0, 90),
                DHParameter(0, 0, 0.5, 0),
                DHParameter(0, 0, 0.5, 0),
                DHParameter(0, 0.2, 0, 90),
                DHParameter(0, 0, 0, -90),
                DHParameter(0, 0.1, 0, 90),
                DHParameter(0, 0.1, 0, 0),
            ],
            "Cylindrical": [
                DHParameter(0, 0.3, 0, 0),
                DHParameter(0, 0, 0, 0, joint_type='prismatic'),
                DHParameter(0, 0, 0, 0, joint_type='prismatic'),
            ],
            "Spherical": [
                DHParameter(0, 0, 0, 90),
                DHParameter(0, 0, 0, 90),
                DHParameter(0, 0, 0.5, 0, joint_type='prismatic'),
            ],
            "Delta Parallel": [
                DHParameter(0, 0.3, 0, 0),
                DHParameter(45, 0, 0.5, 0),
                DHParameter(45, 0, 0.5, 0),
                DHParameter(0, 0.2, 0, 0),
            ],
        }
        
        if preset_name == "Custom":
            return
        
        if preset_name in presets:
            self.dh_params = presets[preset_name]
            for param in self.dh_params:
                param.variable = True
        else:
            return
        
        self.spin_n.setValue(len(self.dh_params))
        self.populate_table()
        QTimer.singleShot(100, self.on_update)
        
        self.statusBar().showMessage(f"✓ Loaded {preset_name} configuration")
    
    def on_n_changed(self, val):
        """Handle joint count change"""
        n_old = len(self.dh_params)
        n_new = val
        
        if n_new > n_old:
            for _ in range(n_new - n_old):
                self.dh_params.append(DHParameter())
        elif n_new < n_old:
            self.dh_params = self.dh_params[:n_new]
        
        self.populate_table()
        self.joint_count_label.setText(f"Joints: {len(self.dh_params)}")
    
    def on_joint_type_changed(self, joint_type, row):
        """Handle joint type change"""
        if row < len(self.dh_params):
            self.dh_params[row].joint_type = joint_type
            self.populate_table()
    
    def on_reset(self):
        """Reset to default configuration"""
        n = self.spin_n.value()
        self.init_default_parameters(n)
        self.populate_table()
        self.on_update()
        self.statusBar().showMessage("✓ Parameters reset to defaults")
    
    def clear_results(self):
        """Clear all results"""
        if hasattr(self, 'results_text'):
            self.results_text.clear()
        self.code_output.clear()
        self.ws_stats_text.clear()
        self.ax.clear()
        self.canvas.draw_idle()
        self.statusBar().showMessage("✓ Results cleared")
    
    def save_configuration(self):
        """Save current configuration to file"""
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "robot_config.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_name:
            try:
                config = {
                    'robot_type': self.preset_combo.currentText(),
                    'n_joints': len(self.dh_params),
                    'dh_params': [p.to_dict() for p in self.dh_params],
                }
                
                with open(file_name, 'w') as f:
                    json.dump(config, f, indent=4)
                
                self.statusBar().showMessage(f"✓ Configuration saved to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))
    
    def load_configuration(self):
        """Load configuration from file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    config = json.load(f)
                
                self.dh_params = [
                    DHParameter(**p) for p in config['dh_params']
                ]
                
                self.spin_n.setValue(len(self.dh_params))
                self.populate_table()
                self.on_update()
                
                self.statusBar().showMessage(f"✓ Configuration loaded from {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))
    
    def export_data(self):
        """Export kinematics data"""
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            "kinematics_data.json",
            "JSON Files (*.json);;CSV Files (*.csv)"
        )
        
        if file_name:
            try:
                Ts, positions = self.kinematics.compute_forward_kinematics(self.dh_params)
                
                export_data = {
                    'dh_params': [p.to_dict() for p in self.dh_params],
                    'transformations': [T.tolist() for T in Ts],
                    'positions': [p.tolist() for p in positions],
                    'end_effector': positions[-1].tolist(),
                }
                
                with open(file_name, 'w') as f:
                    json.dump(export_data, f, indent=4)
                
                self.statusBar().showMessage(f"✓ Data exported to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
    
    def export_plot(self):
        """Export current plot"""
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "robot_plot.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        
        if file_name:
            try:
                self.fig.savefig(file_name, dpi=300, bbox_inches='tight',
                               facecolor=self.fig.get_facecolor())
                self.statusBar().showMessage(f"✓ Plot saved to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
    
    def generate_code(self):
        """Generate kinematics code"""
        lang = self.code_lang_combo.currentText()
        
        if "Python" in lang:
            code = self.generate_python_code()
        elif "C++" in lang:
            code = self.generate_cpp_code()
        elif "MATLAB" in lang:
            code = self.generate_matlab_code()
        elif "URDF" in lang:
            code = self.generate_urdf()
        else:
            code = "# Unsupported language"
        
        self.code_output.setPlainText(code)
        self.tabs.setCurrentIndex(4)  # Switch to code tab
    
    def generate_python_code(self) -> str:
        """Generate Python kinematics code"""
        code = "import numpy as np\n\n"
        code += "def dh_transform(theta, d, a, alpha):\n"
        code += "    '''Denavit-Hartenberg transformation matrix'''\n"
        code += "    ct = np.cos(theta)\n"
        code += "    st = np.sin(theta)\n"
        code += "    ca = np.cos(alpha)\n"
        code += "    sa = np.sin(alpha)\n"
        code += "    return np.array([\n"
        code += "        [ct, -st*ca, st*sa, a*ct],\n"
        code += "        [st, ct*ca, -ct*sa, a*st],\n"
        code += "        [0, sa, ca, d],\n"
        code += "        [0, 0, 0, 1]\n"
        code += "    ])\n\n"
        code += "def forward_kinematics(joint_angles):\n"
        code += "    '''Compute forward kinematics'''\n"
        code += "    # DH Parameters\n"
        code += "    dh_params = [\n"
        
        for param in self.dh_params:
            code += f"        # Joint: {param.joint_type}, "
            code += f"a={param.a}, d={param.d}, alpha={param.alpha_deg}°\n"
            code += f"        [joint_angles[0], {param.d}, {param.a}, np.radians({param.alpha_deg})],\n"
        
        code += "    ]\n\n"
        code += "    T = np.eye(4)\n"
        code += "    for params in dh_params:\n"
        code += "        T = T @ dh_transform(*params)\n"
        code += "    return T\n"
        
        return code
    
    def generate_cpp_code(self) -> str:
        """Generate C++ Eigen code"""
        code = "#include <Eigen/Dense>\n"
        code = "#include <cmath>\n\n"
        code = "Eigen::Matrix4d dh_transform(double theta, double d, double a, double alpha) {\n"
        code = "    Eigen::Matrix4d T;\n"
        code += "    T << cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta),\n"
        code += "         sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta),\n"
        code += "         0, sin(alpha), cos(alpha), d,\n"
        code += "         0, 0, 0, 1;\n"
        code += "    return T;\n"
        code += "}\n\n"
        code += "Eigen::Matrix4d forward_kinematics(const Eigen::VectorXd& joint_angles) {\n"
        code += "    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();\n"
        code += "    // Add DH computations here\n"
        code += "    return T;\n"
        code += "}\n"
        
        return code
    
    def generate_matlab_code(self) -> str:
        """Generate MATLAB code"""
        code = "function T = forward_kinematics(joint_angles)\n"
        code += "% Forward kinematics using DH parameters\n\n"
        code += "T = eye(4);\n\n"
        
        for i, param in enumerate(self.dh_params):
            code += f"% Joint {i+1}: {param.joint_type}\n"
            code += f"theta = joint_angles({i+1});\n"
            code += f"d = {param.d};\n"
            code += f"a = {param.a};\n"
            code += f"alpha = deg2rad({param.alpha_deg});\n\n"
            code += "Ti = [cos(theta) -sin(theta)*cos(alpha) sin(theta)*sin(alpha) a*cos(theta);\n"
            code += "      sin(theta) cos(theta)*cos(alpha) -cos(theta)*sin(alpha) a*sin(theta);\n"
            code += "      0 sin(alpha) cos(alpha) d;\n"
            code += "      0 0 0 1];\n\n"
            code += "T = T * Ti;\n\n"
        
        code += "end\n"
        return code
    
    def generate_urdf(self) -> str:
        """Generate URDF robot description"""
        urdf = '<?xml version="1.0"?>\n'
        urdf += '<robot name="robot">\n\n'
        urdf += '  <link name="base_link">\n'
        urdf += '    <visual>\n'
        urdf += '      <geometry><box size="0.1 0.1 0.1"/></geometry>\n'
        urdf += '    </visual>\n'
        urdf += '  </link>\n\n'
        
        for i, param in enumerate(self.dh_params):
            urdf += f'  <joint name="joint_{i+1}" type="{param.joint_type}">\n'
            urdf += f'    <parent link="link_{i}"/>\n' if i > 0 else '    <parent link="base_link"/>\n'
            urdf += f'    <child link="link_{i+1}"/>\n'
            urdf += f'    <origin xyz="{param.a} {0} {param.d}" rpy="{0} {0} {param.alpha_deg}"/>\n'
            urdf += '  </joint>\n\n'
            urdf += f'  <link name="link_{i+1}">\n'
            urdf += '    <visual>\n'
            urdf += '      <geometry><cylinder length="0.1" radius="0.02"/></geometry>\n'
            urdf += '    </visual>\n'
            urdf += '  </link>\n\n'
        
        urdf += '</robot>\n'
        return urdf
    
    def toggle_theme(self):
        """Toggle between dark and light theme"""
        self.color_scheme = 'light' if self.color_scheme == 'dark' else 'dark'
        self.apply_professional_theme()
        self.save_settings()
        self.statusBar().showMessage(f"✓ Theme changed to {self.color_scheme}")
    
    def toggle_auto_update(self, checked):
        """Toggle auto-update feature"""
        self.statusBar().showMessage(
            f"{'✓ Auto-update enabled' if checked else '⏸ Manual update mode'}"
        )
    
    def toggle_advanced_mode(self, checked):
        """Toggle advanced mode"""
        # Show/hide advanced columns in table
        for col in range(6, 10):
            self.table.setColumnHidden(col, not checked)
        
        self.statusBar().showMessage(
            f"{'Advanced mode' if checked else 'Basic mode'}"
        )
    
    def show_progress(self):
        """Show progress bar"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        QApplication.processEvents()
    
    def hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.setVisible(False)
    
    def on_plot_hover(self, event):
        """Handle mouse hover on plot"""
        if event.inaxes == self.ax:
            # Could show tooltips with coordinate info
            pass
    
    def on_plot_click(self, event):
        """Handle mouse click on plot for interactive IK"""
        if event.inaxes == self.ax and event.button == 1:  # Left click
            # Could implement interactive target selection
            pass
    
    def closeEvent(self, event):
        """Handle application close"""
        self.save_settings()
        event.accept()


class IKTargetDialog(QDialog):
    """Dialog for specifying IK target"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inverse Kinematics Target")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Position inputs
        pos_group = QGroupBox("Target Position")
        pos_layout = QGridLayout()
        
        pos_layout.addWidget(QLabel("X (m):"), 0, 0)
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-10, 10)
        self.x_spin.setValue(0.5)
        self.x_spin.setDecimals(4)
        pos_layout.addWidget(self.x_spin, 0, 1)
        
        pos_layout.addWidget(QLabel("Y (m):"), 1, 0)
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-10, 10)
        self.y_spin.setValue(0.5)
        self.y_spin.setDecimals(4)
        pos_layout.addWidget(self.y_spin, 1, 1)
        
        pos_layout.addWidget(QLabel("Z (m):"), 2, 0)
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(-10, 10)
        self.z_spin.setValue(0.5)
        self.z_spin.setDecimals(4)
        pos_layout.addWidget(self.z_spin, 2, 1)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Orientation inputs (optional)
        ori_group = QGroupBox("Target Orientation (Optional)")
        ori_layout = QGridLayout()
        
        ori_layout.addWidget(QLabel("Roll (°):"), 0, 0)
        self.roll_spin = QDoubleSpinBox()
        self.roll_spin.setRange(-180, 180)
        self.roll_spin.setValue(0)
        ori_layout.addWidget(self.roll_spin, 0, 1)
        
        ori_layout.addWidget(QLabel("Pitch (°):"), 1, 0)
        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(-90, 90)
        self.pitch_spin.setValue(0)
        ori_layout.addWidget(self.pitch_spin, 1, 1)
        
        ori_layout.addWidget(QLabel("Yaw (°):"), 2, 0)
        self.yaw_spin = QDoubleSpinBox()
        self.yaw_spin.setRange(-180, 180)
        self.yaw_spin.setValue(0)
        ori_layout.addWidget(self.yaw_spin, 2, 1)
        
        ori_group.setLayout(ori_layout)
        layout.addWidget(ori_group)
        
        # Use target position from current EE
        btn_current = QPushButton("Use Current EE Position")
        btn_current.clicked.connect(self.use_current_position)
        layout.addWidget(btn_current)
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def use_current_position(self):
        """Use current end-effector position as target"""
        parent = self.parent()
        if hasattr(parent, 'dh_params'):
            from kinematics_utils import compute_forward_kinematics
            Ts, positions = compute_forward_kinematics(parent.dh_params)
            if positions:
                ee = positions[-1]
                self.x_spin.setValue(ee[0])
                self.y_spin.setValue(ee[1])
                self.z_spin.setValue(ee[2])
    
    def get_target_matrix(self) -> np.ndarray:
        """Get target transformation matrix"""
        T = np.eye(4)
        T[0, 3] = self.x_spin.value()
        T[1, 3] = self.y_spin.value()
        T[2, 3] = self.z_spin.value()
        
        # Add orientation if specified
        roll = radians(self.roll_spin.value())
        pitch = radians(self.pitch_spin.value())
        yaw = radians(self.yaw_spin.value())
        
        if roll != 0 or pitch != 0 or yaw != 0:
            # RPY rotation matrix
            R_x = np.array([
                [1, 0, 0],
                [0, cos(roll), -sin(roll)],
                [0, sin(roll), cos(roll)]
            ])
            R_y = np.array([
                [cos(pitch), 0, sin(pitch)],
                [0, 1, 0],
                [-sin(pitch), 0, cos(pitch)]
            ])
            R_z = np.array([
                [cos(yaw), -sin(yaw), 0],
                [sin(yaw), cos(yaw), 0],
                [0, 0, 1]
            ])
            T[:3, :3] = R_z @ R_y @ R_x
        
        return T


class TrajectoryDialog(QDialog):
    """Dialog for trajectory planning parameters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trajectory Planning")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Start point
        start_group = QGroupBox("Start Point (current EE position)")
        start_layout = QGridLayout()
        
        for i, label in enumerate(['X:', 'Y:', 'Z:']):
            start_layout.addWidget(QLabel(label), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-10, 10)
            spin.setDecimals(4)
            spin.setValue(0.5 if i == 0 else (0.5 if i == 1 else 0.5))
            start_layout.addWidget(spin, i, 1)
            setattr(self, f'start_{label.lower()[0]}_spin', spin)
        
        start_group.setLayout(start_layout)
        layout.addWidget(start_group)
        
        # End point
        end_group = QGroupBox("End Point")
        end_layout = QGridLayout()
        
        for i, label in enumerate(['X:', 'Y:', 'Z:']):
            end_layout.addWidget(QLabel(label), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-10, 10)
            spin.setDecimals(4)
            spin.setValue(0.8 if i == 0 else (0.3 if i == 1 else 0.6))
            end_layout.addWidget(spin, i, 1)
            setattr(self, f'end_{label.lower()[0]}_spin', spin)
        
        end_group.setLayout(end_layout)
        layout.addWidget(end_group)
        
        # Parameters
        params_layout = QGridLayout()
        params_layout.addWidget(QLabel("Waypoints:"), 0, 0)
        self.waypoints_spin = QSpinBox()
        self.waypoints_spin.setRange(10, 500)
        self.waypoints_spin.setValue(50)
        params_layout.addWidget(self.waypoints_spin, 0, 1)
        
        params_layout.addWidget(QLabel("Type:"), 1, 0)
        self.traj_type = QComboBox()
        self.traj_type.addItems(['cubic', 'linear', 'trapezoidal'])
        params_layout.addWidget(self.traj_type, 1, 1)
        
        layout.addLayout(params_layout)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray, int, str]:
        """Get trajectory parameters"""
        start = np.array([
            self.start_x_spin.value(),
            self.start_y_spin.value(),
            self.start_z_spin.value()
        ])
        
        end = np.array([
            self.end_x_spin.value(),
            self.end_y_spin.value(),
            self.end_z_spin.value()
        ])
        
        return start, end, self.waypoints_spin.value(), self.traj_type.currentText()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set high DPI scaling
    app.setStyle('Fusion')
    
    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Create and show main window
    window = DHManipulatorGUI()
    window.show()
    
    sys.exit(app.exec())
