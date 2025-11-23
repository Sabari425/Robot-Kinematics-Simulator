import sys
import numpy as np
import math as m
import sympy as sp
from math import radians, atan2, degrees
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QSlider, QGroupBox, QFormLayout, QLineEdit, QTextEdit, QSizePolicy,
    QMessageBox, QComboBox, QDoubleSpinBox, QCheckBox, QSplitter, QFrame,
    QScrollArea, QGridLayout, QProgressBar, QHeaderView
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt



# =============================================================================
# ROBOT KINEMATICS OPERATIONS
# =============================================================================

def identity_matrix(n=4):
    id_matrix = [[0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                id_matrix[i][j] = 1
    return id_matrix


def matrix_multiplication(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        return None
    Result = [[0.0 for j in range(len(matrix2[0]))] for i in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix1[0])):
                Result[i][j] += matrix1[i][k] * matrix2[k][j]
    return Result


def trans(x=5.0, y=5.0, z=5.0):
    Trans = [[0.0 for j in range(4)] for i in range(4)]
    for i in range(4):
        for j in range(4):
            if i == j: Trans[i][j] = 1
    Trans[0][3], Trans[1][3], Trans[2][3] = x, y, z
    return Trans


def rot_x(x_alpha=90.0):
    Rot_x = [[0.0 for j in range(4)] for i in range(4)]
    Rot_x[0][0] = Rot_x[3][3] = 1
    Rot_x[1][1] = Rot_x[2][2] = m.cos(radians(x_alpha))
    Rot_x[1][2] = -m.sin(radians(x_alpha))
    Rot_x[2][1] = m.sin(radians(x_alpha))
    return Rot_x


def rot_y(y_beta=90.0):
    Rot_y = [[0.0 for j in range(4)] for i in range(4)]
    Rot_y[1][1] = Rot_y[3][3] = 1
    Rot_y[0][0] = Rot_y[2][2] = m.cos(radians(y_beta))
    Rot_y[2][0] = -m.sin(radians(y_beta))
    Rot_y[0][2] = m.sin(radians(y_beta))
    return Rot_y


def rot_z(z_gamma=90.0):
    Rot_z = [[0.0 for j in range(4)] for i in range(4)]
    Rot_z[2][2] = Rot_z[3][3] = 1
    Rot_z[0][0] = Rot_z[1][1] = m.cos(radians(z_gamma))
    Rot_z[0][1] = -m.sin(radians(z_gamma))
    Rot_z[1][0] = m.sin(radians(z_gamma))
    return Rot_z


def transpose_matrix(matrix):
    transpose = [[0.0 for j in range(len(matrix))] for i in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            transpose[j][i] = matrix[i][j]
    return transpose


def ordered_matrix_multiplication(t_list):
    if len(t_list) == 1:
        return t_list[0]
    result = matrix_multiplication(t_list[0], t_list[1])
    for i in range(2, len(t_list)):
        result = matrix_multiplication(result, t_list[i])
    return result


def inverse_htm_matrix(matrix):
    inverse = transpose_matrix(matrix)
    rm = [[0.0 for j in range(len(matrix[0]) - 1)] for i in range(len(matrix) - 1)]
    p = [[0.0 for j in range(1)] for i in range(len(matrix) - 1)]

    for i in range(len(matrix) - 1):
        for j in range(len(matrix[0]) - 1):
            rm[i][j] = -inverse[i][j]

    for i in range(len(matrix) - 1):
        p[i][0] = matrix[i][3]

    p_i = matrix_multiplication(rm, p)

    for i in range(len(matrix) - 1):
        inverse[i][3] = p_i[i][0]

    inverse[3] = [0.0, 0.0, 0.0, 1.0]
    return inverse


def dh_transform(theta_rad, d, a, alpha_rad):
    """Return 4x4 DH homogeneous transform given parameters (radians for angles)."""
    ct = np.cos(theta_rad)
    st = np.sin(theta_rad)
    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)
    T = np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=float)
    return T


from kinematics_utils import compute_forward_kinematics, matrix_to_pretty_string, dh_transform


def analyze_final_transformation_matrix(T_final):
    """Analyze the final DH transformation matrix and extract key parameters"""
    if isinstance(T_final, np.ndarray):
        T_final = T_final.tolist()

    analysis = "DH FINAL TRANSFORMATION MATRIX ANALYSIS\n"
    analysis += "=" * 60 + "\n\n"

    # Extract position
    x, y, z = T_final[0][3], T_final[1][3], T_final[2][3]
    analysis += f"END EFFECTOR POSITION:\n"
    analysis += f"  X = {x:.6f}\n"
    analysis += f"  Y = {y:.6f}\n"
    analysis += f"  Z = {z:.6f}\n"
    analysis += f"  Distance from base: {np.sqrt(x ** 2 + y ** 2 + z ** 2):.6f}\n\n"

    # Extract rotation matrix
    R = [row[:3] for row in T_final[:3]]
    analysis += "ROTATION MATRIX (3x3):\n"
    for i in range(3):
        analysis += f"  [{R[i][0]: .6f}  {R[i][1]: .6f}  {R[i][2]: .6f}]\n"
    analysis += "\n"

    # Extract orientation vectors
    n_vector = [R[0][0], R[1][0], R[2][0]]  # Normal vector
    o_vector = [R[0][1], R[1][1], R[2][1]]  # Orientation vector
    a_vector = [R[0][2], R[1][2], R[2][2]]  # Approach vector

    analysis += "ORIENTATION VECTORS:\n"
    analysis += f"  Normal Vector (n):     [{n_vector[0]: .6f}, {n_vector[1]: .6f}, {n_vector[2]: .6f}]\n"
    analysis += f"  Orientation Vector (o): [{o_vector[0]: .6f}, {o_vector[1]: .6f}, {o_vector[2]: .6f}]\n"
    analysis += f"  Approach Vector (a):    [{a_vector[0]: .6f}, {a_vector[1]: .6f}, {a_vector[2]: .6f}]\n\n"

    # Calculate angles from rotation matrix
    # Roll (X-axis rotation)
    roll = atan2(R[2][1], R[2][2])
    # Pitch (Y-axis rotation)
    pitch = atan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2))
    # Yaw (Z-axis rotation)
    yaw = atan2(R[1][0], R[0][0])

    analysis += "EULER ANGLES (ZYX convention):\n"
    analysis += f"  Roll (φ around X):  {degrees(roll):.4f}°\n"
    analysis += f"  Pitch (θ around Y): {degrees(pitch):.4f}°\n"
    analysis += f"  Yaw (ψ around Z):   {degrees(yaw):.4f}°\n\n"

    # Check orthogonality
    identity_approx = np.dot(R, np.transpose(R))
    orthogonality_error = np.max(np.abs(identity_approx - np.eye(3)))
    analysis += "MATRIX PROPERTIES:\n"
    analysis += f"  Orthogonality Error: {orthogonality_error:.2e}\n"
    analysis += f"  Determinant of R: {np.linalg.det(R):.6f}\n"
    analysis += "  ✓ Valid rotation matrix" if abs(np.linalg.det(R) - 1) < 1e-6 else "  ⚠ Non-valid rotation matrix"

    return analysis


def parameters_from_rpy_method(T_rpy):
    """Extract RPY parameters from transformation matrix"""
    if isinstance(T_rpy, np.ndarray):
        T_rpy = T_rpy.tolist()

    n_x, n_y, n_z = T_rpy[0][0], T_rpy[1][0], T_rpy[2][0]
    o_x, o_y = T_rpy[0][1], T_rpy[1][1]
    a_x, a_y = T_rpy[0][2], T_rpy[1][2]

    phi_a_1 = atan2(n_y, n_x)
    phi_o_1 = atan2(-n_z, ((n_x * m.cos(phi_a_1)) + (n_y * m.sin(phi_a_1))))
    phi_n_1 = atan2(((-a_y * m.cos(phi_a_1)) + (a_x * m.sin(phi_a_1))),
                    ((o_y * m.cos(phi_a_1)) - (o_x * m.sin(phi_a_1))))

    result = "ROLL-PITCH-YAW (RPY) ANALYSIS\n"
    result += "=" * 50 + "\n"
    result += f"Roll (φ_x):  {degrees(phi_a_1):.2f}°\n"
    result += f"Pitch (φ_y): {degrees(phi_o_1):.2f}°\n"
    result += f"Yaw (φ_z):   {degrees(phi_n_1):.2f}°\n"
    result += f"Position: X={T_rpy[0][3]:.3f}, Y={T_rpy[1][3]:.3f}, Z={T_rpy[2][3]:.3f}\n"
    return result


def parameters_from_euler_method(T_euler):
    """Extract Euler parameters from transformation matrix"""
    if isinstance(T_euler, np.ndarray):
        T_euler = T_euler.tolist()

    n_x, n_y = T_euler[0][0], T_euler[1][0]
    o_x, o_y = T_euler[0][1], T_euler[1][1]
    a_x, a_y, a_z = T_euler[0][2], T_euler[1][2], T_euler[2][2]

    phi = atan2(a_y, a_x)
    if phi < 0:
        phi = radians(degrees(phi) + 180)
    shai = atan2((-n_x * m.sin(phi) + n_y * m.cos(phi)),
                 (-o_x * m.sin(phi) + o_y * m.cos(phi)))
    theta = atan2((a_x * m.cos(phi) + a_y * m.sin(phi)), a_z)

    result = "EULER ANGLES ANALYSIS\n"
    result += "=" * 50 + "\n"
    result += f"φ (Precession):   {degrees(phi):.2f}°\n"
    result += f"θ (Nutation):     {degrees(theta):.2f}°\n"
    result += f"ψ (Spin):         {degrees(shai):.2f}°\n"
    result += f"Position: X={T_euler[0][3]:.3f}, Y={T_euler[1][3]:.3f}, Z={T_euler[2][3]:.3f}\n"
    return result


# =============================================================================
# ENHANCED GUI CLASS
# =============================================================================

class DHManipulatorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional Robot Kinematics Simulator")
        self.setMinimumSize(1400, 900)
        self.dh_table = []
        self.sliders = []
        self.slider_map = {}
        self.current_kinematics_results = ""
        self.init_default_table(n=3)
        self.init_ui()
        self.apply_professional_theme()

    def apply_professional_theme(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #404040;
                border: 2px solid #606060;
                padding: 10px 15px;
                border-radius: 6px;
                font-weight: bold;
                min-width: 100px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border: 2px solid #707070;
            }
            QPushButton:pressed {
                background-color: #505050;
            }
            QPushButton:disabled {
                background-color: #353535;
                color: #808080;
            }
            QTableWidget {
                background-color: #353535;
                gridline-color: #505050;
                border: 2px solid #505050;
                border-radius: 6px;
                font-size: 9pt;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #404040;
            }
            QTableWidget::item:selected {
                background-color: #505050;
            }
            QHeaderView::section {
                background-color: #404040;
                padding: 10px;
                border: 1px solid #505050;
                font-weight: bold;
                font-size: 9pt;
            }
            QTabWidget::pane {
                border: 2px solid #505050;
                border-radius: 8px;
                background-color: #353535;
            }
            QTabBar::tab {
                background-color: #404040;
                padding: 12px 20px;
                border: 2px solid #505050;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 3px;
                font-weight: bold;
                font-size: 10pt;
            }
            QTabBar::tab:selected {
                background-color: #505050;
                border-color: #707070;
            }
            QTabBar::tab:hover:!selected {
                background-color: #454545;
            }
            QGroupBox {
                border: 3px solid #606060;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                font-weight: bold;
                background-color: #323232;
                font-size: 10pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: #ffffff;
                font-size: 11pt;
            }
            QSlider::groove:horizontal {
                border: 2px solid #606060;
                height: 12px;
                background: #404040;
                margin: 3px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: #a0a0a0;
                border: 2px solid #707070;
                width: 24px;
                margin: -10px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #b0b0b0;
            }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                background-color: #404040;
                border: 2px solid #606060;
                padding: 8px;
                border-radius: 6px;
                color: white;
                min-height: 25px;
                font-size: 10pt;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
                border: 2px solid #707070;
            }
            QTextEdit {
                background-color: #353535;
                border: 2px solid #505050;
                border-radius: 6px;
                padding: 10px;
                font-size: 9pt;
            }
            QLabel {
                color: #e0e0e0;
                padding: 3px;
            }
            QLabel[title="true"] {
                font-size: 14pt;
                font-weight: bold;
                color: #ffffff;
                padding: 5px;
            }
            QLabel[subtitle="true"] {
                font-size: 11pt;
                font-weight: bold;
                color: #cccccc;
                padding: 3px;
            }
            QProgressBar {
                border: 2px solid #505050;
                border-radius: 6px;
                text-align: center;
                background-color: #353535;
                font-size: 9pt;
            }
            QProgressBar::chunk {
                background-color: #5050a0;
                border-radius: 4px;
            }
            QScrollArea {
                border: 2px solid #505050;
                border-radius: 6px;
                background-color: #353535;
            }
        """)

    def init_default_table(self, n=3):
        self.dh_table = []
        for i in range(n):
            self.dh_table.append({
                'theta_deg': 0.0,
                'd': 0.0,
                'a': 1.0,
                'alpha_deg': 0.0,
                'variable': True,
                'joint_type': 'revolute',
                'theta_min': -180,
                'theta_max': 180
            })

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: controls in a scroll area
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(450)
        left_scroll.setMaximumWidth(600)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(15, 15, 15, 15)

        # Title and description
        title = QLabel("Robot Kinematics Simulator")
        title.setProperty("title", True)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title)

        desc = QLabel("Complete DH Parameter-based forward kinematics analysis with real-time 3D visualization")
        desc.setWordWrap(True)
        desc.setProperty("subtitle", True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(desc)

        # Preset configurations
        preset_group = QGroupBox("Robot Configuration")
        preset_layout = QVBoxLayout()

        preset_combo_layout = QHBoxLayout()
        preset_combo_layout.addWidget(QLabel("Robot Type:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            ["Custom", "3-DOF Planar", "SCARA Robot", "Articulated (RRR)", "6-DOF Industrial", "Cylindrical",
             "Spherical"])
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        preset_combo_layout.addWidget(self.preset_combo)
        preset_layout.addLayout(preset_combo_layout)

        # Number of joints
        joints_layout = QHBoxLayout()
        joints_layout.addWidget(QLabel("Number of Joints:"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 12)
        self.spin_n.setValue(len(self.dh_table))
        self.spin_n.valueChanged.connect(self.on_n_changed)
        joints_layout.addWidget(self.spin_n)
        joints_layout.addStretch()
        preset_layout.addLayout(joints_layout)

        preset_group.setLayout(preset_layout)
        left_layout.addWidget(preset_group)

        # DH Parameters Table
        table_group = QGroupBox("DH Parameters Configuration")
        table_layout = QVBoxLayout()

        table_instructions = QLabel("Configure DH parameters. Set 'Variable' to 1 for interactive joint control.")
        table_instructions.setWordWrap(True)
        table_layout.addWidget(table_instructions)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["θ (deg)", "d", "a", "α (deg)", "Variable", "Joint Type", "Min", "Max"])
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        # Increase row height and font size
        self.table.verticalHeader().setDefaultSectionSize(40)
        font = QFont()
        font.setPointSize(11)
        self.table.setFont(font)
        # Set minimum width for each column
        min_widths = [80, 60, 60, 80, 70, 100, 70, 70]
        for i, width in enumerate(min_widths):
            self.table.setColumnWidth(i, width)
        # Make headers bold
        header_font = QFont()
        header_font.setPointSize(11)
        header_font.setBold(True)
        self.table.horizontalHeader().setFont(header_font)
        self.populate_table()
        table_layout.addWidget(self.table)

        table_group.setLayout(table_layout)
        left_layout.addWidget(table_group)

        # Control Buttons
        control_group = QGroupBox("Simulation Controls")
        control_layout = QGridLayout()

        self.btn_update = QPushButton("Compute Forward Kinematics")
        self.btn_update.clicked.connect(self.on_update)
        control_layout.addWidget(self.btn_update, 0, 0, 1, 2)

        self.btn_reset = QPushButton("Reset Parameters")
        self.btn_reset.clicked.connect(self.on_reset)
        control_layout.addWidget(self.btn_reset, 1, 0)

        self.btn_clear = QPushButton("Clear Results")
        self.btn_clear.clicked.connect(self.clear_results)
        control_layout.addWidget(self.btn_clear, 1, 1)

        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)

        # Joint Control Sliders
        self.sliders_group = QGroupBox("Real-time Joint Control")
        self.sliders_layout = QVBoxLayout()
        self.sliders_group.setLayout(self.sliders_layout)
        left_layout.addWidget(self.sliders_group)

        # Status and Information
        info_group = QGroupBox("Robot Status")
        info_layout = QVBoxLayout()

        self.status_label = QLabel("Ready to compute forward kinematics")
        self.status_label.setWordWrap(True)
        info_layout.addWidget(self.status_label)

        # End effector position
        self.ee_label = QLabel("End Effector: X = 0.000, Y = 0.000, Z = 0.000")
        self.ee_label.setWordWrap(True)
        info_layout.addWidget(self.ee_label)

        # Workspace information
        self.workspace_label = QLabel("Workspace: Calculating...")
        self.workspace_label.setWordWrap(True)
        info_layout.addWidget(self.workspace_label)

        # Progress indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        info_layout.addWidget(self.progress_bar)

        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)

        # Right: Visualization and Results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)
        right_layout.setContentsMargins(15, 15, 15, 15)

        self.tabs = QTabWidget()

        # Tab 1: 3D Visualization
        tab_plot = QWidget()
        tp_layout = QVBoxLayout(tab_plot)

        # Plot controls
        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("View Configuration:"))

        self.btn_view_xy = QPushButton("XY Plane")
        self.btn_view_xy.clicked.connect(lambda: self.set_view(0, 0))
        plot_controls.addWidget(self.btn_view_xy)

        self.btn_view_xz = QPushButton("XZ Plane")
        self.btn_view_xz.clicked.connect(lambda: self.set_view(0, 90))
        plot_controls.addWidget(self.btn_view_xz)

        self.btn_view_yz = QPushButton("YZ Plane")
        self.btn_view_yz.clicked.connect(lambda: self.set_view(90, 0))
        plot_controls.addWidget(self.btn_view_yz)

        self.btn_view_iso = QPushButton("Isometric View")
        self.btn_view_iso.clicked.connect(lambda: self.set_view(30, 45))
        plot_controls.addWidget(self.btn_view_iso)

        self.btn_view_reset = QPushButton("Reset View")
        self.btn_view_reset.clicked.connect(lambda: self.set_view(25, 45))
        plot_controls.addWidget(self.btn_view_reset)

        plot_controls.addStretch()
        tp_layout.addLayout(plot_controls)

        # The plot itself
        plot_frame = QFrame()
        plot_frame.setFrameStyle(QFrame.Shape.Box)
        plot_frame.setStyleSheet("background-color: #353535; border: 2px solid #505050; border-radius: 8px;")
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(10, 10, 10, 10)

        self.fig = Figure(figsize=(10, 8), facecolor='#2b2b2b')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_box_aspect([1, 1, 1])
        # Center the plot and set equal aspect ratio
        self.ax.set_position([0.1, 0.1, 0.8, 0.8])
        self.ax.set_proj_type('ortho')
        # Set initial view limits
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        # Add grid and make it more visible
        self.ax.grid(True, linestyle='--', alpha=0.5)
        # Make axis labels more visible
        self.ax.set_xlabel('X', fontsize=12, labelpad=10)
        self.ax.set_ylabel('Y', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Z', fontsize=12, labelpad=10)
        self.canvas.mpl_connect("motion_notify_event", self.on_plot_hover)
        plot_layout.addWidget(self.canvas)

        tp_layout.addWidget(plot_frame)
        self.tabs.addTab(tab_plot, "3D Visualization")

        # Tab 2: Kinematics Analysis
        tab_results = QWidget()
        tr_layout = QVBoxLayout(tab_results)

        results_controls = QHBoxLayout()
        results_controls.addWidget(QLabel("Kinematics Analysis Tools:"))

        self.btn_rpy_analysis = QPushButton("RPY Analysis")
        self.btn_rpy_analysis.clicked.connect(self.perform_rpy_analysis)
        results_controls.addWidget(self.btn_rpy_analysis)

        self.btn_euler_analysis = QPushButton("Euler Analysis")
        self.btn_euler_analysis.clicked.connect(self.perform_euler_analysis)
        results_controls.addWidget(self.btn_euler_analysis)

        self.btn_jacobian = QPushButton("Jacobian Matrix")
        self.btn_jacobian.clicked.connect(self.calculate_jacobian)
        results_controls.addWidget(self.btn_jacobian)

        self.btn_dh_analysis = QPushButton("DH Final Analysis")
        self.btn_dh_analysis.clicked.connect(self.perform_dh_final_analysis)
        results_controls.addWidget(self.btn_dh_analysis)

        results_controls.addStretch()
        tr_layout.addLayout(results_controls)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

        # Use monospace font for matrix display
        text_font = QFont()
        text_font.setFamily("Consolas")
        text_font.setStyleHint(QFont.StyleHint.Monospace)
        text_font.setPointSize(9)
        self.results_text.setFont(text_font)

        tr_layout.addWidget(self.results_text)
        self.tabs.addTab(tab_results, "Kinematics Analysis")

        # Tab 3: Transformation Matrices
        tab_matrices = QWidget()
        tm_layout = QVBoxLayout(tab_matrices)

        matrices_label = QLabel("Transformation Matrices (T0_i):")
        matrices_label.setProperty("subtitle", True)
        tm_layout.addWidget(matrices_label)

        self.matrices_text = QTextEdit()
        self.matrices_text.setReadOnly(True)
        self.matrices_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.matrices_text.setFont(text_font)
        tm_layout.addWidget(self.matrices_text)

        self.tabs.addTab(tab_matrices, "Transformation Matrices")

        # Tab 4: DH Final Transformation
        tab_dh_final = QWidget()
        tdf_layout = QVBoxLayout(tab_dh_final)

        dh_final_label = QLabel("DH Final Transformation Matrix Analysis")
        dh_final_label.setProperty("subtitle", True)
        tdf_layout.addWidget(dh_final_label)

        self.dh_final_text = QTextEdit()
        self.dh_final_text.setReadOnly(True)
        self.dh_final_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.dh_final_text.setFont(text_font)
        tdf_layout.addWidget(self.dh_final_text)

        self.tabs.addTab(tab_dh_final, "DH Final Transformation")

        right_layout.addWidget(self.tabs)

        # Add widgets to splitter
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setSizes([500, 900])

        main_layout.addWidget(splitter)

        # Perform initial computation
        QTimer.singleShot(100, self.on_update)

    def populate_table(self):
        self.table.setRowCount(len(self.dh_table))
        for i, row in enumerate(self.dh_table):
            titem = QTableWidgetItem(f"{row['theta_deg']:.3f}")
            ditem = QTableWidgetItem(f"{row['d']:.3f}")
            aitem = QTableWidgetItem(f"{row['a']:.3f}")
            aalpha = QTableWidgetItem(f"{row['alpha_deg']:.3f}")
            var = QTableWidgetItem("1" if row.get('variable', False) else "0")

            joint_type_combo = QComboBox()
            joint_type_combo.addItems(['revolute', 'prismatic'])
            joint_type_combo.setCurrentText(row.get('joint_type', 'revolute'))
            joint_type_combo.currentTextChanged.connect(lambda text, row=i: self.on_joint_type_changed(text, row))

            if row.get('joint_type', 'revolute') == 'revolute':
                min_value = row.get('theta_min', -180)
                max_value = row.get('theta_max', 180)
            else:
                min_value = row.get('d_min', -0.5)
                max_value = row.get('d_max', 0.5)

            min_item = QTableWidgetItem(str(min_value))
            max_item = QTableWidgetItem(str(max_value))

            self.table.setItem(i, 0, titem)
            self.table.setItem(i, 1, ditem)
            self.table.setItem(i, 2, aitem)
            self.table.setItem(i, 3, aalpha)
            self.table.setItem(i, 4, var)
            self.table.setCellWidget(i, 5, joint_type_combo)
            self.table.setItem(i, 6, min_item)
            self.table.setItem(i, 7, max_item)

    def on_joint_type_changed(self, joint_type, row):
        if joint_type == 'revolute':
            min_value = -180
            max_value = 180
        else:  # prismatic
            min_value = -0.5
            max_value = 0.5
        
        self.table.item(row, 6).setText(str(min_value))
        self.table.item(row, 7).setText(str(max_value))
        self.on_update()

    def read_table(self):
        new_table = []
        for i in range(self.table.rowCount()):
            try:
                theta = float(self.table.item(i, 0).text())
                d = float(self.table.item(i, 1).text())
                a = float(self.table.item(i, 2).text())
                alpha = float(self.table.item(i, 3).text())
                var = int(float(self.table.item(i, 4).text())) != 0
                joint_type = self.table.cellWidget(i, 5).currentText()
                min_val = float(self.table.item(i, 6).text())
                max_val = float(self.table.item(i, 7).text())
                
                params = {
                    'theta_deg': theta,
                    'd': d,
                    'a': a,
                    'alpha_deg': alpha,
                    'variable': var,
                    'joint_type': joint_type
                }
                
                if joint_type == 'revolute':
                    params.update({
                        'theta_min': min_val,
                        'theta_max': max_val
                    })
                else:  # prismatic
                    params.update({
                        'd_min': min_val,
                        'd_max': max_val
                    })
                
                new_table.append(params)
            except (ValueError, AttributeError) as e:
                print(f"Error reading table row {i}: {e}")
                return False

        self.dh_table = new_table
        return True

    def on_n_changed(self, val):
        n_old = len(self.dh_table)
        n_new = val
        if n_new > n_old:
            for _ in range(n_new - n_old):
                self.dh_table.append({
                    'theta_deg': 0.0,
                    'd': 0.0,
                    'a': 1.0,
                    'alpha_deg': 0.0,
                    'variable': True,
                    'joint_type': 'revolute',
                    'theta_min': -180,
                    'theta_max': 180
                })
        elif n_new < n_old:
            self.dh_table = self.dh_table[:n_new]
        self.populate_table()
        self.status_label.setText(f"Configuration updated to {n_new} joints")

    def on_reset(self):
        n = self.spin_n.value()
        self.init_default_table(n)
        self.populate_table()
        self.on_update()
        self.status_label.setText("Parameters reset to defaults")

    def clear_sliders(self):
        for i in reversed(range(self.sliders_layout.count())):
            widget = self.sliders_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.slider_map.clear()

    def create_sliders_for_variables(self):
        self.clear_sliders()
        for i, row in enumerate(self.dh_table):
            if row.get('variable', False):
                # Create a horizontal layout for each slider
                slider_container = QWidget()
                slider_layout = QHBoxLayout(slider_container)
                slider_layout.setContentsMargins(0, 5, 0, 5)

                # Set label and range based on joint type
                if row['joint_type'] == 'revolute':
                    label = QLabel(f"Joint {i + 1} (θ):")
                    min_val = int(row.get('theta_min', -180))
                    max_val = int(row.get('theta_max', 180))
                    current_val = int(row['theta_deg'])
                    suffix = '°'
                else:  # prismatic
                    label = QLabel(f"Joint {i + 1} (d):")
                    min_val = int(row.get('d_min', -0.5) * 1000)
                    max_val = int(row.get('d_max', 0.5) * 1000)
                    current_val = int(row['d'] * 1000)
                    suffix = 'mm'

                label.setMinimumWidth(80)
                slider_layout.addWidget(label)

                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(min_val, max_val)
                slider.setValue(current_val)

                value_label = QLabel(f"{current_val:6.1f}{suffix}")
                value_label.setMinimumWidth(60)
                value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
                slider.valueChanged.connect(self.make_slider_handler(i, value_label))

                slider_layout.addWidget(slider, 1)
                slider_layout.addWidget(value_label)

                self.sliders_layout.addWidget(slider_container)
                self.slider_map[i] = (slider, value_label)

    def make_slider_handler(self, idx, label_widget):
        def handler(val):
            if self.dh_table[idx]['joint_type'] == 'revolute':
                self.dh_table[idx]['theta_deg'] = float(val)
                label_widget.setText(f"{val:6.1f}°")
                self.table.item(idx, 0).setText(f"{val:.3f}")
            else:  # prismatic
                d_value = val / 1000.0  # Convert from millimeters to meters
                self.dh_table[idx]['d'] = d_value
                label_widget.setText(f"{val:6.1f}mm")
                self.table.item(idx, 1).setText(f"{d_value:.3f}")
            self.on_update(plot_only=True)

        return handler

    def on_update(self, plot_only=False):
        if not plot_only:
            self.read_table()

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(20)
        QApplication.processEvents()

        try:
            Ts, positions = compute_forward_kinematics(self.dh_table)
            self.status_label.setText("Forward kinematics computed successfully")
            self.progress_bar.setValue(60)
        except Exception as e:
            self.status_label.setText(f"Computation error: {str(e)}")
            self.progress_bar.setVisible(False)
            return

        # Update results
        self.update_results_text(Ts, positions)
        self.progress_bar.setValue(80)

        if not plot_only:
            self.create_sliders_for_variables()

        self.plot_manipulator(positions, Ts)
        self.progress_bar.setValue(100)

        # Update status information
        ee_pos = positions[-1]
        self.ee_label.setText(f"End Effector Position:\nX = {ee_pos[0]:.3f}, Y = {ee_pos[1]:.3f}, Z = {ee_pos[2]:.3f}")

        # Calculate workspace information
        all_positions = np.array(positions)
        workspace_volume = np.ptp(all_positions, axis=0)
        self.workspace_label.setText(
            f"Workspace Volume:\nX-range: {workspace_volume[0]:.2f}, Y-range: {workspace_volume[1]:.2f}, Z-range: {workspace_volume[2]:.2f}")

        # Update DH final transformation analysis
        if Ts:
            self.update_dh_final_analysis(Ts[-1])

        # Hide progress bar after a delay
        QTimer.singleShot(800, lambda: self.progress_bar.setVisible(False))

    def update_results_text(self, Ts, positions):
        txt_lines = []
        txt_lines.append("=" * 80 + "\n")
        txt_lines.append("ROBOT KINEMATICS ANALYSIS REPORT\n")
        txt_lines.append("=" * 80 + "\n\n")

        txt_lines.append("DH PARAMETERS CONFIGURATION:\n")
        txt_lines.append("-" * 60 + "\n")
        txt_lines.append(f"{'Joint':>5}  {'θ(deg)':>10}  {'d':>8}  {'a':>8}  {'α(deg)':>10}  {'Ctrl':>6}\n")
        txt_lines.append("-" * 60 + "\n")
        for i, row in enumerate(self.dh_table):
            txt_lines.append(
                f"{i + 1:>5}  {row['theta_deg']:>10.4f}  {row['d']:>8.4f}  {row['a']:>8.4f}  {row['alpha_deg']:>10.4f}  {'Yes' if row['variable'] else 'No':>6}\n")

        txt_lines.append("\n" + "=" * 80 + "\n")
        txt_lines.append("JOINT POSITIONS (World Coordinates):\n")
        txt_lines.append("-" * 60 + "\n")
        for i, p in enumerate(positions):
            marker = " ⭐ END EFFECTOR" if i == len(positions) - 1 else ""
            txt_lines.append(f"P{i}: X={p[0]:.4f}, Y={p[1]:.4f}, Z={p[2]:.4f}{marker}\n")

        self.current_kinematics_results = "".join(txt_lines)
        self.results_text.setPlainText(self.current_kinematics_results)

        # Update matrices tab
        matrices_text = "TRANSFORMATION MATRICES (T0_i):\n" + "=" * 60 + "\n\n"
        for i, T in enumerate(Ts):
            matrices_text += f"T0_{i + 1} (Base → Joint {i + 1}):\n"
            matrices_text += matrix_to_pretty_string(T) + "\n\n"

        # Add final transformation matrix
        if Ts:
            matrices_text += "=" * 60 + "\n"
            matrices_text += "FINAL TRANSFORMATION MATRIX (T0_EE):\n"
            matrices_text += "=" * 60 + "\n\n"
            matrices_text += matrix_to_pretty_string(Ts[-1])

        self.matrices_text.setPlainText(matrices_text)

    def update_dh_final_analysis(self, T_final):
        """Update the DH Final Transformation analysis tab"""
        analysis = analyze_final_transformation_matrix(T_final)
        self.dh_final_text.setPlainText(analysis)

    def perform_dh_final_analysis(self):
        """Perform detailed analysis of the final DH transformation matrix"""
        try:
            Ts, positions = compute_forward_kinematics(self.dh_table)
            if not Ts:
                self.status_label.setText("No transformation matrices available")
                return

            final_matrix = Ts[-1]
            analysis = analyze_final_transformation_matrix(final_matrix)

            # Switch to DH Final Transformation tab and show analysis
            self.tabs.setCurrentIndex(3)  # Switch to DH Final Transformation tab
            self.dh_final_text.setPlainText(analysis)
            self.status_label.setText("DH Final Transformation analysis completed")

        except Exception as e:
            self.status_label.setText(f"DH Final analysis failed: {str(e)}")

    def plot_manipulator(self, positions, Ts):
        self.ax.clear()

        # Enhanced plot styling
        self.ax.set_facecolor('#2b2b2b')
        self.ax.grid(True, color='#505050', linestyle='--', alpha=0.6)
        self.ax.tick_params(colors='white', labelsize=9)

        # Calculate appropriate limits with margin
        all_positions = np.array(positions)
        if len(all_positions) > 0:
            max_range = np.max(np.abs(all_positions)) * 1.3
            limit = max(2.0, max_range)
        else:
            limit = 2.0

        self.ax.set_xlim3d(-limit, limit)
        self.ax.set_ylim3d(-limit, limit)
        self.ax.set_zlim3d(-limit, limit)

        # Plot links with increased thickness (25% more than before)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]

        # Main robot links - 25% thicker
        self.ax.plot(xs, ys, zs, 'o-', linewidth=5.0, markersize=10,
                     color='#4285F4', markerfacecolor='lightblue',
                     markeredgecolor='white', markeredgewidth=1.5, zorder=5)

        # Draw coordinate frames at each joint with optimal thickness
        frame_length = limit * 0.18
        colors = ['red', 'green', 'blue']
        axis_labels = ['X', 'Y', 'Z']

        for i, T in enumerate(Ts):
            origin = T[:3, 3]

            # Draw coordinate axes with optimal line thickness
            for j in range(3):
                axis = T[:3, j] * frame_length
                self.ax.quiver(*origin, *axis, color=colors[j], linewidth=2.2,
                               arrow_length_ratio=0.15, alpha=0.9, zorder=10)

                # Label axes with proper spacing
                label_pos = origin + axis * 1.25
                self.ax.text(*label_pos, axis_labels[j], color=colors[j],
                             fontsize=4, fontweight='bold', zorder=15)

            # Label joints with proper spacing above the joint
            joint_label_pos = origin + [0, 0, frame_length * 0.4]
            self.ax.text(*joint_label_pos, f"J{i + 1}", fontsize=5, color='white',
                         fontweight='bold', ha='center', va='bottom',
                         bbox=dict(boxstyle="round,pad=0.4", facecolor='#505050',
                                   alpha=0.8, edgecolor='white'), zorder=20)

        # Base frame with enhanced styling
        base_text_pos = [0, 0, -frame_length * 0.6]
        self.ax.text(*base_text_pos, "BASE", fontsize=7, color='white',
                     fontweight='bold', ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='red',
                               alpha=0.9, edgecolor='white'), zorder=20)

        # End effector with enhanced visibility
        ee = positions[-1]
        self.ax.scatter([ee[0]], [ee[1]], [ee[2]], color='red', s=170,
                        label='End Effector', edgecolors='white', linewidth=2.5, zorder=25)

        # Enhanced axis labels
        self.ax.set_xlabel('X Axis', fontweight='bold', color='white', fontsize=11, labelpad=10)
        self.ax.set_ylabel('Y Axis', fontweight='bold', color='white', fontsize=11, labelpad=10)
        self.ax.set_zlabel('Z Axis', fontweight='bold', color='white', fontsize=11, labelpad=10)

        # Professional title
        self.ax.set_title('Robot Manipulator - 3D Kinematics Visualization',
                          fontsize=13, fontweight='bold', color='white', pad=20)

        # Enhanced legend
        self.ax.legend(loc='upper left', bbox_to_anchor=(0, 1),
                       facecolor='#404040', edgecolor='white', fontsize=10)

        # Smooth rendering
        self.canvas.draw_idle()

    def set_view(self, elev, azim):
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()
        view_names = {
            (0, 0): "XY Plane (Top View)",
            (0, 90): "XZ Plane (Front View)",
            (90, 0): "YZ Plane (Side View)",
            (30, 45): "Isometric View",
            (25, 45): "Default View"
        }
        view_name = view_names.get((elev, azim), "Custom View")
        self.status_label.setText(f"View changed to: {view_name}")

    def load_preset(self, preset_name):
        if preset_name == "Custom":
            return

        self.status_label.setText(f"Loading {preset_name} configuration...")

        if preset_name == "3-DOF Planar":
            self.dh_table = [
                {'theta_deg': 45.0, 'd': 0.0, 'a': 1.0, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': -30.0, 'd': 0.0, 'a': 1.0, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': 15.0, 'd': 0.0, 'a': 0.8, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180}
            ]
        elif preset_name == "SCARA Robot":
            self.dh_table = [
                {'theta_deg': 30.0, 'd': 0.3, 'a': 1.0, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': -45.0, 'd': 0.0, 'a': 0.8, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': 0.0, 'd': 0.0, 'a': 0.0, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'prismatic', 'd_min': -0.5, 'd_max': 0.0}
            ]
        elif preset_name == "Articulated (RRR)":
            self.dh_table = [
                {'theta_deg': 45.0, 'd': 0.2, 'a': 0.0, 'alpha_deg': 90.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': -30.0, 'd': 0.0, 'a': 1.0, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': 15.0, 'd': 0.0, 'a': 0.8, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180}
            ]
        elif preset_name == "Cylindrical":
            self.dh_table = [
                {'theta_deg': 0.0, 'd': 0.3, 'a': 0.0, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': 0.0, 'd': 0.0, 'a': 0.0, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'prismatic', 'd_min': 0.0, 'd_max': 0.5},
                {'theta_deg': 0.0, 'd': 0.0, 'a': 0.0, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'prismatic', 'd_min': 0.0, 'd_max': 0.3}
            ]
        elif preset_name == "Spherical":
            self.dh_table = [
                {'theta_deg': 0.0, 'd': 0.0, 'a': 0.0, 'alpha_deg': 90.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': 0.0, 'd': 0.0, 'a': 0.0, 'alpha_deg': 90.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -90, 'theta_max': 90},
                {'theta_deg': 0.0, 'd': 0.0, 'a': 0.5, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'prismatic', 'd_min': 0.0, 'd_max': 0.5}
            ]
        elif preset_name == "6-DOF Industrial":
            self.dh_table = [
                {'theta_deg': 30.0, 'd': 0.3, 'a': 0.0, 'alpha_deg': 90.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': -45.0, 'd': 0.0, 'a': 1.0, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': 60.0, 'd': 0.0, 'a': 0.5, 'alpha_deg': 90.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': -30.0, 'd': 0.4, 'a': 0.0, 'alpha_deg': -90.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': 45.0, 'd': 0.0, 'a': 0.0, 'alpha_deg': 90.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180},
                {'theta_deg': 0.0, 'd': 0.1, 'a': 0.0, 'alpha_deg': 0.0, 'variable': True, 'joint_type': 'revolute', 'theta_min': -180, 'theta_max': 180}
            ]

        self.spin_n.setValue(len(self.dh_table))
        self.populate_table()
        QTimer.singleShot(100, self.on_update)

    def perform_rpy_analysis(self):
        try:
            Ts, positions = compute_forward_kinematics(self.dh_table)
            if not Ts:
                self.status_label.setText("No transformation matrices available")
                return

            end_effector_matrix = Ts[-1]
            rpy_analysis = parameters_from_rpy_method(end_effector_matrix)

            current_text = self.results_text.toPlainText()
            self.results_text.setPlainText(current_text + "\n\n" + "=" * 60 + "\n" + rpy_analysis)
            self.status_label.setText("RPY analysis completed successfully")

        except Exception as e:
            self.status_label.setText(f"RPY analysis failed: {str(e)}")

    def perform_euler_analysis(self):
        try:
            Ts, positions = compute_forward_kinematics(self.dh_table)
            if not Ts:
                self.status_label.setText("No transformation matrices available")
                return

            end_effector_matrix = Ts[-1]
            euler_analysis = parameters_from_euler_method(end_effector_matrix)

            current_text = self.results_text.toPlainText()
            self.results_text.setPlainText(current_text + "\n\n" + "=" * 60 + "\n" + euler_analysis)
            self.status_label.setText("Euler angles analysis completed")

        except Exception as e:
            self.status_label.setText(f"Euler analysis failed: {str(e)}")

    def calculate_jacobian(self):
        # Placeholder for Jacobian calculation
        try:
            Ts, positions = compute_forward_kinematics(self.dh_table)
            jacobian_text = "JACOBIAN MATRIX ANALYSIS\n" + "=" * 50 + "\n"
            jacobian_text += "Jacobian matrix calculation requires symbolic computation.\n"
            jacobian_text += "This feature will be implemented in future versions.\n"
            jacobian_text += "Current implementation focuses on forward kinematics.\n"

            current_text = self.results_text.toPlainText()
            self.results_text.setPlainText(current_text + "\n\n" + jacobian_text)
            self.status_label.setText("Jacobian analysis placeholder added")

        except Exception as e:
            self.status_label.setText(f"Jacobian analysis failed: {str(e)}")

    def clear_results(self):
        self.results_text.clear()
        self.matrices_text.clear()
        self.dh_final_text.clear()
        self.status_label.setText("All results cleared")

    def on_plot_hover(self, event):
        # Optional: Add tooltip or coordinate display on hover
        if event.inaxes == self.ax:
            pass  # Could implement coordinate display


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    gui = DHManipulatorGUI()
    gui.show()

    sys.exit(app.exec())
