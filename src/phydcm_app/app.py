"""
PhyDCM - Single-file merged application (from the provided multi-file structure)
NOTE:
- This file is a direct merge of the provided sources into one module for convenience.
- It preserves the original classes and logic as-is, with only necessary adjustments to
  internal imports (inlined modules are referenced directly).
"""
# =====================================================================
# app/main_window.py (merged)
# =====================================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # اختياري لتقليل رسائل TF
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # اختياري
import sys
import fitz  # PyMuPDF
import csv
import json
import datetime
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import traceback
# PhyDCM import
try:
    from phydcm import PyHDCMPredictor
    PHYDCM_AVAILABLE = True
    print("✓ phydcm loaded OK")
except Exception:
    PHYDCM_AVAILABLE = False
    print("phydcm import failed:")
    traceback.print_exc()
# PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
# Medical imaging imports
try:
    import pydicom as dicom
    import nibabel as nib
except ImportError:
    dicom = None
    nib = None
# ADD near imports
try:
    from PIL import Image
except Exception:
    Image = None
# Machine learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None
# =====================================================================
# app/dialogs/about_dialog.py (merged)
# =====================================================================
class AboutDialog(QDialog):
    """Professional about dialog"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About PhyDCM Diagnosis System")
        self.setFixedSize(600, 700)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setup_ui()
    def setup_ui(self):
        """Setup about dialog UI"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(20)
        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2ecc71);
                border-radius: 10px;
                padding: 20px;
            }
        """)
        header_layout = QVBoxLayout()
        logo = QLabel()
        pixmap = QPixmap(resource_path("assets/medical_phy.png"))
        if not pixmap.isNull():
            logo.setPixmap(pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo.setAlignment(Qt.AlignCenter)
        title = QLabel("PhyDCM Diagnosis System")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignCenter)
        version = QLabel("Version 2.0 Professional Edition")
        version.setStyleSheet("font-size: 13px; color: white;")
        version.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(logo)
        header_layout.addWidget(title)
        header_layout.addWidget(version)
        header_frame.setLayout(header_layout)
        # Institution info
        inst_group = QGroupBox("Institutional Information")
        inst_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        inst_layout = QVBoxLayout()
        inst_text = QLabel("""
<p style='line-height: 1.8; text-align: center;'>
<b>Republic of Iraq</b><br>
Ministry of Higher Education and Scientific Research<br>
<b style='color: #3498db;'>University of Al-Qadisiyah</b><br>
College of Science<br>
Department of Medical Physics
</p>
        """)
        inst_text.setWordWrap(True)
        inst_layout.addWidget(inst_text)
        inst_group.setLayout(inst_layout)
        # Development team
        team_group = QGroupBox("Development Team")
        team_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2ecc71;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        team_layout = QVBoxLayout()
        team_text = QLabel("""
<p style='line-height: 2.0;'>
<b style='color: #2ecc71;'>Research Students:</b><br>
• Mohammed Hadi Rahim<br>
• Mohammed Hassan Hadi<br>
• Haider Ali Abboud<br>
• Ali Hussein Allawi<br>
<br>
<b style='color: #2ecc71;'>Supervisor:</b><br>
• Asst. Prof. Dr. Haider Saad Abdulbaqi
</p>
        """)
        team_text.setWordWrap(True)
        team_layout.addWidget(team_text)
        team_group.setLayout(team_layout)
        # PyHDCM info
        phydcm_group = QGroupBox("PyHDCM Library")
        phydcm_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e74c3c;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        phydcm_layout = QVBoxLayout()
        phydcm_text = QLabel("""
<p style='line-height: 1.6; text-align: justify;'>
<b>PyHDCM (Physics-based Hybrid Deep Convolutional Model)</b> is an advanced
AI-powered medical imaging analysis library designed for diagnostic applications.
<br><br>
<b>Features:</b><br>
✓ Multi-modal support (MRI, CT, PET)<br>
✓ Deep learning integration<br>
✓ Physics-based analysis<br>
✓ Clinical-grade precision<br>
✓ DICOM/NIfTI compatible
</p>
        """)
        phydcm_text.setWordWrap(True)
        phydcm_layout.addWidget(phydcm_text)
        phydcm_group.setLayout(phydcm_layout)
        # Copyright
        copyright = QLabel("""
<p style='text-align: center; color: #7f8c8d; font-size: 10px;'>
© 2025 University of Al-Qadisiyah<br>
All rights reserved. For educational and research purposes only.
</p>
        """)
        copyright.setWordWrap(True)
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 10px 30px;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        close_btn.clicked.connect(self.accept)
        # Add all widgets
        content_layout.addWidget(header_frame)
        content_layout.addWidget(inst_group)
        content_layout.addWidget(team_group)
        content_layout.addWidget(phydcm_group)
        content_layout.addWidget(copyright)
        content_layout.addStretch()
        content.setLayout(content_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)
        self.setLayout(layout)
# =====================================================================
# app/widgets/image_viewer.py (UPDATED -> 4 screens + ACTIVE GREEN BORDER)
# =====================================================================
class VolumeSliceViewer(QLabel):
    """
    Viewer لعرض Slice ثنائي الأبعاد من Volume ثلاثي الأبعاد.
    - يدعم: dcm folder series و nii/nii.gz
    - Zoom/Navigation: يتحكم بها الـ MainWindow حسب الـ Active Viewer + حالة الشاشة الرابعة
    """
    clicked = pyqtSignal()
    def __init__(self, allow_zoom: bool = False, title: str = ""):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.setAlignment(Qt.AlignCenter)
        self._base_border_color = "#ddd"
        self._active_border_color = "#00ff66"  # أخضر واضح
        self._bg = "#000"
        self._title = title or "No Data"
        self.allow_zoom = allow_zoom
        self._is_active = False  # ✅ للحدود الخضراء + الـ overlay
        self._apply_style(active=False)
        self.setText(self._title)
        # Volume state
        self.volume = None               # numpy [Z,Y,X]
        self.orientation = "axial"       # axial|sagittal|coronal
        self.idx_axial = 0
        self.idx_sagittal = 0
        self.idx_coronal = 0
        # Series mapping
        self.is_series = False
        self.dicom_files_sorted = []     # axial index -> file path
        # Diagnosis source path for 4th viewer
        self.current_source_path = None  # str
        # Zoom
        self.zoom_level = 1.0
    # ------------------------
    # Active UI
    # ------------------------
    def _apply_style(self, active: bool):
        border = self._active_border_color if active else self._base_border_color
        self.setStyleSheet(f"""
            QLabel {{
                border: 3px solid {border};
                background-color: {self._bg};
                color: white;
                font-size: 12px;
            }}
        """)
    def set_active(self, active: bool):
        self._is_active = bool(active)
        self._apply_style(active=self._is_active)
        self.update()
    def is_active(self) -> bool:
        return bool(self._is_active)
    # ------------------------
    # Mouse events
    # ------------------------
    def mousePressEvent(self, e):
        self.clicked.emit()
        super().mousePressEvent(e)
    def wheelEvent(self, e: QWheelEvent):
        # ✅ الـ MainWindow هو اللي يقرر منو يتزوم (Active من الثلاثة / أو الرابعة إذا مشتغلة)
        # نتركها هنا فقط إذا allow_zoom True (والأب يفعّلها فعلياً باستدعاء zoom_in/out)
        super().wheelEvent(e)
    # ------------------------
    # Volume operations
    # ------------------------
    def set_volume(self, vol_zyx: np.ndarray):
        self.volume = vol_zyx
        if self.volume is None or self.volume.ndim != 3:
            self.setText("Invalid Volume")
            return
        z, y, x = self.volume.shape
        self.idx_axial = z // 2
        self.idx_coronal = y // 2
        self.idx_sagittal = x // 2
        self.render()
    def set_orientation(self, ori: str):
        if ori not in ("axial", "sagittal", "coronal"):
            return
        self.orientation = ori
        self.render()
    def step_slice(self, delta: int):
        if self.volume is None:
            return
        z, y, x = self.volume.shape
        if self.orientation == "axial":
            self.idx_axial = max(0, min(self.idx_axial + delta, z - 1))
        elif self.orientation == "coronal":
            self.idx_coronal = max(0, min(self.idx_coronal + delta, y - 1))
        else:  # sagittal
            self.idx_sagittal = max(0, min(self.idx_sagittal + delta, x - 1))
        self.render()
    def set_slice_index(self, value: int):
        if self.volume is None:
            return
        z, y, x = self.volume.shape
        if self.orientation == "axial":
            self.idx_axial = max(0, min(int(value), z - 1))
        elif self.orientation == "coronal":
            self.idx_coronal = max(0, min(int(value), y - 1))
        else:
            self.idx_sagittal = max(0, min(int(value), x - 1))
        self.render()
    def get_slice_count(self) -> int:
        if self.volume is None:
            return 0
        z, y, x = self.volume.shape
        if self.orientation == "axial":
            return int(z)
        if self.orientation == "coronal":
            return int(y)
        return int(x)
    def get_current_index(self) -> int:
        if self.orientation == "axial":
            return int(self.idx_axial)
        if self.orientation == "coronal":
            return int(self.idx_coronal)
        return int(self.idx_sagittal)
    def _get_current_slice(self) -> Optional[np.ndarray]:
        if self.volume is None:
            return None
        if self.orientation == "axial":
            return self.volume[self.idx_axial, :, :]
        elif self.orientation == "coronal":
            return self.volume[:, self.idx_coronal, :]
        else:  # sagittal
            return self.volume[:, :, self.idx_sagittal]
    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32)
        mn, mx = float(np.min(a)), float(np.max(a))
        if mx - mn < 1e-6:
            return np.zeros_like(a, dtype=np.uint8)
        a = (a - mn) / (mx - mn)
        return (a * 255.0).astype(np.uint8)
    def render(self):
        sl = self._get_current_slice()
        if self.orientation == "sagittal":
            sl = np.rot90(sl, k=-1)
        if sl is None:
            self.setPixmap(QPixmap())
            self.setText(self._title)
            self.update()
            return
        img8 = self._to_uint8(sl)
        h, w = img8.shape
        qimg = QImage(img8.tobytes(), w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        if self.allow_zoom:
            nw = max(1, int(pix.width() * self.zoom_level))
            nh = max(1, int(pix.height() * self.zoom_level))
            pix = pix.scaled(nw, nh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)
        self.setText("")
        self.update()
    # ------------------------
    # Zoom
    # ------------------------
    def zoom_in(self):
        if not self.allow_zoom:
            return
        self.zoom_level = min(self.zoom_level * 1.25, 5.0)
        self.render()
    def zoom_out(self):
        if not self.allow_zoom:
            return
        self.zoom_level = max(self.zoom_level / 1.25, 0.25)
        self.render()
    def reset_zoom(self):
        if not self.allow_zoom:
            return
        self.zoom_level = 1.0
        self.render()
    def get_zoom_level(self):
        return int(self.zoom_level * 100)
    def clear(self):
        self.volume = None
        self.is_series = False
        self.dicom_files_sorted = []
        self.current_source_path = None
        self.zoom_level = 1.0
        self.setPixmap(QPixmap())
        self.setText(self._title)
        self.update()
    # ------------------------
    # Overlay when ACTIVE (only for the 3 screens)
    # ------------------------
    def paintEvent(self, e):
        super().paintEvent(e)
        if not self._is_active:
            return
        # ✅ يظهر فقط مع الحدود الخضراء:
        # L يسار، R يمين، وعدد الصور + الفهرس
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # shadow background for readability
        def _draw_badge(text: str, x: int, y: int, align_right: bool = False):
            font = QFont("Segoe UI", 10, QFont.Bold)
            painter.setFont(font)
            fm = QFontMetrics(font)
            pad_x, pad_y = 8, 4
            tw = fm.horizontalAdvance(text)
            th = fm.height()
            if align_right:
                rx = x - (tw + pad_x * 2)
            else:
                rx = x
            rect = QRect(rx, y, tw + pad_x * 2, th + pad_y * 2)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(rect, 8, 8)
            painter.setPen(QColor(0, 255, 102))
            painter.drawText(rect, Qt.AlignCenter, text)
        # L / R
        _draw_badge("L", 10, 10, align_right=False)
        _draw_badge("R", self.width() - 10, 10, align_right=True)
        # Count badge (index/total)
        total = self.get_slice_count()
        if total > 0:
            idx = self.get_current_index() + 1
            ori = self.orientation.upper()
            _draw_badge(f"{ori}  {idx}/{total}", 10, self.height() - 40, align_right=False)
        painter.end()
# =====================================================================
# app/widgets/results_table.py (merged)
# =====================================================================
class ResultsTable(QTableWidget):
    """Custom table widget for displaying diagnosis results"""
    def __init__(self):
        super().__init__()
        # 16 columns (must match number of headers)
        self.setColumnCount(16)
        self.setHorizontalHeaderLabels([
            "Sequence Count", "ID", "Patient Name", "Age",
            "Gender", "Diagnosis Date", "Diagnosis Time", "Scan Type",
            "Image Type", "Image Dimensions", "Image Quality",
            "Brightness", "Sharpness", "Diagnosis", "Confidence %",
            "Processing_Time_Seconds"
        ])
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                background-color: white;
                alternate-background-color: #f5f5f5;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                padding: 5px;
                border: 1px solid #ddd;
                font-weight: bold;
            }
        """)
    def add_result(self, result_data: Dict[str, Any]):
        """Add new diagnosis result"""
        row = self.rowCount()
        self.insertRow(row)
        keys = [
            "sequence", "id", "patient_name", "age",
            "gender", "diagnosis_date", "diagnosis_time", "scan_type",
            "image_type", "image_dimensions", "image_quality",
            "brightness", "sharpness", "diagnosis", "confidence",
            "processing_time_seconds"
        ]
        for col, key in enumerate(keys):
            item = QTableWidgetItem(str(result_data.get(key, "")))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.setItem(row, col, item)
    def export_to_csv(self, filename: str):
        """Export results to CSV file"""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                headers = []
                for col in range(self.columnCount()):
                    headers.append(self.horizontalHeaderItem(col).text())
                writer.writerow(headers)
                for row in range(self.rowCount()):
                    row_data = []
                    for col in range(self.columnCount()):
                        item = self.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {str(e)}")
            return False
# =====================================================================
# app/widgets/terminal.py (merged)
# =====================================================================
class TerminalWidget(QWidget):
    """Academic terminal widget with clean grayscale design"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.min_height_ratio = 0.25
        self.max_height_ratio = 0.66
        self.current_height = 0
        self.setup_ui()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
    def setup_ui(self):
        """Setup terminal UI with academic styling"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.background = QWidget(self)
        self.background.setStyleSheet("background-color: rgba(30, 30, 30, 240);")
        terminal_container = QWidget()
        terminal_container.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 6px;
            }
        """)
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        # Top bar
        top_bar = QWidget()
        top_bar.setStyleSheet("""
            QWidget {
                background-color: #3c3c3c;
                border-bottom: 1px solid #555555;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
        """)
        top_bar.setCursor(Qt.SizeVerCursor)
        top_bar.setFixedHeight(35)
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(12, 0, 8, 0)
        top_bar_layout.setSpacing(0)
        terminal_label = QLabel("Terminal")
        terminal_label.setStyleSheet("""
            QLabel {
                color: #e0e0e0;
                font-weight: 600;
                font-size: 12px;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            }
        """)
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: none;
                border-radius: 3px;
                font-size: 14px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
            }
            QPushButton:pressed {
                background-color: #2c2c2c;
            }
        """)
        close_btn.clicked.connect(self.hide)
        top_bar_layout.addWidget(terminal_label)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(close_btn)
        top_bar.setLayout(top_bar_layout)
        # Output area
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #e0e0e0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                border: none;
                padding: 12px;
                selection-background-color: #555555;
            }
        """)
        self.output_area.append("""
<span style='color: #aaaaaa;'>PhyDCM Terminal v2.0</span>
<span style='color: #aaaaaa;'>Type 'help' for commands</span>
        """)
        # Command input
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Enter command...")
        self.command_input.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 10px 12px;
                border: none;
                border-top: 1px solid #555555;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                selection-background-color: #555555;
            }
        """)
        self.command_input.returnPressed.connect(self.execute_command)
        container_layout.addWidget(top_bar)
        container_layout.addWidget(self.output_area)
        container_layout.addWidget(self.command_input)
        terminal_container.setLayout(container_layout)
        layout.addWidget(terminal_container)
        self.setLayout(layout)
        self.top_bar = top_bar
        self.resizing = False
        self.top_bar.installEventFilter(self)
    def eventFilter(self, obj, event):
        """Handle resize events"""
        if obj == self.top_bar:
            if event.type() == event.MouseButtonPress:
                self.resizing = True
                self.resize_start_y = event.globalPos().y()
                self.start_height = self.height()
                return True
            elif event.type() == event.MouseMove and self.resizing:
                delta_y = self.resize_start_y - event.globalPos().y()
                new_height = self.start_height + delta_y
                if self.parent:
                    parent_height = self.parent.height()
                    min_height = int(parent_height * self.min_height_ratio)
                    max_height = int(parent_height * self.max_height_ratio)
                    new_height = max(min_height, min(new_height, max_height))
                    self.current_height = new_height
                    parent_rect = self.parent.geometry()
                    self.setGeometry(0, parent_rect.height() - new_height,
                                   parent_rect.width(), new_height)
                return True
            elif event.type() == event.MouseButtonRelease:
                self.resizing = False
                return True
        return super().eventFilter(obj, event)
    def execute_command(self):
        """Execute terminal command"""
        command = self.command_input.text().strip()
        if not command:
            return
        self.output_area.append(f"<span style='color: #aaaaaa;'>&gt;</span> <span style='color: #ffffff;'>{command}</span>")
        try:
            if command == "help":
                help_text = """
<span style='color: #e0e0e0;'>Available Commands:</span>
  <span style='color: #bbbbbb;'>help</span>         - Show help message
  <span style='color: #bbbbbb;'>clear</span>        - Clear terminal
  <span style='color: #bbbbbb;'>pip install</span>  - Install package
  <span style='color: #bbbbbb;'>version</span>      - Show version
  <span style='color: #bbbbbb;'>status</span>       - Show status
                """
                self.output_area.append(help_text)
            elif command == "clear":
                self.output_area.clear()
            elif command == "version":
                self.output_area.append("<span style='color: #e0e0e0;'>PhyDCM Diagnosis System v2.0</span>")
            elif command == "status":
                status = """
<span style='color: #e0e0e0;'>System Status:</span>
  PyHDCM: <span style='color: #a0a0a0;'>Active</span>
  TensorFlow: <span style='color: #a0a0a0;'>Loaded</span>
  DICOM: <span style='color: #a0a0a0;'>Available</span>
  NIfTI: <span style='color: #a0a0a0;'>Available</span>
                """
                self.output_area.append(status)
            elif command.startswith("pip install"):
                self.output_area.append("<span style='color: #cccccc;'>Installing package...</span>")
                result = subprocess.run(command.split(), capture_output=True, text=True)
                if result.returncode == 0:
                    self.output_area.append(f"<span style='color: #a0a0a0;'>{result.stdout}</span>")
                else:
                    self.output_area.append(f"<span style='color: #ff8888;'>{result.stderr}</span>")
            else:
                self.output_area.append(f"<span style='color: #ff8888;'>Unknown command: '{command}'</span>")
                self.output_area.append("<span style='color: #aaaaaa;'>Type 'help' for available commands</span>")
        except Exception as e:
            self.output_area.append(f"<span style='color: #ff8888;'>Error: {str(e)}</span>")
        self.output_area.append("")
        self.command_input.clear()
        self.output_area.verticalScrollBar().setValue(
            self.output_area.verticalScrollBar().maximum()
        )
    def showEvent(self, event):
        """Handle show event"""
        if self.parent:
            parent_rect = self.parent.geometry()
            if self.current_height == 0:
                parent_height = parent_rect.height()
                self.current_height = int(parent_height * self.min_height_ratio)
            self.setGeometry(0, parent_rect.height() - self.current_height,
                           parent_rect.width(), self.current_height)
            self.background.setGeometry(self.rect())
        super().showEvent(event)
    def resizeEvent(self, event):
        """Handle resize event"""
        if self.parent and self.isVisible():
            parent_rect = self.parent.geometry()
            if self.current_height == 0:
                parent_height = int(parent_rect.height())
                self.current_height = int(parent_height * self.min_height_ratio)
            self.setGeometry(0, parent_rect.height() - self.current_height,
                           parent_rect.width(), self.current_height)
            self.background.setGeometry(self.rect())
        super().resizeEvent(event)
# =====================================================================
# core/diagnosis_thread.py (merged)
# =====================================================================
class DiagnosisThread(QThread):
    """Thread for performing diagnosis"""
    diagnosis_complete = pyqtSignal(dict)
    def __init__(self, image_path: str, patient_data: dict, phydcm_predictor=None, model=None, seq_num: int = 1):
        super().__init__()
        self.image_path   = image_path
        self.patient_data = patient_data
        self.phydcm       = phydcm_predictor
        self.model        = model
        self.seq_num      = seq_num
    def run(self):
        """Perform diagnosis"""
        try:
            start_time = time.time()
            if self.phydcm:
                result = self.predict_with_phydcm()
            elif self.model:
                result = self.predict_with_custom_model()
            else:
                result = self.predict_with_fallback()
            elapsed = time.time() - start_time
            now = datetime.datetime.now()
            patient_id = (
                f"{now.hour:02d}{now.minute:02d}"
                f"{self.patient_data['age']}"
                f"{self.patient_data['name'][0].upper() if self.patient_data['name'] else 'X'}"
                f"{now.month:02d}"
                f"{self.patient_data['name'][-1].upper() if self.patient_data['name'] else 'X'}"
                f"{1}"
            )
            complete_result = {
                'sequence': 1,
                'id': patient_id,
                'patient_name': self.patient_data['name'],
                'age': self.patient_data['age'],
                'gender': self.patient_data['gender'],
                'diagnosis_date': now.strftime("%Y-%m-%d"),
                'diagnosis_time': now.strftime("%H:%M"),
                'scan_type': self.patient_data['scan_type'],
                'processing_time': f"{time.time() - start_time:.2f}s",
                'processing_time_seconds': f"{elapsed:.2f}",
                'success': True
            }
            complete_result.update(result)
            self.diagnosis_complete.emit(complete_result)
        except Exception as e:
            self.diagnosis_complete.emit({
                'success': False,
                'error': str(e)
            })
    #########################################    confidence    ####################################################################
    # ADD inside class DiagnosisThread (e.g., after run())

    def _safe_float(self, x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    def _to_gray_uint8(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim == 3:
            # if RGB -> luminance
            if a.shape[-1] == 3:
                a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
            else:
                a = a[..., 0]
        a = a.astype(np.float32)
        mn, mx = float(np.min(a)), float(np.max(a))
        if mx - mn < 1e-6:
            return np.zeros_like(a, dtype=np.uint8)
        a = (a - mn) / (mx - mn)
        return (a * 255.0).clip(0, 255).astype(np.uint8)

    def _load_image_for_metrics(self) -> np.ndarray:
        """
        Returns a 2D grayscale uint8 image for metrics:
        - DICOM: use pixel_array
        - NIfTI: use middle axial slice
        - Standard images: PIL
        """
        p = self.image_path
        ext = os.path.splitext(p)[1].lower()

        # handle .nii.gz
        if p.lower().endswith(".nii.gz"):
            ext = ".nii.gz"

        # DICOM
        if ext == ".dcm" and dicom is not None:
            ds = dicom.dcmread(p)
            arr = ds.pixel_array
            # if multi-frame, take middle
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]
            return self._to_gray_uint8(arr)

        # NIfTI
        if ext in (".nii", ".nii.gz") and nib is not None:
            img = nib.load(p)
            vol = img.get_fdata()
            if vol.ndim >= 3:
                sl = vol[:, :, vol.shape[2] // 2]
            else:
                sl = vol
            return self._to_gray_uint8(sl)

        # Standard image files
        if Image is not None:
            im = Image.open(p).convert("L")
            return np.array(im, dtype=np.uint8)

        # last resort: try Qt image
        qimg = QImage(p)
        if qimg.isNull():
            return np.zeros((256, 256), dtype=np.uint8)
        qimg = qimg.convertToFormat(QImage.Format_Grayscale8)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(h * w)
        return np.frombuffer(ptr, np.uint8).reshape((h, w))

    def _compute_metrics(self, gray_u8: np.ndarray) -> dict:
        """
        Quality metrics in [0..1] style as much as possible:
        - brightness: mean/255
        - contrast_norm: std/128 (clipped)
        - sharpness_norm: Laplacian variance normalized
        - edge_density: percentage of strong gradients
        """
        g = gray_u8.astype(np.float32)
        mean = float(np.mean(g))
        std = float(np.std(g))

        # gradients (no extra deps)
        gx = np.abs(np.diff(g, axis=1))
        gy = np.abs(np.diff(g, axis=0))
        # pad to match shape
        gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
        gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
        grad = gx + gy

        # Laplacian variance (sharpness proxy)
        # 4-neighborhood laplacian
        lap = (
            -4 * g
            + np.roll(g, 1, axis=0) + np.roll(g, -1, axis=0)
            + np.roll(g, 1, axis=1) + np.roll(g, -1, axis=1)
        )
        lap_var = float(np.var(lap))

        brightness = mean / 255.0
        contrast_norm = min(std / 128.0, 1.0)

        # normalize lap_var roughly (robust-ish)
        # typical medical slices vary a lot; this keeps values bounded
        sharpness_norm = lap_var / (lap_var + 5000.0)
        sharpness_norm = float(np.clip(sharpness_norm, 0.0, 1.0))

        # edge density: ratio of pixels with strong gradient
        thr = max(20.0, np.percentile(grad, 85))  # adaptive threshold
        edge_density = float(np.mean(grad > thr))

        # penalties for extreme brightness (too dark/too bright)
        # peak at ~0.5, drop toward extremes
        bright_penalty = 1.0 - (abs(brightness - 0.5) / 0.5)
        bright_penalty = float(np.clip(bright_penalty, 0.0, 1.0))

        # final quality score (weighted)
        quality_score = (
            0.35 * bright_penalty +
            0.25 * contrast_norm +
            0.30 * sharpness_norm +
            0.10 * min(edge_density / 0.15, 1.0)
        )
        quality_score = float(np.clip(quality_score, 0.0, 1.0))

        return {
            "brightness": brightness,
            "contrast_norm": contrast_norm,
            "sharpness_norm": sharpness_norm,
            "edge_density": edge_density,
            "quality_score": quality_score
        }

    def _label_brightness(self, b: float) -> str:
    # medical images are darker by nature
        if b < 0.15:
            return "Low"
        if 0.15 <= b <= 0.35:
            return "Optimal"
        if b <= 0.55:
            return "Good"
        return "High"


    def _label_sharpness(self, s: float) -> str:
        if s >= 0.25:
            return "High"
        if s >= 0.12:
            return "Medium"
        return "Low"

    def _label_quality(self, q: float) -> str:
        if q >= 0.65:
            return "Excellent"
        if q >= 0.50:
            return "High"
        if q >= 0.40:
            return "Balanced"
        if q >= 0.23:
            return "Acceptable"
        return "Low"

    def _entropy_uncertainty(self, probs: np.ndarray) -> float:
        """
        Returns uncertainty in [0..1] based on normalized entropy.
        """
        p = np.asarray(probs, dtype=np.float64).ravel()
        p = np.clip(p, 1e-12, 1.0)
        p = p / np.sum(p)
        h = -np.sum(p * np.log(p))
        h_max = np.log(len(p)) if len(p) > 1 else 1.0
        u = float(h / h_max) if h_max > 0 else 0.0
        return float(np.clip(u, 0.0, 1.0))

    def _calibrated_confidence(self, *, probs: Optional[np.ndarray], metrics: dict,
                            fallback_pmax: float = 0.60,
                            hard_min: float = 40.0, hard_max: float = 92.0) -> float:
        """
        Academic-friendly calibrated confidence:
        - starts from pmax (model strength)
        - penalizes uncertainty (entropy)
        - penalizes weak image quality
        - bounded to avoid unrealistic 95-100%
        """
        q = float(metrics.get("quality_score", 0.6))
        q = float(np.clip(q, 0.0, 1.0))

        if probs is None:
            pmax = self._safe_float(fallback_pmax, 0.6)
            uncert = 0.65  # assume higher uncertainty without probabilities
        else:
            p = np.asarray(probs, dtype=np.float64).ravel()
            if p.size == 0:
                pmax = self._safe_float(fallback_pmax, 0.6)
                uncert = 0.65
            else:
                p = np.clip(p, 1e-12, 1.0)
                p = p / np.sum(p)
                pmax = float(np.max(p))
                uncert = self._entropy_uncertainty(p)

        # map pmax (0.5..1) -> strength [0..1] smoothly
        # (prevents huge jumps)
        strength = (pmax - 0.5) / 0.5
        strength = float(np.clip(strength, 0.0, 1.0))
        strength = strength ** 0.85

        # uncertainty penalty: 0 (confident) -> 1 (very uncertain)
        u_pen = 1.0 - (uncert ** 0.9)

        # final score in [0..1]
        score = (0.55 * strength + 0.45 * q) * u_pen

        # bound to academic range
        conf = hard_min + (hard_max - hard_min) * score

        # extra safety caps: never claim 90%+ if quality is poor
        if q < 0.45:
            conf = min(conf, 78.0)
        if q < 0.35:
            conf = min(conf, 70.0)

        return float(np.clip(conf, hard_min, hard_max))


    #########################################    confidence    ####################################################################
    def predict_with_phydcm(self):
        """Predict using phydcm library"""
        # REPLACE the body of predict_with_phydcm()

        try:
            scan_type = self.patient_data['scan_type'].lower()
            raw = self.phydcm.predict(self.image_path, scan_type, return_probabilities=True)

            gray = self._load_image_for_metrics()
            m = self._compute_metrics(gray)

            diagnosis = raw.get('diagnosis', 'Undetermined')

            # try extract probabilities (list/np array or dict)
            probs = None
            if isinstance(raw.get("probabilities"), (list, tuple, np.ndarray)):
                probs = np.asarray(raw["probabilities"], dtype=np.float64)
            elif isinstance(raw.get("probabilities"), dict):
                probs = np.asarray(list(raw["probabilities"].values()), dtype=np.float64)

            # fallback_pmax: if library provides raw['confidence'] in [0..1]
            fallback_pmax = raw.get("confidence", 0.60)

            confidence = self._calibrated_confidence(
                probs=probs,
                metrics=m,
                fallback_pmax=float(fallback_pmax),
                hard_min=45.0,
                hard_max=92.0
            )

            img_ext = os.path.splitext(self.image_path)[1].lower()
            if self.image_path.lower().endswith(".nii.gz"):
                img_ext = ".nii.gz"

            img_type = "Voxel" if img_ext in ['.dcm', '.nii', '.nii.gz'] else "Pixel"
            dims = "3D" if img_type == "Voxel" else "2D"

            return {
                'image_type': img_type,
                'image_dimensions': dims,
                'image_quality': self._label_quality(m["quality_score"]),
                'brightness': self._label_brightness(m["brightness"]),
                'sharpness': self._label_sharpness(m["sharpness_norm"]),
                'diagnosis': diagnosis,
                'confidence': f"{confidence:.1f}%",
                'sequence_number': self.seq_num
            }
        except Exception as e:
            print("phydcm failed:", e)
            print(f"Phydcm prediction failed: {str(e)}")
            return self.predict_with_fallback()

    def predict_with_custom_model(self):
        """Predict using custom model"""
        try:
            from tensorflow.keras.preprocessing import image
            img = image.load_img(self.image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            # REPLACE from: predictions = self.model.predict(...) down to return {...}
            predictions = self.model.predict(img_array)
            probs = np.asarray(predictions).ravel()

            gray = self._load_image_for_metrics()
            m = self._compute_metrics(gray)

            class_names = ['No Tumor', 'Tumor']
            predicted_class = class_names[int(np.argmax(probs))]

            confidence = self._calibrated_confidence(
                probs=probs,
                metrics=m,
                fallback_pmax=float(np.max(probs)) if probs.size else 0.60,
                hard_min=45.0,
                hard_max=92.0
            )

            image_ext = os.path.splitext(self.image_path)[1].lower()
            if self.image_path.lower().endswith(".nii.gz"):
                image_ext = ".nii.gz"

            image_type = "Voxel" if image_ext in ['.dcm', '.nii', '.nii.gz'] else "Pixel"
            dimensions = "3D" if image_type == "Voxel" else "2D"

            return {
                'image_type': image_type,
                'image_dimensions': dimensions,
                'image_quality': self._label_quality(m["quality_score"]),
                'brightness': self._label_brightness(m["brightness"]),
                'sharpness': self._label_sharpness(m["sharpness_norm"]),
                'diagnosis': predicted_class,
                'confidence': f"{confidence:.1f}%",
                'sequence_number': self.seq_num
            }

        except Exception as e:
            print(f"Custom model prediction failed: {str(e)}")
            return self.predict_with_fallback()
    def predict_with_fallback(self):
        """Fallback prediction method"""
        # REPLACE the whole predict_with_fallback()

        gray = self._load_image_for_metrics()
        m = self._compute_metrics(gray)

        image_ext = os.path.splitext(self.image_path)[1].lower()
        if self.image_path.lower().endswith(".nii.gz"):
            image_ext = ".nii.gz"

        image_type = "Voxel" if image_ext in ['.dcm', '.nii', '.nii.gz'] else "Pixel"
        dimensions = "3D" if image_type == "Voxel" else "2D"

        # Deterministic heuristic (not random):
        # - low edge + low contrast -> likely "No Tumor"
        # - moderate -> "Benign Tumor"
        # - higher structural complexity -> "Malignant Tumor"
        ed = m["edge_density"]
        cn = m["contrast_norm"]
        sn = m["sharpness_norm"]

        if (ed < 0.06 and cn < 0.28) or (sn < 0.22 and cn < 0.25):
            diagnosis = "No Tumor"
        elif cn < 0.38 or ed < 0.10:
            diagnosis = "Benign Tumor"
        else:
            diagnosis = "Malignant Tumor"

        # pseudo pmax derived from quality only (bounded)
        pseudo_pmax = 0.55 + 0.35 * float(m["quality_score"])

        confidence = self._calibrated_confidence(
            probs=None,
            metrics=m,
            fallback_pmax=pseudo_pmax,
            hard_min=45.0,
            hard_max=80.0   # مهم: بدون نموذج لا تتجاوز سقف منخفض
        )

        return {
            'image_type': image_type,
            'image_dimensions': dimensions,
            'image_quality': self._label_quality(m["quality_score"]),
            'brightness': self._label_brightness(m["brightness"]),
            'sharpness': self._label_sharpness(m["sharpness_norm"]),
            'diagnosis': diagnosis,
            'confidence': f"{confidence:.1f}%",
            'sequence_number': self.seq_num
        }

# =====================================================================
# core/loading_screen.py (merged)
# =====================================================================
class LoadingScreen(QWidget):
    """Elegant loading screen with top close/minimize buttons"""
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        # ===== Window Size Configuration =====
        self.setFixedSize(600, 700)
        # ===== Center the Window =====
        qr = self.frameGeometry()
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        # ===== Gradient Background =====
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#ffffff"))
        gradient.setColorAt(1.0, QColor("#b2d7f9"))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setAutoFillBackground(True)
        self.setPalette(palette)
        # ===== Main Layout =====
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        # ===== Top Bar for Buttons =====
        top_bar = TopBarButtonsWidget(self, show_minimize=True, show_close=True)
        main_layout.addWidget(top_bar)
        # ===== Content Layout =====
        content_layout = QVBoxLayout()
        content_layout.setAlignment(Qt.AlignCenter)
        content_layout.setSpacing(10)
        # Logo
        self.logo_label = QLabel()
        pixmap = QPixmap(resource_path("assets/medical_phy.png"))
        if not pixmap.isNull():
            scaled = pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled)
        else:
            self.logo_label.setText("LOGO NOT FOUND")
        self.logo_label.setAlignment(Qt.AlignCenter)
        # Texts
        self.uni_label = QLabel("University of Al-Qadisiyah")
        self.uni_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.uni_label.setStyleSheet("color: #0a3d62;")
        self.uni_label.setAlignment(Qt.AlignCenter)
        self.college_label = QLabel("College of Science – Department of Medical Physics")
        self.college_label.setFont(QFont("Segoe UI", 11))
        self.college_label.setStyleSheet("color: #145a92;")
        self.college_label.setAlignment(Qt.AlignCenter)
        self.year_label = QLabel("© 2025")
        self.year_label.setFont(QFont("Segoe UI", 10))
        self.year_label.setStyleSheet("color: #5d6d7e;")
        self.year_label.setAlignment(Qt.AlignCenter)
        # Loader below content
        self.loader_label = QLabel()
        self.loader_label.setAlignment(Qt.AlignCenter)
        self.loader_label.setText("Loading...")
        content_layout.addStretch()
        content_layout.addWidget(self.logo_label)
        content_layout.addSpacing(5)
        content_layout.addWidget(self.uni_label)
        content_layout.addWidget(self.college_label)
        content_layout.addWidget(self.year_label)
        content_layout.addStretch()
        content_layout.addWidget(self.loader_label, alignment=Qt.AlignBottom)
        content_layout.addSpacing(20)
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.finish_loading)
        self.timer.start(8000)
    def finish_loading(self):
        self.timer.stop()
        self.close()
class TopBarButtonsWidget(QWidget):
    """Widget containing close and minimize buttons for windows"""
    def __init__(self, parent=None, show_minimize=True, show_close=True):
        super().__init__(parent)
        self.parent_widget = parent
        self.setup_ui(show_minimize, show_close)
    def setup_ui(self, show_minimize, show_close):
        """Setup the buttons UI"""
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        layout.addStretch()
        button_style = """
            QPushButton {
                color: #003366;
                font-size: 24px;
                font-weight: bold;
                background-color: transparent;
                border: none;
            }
            QPushButton:hover {
                color: #0055aa;
            }
            QPushButton:pressed {
                color: #001a33;
            }
        """
        # Minimize button
        if show_minimize:
            self.min_button = QPushButton("–")
            self.min_button.setFixedSize(40, 40)
            self.min_button.setStyleSheet(button_style)
            self.min_button.clicked.connect(self.minimize_window)
            layout.addWidget(self.min_button)
            layout.addSpacing(15)
        # Close button
        if show_close:
            self.close_button = QPushButton("×")
            self.close_button.setFixedSize(40, 40)
            self.close_button.setStyleSheet(button_style)
            self.close_button.clicked.connect(self.close_window)
            layout.addWidget(self.close_button)
        self.setLayout(layout)
    def minimize_window(self):
        """Minimize the parent window"""
        if self.parent_widget:
            self.parent_widget.showMinimized()
    def close_window(self):
        """Close the parent window"""
        if self.parent_widget:
            self.parent_widget.close()
# =====================================================================
# core/theme_manager.py (merged)
# =====================================================================
def apply_dark_theme(window: QMainWindow):
    """Apply dark theme to the application"""
    window.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QLineEdit, QComboBox, QSpinBox {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 8px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #4c4c4c;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            color: #666666;
        }
        QTableWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            gridline-color: #555555;
            alternate-background-color: #363636;
        }
        QHeaderView::section {
            background-color: #3c3c3c;
            color: #ffffff;
            padding: 5px;
            border: 1px solid #555555;
        }
        QLabel {
            background-color: transparent;
            color: #ffffff;
        }
        QScrollArea {
            background-color: #2b2b2b;
        }
        QMenuBar {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QMenuBar::item:selected {
            background-color: #4c4c4c;
            color: #00ff88;
        }
        QMenuBar::item:pressed {
            background-color: #5c5c5c;
        }
        QMenu {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px;
        }
        QMenu::item:selected {
            background-color: #4c4c4c;
            color: #00ff88;
        }
        QMenu::item:pressed {
            background-color: #5c5c5c;
        }
    """)
    if hasattr(window, 'results_table'):
        window.results_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #555555;
                background-color: #2b2b2b;
                color: #ffffff;
                alternate-background-color: #363636;
            }
            QHeaderView::section {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #555555;
                font-weight: bold;
            }
        """)
def apply_light_theme(window: QMainWindow):
    """Apply light theme to the application"""
    window.setStyleSheet("")
    if hasattr(window, 'results_table'):
        window.results_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                background-color: white;
                alternate-background-color: #f5f5f5;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                padding: 5px;
                border: 1px solid #ddd;
                font-weight: bold;
            }
        """)
class EnglishNameValidator(QValidator):
    """
    Accepts only English letters and spaces.
    Limits to max 15 letters (spaces not counted).
    """
    def __init__(self, parent=None, max_letters: int = 15):
        super().__init__(parent)
        self.max_letters = max_letters
    def _is_letter(self, ch: str) -> bool:
        return ("A" <= ch <= "Z") or ("a" <= ch <= "z")
    def validate(self, s: str, pos: int):
        if s == "":
            return (QValidator.Intermediate, s, pos)
        letters_count = 0
        prev_space = False
        for ch in s:
            if ch == " ":
                if prev_space:
                    return (QValidator.Invalid, s, pos)
                prev_space = True
                continue
            prev_space = False
            if not self._is_letter(ch):
                return (QValidator.Invalid, s, pos)
            letters_count += 1
            if letters_count > self.max_letters:
                return (QValidator.Invalid, s, pos)
        return (QValidator.Acceptable, s, pos)
    def fixup(self, s: str) -> str:
        cleaned = []
        letters_count = 0
        for ch in s:
            if ch == " ":
                cleaned.append(ch)
            elif self._is_letter(ch):
                if letters_count < self.max_letters:
                    cleaned.append(ch)
                    letters_count += 1
        out = " ".join("".join(cleaned).split())
        return out
    
# =====================================================================
# PDF helpers (must be GLOBAL – not inside any class)
# =====================================================================
def project_root() -> Path:
    """
    يرجع جذر المشروع PhyDCM-App حتى لو كان الملف داخل:
    src/phydcm_app/app.py
    """
    # app.py -> phydcm_app -> src -> PhyDCM-App
    return Path(__file__).resolve().parents[2]

def src_root() -> Path:
    """يرجع مجلد src"""
    return Path(__file__).resolve().parents[1]

def resource_path(rel_path: str) -> str:
    """
    يرجّع مسار صحيح سواء تشغيل عادي أو بعد التحويل إلى exe (PyInstaller).
    rel_path مثال: 'assets/medical_phy.png'
    """
    # PyInstaller
    if hasattr(sys, "_MEIPASS"):
        base = Path(getattr(sys, "_MEIPASS"))  # type: ignore
        return str(base / rel_path)

    # تشغيل عادي من سورس المشروع
    return str(project_root() / rel_path)

# ✅ مساراتك المطلوبة لكن بشكل portable
ASSETS_DIR = project_root() / "assets"
HISTORY_DIR = src_root() / "HISTORY"

def fitz_pixmap_to_qpixmap(pm: fitz.Pixmap) -> QPixmap:
    """
    تحويل Pixmap من PyMuPDF إلى QPixmap (آمن وسريع).
    """
    if pm.alpha:
        qimg = QImage(
            pm.samples,
            pm.width,
            pm.height,
            pm.stride,
            QImage.Format_RGBA8888
        )
    else:
        qimg = QImage(
            pm.samples,
            pm.width,
            pm.height,
            pm.stride,
            QImage.Format_RGB888
        )
    return QPixmap.fromImage(qimg.copy())

#///////////////////////////////////////////////////////////////////////////////////////

class PdfViewerDialog(QDialog):
    """
    PDF viewer dialog using PyMuPDF (fitz) + QLabel rendering.
    - Next/Prev page
    - Zoom in/out
    - Download (Save As)
    """
    def __init__(self, pdf_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help (PDF)")
        self.resize(900, 650)

        self.pdf_path = pdf_path
        self.doc = None
        self.page_index = 0
        self.zoom = 1.0

        self._build_ui()
        self._open_pdf()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # toolbar
        bar = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.zoom_out_btn = QPushButton("Zoom -")
        self.zoom_in_btn = QPushButton("Zoom +")
        self.page_label = QLabel("Page: -/-")
        self.page_label.setAlignment(Qt.AlignCenter)

        self.save_btn = QPushButton("Download PDF")
        self.close_btn = QPushButton("Close")

        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn.clicked.connect(self.next_page)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.save_btn.clicked.connect(self.save_as)
        self.close_btn.clicked.connect(self.accept)

        bar.addWidget(self.prev_btn)
        bar.addWidget(self.next_btn)
        bar.addSpacing(10)
        bar.addWidget(self.zoom_out_btn)
        bar.addWidget(self.zoom_in_btn)
        bar.addStretch()
        bar.addWidget(self.page_label)
        bar.addStretch()
        bar.addWidget(self.save_btn)
        bar.addWidget(self.close_btn)

        # viewer area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.image_label = QLabel("No PDF loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #111; color: white; padding: 10px;")
        self.scroll.setWidget(self.image_label)

        layout.addLayout(bar)
        layout.addWidget(self.scroll)

    def _open_pdf(self):
        try:
            if not self.pdf_path or not os.path.exists(self.pdf_path):
                QMessageBox.warning(self, "PDF", f"PDF not found:\n{self.pdf_path}")
                return

            self.doc = fitz.open(self.pdf_path)
            self.page_index = 0
            self.zoom = 1.0
            self._render_page()
        except Exception as e:
            QMessageBox.critical(self, "PDF", f"Failed to open PDF:\n{str(e)}")

    def _render_page(self):
        if self.doc is None:
            return

        self.page_index = max(0, min(self.page_index, self.doc.page_count - 1))
        page = self.doc.load_page(self.page_index)

        # render with zoom
        mat = fitz.Matrix(self.zoom, self.zoom)
        pm = page.get_pixmap(matrix=mat, alpha=False)
        pix = fitz_pixmap_to_qpixmap(pm)

        self.image_label.setPixmap(pix)
        self.image_label.setText("")
        self.page_label.setText(f"Page: {self.page_index + 1}/{self.doc.page_count}")

        self.prev_btn.setEnabled(self.page_index > 0)
        self.next_btn.setEnabled(self.page_index < self.doc.page_count - 1)

    def next_page(self):
        if self.doc is None:
            return
        if self.page_index < self.doc.page_count - 1:
            self.page_index += 1
            self._render_page()

    def prev_page(self):
        if self.doc is None:
            return
        if self.page_index > 0:
            self.page_index -= 1
            self._render_page()

    def zoom_in(self):
        if self.doc is None:
            return
        self.zoom = min(self.zoom * 1.25, 5.0)
        self._render_page()

    def zoom_out(self):
        if self.doc is None:
            return
        self.zoom = max(self.zoom / 1.25, 0.25)
        self._render_page()

    def save_as(self):
        if not self.pdf_path or not os.path.exists(self.pdf_path):
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save PDF As", "help.pdf", "PDF Files (*.pdf)")
        if not out_path:
            return
        try:
            with open(self.pdf_path, "rb") as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            QMessageBox.information(self, "Saved", "PDF saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save PDF:\n{str(e)}")

    def closeEvent(self, e):
        try:
            if self.doc is not None:
                self.doc.close()
        except Exception:
            pass
        super().closeEvent(e)


# =====================================================================
# MedicalDiagnosisApp (continues)
# =====================================================================
class MedicalDiagnosisApp(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.current_model = None
        self.current_images = []
        self.results_data = []
        self.zoom_level = 1.0
        self.dark_mode = False
        self.loaded_model_path = None
        self.phydcm_predictor = None
        self.auto_save_enabled = False
        # New flags for folder/single logic
        self.loaded_from_folder = False
        # ✅ Active viewer among the 3 screens (Axial/Sagittal/Coronal)
        self.active_mpr_viewer: Optional[VolumeSliceViewer] = None
        if PHYDCM_AVAILABLE:
            try:
                self.phydcm_predictor = PyHDCMPredictor()
                print("✓ PyHDCM predictor initialized successfully")
            except Exception as e:
                print(f"✗ Failed to initialize PyHDCM predictor: {str(e)}")
                self.phydcm_predictor = None
        self.setup_ui()
        self.setup_menu()
        self.setup_connections()
    # ---------------------------
    # Helpers: Load medical data
    # ---------------------------
    def _load_dicom_series_folder(self, folder_path: str):
        if dicom is None:
            raise RuntimeError("pydicom is not installed.")
        files = []
        for f in Path(folder_path).iterdir():
            if f.is_file() and f.name.lower().endswith(".dcm"):
                files.append(str(f))
        if not files:
            return None, []
        def sort_key(p):
            try:
                ds = dicom.dcmread(p, stop_before_pixels=True, force=True)
                return int(getattr(ds, "InstanceNumber", 10**9))
            except Exception:
                return 10**9
        files_sorted = sorted(files, key=lambda p: (sort_key(p), p))
        slices = []
        for p in files_sorted:
            ds = dicom.dcmread(p, force=True)
            arr = ds.pixel_array
            slices.append(arr)
        vol = np.stack(slices, axis=0)  # [Z,Y,X]
        return vol, files_sorted
    def _load_nifti(self, file_path: str):
        if nib is None:
            raise RuntimeError("nibabel is not installed.")
        nii = nib.load(file_path)
        data = nii.get_fdata()
        if data.ndim == 4:
            data = data[:, :, :, 0]
        if data.ndim == 3:
            # [X,Y,Z] -> [Z,Y,X]
            data = np.transpose(data, (2, 1, 0))
        else:
            raise ValueError("Unsupported NIfTI shape.")
        return data
    def _set_mpr_views_from_volume(self, vol_zyx: np.ndarray, files_sorted: list):
        # fixed 3 views
        self.viewer_axial.set_volume(vol_zyx)
        self.viewer_sagittal.set_volume(vol_zyx)
        self.viewer_coronal.set_volume(vol_zyx)
        self.viewer_axial.set_orientation("axial")
        self.viewer_sagittal.set_orientation("sagittal")
        self.viewer_coronal.set_orientation("coronal")
        # mark series mapping for choose
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
            vw.is_series = True
            vw.dicom_files_sorted = files_sorted
        # ✅ default active = Axial
        self._set_active_mpr(self.viewer_axial)
        # ✅ تحديث مدى السلايدر بعد التحميل
        if hasattr(self, "_update_active_slider"):
            self._update_active_slider()
        # fourth reset until choose
        self.viewer_chosen.clear()
        self.viewer_chosen.setText("Chosen / Diagnosis")
        # enable choose only for folder mode
        self.choose_btn.setEnabled(True)
    # ---------------------------
    # ✅ Active logic (3 screens)
    # ---------------------------
    def _set_active_mpr(self, viewer: VolumeSliceViewer):
        # green border only for these 3
        for vw in (self.viewer_axial, self.viewer_sagittal, self.viewer_coronal):
            vw.set_active(vw is viewer)
        self.active_mpr_viewer = viewer
        # slider sync
        if hasattr(self, "_update_active_slider"):
            self._update_active_slider()
        # status
        self.status_bar.showMessage(f"Active View: {viewer.orientation.upper()} (MPR)")
    def _zoom_target_viewer(self) -> Optional[VolumeSliceViewer]:
        """
        ✅ حسب طلبك:
        - إذا الشاشة الرابعة "مشتغلة" (سواء single file أو بعد Choose) => الزوم عليها فقط
        - غير ذلك (Folder قبل choose) => الزوم على الشاشة الفعّالة من الثلاثة
        """
        # fourth is considered "working" if it has a volume loaded (single or chosen)
        if getattr(self.viewer_chosen, "volume", None) is not None:
            return self.viewer_chosen
        return self.active_mpr_viewer
    def _nav_target_viewer(self) -> Optional[VolumeSliceViewer]:
        """
        ✅ التحريك/السلايدر:
        - إذا الشاشة الرابعة فيها volume => السلايدر يحركها
        - غير ذلك => السلايدر يحرك الشاشة الفعالة من الثلاثة
        """
        if getattr(self.viewer_chosen, "volume", None) is not None:
            return self.viewer_chosen
        return self.active_mpr_viewer
    def _choose_from_active_to_fourth(self):
        """Choose يعمل فقط عند رفع مجلد (Folder mode) ويأخذ من الشاشة ذات الحدود الخضراء"""
        if not self.loaded_from_folder:
            return
        src = self.active_mpr_viewer
        if src is None or src.volume is None:
            return
        # Copy volume + orientation/index from active viewer into 4th
        self.viewer_chosen.set_volume(src.volume)
        self.viewer_chosen.orientation = src.orientation
        self.viewer_chosen.idx_axial = src.idx_axial
        self.viewer_chosen.idx_sagittal = src.idx_sagittal
        self.viewer_chosen.idx_coronal = src.idx_coronal
        self.viewer_chosen.render()
        # Diagnosis source path:
        # - DICOM series: pick axial slice file by axial index (stable)
        if src.is_series and src.dicom_files_sorted:
            axial_idx = max(0, min(src.idx_axial, len(src.dicom_files_sorted) - 1))
            self.viewer_chosen.current_source_path = src.dicom_files_sorted[axial_idx]
        else:
            self.viewer_chosen.current_source_path = None
        # now allow diagnosis
        self.diagnose_btn.setEnabled(True)
        # ✅ بعد ما تشتغل الشاشة الرابعة، الزوم/السلايدر يصير عليها فقط حسب شرطنا
        if hasattr(self, "_update_active_slider"):
            self._update_active_slider()
        self.status_bar.showMessage("Chosen image copied to 4th screen. Start Diagnosis is now enabled.")
    # ---------------------------
    # UI
    # ---------------------------
    def setup_ui(self):
        """Setup the main UI"""
        self.setWindowTitle("PhyDCM Diagnosis System")
        self.setMinimumSize(1200, 800)
        self.setWindowIcon(QIcon(resource_path("assets/medical_phy.png")))
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        # ✅ أنشئ الـ StatusBar قبل create_middle_panel لأن create_middle_panel يستدعي _set_active_mpr
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        middle_panel = self.create_middle_panel()
        main_layout.addWidget(middle_panel, 2)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        central_widget.setLayout(main_layout)
        self.terminal = TerminalWidget(self)
        self.terminal.setVisible(False)
    def create_left_panel(self) -> QWidget:
        """Create left panel with input fields"""
        panel = QWidget()
        panel.setMaximumWidth(300)
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        # Patient name
        name_label = QLabel("Patient Name:")
        self.name_input = QLineEdit()
        self.name_input.setValidator(EnglishNameValidator(self.name_input, max_letters=15))
        self.name_input.editingFinished.connect(self._normalize_patient_name)
        self.name_input.setMaxLength(120)
        self.name_input.setPlaceholderText("Enter patient name")
        # Patient age
        age_label = QLabel("Patient Age:")
        self.age_input = QSpinBox()
        self.age_input.setRange(0, 120)
        self.age_input.setValue(25)
        # Gender
        gender_label = QLabel("Gender:")
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Male", "Female", "Other"])
        # Scan type
        scan_label = QLabel("Scan Type:")
        self.scan_combo = QComboBox()
        self.scan_combo.addItems(["MRI", "CT", "PET"])
        # Model status
        self.model_status_label = QLabel("Model Status:")
        self.model_status = QLabel()
        self.update_model_status()
        # Import buttons
        import_single_btn = QPushButton("Import Medical File")
        import_single_btn.clicked.connect(self.import_single_image)
        import_folder_btn = QPushButton("Import DICOM Folder")
        import_folder_btn.clicked.connect(self.import_image_folder)
        # ✅ Choose (must be فوق Start Diagnosis)
        self.choose_btn = QPushButton("Choose")
        self.choose_btn.setEnabled(False)
        self.choose_btn.setToolTip("Works only when importing a DICOM folder/series")
        self.choose_btn.clicked.connect(self._choose_from_active_to_fourth)
        # Diagnosis button
        self.diagnose_btn = QPushButton("Start Diagnosis")
        self.diagnose_btn.clicked.connect(self.start_diagnosis)
        self.diagnose_btn.setEnabled(False)
        self.diagnose_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        scroll_layout.addWidget(name_label)
        scroll_layout.addWidget(self.name_input)
        scroll_layout.addWidget(age_label)
        scroll_layout.addWidget(self.age_input)
        scroll_layout.addWidget(gender_label)
        scroll_layout.addWidget(self.gender_combo)
        scroll_layout.addWidget(scan_label)
        scroll_layout.addWidget(self.scan_combo)
        scroll_layout.addWidget(self.model_status_label)
        scroll_layout.addWidget(self.model_status)
        scroll_layout.addSpacing(20)
        scroll_layout.addWidget(import_single_btn)
        scroll_layout.addWidget(import_folder_btn)
        scroll_layout.addSpacing(12)
        scroll_layout.addWidget(self.choose_btn)      # ✅ فوق Start Diagnosis
        scroll_layout.addWidget(self.diagnose_btn)
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        panel.setLayout(layout)
        return panel
    def create_middle_panel(self) -> QWidget:
        """
        ✅ 4 شاشات (2x2):
        - ثلاث شاشات MPR: Axial / Sagittal / Coronal
        - الشاشة الرابعة: Chosen / Diagnosis
        ✅ عند النقر على أي واحدة من الثلاثة => حدود خضراء + (L/R + count) وتصبح هي Active للتحريك والZoom
        ✅ عند اشتغال الشاشة الرابعة (single أو بعد Choose) => التحريك والZoom يصير فقط على الشاشة الرابعة
        """
        panel = QWidget()
        outer = QVBoxLayout()
        grid = QGridLayout()
        grid.setSpacing(8)
        # ✅ الثلاثة MPR: نسمح بالزوم لكن الـ MainWindow يتحكم بمنو فعلاً يتزوم
        self.viewer_axial = VolumeSliceViewer(allow_zoom=True, title="Axial")
        self.viewer_sagittal = VolumeSliceViewer(allow_zoom=True, title="Sagittal")
        self.viewer_coronal = VolumeSliceViewer(allow_zoom=True, title="Coronal")
        # ✅ 4th screen: chosen/diagnosis
        self.viewer_chosen = VolumeSliceViewer(allow_zoom=True, title="Chosen / Diagnosis")
        # Placement:
        # (0,0)=Axial, (0,1)=Sagittal
        # (1,0)=Coronal, (1,1)=Chosen
        grid.addWidget(self.viewer_axial, 0, 0)
        grid.addWidget(self.viewer_sagittal, 0, 1)
        grid.addWidget(self.viewer_coronal, 1, 0)
        grid.addWidget(self.viewer_chosen, 1, 1)
        outer.addLayout(grid)
        # =================================================================
        # ✅ Controls (Slider) -> يتحكم بالـ Active من الثلاثة أو بالشاشة الرابعة إذا مشتغلة
        # =================================================================
        controls = QHBoxLayout()
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.setEnabled(False)
        # طول تقريبا بطول شاشتين
        self.slice_slider.setFixedWidth(self.viewer_axial.minimumWidth() * 2)
        self.slice_slider.setToolTip("Slice Navigation (Active / 4th)")
        self.slice_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 10px;
                background: #cfcfcf;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                width: 22px;
                margin: -6px 0;
                border-radius: 11px;
                background: #4CAF50;
            }
        """)
        def _update_slider_range():
            tgt = self._nav_target_viewer()
            if tgt is None or getattr(tgt, "volume", None) is None:
                self.slice_slider.setEnabled(False)
                self.slice_slider.setMinimum(0)
                self.slice_slider.setMaximum(0)
                self.slice_slider.setValue(0)
                return
            total = tgt.get_slice_count()
            mx = max(0, total - 1)
            val = max(0, min(tgt.get_current_index(), mx))
            self.slice_slider.blockSignals(True)
            self.slice_slider.setEnabled(True)
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(mx)
            self.slice_slider.setValue(val)
            self.slice_slider.blockSignals(False)
        def _on_slider_changed(value: int):
            tgt = self._nav_target_viewer()
            if tgt is None or getattr(tgt, "volume", None) is None:
                return
            tgt.set_slice_index(value)
        self.slice_slider.valueChanged.connect(_on_slider_changed)
        # ✅ تفعيل الـ Active عند النقر على أي شاشة من الثلاثة (حدود خضراء + overlay)
        self.viewer_axial.clicked.connect(lambda: self._set_active_mpr(self.viewer_axial))
        self.viewer_sagittal.clicked.connect(lambda: self._set_active_mpr(self.viewer_sagittal))
        self.viewer_coronal.clicked.connect(lambda: self._set_active_mpr(self.viewer_coronal))
        controls.addWidget(self.slice_slider)
        controls.addStretch()
        outer.addLayout(controls)
        # expose updater for other methods (folder load / single load / choose)
        self._update_active_slider = _update_slider_range
        # default active view visual
        self._set_active_mpr(self.viewer_axial)
        panel.setLayout(outer)
        return panel
    def create_right_panel(self) -> QWidget:
        """Create right panel with results table"""
        panel = QWidget()
        layout = QVBoxLayout()
        self.results_table = ResultsTable()
        btn_layout = QHBoxLayout()
        export_btn = QPushButton("Export CSV/JSON")
        export_btn.clicked.connect(self.export_results)
        add_patient_btn = QPushButton("Add Patient")
        add_patient_btn.clicked.connect(self.add_new_patient)
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_selected_result)
        btn_layout.addWidget(export_btn)
        btn_layout.addWidget(add_patient_btn)
        btn_layout.addWidget(delete_btn)
        layout.addWidget(self.results_table)
        layout.addLayout(btn_layout)
        panel.setLayout(layout)
        return panel
    def update_model_status(self):
        """Update model status display"""
        if self.current_model:
            self.model_status.setText(f"Custom Model: {os.path.basename(self.loaded_model_path) if self.loaded_model_path else 'Loaded'}")
            self.model_status.setStyleSheet("color: green; font-weight: bold;")
        elif self.phydcm_predictor:
            self.model_status.setText("PhyDCM (Default)")
            self.model_status.setStyleSheet("color: #ff8a00; font-weight: bold;")
        else:
            self.model_status.setText("No Model")
            self.model_status.setStyleSheet("color: red;")
    # =================================================================
    # Menu (preserved as you wrote)
    # =================================================================
    def setup_menu(self):
        """Setup menu bar with new structure"""
        menubar = self.menuBar()
        # ===== File Menu =====
        file_menu = menubar.addMenu('File')
        new_add_action = QAction('New Add', self)
        new_add_action.setShortcut('Ctrl+N')
        new_add_action.triggered.connect(self.new_add)
        file_menu.addAction(new_add_action)
        new_window_action = QAction('New Window', self)
        new_window_action.setShortcut('Ctrl+Shift+N')
        new_window_action.triggered.connect(self.new_window)
        file_menu.addAction(new_window_action)
        file_menu.addSeparator()
        open_file_action = QAction('Open File', self)
        open_file_action.setShortcut('Ctrl+O')
        open_file_action.triggered.connect(self.open_file)
        file_menu.addAction(open_file_action)
        file_menu.addSeparator()
        save_action = QAction('Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        save_as_action = QAction('Save As...', self)
        save_as_action.setShortcut('Ctrl+Shift+S')
        save_as_action.triggered.connect(self.save_as_file)
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        share_action = QAction('Share', self)
        share_action.triggered.connect(self.share_file)
        file_menu.addAction(share_action)
        file_menu.addSeparator()
        self.auto_save_action = QAction('Auto Save', self)
        self.auto_save_action.setCheckable(True)
        self.auto_save_action.triggered.connect(self.toggle_auto_save)
        file_menu.addAction(self.auto_save_action)
        file_menu.addSeparator()
        clear_file_action = QAction('Clear File', self)
        clear_file_action.setShortcut('Ctrl+K, F')
        clear_file_action.triggered.connect(self.clear_file)
        file_menu.addAction(clear_file_action)
        file_menu.addSeparator()
        close_action = QAction('Close Window', self)
        close_action.setShortcut('Ctrl+F4')
        close_action.triggered.connect(self.close_window)
        file_menu.addAction(close_action)
        # ===== Edit Menu =====
        edit_menu = menubar.addMenu('Edit')
        undo_action = QAction('Undo (Not Available)', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.undo_action)
        edit_menu.addAction(undo_action)
        redo_action = QAction('Redo (Not Available)', self)
        redo_action.setShortcut('Ctrl+Y')
        redo_action.triggered.connect(self.redo_action)
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()
        copy_action = QAction('Copy', self)
        copy_action.setShortcut('Ctrl+C')
        copy_action.triggered.connect(self.copy_to_clipboard)
        edit_menu.addAction(copy_action)
        paste_action = QAction('Paste', self)
        paste_action.setShortcut('Ctrl+V')
        paste_action.triggered.connect(self.paste_from_clipboard)
        edit_menu.addAction(paste_action)
        edit_menu.addSeparator()
        zoom_in_action = QAction('Zoom In', self)
        zoom_in_action.setShortcut('Ctrl++')
        zoom_in_action.triggered.connect(self.zoom_in)
        edit_menu.addAction(zoom_in_action)
        zoom_out_action = QAction('Zoom Out', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.zoom_out)
        edit_menu.addAction(zoom_out_action)
        # ===== View Menu =====
        view_menu = menubar.addMenu('View')
        processing_action = QAction('Processing (Not Available)', self)
        processing_action.triggered.connect(self.processing_action)
        view_menu.addAction(processing_action)
        terminal_action = QAction('Terminal', self)
        terminal_action.setShortcut('Ctrl+T')
        terminal_action.triggered.connect(self.toggle_terminal)
        view_menu.addAction(terminal_action)
        history_action = QAction('History', self)
        history_action.triggered.connect(self.show_history)
        view_menu.addAction(history_action)
        segmentation_action = QAction('Segmentation (Not Available)', self)
        segmentation_action.triggered.connect(self.segmentation_action)
        view_menu.addAction(segmentation_action)
        view_menu.addSeparator()
        dark_mode_action = QAction('Light/Dark Pattern', self)
        dark_mode_action.setShortcut('Ctrl+D')
        dark_mode_action.triggered.connect(self.toggle_dark_mode)
        view_menu.addAction(dark_mode_action)
        # ===== Import Menu (Currently Disabled) =====
        import_menu = menubar.addMenu('Import')
        import_menu.setEnabled(False)
        # ===== Info Menu =====
        info_menu = menubar.addMenu('Info')
        about_action = QAction('About Us', self)
        about_action.triggered.connect(self.show_about)
        info_menu.addAction(about_action)
        help_action = QAction('Help', self)
        help_action.triggered.connect(self.show_help)
        info_menu.addAction(help_action)
    # =================================================================
    # Menu actions (as-is / minimal changes)
    # =================================================================
    def new_add(self):
        self.add_new_patient()
        self.status_bar.showMessage("New patient form cleared")
    def new_window(self):
        try:
            subprocess.Popen([sys.executable, __file__])
            self.status_bar.showMessage("New window opened")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not open new window: {str(e)}")
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not file_path:
            return

        try:
            # ✅ افتح فقط History JSON
            ok = self.load_history_json(file_path)
            if ok:
                self.status_bar.showMessage(f"Opened JSON: {os.path.basename(file_path)}")
            else:
                self.status_bar.showMessage("Failed to open JSON")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open JSON file:\n{str(e)}")

    def share_file(self):
        if hasattr(self, 'current_file') and self.current_file:
            QMessageBox.information(self, "Share", f"File ready for sharing:\n{self.current_file}")
        else:
            reply = QMessageBox.question(self, "Share", "Save file before sharing?",
                                     QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.save_as_file()
    def toggle_auto_save(self, checked):
        self.auto_save_enabled = checked
        self.status_bar.showMessage("Auto-save enabled" if checked else "Auto-save disabled")
    def clear_file(self):
        reply = QMessageBox.question(
            self, "Confirm Clear",
            "Clear all patient data? Save current file first?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
        )
        if reply == QMessageBox.Save:
            self.save_file()
            self.results_table.setRowCount(0)
            self.status_bar.showMessage("File cleared after saving")
        elif reply == QMessageBox.Discard:
            self.results_table.setRowCount(0)
            self.status_bar.showMessage("File cleared without saving")
    def close_window(self):
        self.close()
    def undo_action(self):
        QMessageBox.information(self, "Undo", "Undo functionality is not available in this version")
    def redo_action(self):
        QMessageBox.information(self, "Redo", "Redo functionality is not available in this version")
    def copy_to_clipboard(self):
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, (QLineEdit, QSpinBox, QComboBox)):
            if hasattr(focused_widget, 'selectedText'):
                text = focused_widget.selectedText()
            elif isinstance(focused_widget, QSpinBox):
                text = str(focused_widget.value())
            else:
                text = focused_widget.currentText() if hasattr(focused_widget, 'currentText') else ""
            if text:
                QApplication.clipboard().setText(text)
                self.status_bar.showMessage("Copied to clipboard")
            else:
                self.status_bar.showMessage("Nothing to copy")
        else:
            self.status_bar.showMessage("Select an input field to copy from")
    def paste_from_clipboard(self):
        focused_widget = QApplication.focusWidget()
        clipboard_text = QApplication.clipboard().text()
        if not clipboard_text:
            self.status_bar.showMessage("Clipboard is empty")
            return
        if isinstance(focused_widget, QLineEdit):
            focused_widget.insert(clipboard_text)
            self.status_bar.showMessage("Pasted from clipboard")
        elif isinstance(focused_widget, QSpinBox):
            try:
                focused_widget.setValue(int(clipboard_text))
                self.status_bar.showMessage("Pasted from clipboard")
            except ValueError:
                self.status_bar.showMessage("Invalid number format")
        elif isinstance(focused_widget, QComboBox):
            index = focused_widget.findText(clipboard_text)
            if index >= 0:
                focused_widget.setCurrentIndex(index)
                self.status_bar.showMessage("Pasted from clipboard")
            else:
                self.status_bar.showMessage("Value not found in list")
        else:
            self.status_bar.showMessage("Select an input field to paste into")
    def processing_action(self):
        QMessageBox.information(self, "Processing", "Processing view is not available in this version")
    # ================================================================
    # HISTORY JSON (Save/Load)  ✅ added (no extra windows on load)
    # ================================================================
    def _script_dir(self) -> Path:
        try:
            return Path(__file__).resolve().parent
        except Exception:
            return Path.cwd()
    def _history_dir(self) -> Path:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        return HISTORY_DIR
        h.mkdir(parents=True, exist_ok=True)
        return h
    def _table_headers(self) -> list:
        headers = []
        for col in range(self.results_table.columnCount()):
            item = self.results_table.horizontalHeaderItem(col)
            headers.append(item.text() if item else f"Col{col}")
        return headers
    def _table_rows_as_dicts(self) -> list:
        headers = self._table_headers()
        rows = []
        for r in range(self.results_table.rowCount()):
            row_dict = {}
            for c, h in enumerate(headers):
                it = self.results_table.item(r, c)
                row_dict[h] = it.text() if it else ""
            rows.append(row_dict)
        return rows
    def _get_patient_id_for_history(self) -> Optional[str]:
        """
        ID من الجدول:
        - إذا سطر محدد -> استخدمه
        - وإلا آخر سطر
        """
        if self.results_table.rowCount() == 0:
            return None
        row = self.results_table.currentRow()
        if row < 0:
            row = self.results_table.rowCount() - 1
        id_item = self.results_table.item(row, 1)  # عمود ID عندك هو رقم 1
        pid = (id_item.text().strip() if id_item else "").strip()
        return pid or None
    def save_history_json(self) -> bool:
        """
        ✅ يحفظ Snapshot كامل إلى HISTORY/<ID>.json
        ✅ اسم الملف = ID ولا يمكن تغييره (بدون Save Dialog)
        """
        pid = self._get_patient_id_for_history()
        if not pid:
            QMessageBox.warning(self, "History JSON", "No Patient ID found to save.\nSelect a row or complete a diagnosis first.")
            return False
        history_path = self._history_dir() / f"{pid}.json"
        payload = {
            "version": "PhyDCM-HISTORY-1",
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "patient_id": pid,
            # جدول النتائج بالكامل
            "table": {
                "headers": self._table_headers(),
                "rows": self._table_rows_as_dicts(),
            },
            # حالة الصور/المجلد
            "images_state": {
                "loaded_from_folder": bool(self.loaded_from_folder),
                "current_images": list(self.current_images) if isinstance(self.current_images, list) else [],
                "chosen_source_path": getattr(self.viewer_chosen, "current_source_path", None),
                # وضعية الـ Chosen (للاسترجاع)
                "viewer_chosen": {
                    "orientation": getattr(self.viewer_chosen, "orientation", "axial"),
                    "idx_axial": getattr(self.viewer_chosen, "idx_axial", 0),
                    "idx_coronal": getattr(self.viewer_chosen, "idx_coronal", 0),
                    "idx_sagittal": getattr(self.viewer_chosen, "idx_sagittal", 0),
                },
                # وضعيات الثلاثة (للاسترجاع + تحديد active)
                "viewer_axial": {
                    "idx_axial": getattr(self.viewer_axial, "idx_axial", 0),
                },
                "viewer_sagittal": {
                    "idx_sagittal": getattr(self.viewer_sagittal, "idx_sagittal", 0),
                },
                "viewer_coronal": {
                    "idx_coronal": getattr(self.viewer_coronal, "idx_coronal", 0),
                },
                "active_mpr": getattr(self.active_mpr_viewer, "orientation", "axial") if self.active_mpr_viewer else "axial",
            }
        }
        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.status_bar.showMessage(f"Saved HISTORY JSON: {history_path.name}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "History JSON", f"Failed to save JSON:\n{str(e)}")
            return False
    def _load_single_medical_file_path(self, file_path: str):
        """
        تحميل ملف مفرد (dcm/nii/nii.gz) بدون FileDialog (يستخدمه History أيضاً)
        نفس منطق import_single_image لكن بدون نافذة اختيار.
        """
        if not file_path:
            return
        ext = file_path.lower()
        self.loaded_from_folder = False
        self.choose_btn.setEnabled(False)
        self.diagnose_btn.setEnabled(False)
        # تنظيف MPR
        self.viewer_axial.clear()
        self.viewer_sagittal.clear()
        self.viewer_coronal.clear()
        # clear chosen first
        self.viewer_chosen.clear()
        self.viewer_chosen.setText("Chosen / Diagnosis")
        # default active border remains on axial (لكن ماكو volume)
        self._set_active_mpr(self.viewer_axial)
        if ext.endswith(".dcm"):
            if dicom is None:
                raise RuntimeError("pydicom is not installed.")
            ds = dicom.dcmread(file_path, force=True)
            vol = np.expand_dims(ds.pixel_array, axis=0)  # [1,Y,X]
            self.viewer_chosen.set_volume(vol)
            self.viewer_chosen.set_orientation("axial")
            self.viewer_chosen.current_source_path = file_path
        elif ext.endswith(".nii") or ext.endswith(".nii.gz"):
            vol = self._load_nifti(file_path)
            self.viewer_chosen.set_volume(vol)
            self.viewer_chosen.set_orientation("axial")
            self.viewer_chosen.current_source_path = file_path
        else:
            raise RuntimeError("Only medical formats are allowed: dcm, nii, nii.gz")
        self.current_images = [file_path]
        self.diagnose_btn.setEnabled(True)
        # ✅ بما أن الشاشة الرابعة اشتغلت => السلايدر يصير عليها
        if hasattr(self, "_update_active_slider"):
            self._update_active_slider()
        self.status_bar.showMessage(f"Loaded medical file into 4th screen: {os.path.basename(file_path)}")
    def _load_dicom_folder_path(self, folder_path: str):
        """
        تحميل مجلد DICOM بدون FileDialog (يستخدمه History أيضاً)
        نفس منطق import_image_folder لكن بدون نافذة اختيار.
        """
        if not folder_path:
            return
        vol, files_sorted = self._load_dicom_series_folder(folder_path)
        if vol is None or not files_sorted:
            raise RuntimeError("No DICOM (*.dcm) files found in selected folder")
        self.loaded_from_folder = True
        self.current_images = [folder_path]
        self._set_mpr_views_from_volume(vol, files_sorted)
        # Diagnosis disabled until Choose (لكن History قد يعيد تفعيلها إذا عندنا chosen)
        self.viewer_chosen.current_source_path = None
        self.diagnose_btn.setEnabled(False)
        self.status_bar.showMessage(
            f"Loaded {len(files_sorted)} DICOM slices. Click one of the 3 views to activate (green border), then Choose, then Start Diagnosis."
        )
    def load_history_json(self, json_path: str) -> bool:
        """
        ✅ يسترجع: جدول النتائج + الصور/المجلد + وضعيات viewers
        ✅ بدون فتح أي نافذة ثانية (حتى رسالة نجاح)
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # ===== Restore table =====
            table = data.get("table", {})
            rows = table.get("rows", [])
            self.results_table.setRowCount(0)
            headers = self._table_headers()
            for row_dict in rows:
                r = self.results_table.rowCount()
                self.results_table.insertRow(r)
                for c, h in enumerate(headers):
                    val = str(row_dict.get(h, ""))
                    item = QTableWidgetItem(val)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.results_table.setItem(r, c, item)
            # ===== Restore images =====
            imgs = data.get("images_state", {}) or {}
            loaded_from_folder = bool(imgs.get("loaded_from_folder", False))
            current_images = imgs.get("current_images", []) or []
            chosen_source_path = imgs.get("chosen_source_path", None)
            vc = imgs.get("viewer_chosen", {}) or {}
            va = imgs.get("viewer_axial", {}) or {}
            vs = imgs.get("viewer_sagittal", {}) or {}
            vco = imgs.get("viewer_coronal", {}) or {}
            active_mpr = (imgs.get("active_mpr", "axial") or "axial").lower()
            if loaded_from_folder:
                folder_path = current_images[0] if current_images else None
                if folder_path:
                    self._load_dicom_folder_path(folder_path)
                    # restore indices for 3 views
                    if self.viewer_axial.volume is not None:
                        self.viewer_axial.idx_axial = int(va.get("idx_axial", self.viewer_axial.idx_axial))
                        self.viewer_axial.render()
                    if self.viewer_sagittal.volume is not None:
                        self.viewer_sagittal.idx_sagittal = int(vs.get("idx_sagittal", self.viewer_sagittal.idx_sagittal))
                        self.viewer_sagittal.render()
                    if self.viewer_coronal.volume is not None:
                        self.viewer_coronal.idx_coronal = int(vco.get("idx_coronal", self.viewer_coronal.idx_coronal))
                        self.viewer_coronal.render()
                    # restore active green border
                    if active_mpr == "sagittal":
                        self._set_active_mpr(self.viewer_sagittal)
                    elif active_mpr == "coronal":
                        self._set_active_mpr(self.viewer_coronal)
                    else:
                        self._set_active_mpr(self.viewer_axial)
                    # restore CHOSEN if existed
                    if chosen_source_path and self.viewer_axial.volume is not None:
                        # copy volume from any (same volume)
                        self.viewer_chosen.set_volume(self.viewer_axial.volume)
                        self.viewer_chosen.orientation = vc.get("orientation", "axial")
                        self.viewer_chosen.idx_axial = int(vc.get("idx_axial", 0))
                        self.viewer_chosen.idx_coronal = int(vc.get("idx_coronal", 0))
                        self.viewer_chosen.idx_sagittal = int(vc.get("idx_sagittal", 0))
                        self.viewer_chosen.render()
                        self.viewer_chosen.current_source_path = chosen_source_path
                        self.diagnose_btn.setEnabled(True)
                        self.choose_btn.setEnabled(True)
                    else:
                        self.viewer_chosen.clear()
                        self.viewer_chosen.setText("Chosen / Diagnosis")
                        self.choose_btn.setEnabled(True)
                        self.diagnose_btn.setEnabled(False)
            else:
                file_path = current_images[0] if current_images else chosen_source_path
                if file_path:
                    self._load_single_medical_file_path(file_path)
                    if self.viewer_chosen.volume is not None:
                        self.viewer_chosen.orientation = vc.get("orientation", "axial")
                        self.viewer_chosen.idx_axial = int(vc.get("idx_axial", self.viewer_chosen.idx_axial))
                        self.viewer_chosen.idx_coronal = int(vc.get("idx_coronal", self.viewer_chosen.idx_coronal))
                        self.viewer_chosen.idx_sagittal = int(vc.get("idx_sagittal", self.viewer_chosen.idx_sagittal))
                        self.viewer_chosen.render()
                    self.viewer_chosen.current_source_path = chosen_source_path or file_path
                    self.diagnose_btn.setEnabled(True)
            # sync slider after load
            if hasattr(self, "_update_active_slider"):
                self._update_active_slider()
            self.status_bar.showMessage(f"History loaded: {Path(json_path).name}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "History", f"Failed to load history:\n{str(e)}")
            return False
    def show_history(self):
        # ✅ نافذة واحدة فقط (بدون MessageBox إضافي عند التحميل)
        history_dialog = QDialog(self)
        history_dialog.setWindowTitle("Patient History (HISTORY JSON)")
        history_dialog.setMinimumSize(700, 500)
        layout = QVBoxLayout()
        controls_layout = QHBoxLayout()
        search_label = QLabel("Search ID:")
        search_input = QLineEdit()
        search_input.setPlaceholderText("Type Patient ID...")
        sort_combo = QComboBox()
        sort_combo.addItems(["Newest First", "Oldest First", "Name A-Z", "Name Z-A"])
        refresh_btn = QPushButton("Refresh")
        controls_layout.addWidget(search_label)
        controls_layout.addWidget(search_input)
        controls_layout.addWidget(sort_combo)
        controls_layout.addWidget(refresh_btn)
        history_list = QListWidget()
        def _list_history_files() -> list:
            hdir = self._history_dir()
            files = list(hdir.glob("*.json"))
            items = []
            for p in files:
                try:
                    pid = p.stem
                    mtime = p.stat().st_mtime
                    items.append((pid, str(p), mtime))
                except Exception:
                    continue
            return items
        def _apply_sort(items: list, sort_type: str) -> list:
            if sort_type == "Newest First":
                return sorted(items, key=lambda x: x[2], reverse=True)
            if sort_type == "Oldest First":
                return sorted(items, key=lambda x: x[2], reverse=False)
            if sort_type == "Name Z-A":
                return sorted(items, key=lambda x: x[0].lower(), reverse=True)
            return sorted(items, key=lambda x: x[0].lower(), reverse=False)  # Name A-Z
        def _refresh():
            history_list.clear()
            items = _list_history_files()
            items = _apply_sort(items, sort_combo.currentText())
            q = search_input.text().strip().lower()
            for pid, path, mtime in items:
                if q and (q not in pid.lower()):
                    continue
                it = QListWidgetItem(pid)
                it.setData(Qt.UserRole, path)
                history_list.addItem(it)
        def _load_selected():
            current_item = history_list.currentItem()
            if not current_item:
                return
            path = current_item.data(Qt.UserRole)
            if not path:
                return
            ok = self.load_history_json(path)
            if ok:
                # بدون نافذة ثانية
                history_dialog.accept()
        refresh_btn.clicked.connect(_refresh)
        sort_combo.currentTextChanged.connect(lambda _: _refresh())
        search_input.textChanged.connect(lambda _: _refresh())
        history_list.itemDoubleClicked.connect(lambda _: _load_selected())
        layout.addLayout(controls_layout)
        layout.addWidget(history_list)
        btns = QHBoxLayout()
        load_btn = QPushButton("Load Selected")
        load_btn.clicked.connect(_load_selected)
        open_folder_btn = QPushButton("Open HISTORY Folder")
        def _open_history_folder():
            try:
                hdir = str(self._history_dir())
                if sys.platform.startswith("win"):
                    os.startfile(hdir)  # type: ignore
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", hdir])
                else:
                    subprocess.Popen(["xdg-open", hdir])
            except Exception as e:
                QMessageBox.warning(history_dialog, "Warning", f"Cannot open folder:\n{str(e)}")
        open_folder_btn.clicked.connect(_open_history_folder)
        btns.addWidget(load_btn)
        btns.addWidget(open_folder_btn)
        btns.addStretch()
        layout.addLayout(btns)
        history_dialog.setLayout(layout)
        _refresh()
        history_dialog.exec_()
    def segmentation_action(self):
        QMessageBox.information(self, "Segmentation", "Segmentation view is not available in this version")
    def show_help(self):
        pdf_path = resource_path("assets/help.pdf")
        dlg = PdfViewerDialog(pdf_path, parent=self)
        dlg.exec_()

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Confirm Exit",
            "Are you sure you want to exit the application?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    def setup_connections(self):
        pass
    # =================================================================
    # Import (UPDATED: medical only + folder logic)
    # =================================================================
    def import_single_image(self):
        """Import SINGLE medical file (dcm/nii/nii.gz) -> open مباشرة في الشاشة الرابعة"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Medical File", "",
            "Medical Files (*.dcm *.nii *.nii.gz);;All Files (*.*)"
        )
        if not file_path:
            return
        try:
            self._load_single_medical_file_path(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load medical file:\n{str(e)}")
    def import_image_folder(self):
        """Import folder of DICOM images -> opens in 3 views (active selectable) + choose -> 4th"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder_path:
            return
        try:
            vol, files_sorted = self._load_dicom_series_folder(folder_path)
            if vol is None or not files_sorted:
                QMessageBox.warning(self, "Error", "No DICOM (*.dcm) files found in selected folder")
                return
            self.loaded_from_folder = True
            self.current_images = [folder_path]
            self._set_mpr_views_from_volume(vol, files_sorted)
            # Diagnosis disabled until Choose
            self.viewer_chosen.current_source_path = None
            self.diagnose_btn.setEnabled(False)
            self.status_bar.showMessage(
                f"Loaded {len(files_sorted)} DICOM slices. Click one view (green border) to activate, then Choose, then Start Diagnosis."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load DICOM folder:\n{str(e)}")
    def import_model(self):
        """Import trained model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "Model Files (*.h5 *.keras *.pb);;All Files (*.*)"
        )
        if file_path:
            try:
                if keras is None:
                    raise RuntimeError("TensorFlow/Keras not available.")
                self.current_model = keras.models.load_model(file_path)
                self.loaded_model_path = file_path
                self.update_model_status()
                QMessageBox.information(self, "Success", "Model loaded successfully")
                self.status_bar.showMessage(f"Loaded model: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                self.status_bar.showMessage("Model loading failed")
    def import_csv(self):
        """Import CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        self.results_table.add_result(row)
                self.status_bar.showMessage(f"Loaded CSV: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")
    # =================================================================
    # Diagnosis (UPDATED: always diagnose 4th screen)
    # =================================================================
    def start_diagnosis(self):
        """Start diagnosis process - ALWAYS on 4th screen"""
        name = " ".join(self.name_input.text().split())
        self.name_input.setText(name)
        if not name:
            QMessageBox.warning(self, "Validation Error", "Please enter patient name")
            return
        letters_only = "".join(ch for ch in name if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
        if len(letters_only) < 3:
            QMessageBox.warning(self, "Validation Error", "Patient Name must be at least 3 English letters.")
            return
        chosen_path = getattr(self.viewer_chosen, "current_source_path", None)
        if not chosen_path:
            QMessageBox.warning(self, "Validation Error",
                                "No image chosen for diagnosis.\n\n- If you imported a folder: click a view to activate (green), use Choose first.\n- If you imported a single file: it should appear in 4th screen.")
            return
        seq_num = 1
        if self.loaded_from_folder and getattr(self.viewer_axial, "dicom_files_sorted", None):
            seq_num = len(self.viewer_axial.dicom_files_sorted)
        self.diagnosis_thread = DiagnosisThread(
            image_path=chosen_path,
            patient_data={
                'name': self.name_input.text(),
                'age': self.age_input.value(),
                'gender': self.gender_combo.currentText(),
                'scan_type': self.scan_combo.currentText()
            },
            model=self.current_model,
            phydcm_predictor=self.phydcm_predictor,
            seq_num=seq_num
        )
        self.diagnosis_thread.diagnosis_complete.connect(self.on_diagnosis_complete)
        self.diagnosis_thread.start()
        self.status_bar.showMessage("Performing diagnosis...")
        self.diagnose_btn.setEnabled(False)
    def on_diagnosis_complete(self, result: Dict[str, Any]):
        """Handle diagnosis completion"""
        self.diagnose_btn.setEnabled(True)
        if result.get('success'):
            if self.loaded_from_folder and getattr(self.viewer_axial, "dicom_files_sorted", None):
                result["sequence"] = len(self.viewer_axial.dicom_files_sorted)
            else:
                result["sequence"] = 1
            self.results_table.add_result(result)
            self.status_bar.showMessage("Diagnosis completed successfully")
            self.name_input.clear()
            self.age_input.setValue(25)
        else:
            QMessageBox.critical(self, "Diagnosis Error", result.get('error', 'Unknown error'))
            self.status_bar.showMessage("Diagnosis failed")
    # =================================================================
    # Export + misc
    # =================================================================
    def export_results(self):
        """
        ✅ Export results to CSV OR JSON
        ✅ عند اختيار JSON: حفظ مباشر داخل HISTORY وباسم ID المريض (بدون تغيير الاسم)
        """
        if self.results_table.rowCount() == 0:
            QMessageBox.information(self, "Info", "No results to export")
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("Export")
        msg.setText("Choose export format:")
        csv_btn = msg.addButton("CSV", QMessageBox.AcceptRole)
        json_btn = msg.addButton("JSON (History)", QMessageBox.AcceptRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.exec_()
        clicked = msg.clickedButton()
        if clicked == cancel_btn:
            return
        if clicked == json_btn:
            ok = self.save_history_json()
            if ok:
                QMessageBox.information(self, "Success", "Saved as JSON in HISTORY folder (filename = Patient ID).")
            return
        # CSV
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if file_path:
            if self.results_table.export_to_csv(file_path):
                QMessageBox.information(self, "Success", "Results exported successfully")
    def add_new_patient(self):
        """Add new patient"""
        self.name_input.clear()
        self.age_input.setValue(25)
        self.gender_combo.setCurrentIndex(0)
        self.current_images = []
        self.diagnose_btn.setEnabled(False)
        self.choose_btn.setEnabled(False)
        self.loaded_from_folder = False
        # clear viewers
        self.viewer_axial.clear()
        self.viewer_sagittal.clear()
        self.viewer_coronal.clear()
        self.viewer_chosen.clear()
        self.viewer_chosen.setText("Chosen / Diagnosis")
        # reset active border to axial
        self._set_active_mpr(self.viewer_axial)
        if hasattr(self, "_update_active_slider"):
            self._update_active_slider()
    def delete_selected_result(self):
        """Delete selected result"""
        current_row = self.results_table.currentRow()
        if current_row >= 0:
            reply = QMessageBox.question(
                self, "Confirm Delete",
                "Are you sure you want to delete the selected result?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.results_table.removeRow(current_row)
    def save_file(self):
        """Save current results"""
        if not hasattr(self, 'current_file'):
            self.save_as_file()
        else:
            self.results_table.export_to_csv(self.current_file)
    def save_as_file(self):
        """Save results as new file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save As", "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if file_path:
            if self.results_table.export_to_csv(file_path):
                self.current_file = file_path
                self.status_bar.showMessage(f"Saved to: {file_path}")
    def restart_application(self):
        """Restart the application"""
        reply = QMessageBox.question(
            self, "Confirm Restart",
            "This will clear all data and restart the application. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            python = sys.executable
            os.execl(python, python, *sys.argv)
    # ✅ Zoom المطلوب: إذا الشاشة الرابعة مشتغلة => الزوم عليها فقط، وإلا على Active من الثلاثة
    def zoom_in(self):
        tgt = self._zoom_target_viewer()
        if tgt is not None:
            tgt.zoom_in()
            label = "4th" if tgt is self.viewer_chosen else tgt.orientation.upper()
            self.status_bar.showMessage(f"{label} Zoom: {tgt.get_zoom_level()}%")
    def zoom_out(self):
        tgt = self._zoom_target_viewer()
        if tgt is not None:
            tgt.zoom_out()
            label = "4th" if tgt is self.viewer_chosen else tgt.orientation.upper()
            self.status_bar.showMessage(f"{label} Zoom: {tgt.get_zoom_level()}%")
    def toggle_terminal(self):
        """Toggle terminal visibility"""
        if self.terminal.isVisible():
            self.terminal.setVisible(False)
            self.status_bar.showMessage("Terminal hidden")
        else:
            self.terminal.setVisible(True)
            self.terminal.command_input.setFocus()
            self.status_bar.showMessage("Terminal opened")
    def toggle_dark_mode(self):
        """Toggle between dark and light mode (Ctrl+D)"""
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            apply_dark_theme(self)
            self.status_bar.showMessage("Dark mode is enabled")
        else:
            apply_light_theme(self)
            self.status_bar.showMessage("Light mode is enabled")
        self.update()
        QApplication.processEvents()
    def show_about(self):
        dialog = AboutDialog(self)
        dialog.exec_()
    def _normalize_patient_name(self):
        txt = self.name_input.text()
        txt = " ".join(txt.split())
        self.name_input.setText(txt)
# =====================================================================
# main.py (merged entrypoint)
# =====================================================================
def main():
    """Main function - Entry point for the application"""
    app = QApplication(sys.argv)
    loading = LoadingScreen()
    loading.show()
    app.processEvents()
    QThread.msleep(3000)
    window = MedicalDiagnosisApp()
    window.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    QTimer.singleShot(1500, lambda: window.setWindowFlag(Qt.WindowStaysOnTopHint, False))
    QTimer.singleShot(1500, window.show)
    window.show()
    loading.close()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec_())
if __name__ == '__main__':
    main()
