import sys
import os
import uuid
import tempfile
import shutil
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QWidget, QPushButton,
    QFileDialog, QLineEdit, QSlider, QMessageBox, QProgressDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import pyvips

from workers import OverlayWorker

# ==========================
# Main application
# ==========================
class ImageEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SVS Microscopy Image Post-Processing")
        self.setGeometry(100, 100, 1200, 800)

        # State
        self.fibi_path = None
        self.working_temp_file = None  # working copy of main image
        self.overlay_ready = False

        # Main layout
        main_layout = QHBoxLayout()

        # ----- Left preview container -----
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        # Image preview
        self.image_label = QLabel("Load an image to show preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 100);
                color: white;
                padding: 4px 8px;
                font-weight: bold;
            }
        """)
        preview_layout.addWidget(self.image_label)

        # Overlay "Preview" label
        self.preview_tag = QLabel("PREVIEW ONLY", self.image_label)
        self.preview_tag.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 100);
                color: white;
                padding: 4px 8px;
                font-weight: bold;
            }
        """)
        self.preview_tag.move(10, 10)   # top-left corner
        self.preview_tag.raise_()

        main_layout.addWidget(preview_container, stretch=3)

        # ----- Right side tabs -----
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_color_correction_tab(), "Color Correction")
        self.tabs.addTab(self.create_backlit_overlay_tab(), "Backlit Overlay")
        main_layout.addWidget(self.tabs, stretch=1)


        # Center widget
        cw = QWidget()
        cw.setLayout(main_layout)
        self.setCentralWidget(cw)

        self.create_menu()

    # ========================================
    # GUI components
    # ========================================
    def create_color_correction_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Color Correction Controls (Placeholder)"))
        layout.addWidget(QPushButton("Apply Color Correction"))
        layout.addStretch(1)
        tab.setLayout(layout)
        return tab

    def create_backlit_overlay_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Backlit path
        self.backlit_path_input = QLineEdit()
        self.backlit_path_input.setPlaceholderText("Select Backlit Image")
        backlit_button = QPushButton("Browse Backlit Image")
        backlit_button.clicked.connect(self.select_backlit_image)

        # Opacity slider
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(25)
        self.opacity_slider_label = QLabel("Backlit Opacity: 25%")
        self.opacity_slider.valueChanged.connect(self.update_opacity_label)

        # Overlay button
        overlay_button = QPushButton("Place Overlay")
        overlay_button.clicked.connect(self.start_overlay_preview)

        # Add widgets compactly
        layout.addWidget(QLabel("Backlit Image:"))
        layout.addWidget(self.backlit_path_input)
        layout.addWidget(backlit_button)
        layout.addWidget(self.opacity_slider_label)
        layout.addWidget(self.opacity_slider)
        layout.addWidget(overlay_button)
        layout.addStretch(1)

        tab.setLayout(layout)
        return tab

    # ========================================
    # File loading and display
    # ========================================
    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Images (*.png *.xpm *.jpg *.bmp *.tiff *.svs)",
            options=options
        )
        if not file_path:
            return

        self.fibi_path = file_path
        self.overlay_ready = False

        # Copy to temporary working file
        self.working_temp_file = os.path.join(tempfile.gettempdir(), f"working_{uuid.uuid4()}.tiff")
        if file_path.lower().endswith(".svs"):
            image = pyvips.Image.new_from_file(file_path, access='sequential')
            image.write_to_file(self.working_temp_file)
        else:
            shutil.copy(file_path, self.working_temp_file)

        self.show_preview(self.working_temp_file)

    def show_preview(self, filepath):
        try:
            if filepath.lower().endswith((".svs", ".tiff", ".tif")):
                image = pyvips.Image.new_from_file(filepath, access='sequential')
                scale = max(image.width, image.height) / 800.0
                if scale > 1:
                    image = image.resize(1/scale)
                if image.bands == 3:
                    data = image.write_to_memory()
                    qimage = QImage(data, image.width, image.height, QImage.Format_RGB888)
                elif image.bands == 4:
                    data = image.write_to_memory()
                    qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
                else:
                    data = image.write_to_memory()
                    qimage = QImage(data, image.width, image.height, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qimage)
            else:
                pixmap = QPixmap(filepath)

            scaled = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

        except Exception as e:
            print("Preview load error:", e)
            self.image_label.setText("Failed preview load")

    # ========================================
    # Backlit browsing
    # ========================================
    def select_backlit_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Backlit Image",
            "",
            "Images (*.png *.jpg *.tiff *.svs)"
        )
        if file_path:
            self.backlit_path_input.setText(file_path)

    # ========================================
    # Overlay preview logic
    # ========================================
    def start_overlay_preview(self):
        if not self.working_temp_file:
            QMessageBox.warning(self, "Error", "Load main image first.")
            return
        fibi_path = self.fibi_path # use original FIBI path for overlay (non-temporary, non-color-corrected file)

        backlit = self.backlit_path_input.text().strip()
        if not os.path.isfile(backlit):
            QMessageBox.warning(self, "Error", "Select a valid backlit image.")
            return

        opacity = self.opacity_slider.value()
        tmp_overlay = os.path.join(tempfile.gettempdir(), f"overlay_{uuid.uuid4()}.tiff")

        self.progress = QProgressDialog("Computing overlay...", "Cancel", 0, 0, self)
        self.progress.setWindowModality(Qt.ApplicationModal)
        self.progress.show()

        self.worker = OverlayWorker(
            fibi_path=fibi_path,
            backlit_path=backlit,
            opacity=opacity,
            output_path=tmp_overlay
        )
        self.worker.done.connect(self.overlay_preview_finished)
        self.worker.error.connect(self.overlay_error)
        self.worker.start()

    def overlay_preview_finished(self, output_path):
        self.progress.close()
        # Replace working file with updated overlay
        self.working_temp_file = output_path
        self.overlay_ready = True
        self.show_preview(self.working_temp_file)

    def overlay_error(self, msg):
        self.progress.close()
        QMessageBox.critical(self, "Overlay Error", msg)

    # ========================================
    # Slider label update
    # ========================================
    def update_opacity_label(self):
        value = self.opacity_slider.value()
        self.opacity_slider_label.setText(f"Backlit Opacity: {value}%")

    # ========================================
    # Menu + saving
    # ========================================
    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        load_action = file_menu.addAction("Load Image")
        load_action.triggered.connect(self.load_image)

        write_action = file_menu.addAction("Save As...")
        write_action.triggered.connect(self.save_as)

    def save_as(self):
        if not self.overlay_ready or not self.working_temp_file:
            QMessageBox.warning(self, "Error", "No changes to write.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save overlay TIFF",
            "",
            "TIFF Files (*.tiff)"
        )
        if not save_path:
            return

        try:
            shutil.copy(self.working_temp_file, save_path)
            QMessageBox.information(self, "Success", "File saved!")
        except Exception as e:
            QMessageBox.critical(self, "Error: File Not Saved", str(e))


# ========================================
# main
# ========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageEditorApp()
    window.show()
    sys.exit(app.exec_())
