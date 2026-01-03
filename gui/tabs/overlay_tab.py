import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QSlider
from PyQt5.QtCore import Qt

def create_backlit_overlay_tab(parent) -> QWidget:
    """
    Create the "Backlit Overlay" tab.

    Args:
        parent (QWidget): The parent widget (ImageEditorApp instance).

    Returns:
        QWidget: The tab widget.
    """
    tab = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(6)

    # Backlit path
    parent.backlit_path_input = QLineEdit()
    parent.backlit_path_input.setPlaceholderText("Select Backlit Image")
    backlit_button = QPushButton("Browse Backlit Image")
    backlit_button.clicked.connect(parent.select_backlit_image)

    # Opacity slider
    parent.opacity_slider = QSlider(Qt.Horizontal)
    parent.opacity_slider.setRange(0, 100)
    parent.opacity_slider.setValue(25)
    parent.opacity_slider_label = QLabel("Backlit Opacity: 25%")
    parent.opacity_slider.valueChanged.connect(parent.update_opacity_label)

    # Overlay button
    overlay_button = QPushButton("Place Overlay")
    overlay_button.clicked.connect(parent.start_overlay_preview)

    # Add widgets compactly
    layout.addWidget(QLabel("Backlit Image:"))
    layout.addWidget(parent.backlit_path_input)
    layout.addWidget(backlit_button)
    layout.addWidget(parent.opacity_slider_label)
    layout.addWidget(parent.opacity_slider)
    layout.addWidget(overlay_button)
    layout.addStretch(1)

    tab.setLayout(layout)
    return tab