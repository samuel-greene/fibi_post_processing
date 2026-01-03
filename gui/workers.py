from PyQt5.QtCore import QThread, pyqtSignal
from core import create_overlay  # our custom overlay function

# --------------------------
# Worker thread for overlay
# --------------------------
class OverlayWorker(QThread):
    done = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, fibi_path, backlit_path, opacity, output_path):
        super().__init__()
        self.fibi_path = fibi_path
        self.backlit_path = backlit_path
        self.opacity = opacity
        self.output_path = output_path

    def run(self):
        try:
            print(self.opacity)
            if self.opacity > 1:
                self.opacity /= 100.0
            if self.opacity > 1.0 or self.opacity < 0.0:
                raise ValueError("Opacity must be between 0 and 1.")
            result = create_overlay(
                fibi_path=self.fibi_path,
                backlit_path=self.backlit_path,
                output_path=self.output_path,
                opacity=self.opacity,
            )
            self.done.emit(result)
        except Exception as e:
            self.error.emit(str(e))