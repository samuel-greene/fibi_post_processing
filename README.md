# SVS Microscopy Image Post-Processing

This application provides a graphical interface for post-processing microscopy images, including:

- **Color Correction**: Adjust image colors.
- **Backlit Overlay**: Add and adjust backlit overlays with customizable opacity.

## Features
- Load and preview microscopy images.
- Apply overlays with adjustable opacity.
- Save processed images as TIFF files.

## Requirements
- Python 3.8+
- PyQt5
- pyvips

## Usage
1. Run the application:
   ```bash
   python3 main.py
   ```
2. Use the GUI to load images, apply overlays, and save results.

## Notes
- I often ran into missing dependency issues with pyvips and vips on mac and was able to fix by adding "export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH" to path.