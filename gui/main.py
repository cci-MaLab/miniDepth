"""Application entry point for launching the Napari-based column viewer."""

# Allow running as a script (`python gui/main.py`) as well as a module (`python -m gui`).
if __name__ == "__main__" and __package__ is None:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "gui"

import argparse
import os
import sys

from PyQt5.QtWidgets import QApplication

from .core import load_volume
from .viewer import VolumeViewer


def main():
    """Load the dataset specified on the command line and launch the interactive viewer UI."""
    parser = argparse.ArgumentParser(description="Napari-based 3D column viewer")
    parser.add_argument("data_dir", help="Path to the session directory (e.g. G:\\AK\\Cell_Overlap\\New\\SG006_3D_D3)")
    args = parser.parse_args()

    main_dir = args.data_dir
    global_masks_dir = os.path.join(main_dir, "GLOBAL_PLANE_MASKS")
    num_planes = len(os.listdir(global_masks_dir))

    volume = load_volume(global_masks_dir, num_planes)
    print(volume.shape)

    volume_viewer = VolumeViewer(volume, main_dir)
    app = QApplication.instance() or QApplication([])
    volume_viewer.create_napari_viewer()
    volume_viewer.dock_widgets()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
