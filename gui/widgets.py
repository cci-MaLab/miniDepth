"""Qt dock widgets used by the Napari volume viewer application."""

import os
from datetime import datetime

import numpy as np
from PyQt5.QtWidgets import (
    QFileDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .core import open_minian


class CellListWidget(QWidget):
    """Widget that lists labels and filters the volume by selected IDs."""

    def __init__(self, volume_viewer):
        """Initialize widget with a shared `VolumeViewer` controller."""
        super().__init__()
        self.volume_viewer = volume_viewer
        self.nearby_widget = NearbyWidget()
        self.init_ui()

    def init_ui(self):
        """Build list and action buttons for cell selection."""
        layout = QVBoxLayout()

        self.cell_list = QListWidget()
        self.cell_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.cell_list)

        view_button = QPushButton("View Selected Cells")
        view_button.clicked.connect(self.view_selected_cells)
        layout.addWidget(view_button)

        reset_button = QPushButton("Reset/Default View")
        reset_button.clicked.connect(self.reset_view)
        layout.addWidget(reset_button)

        layout.addStretch(1)
        self.setLayout(layout)
        self.setMinimumWidth(200)
        self.update_cell_list()

    def update_cell_list(self, preserve_selection=False):
        """Refresh listed IDs from viewer and optionally preserve selection."""
        prev_selection = [item.text() for item in self.cell_list.selectedItems()] if preserve_selection else []

        self.cell_list.setUpdatesEnabled(False)
        self.cell_list.clear()
        for cid in self.volume_viewer.get_unique_cell_ids():
            self.cell_list.addItem(str(cid))

        for i in range(self.cell_list.count()):
            item = self.cell_list.item(i)
            if item.text() in prev_selection:
                item.setSelected(True)
        self.cell_list.setUpdatesEnabled(True)

    def view_selected_cells(self):
        """Push selected IDs to viewer and refresh nearby-cell lists."""
        selected_items = self.cell_list.selectedItems()
        selected_cells = [int(item.text()) for item in selected_items]
        self.volume_viewer.selected_cells = selected_cells
        self.volume_viewer.original_selected_cells = selected_cells.copy()
        self.volume_viewer.update_viewer()

        if selected_cells:
            overlapping, adjacent = self.volume_viewer.find_nearby_cells(selected_cells)
            self.volume_viewer.update_nearby_widget(overlapping, adjacent)

    def reset_view(self):
        """Clear active selection and restore full-volume display."""
        self.volume_viewer.selected_cells = None
        self.volume_viewer.update_viewer()


class NearbyWidget(QWidget):
    """Widget for selecting overlapping and adjacent cells."""

    def __init__(self, selection_update_fn=None, reset_fn=None):
        """Initialize callbacks used to synchronize with the viewer."""
        super().__init__()
        self.selection_update_fn = selection_update_fn
        self.reset_fn = reset_fn
        self.init_ui()
        self.setWindowTitle("Nearby Cells Widget")

    def init_ui(self):
        """Build overlapping/adjacent lists and reset button."""
        layout = QVBoxLayout()

        self.overlap_label = QLabel("Overlapping Cells")
        self.overlap_list = QListWidget()

        self.adjacent_label = QLabel("Adjacent Cells")
        self.adjacent_list = QListWidget()

        self.overlap_list.setSelectionMode(QListWidget.MultiSelection)
        self.overlap_list.itemSelectionChanged.connect(self.on_selection_changed)

        self.adjacent_list.setSelectionMode(QListWidget.MultiSelection)
        self.adjacent_list.itemSelectionChanged.connect(self.on_selection_changed)

        self.reset_button = QPushButton("Remove Added Cells")
        self.reset_button.clicked.connect(self.reset_to_original_selection)

        layout.addWidget(self.overlap_label)
        layout.addWidget(self.overlap_list)
        layout.addWidget(self.adjacent_label)
        layout.addWidget(self.adjacent_list)
        layout.addWidget(self.reset_button)

        self.setLayout(layout)
        layout.addStretch(1)

    def update_lists(self, overlap_cells, adjacent_cells):
        """Populate overlap/adjacent lists with new IDs."""
        self.overlap_list.clear()
        self.adjacent_list.clear()
        self.overlap_list.addItems(map(str, overlap_cells))
        self.adjacent_list.addItems(map(str, adjacent_cells))

    def clear_lists(self):
        """Clear overlap/adjacent list widgets."""
        self.overlap_list.clear()
        self.adjacent_list.clear()

    def on_selection_changed(self):
        """Forward currently selected nearby IDs to the viewer callback."""
        overlap_selected = [int(item.text()) for item in self.overlap_list.selectedItems()]
        adjacent_selected = [int(item.text()) for item in self.adjacent_list.selectedItems()]

        selected_nearby_cells = set(overlap_selected + adjacent_selected)
        print(f"Selected nearby cells: {selected_nearby_cells}")
        if self.selection_update_fn:
            self.selection_update_fn(selected_nearby_cells)
        else:
            print("No selection update function provided.")

    def reset_to_original_selection(self):
        """Ask viewer to reset and clear nearby lists."""
        if self.reset_fn:
            self.reset_fn()
            self.clear_lists()
        else:
            print("No reset function provided.")


class MergeCellsWidget(QWidget):
    """Widget to merge label IDs and save edited label volumes."""

    def __init__(self, viewer):
        """Initialize merge and save controls.

        Args:
            viewer: Shared `VolumeViewer` controller.
        """
        super().__init__()
        self.viewer = viewer
        self.setWindowTitle("Merge Cells")
        self.init_ui()

    def init_ui(self):
        """Build merge input fields and action buttons."""
        layout = QVBoxLayout()

        self.label = QLabel("Enter two cell IDs to merge:")
        layout.addWidget(self.label)

        self.input1 = QLineEdit()
        self.input1.setPlaceholderText("Cell ID 1 (target)")
        layout.addWidget(self.input1)

        self.input2 = QLineEdit()
        self.input2.setPlaceholderText("Cell ID 2 (to replace)")
        layout.addWidget(self.input2)

        self.merge_button = QPushButton("Merge Cells")
        self.merge_button.clicked.connect(self.merge_cells)
        layout.addWidget(self.merge_button)

        self.save_button = QPushButton("Save Volume")
        self.save_button.clicked.connect(self.save_volume)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def merge_cells(self):
        """Replace all voxels of source ID with target ID."""
        try:
            id1 = int(self.input1.text())
            id2 = int(self.input2.text())
        except ValueError:
            print("Please enter valid integers.")
            return

        print(f"Merging cell {id2} into {id1}")
        self.viewer.volume[self.viewer.volume == id2] = id1

        self.viewer.update_viewer()
        self.input1.clear()
        self.input2.clear()

    def save_volume(self):
        """Write current edited label volume to an `.npy` file."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Volume", "", "NumPy Files (*.npy)")
        edited = self.viewer.labels_layer.data
        print("Unique values in edited:", np.unique(edited))
        new_edits = np.setdiff1d(np.unique(edited), np.unique(self.viewer.volume))
        print("New label values introduced by editing:", new_edits)

        # Compare against the displayed pre-edit content so only intended edits apply.
        pre_display = np.where(
            np.isin(self.viewer.volume, self.viewer.selected_cells),
            self.viewer.volume,
            0,
        )

        changes_mask = edited != pre_display
        np.copyto(self.viewer.volume, edited, where=changes_mask)

        self.viewer.labels_layer.data = self.viewer.volume
        self.viewer.update_viewer()

        if file_path:
            np.save(file_path, self.viewer.volume)
            print(f"Volume saved to {file_path}")


class LoadLayerWidget(QWidget):
    """Widget for loading external MINIAN layers into the viewer."""

    def __init__(self, volume_viewer):
        """Initialize load control with a shared `VolumeViewer`."""
        super().__init__()
        self.volume_viewer = volume_viewer
        self.setWindowTitle("Load New Layer")
        self.init_ui()

    def init_ui(self):
        """Build single-button folder loader UI."""
        layout = QVBoxLayout()

        self.load_button = QPushButton("Load New Layer")
        self.load_button.clicked.connect(self.load_new_layer)
        layout.addWidget(self.load_button)

        self.setLayout(layout)

    def load_new_layer(self):
        """Open folder chooser and attempt to load MINIAN data."""
        file_dialog = QFileDialog()
        folder_path = file_dialog.getExistingDirectory(None, "Select Layer Folder")
        if not folder_path:
            return

        try:
            minian_data = open_minian(folder_path)
            self.volume_viewer.add_minian_footprint_to_viewer(minian_data)
        except Exception as e:
            print(f"Error loading layer: {e}")


class PairSaveWidget(QWidget):
    """Widget to save, view, and delete Minian-column pair map files."""

    def __init__(self, volume_viewer, save_dir):
        """Initialize pair-management UI and filesystem target directory."""
        super().__init__()
        self.viewer = volume_viewer
        self.save_dir = os.path.join(save_dir, "saved_column_pairs")
        os.makedirs(self.save_dir, exist_ok=True)

        self.setWindowTitle("Save Minian-Column Pair")
        layout = QVBoxLayout()

        self.label = QLabel("Enter Minian ID and Column ID")
        layout.addWidget(self.label)

        self.minian_input = QLineEdit()
        self.minian_input.setPlaceholderText("Minian ID")
        layout.addWidget(self.minian_input)

        self.column_input = QLineEdit()
        self.column_input.setPlaceholderText("Column ID")
        layout.addWidget(self.column_input)

        self.save_button = QPushButton("Save Pair")
        self.save_button.clicked.connect(self.save_pair)
        layout.addWidget(self.save_button)

        self.pair_list = QListWidget()
        self.refresh_list()
        layout.addWidget(self.pair_list)

        self.view_button = QPushButton("View Selected Pair")
        self.view_button.clicked.connect(self.view_pair)
        layout.addWidget(self.view_button)

        self.delete_button = QPushButton("Delete Selected Pair")
        self.delete_button.clicked.connect(self.delete_pair)
        layout.addWidget(self.delete_button)

        self.setLayout(layout)

    def refresh_list(self):
        """Reload the list of saved NPZ pair files."""
        self.pair_list.clear()
        for fname in sorted(os.listdir(self.save_dir)):
            if fname.endswith(".npz"):
                self.pair_list.addItem(fname)

    def save_pair(self):
        """Save selected Minian and column maps into a timestamped NPZ file."""
        try:
            minian_id = int(self.minian_input.text())
            column_id = int(self.column_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integers for both IDs.")
            return

        minian_map = self.get_map(self.viewer.minian_labels_layer.data[0], minian_id)
        column_map = self.get_map(self.viewer.volume, column_id)

        if minian_map is None or column_map is None:
            QMessageBox.warning(self, "Missing Data", "Could not find both maps.")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"minian_{minian_id}_column_{column_id}__{timestamp}.npz"
        save_path = os.path.join(self.save_dir, filename)

        np.savez_compressed(
            save_path,
            minian_map=minian_map,
            column_map=column_map,
            minian_id=minian_id,
            column_id=column_id,
        )

        self.refresh_list()
        self.minian_input.clear()
        self.column_input.clear()
        QMessageBox.information(self, "Saved", f"Saved pair as {filename}")

    def get_map(self, data, cell_id):
        """Return masked map for one ID, or `None` when ID is absent."""
        mask = data == cell_id
        return data * mask if np.any(mask) else None

    def view_pair(self):
        """Load selected pair and add both maps as Napari labels layers."""
        selected = self.pair_list.currentItem()
        if not selected:
            return
        path = os.path.join(self.save_dir, selected.text())
        data = np.load(path)

        self.viewer.viewer.add_labels(data["minian_map"], name="Saved Minian")
        self.viewer.viewer.add_labels(data["column_map"], name="Saved Column")

    def delete_pair(self):
        """Delete selected pair file and refresh UI list."""
        selected = self.pair_list.currentItem()
        if not selected:
            return
        path = os.path.join(self.save_dir, selected.text())
        os.remove(path)
        self.refresh_list()
        QMessageBox.information(self, "Deleted", f"Deleted {selected.text()}")
