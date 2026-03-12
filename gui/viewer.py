"""Napari viewer orchestration for volume exploration and pairing workflows."""

import os

import napari
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QListWidget, QPushButton, QVBoxLayout, QWidget

from .core import near_by_cells, find_overlapping_cells, show_label_stack_pyvista
from .stats import generate_and_save_stats
from .widgets import CellListWidget, LoadLayerWidget, MergeCellsWidget, NearbyWidget, PairSaveWidget


class VolumeViewer:
    """Controller object that wires data, Napari layers, and dock widgets.

    The class centralizes interaction state for selected columns and loaded
    Minian footprints, and exposes methods used by UI widgets.
    """

    def __init__(self, volume, save_dir):
        """Initialize viewer state.

        Args:
            volume: Label volume array of shape `(z, y, x)`.
            save_dir: Session directory where pair files and exports are saved.
        """
        self.volume = volume
        self.original_selected_cells = None
        self.selected_cells = None
        self.viewer = None
        self.labels_layer = None
        self.cell_list_widget = None
        self.nearby_widget = None
        self.overlapping_dict = None
        self.minian_dict = None
        self.minain_data = None
        self.minian_footprints = None
        self.minian_footprints_labels = None
        self.grid_view_enabled = False
        self.save_dir = save_dir

    def create_napari_viewer(self):
        """Create the Napari viewer and add the main labels layer."""
        print("Loading the volume into napari")
        self.viewer = napari.Viewer()
        print("Adding the volume as a labels layer")

        self.labels_layer = self.viewer.add_labels(self.volume, name="Full Stack")
        self.labels_layer.mouse_double_click_callbacks.append(self.on_double_click)
        print("Done")
        return self.viewer

    def dock_widgets(self):
        """Create and attach all right-side dock widgets and action buttons."""
        print("Docking the widgets")
        self.cell_list_widget = CellListWidget(self)
        self.nearby_widget = NearbyWidget(self.handle_nearby_selection, self.reset_to_original_selection)

        print("Adding the cell list widget")
        self.viewer.window.add_dock_widget(self.cell_list_widget, area='right')

        print("Adding the nearby widget")
        self.viewer.window.add_dock_widget(self.nearby_widget, area='right')

        print("Adding the merge cells widget")
        self.merge_cells_widget = MergeCellsWidget(self)
        self.viewer.window.add_dock_widget(self.merge_cells_widget, area='right')

        print("Adding the load layer widget")
        self.load_layer_widget = LoadLayerWidget(self)
        self.viewer.window.add_dock_widget(self.load_layer_widget, area='right')

        plot_button = QPushButton("Plot")
        plot_button.clicked.connect(lambda: show_label_stack_pyvista(self.labels_layer.data))
        
        self.viewer.window.add_dock_widget(plot_button, area='right')

        self.pair_save_widget = PairSaveWidget(self, self.save_dir)
        self.viewer.window.add_dock_widget(self.pair_save_widget, area='right')

        stats_button = QPushButton("Generate Stats")
        stats_button.clicked.connect(self.generate_stats)
        self.viewer.window.add_dock_widget(stats_button, area='right')

    def update_viewer(self):
        """Refresh displayed labels using the current cell selection state."""
        print("Updating the viewer")
        if self.selected_cells is None:
            print("No selection, displaying the full volume")
            self.labels_layer.data = self.volume
        else:
            print(f"Displaying the selected cells: {self.selected_cells}")
            mask = np.isin(self.volume, self.selected_cells)
            filtered = np.where(mask, self.volume, 0)
            self.labels_layer.data = filtered

    def get_unique_cell_ids(self):
        """Return non-zero label IDs currently visible in the labels layer."""
        return [int(cid) for cid in np.unique(self.labels_layer.data) if cid != 0]

    def on_double_click(self, _event, _):
        """Handle double-click selection for both column and Minian label layers."""

        active_layer = self.viewer.layers.selection.active
        if not active_layer:
            print("No active layer selected.")
            return

        layer_name = active_layer.name
        print(f"Double-clicked on layer: {layer_name}")

        status = self.viewer.status
        coord_text = status.get('coordinates', '')
        parts = coord_text.split(':')
        if len(parts) == 2:
            coords_str = parts[0].strip()
            label_str = parts[1].strip()
            try:
                # Coordinates are currently parsed only for debugging/logging.
                coords = tuple(int(num) for num in coords_str.strip('[]').split())
                column_id = int(label_str)
            except Exception as e:
                print("Error parsing coordinates/label:", e)
                return
        else:
            coords = ()
            column_id = None

        if layer_name == "Full Stack":
            if column_id and column_id != 0:
                self.selected_cells = [column_id]
                self.original_selected_cells = self.selected_cells.copy()
                self.update_viewer()
                overlapping, adjacent = self.find_nearby_cells(self.selected_cells)
                self.update_nearby_widget(overlapping, adjacent)

        if layer_name == "Minian Footprints Labels":
            if column_id:
                self.minian_dict = find_overlapping_cells(self.minian_data, self.volume)
                overlapping_ids = self.minian_dict[column_id]
                print(f"Overlapping IDs: {overlapping_ids}")
                self.selected_cells = list(set(overlapping_ids))
                self.original_selected_cells = self.selected_cells.copy()
                self.update_viewer()
                overlapping, adjacent = self.find_nearby_cells(self.selected_cells)
                self.update_nearby_widget(overlapping, adjacent)
                self.update_minain_layer([column_id])

    def find_nearby_cells(self, main_cells):
        """Compute overlapping and adjacent neighbors for selected labels."""
        print(f"Finding nearby cells for: {main_cells}")
        overlapping, adjacent = near_by_cells(main_cells, self.volume)
        print(f"Overlapping: {overlapping}")
        print(f"Adjacent: {adjacent}")
        return overlapping, adjacent

    def update_nearby_widget(self, overlapping, adjacent):
        """Push neighbor results into the nearby-cell widget if mounted."""
        if self.nearby_widget:
            print(f"Updating nearby widget with overlapping: {overlapping} and adjacent: {adjacent}")
            self.nearby_widget.update_lists(overlapping, adjacent)

    def handle_nearby_selection(self, new_cells):
        """Merge nearby widget selections into the current active selection."""
        print(f"Handling nearby selection: {new_cells}")
        if self.original_selected_cells is None:
            print("Original selected cells is None, returning")
            return
        valid_cells = [cell for cell in new_cells if cell in np.unique(self.volume)]
        print(f"Valid nearby cells: {valid_cells}")
        combined = list(set(self.original_selected_cells).union(new_cells))
        self.selected_cells = combined
        print(f"Updating viewer with new selection: {self.selected_cells}")
        self.update_viewer()

    def reset_to_original_selection(self):
        """Reset selection back to the original user-selected cell set."""
        print("Resetting to original selection")
        if self.original_selected_cells is None:
            print("Original selected cells is None, returning")
            return
        print(f"Original selected cells: {self.original_selected_cells}")
        self.selected_cells = self.original_selected_cells
        print(f"Updating viewer with new selection: {self.selected_cells}")
        self.update_viewer()

    def update_minain_layer(self, selected_items):
        """Filter Minian labels layer to only selected Minian unit IDs."""
        minian_mask = np.isin(self.minian_labels_layer.data, selected_items)
        filtered_minian = np.where(minian_mask, self.minian_labels_layer.data, 0)
        self.minian_labels_layer.data = filtered_minian

    def minian_add_to_view(self):
        """Add selected Minian units and their overlapping columns to the view."""
        selected_items = self.minian_list_widget.selectedItems()

        self.minian_dict = find_overlapping_cells(self.minian_data, self.volume)

        selected_minian_ids = [int(item.text()) for item in selected_items]
        print(f"Selected Minian IDs: {selected_minian_ids}")
        overlapping_ids = [item for minian_id in selected_minian_ids if minian_id in self.minian_dict for item in self.minian_dict[minian_id]]

        print(f"Overlapping IDs: {overlapping_ids}")

        self.selected_cells = list(set(overlapping_ids))
        overlapping, adjacent = self.find_nearby_cells(self.selected_cells)
        self.update_nearby_widget(overlapping, adjacent)

        self.update_viewer()

        self.update_minain_layer([selected_minian_ids])

    def add_minian_list_widget(self):
        """Create a dock widget listing Minian IDs for manual filtering."""
        print("Adding Minian list widget")
        widget = QWidget()
        layout = QVBoxLayout()

        self.minian_list_widget = QListWidget()
        self.minian_list_widget.setSelectionMode(QListWidget.MultiSelection)

        ids = self.overlapping_dict.keys()
        self.minian_list_widget.addItems(map(str, ids))

        layout.addWidget(self.minian_list_widget)

        add_button = QPushButton("Add to Viewer")
        layout.addWidget(add_button)
        add_button.clicked.connect(self.minian_add_to_view)

        reset_button = QPushButton("Reset Selection")
        layout.addWidget(reset_button)
        reset_button.clicked.connect(self.minian_reset_to_original_selection)
        widget.setLayout(layout)
        self.viewer.window.add_dock_widget(widget, area='right')

    def add_minian_footprint_to_viewer(self, minian_data):
        """Add MINIAN footprints as label/image layers and precompute overlaps."""
        self.minian_data = minian_data

        self.minian_footprints = minian_data['A']
        unit_ids = minian_data['A']['unit_id'].values

        # Convert per-unit footprints into one 2D labeled ID map.
        updated_footprints = np.zeros(self.minian_footprints.shape[1:], dtype=int)
        for unit_id in unit_ids:
            footprint = self.minian_footprints.sel(unit_id=unit_id).data
            updated_footprints[footprint > 0] = unit_id

        self.minian_footprints_labels = np.expand_dims(updated_footprints, axis=0)

        self.minian_labels_layer = self.viewer.add_labels(self.minian_footprints_labels, name = "Minian Footprints Labels")
        self.minian_labels_layer.mouse_double_click_callbacks.append(self.on_double_click)

        self.minian_image_layer = self.viewer.add_image(np.array(np.max(self.minian_footprints.data, axis=0)), name = "Minian Footprints")

        self.overlapping_dict = find_overlapping_cells(minian_data, self.volume)

        print(f"Found {len(self.overlapping_dict)} overlapping cells in volume.")
        
        overlapping_ids = list(set(item for sublist in self.overlapping_dict.values() for item in sublist))

        print(f"Overlapping IDs: {overlapping_ids}")

        print(f"Found {len(overlapping_ids)} overlapping cells in volume.")

        self.selected_cells = list(overlapping_ids)
        self.update_viewer()

        self.add_minian_list_widget()

    def minian_reset_to_original_selection(self):
        """Restore Minian layer and corresponding overlapping column selection."""
        print("Resetting to original selection")    

        self.minian_labels_layer.data = self.minian_footprints_labels
        self.minian_dict = find_overlapping_cells(self.minian_data, self.volume)
        overlapping_ids = list(set(item for sublist in self.minian_dict.values() for item in sublist))
        print(f"Overlapping IDs: {overlapping_ids}")

        self.selected_cells = list(overlapping_ids)
        self.update_viewer()

    def generate_stats(self):
        """Generate CSV metrics from saved Minian-column pair files."""
        saved_pairs_dir = os.path.join(self.save_dir, "saved_column_pairs")
        tiff_base_path = os.path.join(self.save_dir, "PRE_PROCESSED_TIFF")
        output_csv_path = os.path.join(saved_pairs_dir, "column_metrics.csv")
        generate_and_save_stats(saved_pairs_dir, tiff_base_path, output_csv_path)
        QMessageBox.information(
            self.viewer.window._qt_window,
            "Stats Generated",
            f"Stats saved to:\n{output_csv_path}",
        )









