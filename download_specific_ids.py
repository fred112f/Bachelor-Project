from qgis.core import QgsProject
from qgis.utils import iface
from qgis.PyQt.QtCore import QTimer

# --- Configuration ---
output_folder = r"C:\temp\screenshots_Google"
render_delay_ms = 2000
fixed_scale = 144448
image_ids_to_process = [123, 456, 789] 

project = QgsProject.instance()
canvas = iface.mapCanvas()
vertices_layer = project.mapLayersByName("Vertices")
if not vertices_layer:
    raise Exception("Layer 'Vertices' not found")
vertices_layer = vertices_layer[0]

features_to_process = [
    f for f in vertices_layer.getFeatures()
    if f.id() in image_ids_to_process
]
features_to_process.sort(key=lambda f: f.id())

google_layers = project.mapLayersByName("Google")
if google_layers:
    google_tree_layer = project.layerTreeRoot().findLayer(google_layers[0].id())
    if google_tree_layer:
        google_tree_layer.setItemVisibilityChecked(True)

vertices_tree_layer = project.layerTreeRoot().findLayer(vertices_layer.id())
if vertices_tree_layer:
    vertices_tree_layer.setItemVisibilityChecked(False)

current_index = 0

def finalize_screenshot():
    global current_index
    feat = features_to_process[current_index]
    out_path = f"{output_folder}/screenshot_{feat.id()}.png"
    canvas.saveAsImage(out_path)
    print("Saved screenshot to:", out_path)
    current_index += 1
    if current_index < len(features_to_process):
        process_feature(current_index)
    else:
        print("All requested features processed.")

def on_render_complete():
    canvas.renderComplete.disconnect(on_render_complete)
    QTimer.singleShot(render_delay_ms, finalize_screenshot)

def process_feature(index):
    feat = features_to_process[index]
    vertices_layer.selectByIds([feat.id()])
    canvas.zoomToSelected(vertices_layer)
    canvas.zoomScale(fixed_scale)
    canvas.renderComplete.connect(on_render_complete)
    canvas.refresh()

if features_to_process:
    process_feature(current_index)
else:
    pass