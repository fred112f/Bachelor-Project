from qgis.core import QgsProject
from qgis.utils import iface
from qgis.PyQt.QtCore import QTimer

project = QgsProject.instance()
canvas = iface.mapCanvas()
udtrukket_layer = project.mapLayersByName("Vertices")[0]
osm_layers = project.mapLayersByName("Google")
if osm_layers:
    osm_tree_layer = project.layerTreeRoot().findLayer(osm_layers[0].id())
    if osm_tree_layer:
        osm_tree_layer.setItemVisibilityChecked(True)
udtrukket_tree_item = project.layerTreeRoot().findLayer(udtrukket_layer.id())
udtrukket_tree_item.setItemVisibilityChecked(False)
all_features = list(udtrukket_layer.getFeatures())
all_features.sort(key=lambda f: f.id()) 
output_folder = r"C:\temp\screenshots_Google"
render_delay_ms = 2000
fixed_scale = 144448
current_index = 0

def finalize_screenshot():
    global current_index
    feat = all_features[current_index]
    out_path = f"{output_folder}/screenshot_{feat.id()}.png"
    canvas.saveAsImage(out_path)
    print("Saved screenshot to:", out_path)
    current_index += 1
    if current_index < len(all_features):
        process_feature(current_index)
    else:
        print("All features processed.")
def on_render_complete():
    canvas.renderComplete.disconnect(on_render_complete)
    QTimer.singleShot(render_delay_ms, finalize_screenshot)
def process_feature(index):
    feat = all_features[index]
    udtrukket_layer.selectByIds([feat.id()])
    canvas.zoomToSelected(udtrukket_layer)
    canvas.zoomScale(fixed_scale)
    canvas.renderComplete.connect(on_render_complete)
    canvas.refresh()
process_feature(current_index) 