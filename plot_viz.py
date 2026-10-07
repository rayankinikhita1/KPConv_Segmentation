
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from matplotlib.colors import ListedColormap

label_id_to_name = {
    0: "No Classification",
    1: "High Voltage Power Transformer",
    2: "Grounding Transformer",
    3: "Medium Voltage Power Transformer",
    4: "Voltage Transformer",
    5: "Current Transformer",
    6: "Capacitor Bank",
    7: "Circuit Breaker",
    8: "Recloser",
    9: "Surge Arrester",
    10: "Disconnect Switch",
    11: "Actuator",
    12: "Junction Box",
    13: "Busbar",
    14: "Wire",
    15: "Insulator",
    16: "Metallic Support Structure",
    17: "Concrete Base",
    18: "Building",
    19: "Lamp",
    20: "Pipe",
    21: "Nameplate",
    22: "Artifact",
    23: "Vegetation",
    24: "Ground",
    25: "Fence"
}
# label_id_to_name = {           0: "ground",
#                                 1: "wall",
#                                 2: "busbar",
#                                 3: "wire",
#                                 4: "tower",
#                                 5: "media",
#                                 6: "builds",
#                                 7: "clutter",
#                                 8: "small",
#                                 9: "large"}

# Load prediction PLY
#pred_ply = PlyData.read('/home/nikhita_rayanki/projects/pointcloud-processing-pipeline/KPConv-PyTorch/test/Log_2025-09-12_08-04-15/predictions/03_1.ply')
pred_ply = PlyData.read('/home/nikhita_rayanki/substation1.ply')
x = np.array(pred_ply['vertex']['x'])
y = np.array(pred_ply['vertex']['y'])
z = np.array(pred_ply['vertex']['z'])
preds = np.array(pred_ply['vertex']['preds']).astype(int)

unique_classes = np.unique(preds)
num_classes = len(label_id_to_name)
# Use tab20 for up to 20 classes, otherwise combine tab20b and tab20c
if num_classes <= 20:
    cmap = plt.get_cmap('tab20', num_classes)
else:
    cmap = ListedColormap(
        np.vstack([plt.get_cmap('tab20b').colors, plt.get_cmap('tab20c').colors])[:num_classes]
    )

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=preds, cmap=cmap, s=0.1, vmin=0, vmax=num_classes-1)

# Only show ticks for present classes
cbar = plt.colorbar(sc, ax=ax, pad=0.1, ticks=unique_classes)
cbar.ax.set_yticklabels([label_id_to_name[i] for i in unique_classes])

plt.title('Predicted Point Cloud with Correct Class Colors')
plt.tight_layout()
plt.savefig('HQ2_val_prediction_vis_with_labels_substation1.png')
plt.close()