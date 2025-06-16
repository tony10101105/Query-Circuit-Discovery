import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


data = np.load('edge_scores.npy')

valid_mask = np.isfinite(data)
valid_values = data[valid_mask]

# Normalize valid values to 0~1
min_val, max_val = valid_values.min(), valid_values.max()

normalized_data = (data - min_val) / (max_val - min_val)
normalized_data[~valid_mask] = 0  # assign 0 to -inf entries
print(normalized_data.max())
print(normalized_data.min())

max_idx = np.unravel_index(np.argmax(data), data.shape)
print(max_idx)
# Create the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(data, aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Value')
plt.title('Heatmap of 2D Array')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.tight_layout()
plt.savefig('edge_score.png')