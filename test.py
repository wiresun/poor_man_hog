from typing import cast
import numpy as np
import skimage as ski
from PIL import Image

import main as m

width = 128
height = 256
cell_size = (8, 8)
block_size = (2, 2)
number_of_bins = 9

im = Image.open("arnold.jpg")
image = np.array(im)
image = m.nn_scale(image, height, width)
image = m.grayscale(image).astype(np.uint8)

poor_hog, _ = m.poor_man_hog(
    image,
    cell_size=cell_size,
    block_size=block_size,
    orientations=number_of_bins,
    bilinear_binning=False,
)

ski_hog = ski.feature.hog(
    image,
    orientations=number_of_bins,
    pixels_per_cell=cell_size,
    cells_per_block=block_size,
    block_norm="L2-Hys",
    feature_vector=False,
    visualize=False,
)

hog_delta = cast(m.ArrayF64, poor_hog - ski_hog)
delta = np.linalg.norm(hog_delta)
if np.isclose(poor_hog, ski_hog, rtol=1.0e-5, atol=1.0e-5).all():
    print(f"Test succeed: feature discriptors are similar (delta = {delta})")
else:
    print(f"Test failed: delta = {delta} ")
