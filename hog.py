#!/usr/bin/env python
"""
Histogram of oriented gradients
"""

from typing import cast
from PIL import Image, ImageDraw
import numpy as np
import argparse

# [Height, Width]
type Size = tuple[int, int]

type ArrayU8 = np.typing.NDArray[np.uint8]

type ArrayF64 = np.typing.NDArray[np.float64]


def poor_man_hog(
    image: ArrayU8,
    cell_size: Size,
    block_size: Size,
    orientations: int,
    bilinear_binning: bool,
):
    """
    Compute HOG feature descriptor.

    image: 2-dimensional array of unsigned 8-bit integers.
    cell_size: (height, width) of cells in pixels
    block_size: (height, width) of blocks in cells.
    orientations: the number of orientation bins in the histograms
    bilinear_binning: when true, use bilinearly interpolated bin assignment.
                      Otherwise, use single bin assignments
    """
    assert image.ndim == 2

    h, w = cast(tuple[int, int], image.shape[0:2])

    grad_rows, grad_cols = gradient(image)

    histograms = collect_histograms(
        grad_rows,
        grad_cols,
        cell_size=cell_size,
        orientations=orientations,
        bilinear_binning=bilinear_binning,
    )

    hog_image = visualize_hog(
        histograms, image_size=(h, w), cell_size=cell_size, orientations=orientations
    )

    normalized_blocks = normalize_blocks(
        histograms, block_size=block_size, orientations=orientations
    )

    return normalized_blocks, hog_image


def grayscale(arr: ArrayU8) -> ArrayU8:
    """
    Grayscale
    """
    if arr.ndim != 3:
        return arr

    assert arr.shape[2] == 3

    weights = np.array([0.299, 0.587, 0.114])
    float_result = cast(np.typing.NDArray[np.float32], arr.dot(weights))
    result = float_result.astype(np.uint8, copy=False)

    return result


def nn_scale(arr: ArrayU8, height: int, width: int) -> ArrayU8:
    """
    Nearest neighbor interpolated scaling. Return (height, width)
    """
    assert arr.ndim >= 2

    orig_h, orig_w = cast(tuple[float, float], arr.shape[0:2])

    row_scale = orig_h / height
    col_scale = orig_w / width

    rows = np.uint(row_scale * np.arange(height))
    cols = np.uint(col_scale * np.arange(width))

    scaled_array = arr[np.ix_(rows, cols)]

    return scaled_array


def gradient(
    arr: ArrayU8,
) -> tuple[ArrayF64, ArrayF64]:
    """
    Compute the image gradient using standard [-1, 0, 1] kernel
    """
    arr_f64 = arr.astype(dtype=np.float64, copy=False)

    grad_row = np.zeros(arr.shape)
    grad_row[1:-1, :] = arr_f64[2:, :] - arr_f64[:-2, :]

    grad_col = np.zeros(arr.shape)
    grad_col[:, 1:-1] = arr_f64[:, 2:] - arr_f64[:, :-2]

    return grad_row, grad_col


def unsigned_angles_and_magnitudes(
    grad_row: ArrayF64, grad_col: ArrayF64
) -> tuple[ArrayF64, ArrayF64]:
    angles = np.rad2deg(np.arctan2(grad_row, grad_col)) % 180.0
    magnitudes = np.sqrt(grad_row**2 + grad_col**2)
    return angles, magnitudes


def single_bin(orientations: int, histogram: ArrayF64, angle: float, magnitude: float):
    """
    Assign angle, magnitude to single bin.
    """
    bin_width = 180.0 / orientations
    mid = (angle % 180.0) / bin_width
    bin = int(mid)
    histogram[bin] += magnitude


def bilinear_bin(
    orientations: int,
    histogram: ArrayF64,
    angle: float,
    magnitude: float,
):
    """
    Bilinearly interpolated binning.

    Evenly distribute magnitude over bin_width centered at angle.

    For example, when angle is 45.0 and bin_width is 20.0, assign
    5.0 * magnitude to bin [20, 40) and 15 * magnitude to bin [40, 60)

    Wraps at 0 and 180. For example, angle 0 assigns half magintude to [0, 20)
    and half to [160, 180).
    """
    bin_width = 180.0 / orientations
    mid = ((angle - bin_width / 2) % 180.0) / bin_width

    lo = int(mid)
    hi = (lo + 1) % orientations

    hi_weight = mid - lo
    lo_weight = 1.0 - hi_weight

    histogram[lo] += magnitude * lo_weight
    histogram[hi] += magnitude * hi_weight


def cell_grid(arr: np.ndarray, cell_size: Size) -> np.ndarray:
    """
    Partition arr into rectangular cells, dropping partial cells, which
    only occur when cell dimensions don't divide the array dimensions.

    Assumes arr.ndim == 2.
    """
    cell_h, cell_w = cell_size
    h, w = cast(tuple[int, int], arr.shape[0:2])

    return arr.reshape(h // cell_h, cell_h, w // cell_w, cell_w).transpose([0, 2, 1, 3])


def collect_histograms(
    grad_row: ArrayF64,
    grad_col: ArrayF64,
    cell_size: Size,
    orientations: int,
    bilinear_binning: bool,
) -> ArrayF64:
    """
    Collect gradient angle histograms over cells, weighted by gradient magnitude.

    Return array of histograms by cell grid coordinates [cell_row, cell_col, histogram]
    """
    assert grad_row.shape == grad_col.shape
    assert grad_row.ndim == 2

    cell_h, cell_w = cell_size
    h, w = cast(tuple[int, int], grad_row.shape)

    # Assert cell grid fits image
    assert h % cell_h == 0 and w % cell_w == 0

    cell_grid_h = h // cell_h
    cell_grid_w = w // cell_w

    angle, magnitude = unsigned_angles_and_magnitudes(grad_row, grad_col)
    angle_cells = cell_grid(angle, cell_size)
    magnitude_cells = cell_grid(magnitude, cell_size)
    histograms = np.zeros([cell_grid_h, cell_grid_w, orientations])

    if bilinear_binning:
        bin = bilinear_bin
    else:
        bin = single_bin

    for g_row in range(cell_grid_h):
        for g_col in range(cell_grid_w):
            cell_histogram = cast(ArrayF64, histograms[g_row, g_col])
            angle_cell = cast(ArrayF64, angle_cells[g_row, g_col])
            magnitude_cell = cast(ArrayF64, magnitude_cells[g_row, g_col])

            for cell_r in range(cell_h):
                for cell_c in range(cell_w):
                    bin(
                        orientations=orientations,
                        histogram=cell_histogram,
                        angle=cast(float, angle_cell[cell_r, cell_c]),
                        magnitude=cast(float, magnitude_cell[cell_r, cell_c]),
                    )

    return histograms


def normalize_l2(block: ArrayF64, eps: float) -> ArrayF64:
    sum_squares = np.sum(block**2)
    block_norm = cast(float, np.sqrt(sum_squares + eps**2))
    return block / block_norm


def normalize_l2_hys(block: ArrayF64, eps: float) -> ArrayF64:
    result = normalize_l2(block, eps)
    result = np.minimum(result, 0.2)
    return normalize_l2(result, eps)


def normalize_block(block: ArrayF64, eps: float = 1.0e-7) -> ArrayF64:
    return normalize_l2_hys(block, eps)


def normalize_blocks(
    histograms: ArrayF64, block_size: Size, orientations: int
) -> ArrayF64:
    block_height, block_width = block_size
    blocks_rows = cast(int, histograms.shape[0] + 1 - block_height)
    blocks_cols = cast(int, histograms.shape[1] + 1 - block_width)

    result = np.zeros(
        [blocks_rows, blocks_cols, block_height, block_width, orientations],
        dtype=float,
    )

    for row in range(blocks_rows):
        for col in range(blocks_cols):
            block = histograms[row : row + block_height, col : col + block_width]
            result[row, col] = normalize_block(block)

    return result


def visualize_hog(
    histograms: ArrayF64,
    image_size: Size,
    cell_size: Size,
    orientations: int,
) -> Image.Image:
    """
    Reconstruct image from histograms over cells.

    For each histogram and each bin therein, draw a line perpendicular to
    the bin's orientation whose brightness depends on magnitude.
    """
    hog_image = Image.new("RGB", (image_size[1], image_size[0]), color="black")
    draw = ImageDraw.Draw(hog_image)

    cell_h, cell_w = cell_size
    cell_grid_h, cell_grid_w = cast(tuple[int, int], histograms.shape[0:2])

    for cell_r in range(cell_grid_h):
        for cell_c in range(cell_grid_w):
            histogram = histograms[cell_r, cell_c, :]
            bin_width = 180.0 // orientations

            cell_lines: list[
                tuple[float, tuple[float, float], tuple[float, float]]
            ] = []

            for bin in range(orientations):
                cell_centre_col = cell_c * cell_w + cell_w // 2
                cell_centre_row = cell_r * cell_h + cell_h // 2

                angle = bin * bin_width + 10.0 + 90.0
                angle = cast(float, np.deg2rad(angle % 180.0))

                dr = cast(float, 0.3 * cell_h * np.sin(angle))
                dc = cast(float, 0.3 * cell_w * np.cos(angle))

                mag = cast(float, histogram[bin])
                start = (cell_centre_col - dc, cell_centre_row - dr)
                end = (cell_centre_col + dc, cell_centre_row + dr)

                cell_lines.append((mag, start, end))

            # Sort lines to plot brightest lines on top
            cell_lines.sort(key=lambda line: line[0])

            for idx, (mag, start, end) in enumerate(cell_lines):
                # Experimentally determined line brightness
                color_scale = 0.1 * (idx / orientations) ** 2
                b = int(color_scale * mag)
                draw.line([start, end], fill=(b, b, b), width=1)

    return hog_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="hog.py", description="Compute HOG feature descriptor for image"
    )
    _ = parser.add_argument("--image", help="path to image file", default="arnold.jpg")
    _ = parser.add_argument(
        "--print-fd",
        action="store_true",
        help="print feature descriptor instead of showing a picture",
    )
    args = parser.parse_args()
    image_path = cast(str, args.image)
    print_fd = cast(bool, args.print_fd)

    width = 128
    height = 256
    cell_size = (8, 8)
    block_size = (2, 2)
    number_of_bins = 9

    image = Image.open(image_path)
    image = np.array(image)
    image = nn_scale(image, height, width)
    image = grayscale(image).astype(np.uint8)

    poor_hog, poor_hog_image = poor_man_hog(
        image,
        cell_size=cell_size,
        block_size=block_size,
        orientations=number_of_bins,
        bilinear_binning=True,
    )

    if print_fd:
        import sys

        np.set_printoptions(threshold=sys.maxsize)

        print(poor_hog.ravel())
    else:
        _ = poor_hog_image.show()
