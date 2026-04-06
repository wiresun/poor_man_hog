# About
I want to get _somewhere_ and had to write _something_.

[Histogram of oriented gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) is a method for computing feature descriptors of images.

1. Compute image gradients to obtain gradient angle and magnitude for pixels.
2. Group pixels into cells and compute gradient histograms for each cell.
3. Groups cells into blocks and normalize blocks.

Since this is a poor man's version, input images are resized and converted to grayscale, cells form a rectangular partition of the image, and blocks are rectangular groups of adjecent cells.

# Running

Make sure python has access to the `numpy` and `pillow` packages.
Then simply run the following command and expect to witness heroic stature (see Picture section below):

```bash
python hog.py
```

To compare our feature descriptor to the one produced by scikit-image
[hog](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog) make sure python has access to the `scikit-image` package and
run

```bash
python test_hog.py
```

A `shell.nix` is provided for the enlightened.

# Test
By default the program outputs a visual representation of the histograms over cells (flipped 90 degrees).

As a test I wanted to compare our feature descriptor to that of scikit-image's
[hog](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog).
It turns out sckikit-image uses single bin assignments in histogram creation.

# Picture
The picture features Arnold Schwarzenegger.
It is in the public domain and obtained from [here](https://commons.wikimedia.org/wiki/File:Arnold_Schwarzenegger_1974.jpg)

