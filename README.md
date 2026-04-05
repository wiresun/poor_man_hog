# About
I want to get _somewhere_ and had to write _something_. 

[Histogram of oriented gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) is a method for computing feature descriptors of images. 

1. Compute image gradients to obtain gradient angle and magnitude for pixels.
2. Partition the image into regular grid of rectangular cells.
   For each cell, collect a histogram gradient angles weighted by magnitude. 
3. Normalize gradient histograms over blocks, which are rectangular views of the grid of cells. 


# Running

Make sure python has access to the `numpy`, `pillow` and `scikit-image` packages.
Then simply run the following command and expect to witness heroic stature (see Picture section below): 

```bash
python hog.py
```

To compare our feature descriptor to the one produced by scikit-image
[hog](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog)
run

```bash
python test_hog.py
```

A `shell.nix` is provided for the enlightened. 

# Test

As a test I wanted to compare our feature descriptor to that of scikit-image's
[hog](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog).
It turns out sckikit-image uses single bin assignments in histogram creation.

# Picture
The picture features Arnold Schwarzenegger.
It is in the public domain and obtained from [here](https://commons.wikimedia.org/wiki/File:Arnold_Schwarzenegger_1974.jpg)

