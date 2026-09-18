import numpy as np

from vine360.masking.classical_sky import classify_sky_classical


def test_classifies_bright_blue_top_as_sky_and_green_ground_as_not():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[0:50, :] = [135, 206, 235]  # sky blue, top half
    image[50:100, :] = [34, 90, 34]  # dark green, bottom half

    mask = classify_sky_classical(image)

    assert mask[10, 50] == 255
    assert mask[90, 50] == 0


def test_bright_object_below_top_fraction_is_not_classified_as_sky():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = [20, 60, 20]
    image[80:90, 40:60] = [250, 250, 250]  # a bright white post low in frame

    mask = classify_sky_classical(image)
    assert mask[85, 50] == 0
