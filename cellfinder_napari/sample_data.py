from typing import List

import numpy as np
import pooch
from napari.types import LayerData
from skimage.io import imread

base_url = "https://raw.githubusercontent.com/brainglobe/cellfinder/master/tests/data/integration/detection/crop_planes"


def load_sample() -> List[LayerData]:
    """
    Load some sample data.
    """
    layers = []
    for ch, name in zip([1, 0], ["Background", "Signal"]):
        data = []
        for i in range(30):
            url = f"{base_url}/ch{ch}/ch{ch}{str(i).zfill(4)}.tif"
            file = pooch.retrieve(url=url, known_hash=None)
            data.append(imread(file))

        data = np.stack(data, axis=0)
        layers.append((data, {"name": name}))

    return layers
