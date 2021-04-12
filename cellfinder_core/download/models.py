import os
from imlib.general.system import ensure_directory_exists

from cellfinder_core.download.download import download


model_weight_urls = {
    "resnet50_tv": "https://gin.g-node.org/cellfinder/models/raw/"
    "master/resnet50_tv.h5",
    "resnet50_all": "https://gin.g-node.org/cellfinder/models/raw/"
    "master/resnet50_weights.h5",
}

download_requirements_gb = {
    "resnet50_tv": 0.18,
    "resnet50_all": 0.18,
}


def main(model, download_path):
    model_weight_dir = os.path.join(download_path, "model_weights")
    model_path = os.path.join(model_weight_dir, model + ".h5")
    if not os.path.exists(model_path):
        ensure_directory_exists(model_weight_dir)

        download_path = os.path.join(model_weight_dir, model + ".h5")
        download(
            download_path,
            model_weight_urls[model],
            model,
            download_requires=download_requirements_gb[model],
        )

    else:
        print(f"Model already exists at {model_path}. Skipping download")

    return model_path
