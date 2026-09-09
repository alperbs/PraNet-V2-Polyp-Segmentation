import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lib.pranet import PraNet_V2
from utils.dataloader import test_dataset


# =========================================================
# Configuration
# =========================================================

TEST_SIZE = 352

DATASETS = [
    "CVC-300",
    "CVC-ClinicDB",
    "Kvasir",
    "ETIS-LaribPolypDB",
]

CHECKPOINT = "./snapshots/PraNet-V2/RES-V2.pth"
RESULTS_DIR = "./results/PraNet-V2"


# =========================================================
# Model
# =========================================================

def load_model():
    """Load the pretrained PraNet-V2 model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("=" * 50)
    print("PraNet-V2")
    print("=" * 50)
    print("Device:", device)

    model = PraNet_V2(num_class=1)
    print("Model created.")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint,
        strict=False
    )

    if missing_keys:
        print("Warning - missing keys:", len(missing_keys))

    if unexpected_keys:
        print("Warning - unexpected keys:", len(unexpected_keys))

    print("Checkpoint loaded:", CHECKPOINT)

    model = model.to(device)
    model.eval()

    print("Model ready.")
    print()

    return model, device


# =========================================================
# Dataset Testing
# =========================================================

def test_dataset_model(model, device, dataset_name):
    """Run inference on one dataset."""

    data_path = os.path.join(
        "./data/TestDataset",
        dataset_name
    )

    image_root = os.path.join(data_path, "images") + "/"
    gt_root = os.path.join(data_path, "masks") + "/"

    save_path = os.path.join(
        RESULTS_DIR,
        dataset_name
    )

    os.makedirs(save_path, exist_ok=True)

    print()
    print("=" * 50)
    print("Dataset:", dataset_name)
    print("=" * 50)

    test_loader = test_dataset(
        image_root,
        gt_root,
        TEST_SIZE
    )

    print("Number of images:", test_loader.size)

    with torch.inference_mode():

        for _ in tqdm(
            range(test_loader.size),
            desc=dataset_name
        ):

            image, gt, name = test_loader.load_data()

            # Ground-truth is only used to obtain the
            # original image dimensions for interpolation.
            gt = np.asarray(gt, dtype=np.float32)

            image = image.to(device)

            # PraNet-V2 multi-level outputs
            (
                res2,
                res3,
                res4,
                res5,
                _res2_bg,
                _res3_bg,
                _res4_bg,
                _res5_bg
            ) = model(image)

            # Combine foreground predictions
            prediction = res2 + res3 + res4 + res5

            # Restore original image dimensions
            prediction = F.interpolate(
                prediction,
                size=gt.shape,
                mode="bilinear",
                align_corners=False
            )

            # Convert logits to probabilities
            prediction = torch.sigmoid(prediction)

            # Move to CPU and convert to NumPy
            prediction = (
                prediction
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )

            # Normalize to [0, 1]
            prediction = (
                prediction - prediction.min()
            ) / (
                prediction.max() - prediction.min() + 1e-8
            )

            # Convert to 8-bit grayscale mask
            prediction_uint8 = (
                prediction * 255
            ).astype(np.uint8)

            # Save prediction
            imageio.imwrite(
                os.path.join(save_path, name),
                prediction_uint8
            )

    print()
    print("Completed:", dataset_name)
    print("Results:", save_path)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    model, device = load_model()

    for dataset_name in DATASETS:
        test_dataset_model(
            model,
            device,
            dataset_name
        )

    print()
    print("=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)
    print()
    print("Results:", RESULTS_DIR)