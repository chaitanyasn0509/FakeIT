from pathlib import Path
import torch
import rasterio

from models.restormer_model import RestormerReconstructor


def load_image(path: Path):
    with rasterio.open(path) as src:
        image = src.read().astype("float32")
        profile = src.profile

    image = torch.from_numpy(image).unsqueeze(0)

    return image, profile


def save_image(image, profile, output_path):
    image = image.squeeze(0).cpu().numpy()

    profile.update(
        count=image.shape[0],
        dtype="float32",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(image)


def main():
    checkpoint = "checkpoints/best.ckpt"

    model = RestormerReconstructor.load_from_checkpoint(checkpoint)

    model.eval()

    image, profile = load_image(Path("sample.tif"))

    with torch.no_grad():
        prediction = model(image)

    save_image(prediction, profile, "output.tif")

    print("Saved output.tif")


if __name__ == "__main__":
    main()