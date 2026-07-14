from pathlib import Path
import time

import numpy as np
import rasterio
import torch

from evaluation.metrics import compute_metrics
from models.restormer_model import RestormerReconstructor


class InferenceService:
    """Inference service for FakeIT cloud removal."""

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = Path("checkpoints/best-000-0.0000.ckpt")

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        print(f"Loading model from {checkpoint}...")

        self.model = RestormerReconstructor.load_from_checkpoint(
            checkpoint,
            map_location=self.device,
        )

        self.model.eval()
        self.model.to(self.device)

        print(f"Model loaded on {self.device}")

    def preprocess(self, image_path):

        image_path = Path(image_path)

        with rasterio.open(image_path) as src:

            image = src.read().astype(np.float32)

            profile = src.profile.copy()

        image = np.clip(image, 0, 10000)

        image /= 10000.0

        tensor = (
            torch.from_numpy(image)
            .unsqueeze(0)
            .to(self.device)
        )

        return tensor, profile

    def predict(self, image_path, target_path=None):

        tensor, profile = self.preprocess(image_path)

        start_time = time.time()

        with torch.no_grad():

            prediction = self.model(tensor)

        inference_time = time.time() - start_time

        prediction = (
            prediction.squeeze(0)
            .cpu()
            .numpy()
        )

        prediction = np.clip(prediction, 0.0, 1.0)

        metrics = None

        if target_path is not None:

            with rasterio.open(target_path) as src:

                target = src.read().astype(np.float32)

            target = np.clip(target, 0, 10000)

            target /= 10000.0

            metrics = compute_metrics(
                prediction,
                target,
            )

        return prediction, metrics, profile, inference_time

    def save_prediction(
        self,
        prediction,
        profile,
        output_path,
    ):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = (
            prediction * 10000.0
        ).astype(np.uint16)

        profile.update(
            driver="GTiff",
            dtype="uint16",
            count=output.shape[0],
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:

            dst.write(output)

        return str(output_path)

    def infer(
        self,
        image_path,
        output_path="results/prediction.tif",
        target_path=None,
    ):

        prediction, metrics, profile, inference_time = self.predict(
            image_path=image_path,
            target_path=target_path,
        )

        saved_file = self.save_prediction(
            prediction,
            profile,
            output_path,
        )

        return {
            "prediction_path": saved_file,
            "metrics": metrics,
            "inference_time": round(inference_time, 3),
            "device": str(self.device),
            "model": "Restormer",
        }