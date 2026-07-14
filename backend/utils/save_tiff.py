import rasterio


def save_prediction(reference_path, prediction, output_path):

    with rasterio.open(reference_path) as src:

        profile = src.profile

    profile.update(dtype="float32")

    with rasterio.open(output_path, "w", **profile) as dst:

        dst.write(prediction.astype("float32"))