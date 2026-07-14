from pathlib import Path
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.inference_service import InferenceService

router = APIRouter(
    prefix="/predict",
    tags=["Inference"],
)

service = InferenceService()


@router.post("/")
async def predict(
    image: UploadFile = File(...),
):
    """
    Upload a cloudy TIFF image and receive a cloud-free prediction.
    """

    if not image.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(
            status_code=400,
            detail="Only TIFF images are supported.",
        )

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    input_path = upload_dir / image.filename

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    output_name = input_path.stem + "_prediction.tif"

    result = service.infer(
        image_path=str(input_path),
        output_path=f"results/{output_name}",
    )

    return {
        "status": "success",
        **result,
    }