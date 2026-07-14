from fastapi import APIRouter

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"]
)


@router.get("/")
def datasets():
    return {"datasets": "available"}