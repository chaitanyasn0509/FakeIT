from fastapi import APIRouter

router = APIRouter(
    prefix="/training",
    tags=["Training"]
)


@router.get("/")
def status():
    return {"training": "ready"}