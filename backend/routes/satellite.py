from fastapi import APIRouter

router = APIRouter(
    prefix="/satellite",
    tags=["Satellite"]
)


@router.get("/")
def satellite():
    return {"satellite": "connected"}