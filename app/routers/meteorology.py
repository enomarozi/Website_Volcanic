from pathlib import Path
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.dependencies import get_current_user
from app.models.meteorology import MeteorologyDataset
from app.models.simulation import Simulation

router = APIRouter(prefix="/meteorology", tags=["Meteorology"])
BASE_DIR = Path(__file__).resolve().parent.parent.parent

@router.post("/{dataset_id}/delete")
def delete_dataset(
    dataset_id: int,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not user:
        return JSONResponse(status_code=401, content={
            "success": False,
            "message": "Session login berakhir."
        })

    dataset = db.get(MeteorologyDataset, dataset_id)
    if not dataset:
        return JSONResponse(status_code=404, content={
            "success": False,
            "message": "Dataset tidak ditemukan."
        })

    if dataset.uploaded_by != user.id and user.role != "admin":
        return JSONResponse(status_code=403, content={
            "success": False,
            "message": "Anda tidak memiliki izin menghapus dataset ini."
        })

    used_count = db.query(Simulation).filter(
        Simulation.meteorology_dataset_id == dataset.id
    ).count()

    if used_count > 0:
        return JSONResponse(status_code=409, content={
            "success": False,
            "message": f"Dataset tidak dapat dihapus karena masih digunakan oleh {used_count} simulasi."
        })

    paths = [
        BASE_DIR / value
        for value in (
            dataset.upper_air_path,
            dataset.surface_instant_path,
            dataset.surface_accum_path
        )
        if value
    ]

    try:
        db.delete(dataset)
        db.commit()

        for path in paths:
            path.unlink(missing_ok=True)

        return JSONResponse(status_code=200, content={
            "success": True,
            "message": "Dataset meteorologi berhasil dihapus."
        })

    except Exception as exc:
        db.rollback()
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": str(exc)
        })