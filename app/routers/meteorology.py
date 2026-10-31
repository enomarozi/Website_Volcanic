from pathlib import Path
import json
from fastapi import APIRouter,Depends,File,Form,Request,UploadFile
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.database import get_db
from app.dependencies import get_current_user
from app.models.meteorology import MeteorologyDataset
from app.models.simulation import Simulation
from app.services.netcdf_upload import inspect_netcdf,save_upload,unique_filename

router=APIRouter(prefix="/meteorology",tags=["Meteorology"])
templates=Jinja2Templates(directory="app/templates")
BASE_DIR=Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR=BASE_DIR/"data"/"meteorology"

@router.get("")
def index(request:Request,user=Depends(get_current_user),db:Session=Depends(get_db)):
    datasets=db.query(MeteorologyDataset).order_by(MeteorologyDataset.created_at.desc()).all()
    return templates.TemplateResponse(
        request=request,
        name="meteorology/index.html",
        context={"request":request,"user":user,"datasets":datasets}
    )
    
@router.get("/upload")
def upload_page(request:Request,user=Depends(get_current_user)):
    return templates.TemplateResponse(
        request=request,
        name="meteorology/upload.html",
        context={"request":request,"user":user}
    )

@router.post("/upload")
def upload(
    request:Request,
    name:str=Form(...),
    description:str=Form(""),
    upper_air:UploadFile|None=File(None),
    surface_instant:UploadFile|None=File(None),
    surface_accum:UploadFile|None=File(None),
    user=Depends(get_current_user),
    db:Session=Depends(get_db)
):
    files=[upper_air,surface_instant,surface_accum]
    if not any(f and f.filename for f in files):
        return templates.TemplateResponse("meteorology/upload.html",{"request":request,"user":user,"error":"Minimal satu file NetCDF harus dipilih."},status_code=400)

    paths={}
    metadata={}
    try:
        for file,kind,key in [
            (upper_air,"upper_air","upper_air_path"),
            (surface_instant,"surface","surface_instant_path"),
            (surface_accum,"surface","surface_accum_path")
        ]:
            if not file or not file.filename:
                continue
            if Path(file.filename).suffix.lower()!=".nc":
                raise ValueError(f"{file.filename}: file harus berekstensi .nc")
            filename=unique_filename(file.filename,kind)
            path=UPLOAD_DIR/filename
            save_upload(file,path)
            try:
                info=inspect_netcdf(path,kind)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            paths[key]=str(path.relative_to(BASE_DIR))
            metadata[kind]=info

        selected=list(metadata.values())
        time_count=max([x["time_count"] for x in selected if x.get("time_count") is not None],default=None)
        start_time=next((x["start_time"] for x in selected if x.get("start_time")),None)
        end_time=next((x["end_time"] for x in reversed(selected) if x.get("end_time")),None)
        intervals=[x["interval_hours"] for x in selected if x.get("interval_hours") is not None]
        interval_hours=float(intervals[0]) if intervals else None
        variables=[]
        for item in selected:
            variables.extend(item.get("variables",[]))
        dataset=MeteorologyDataset(
            name=name,
            description=description or None,
            upper_air_path=paths.get("upper_air_path"),
            surface_instant_path=paths.get("surface_instant_path"),
            surface_accum_path=paths.get("surface_accum_path"),
            start_time=start_time,
            end_time=end_time,
            time_count=time_count,
            interval_hours=interval_hours,
            variables=json.dumps(sorted(set(variables))),
            status="valid",
            uploaded_by=user.id
        )
        db.add(dataset)
        db.commit()
    except Exception as exc:
        for path in paths.values():
            Path(BASE_DIR/path).unlink(missing_ok=True)
        return templates.TemplateResponse("meteorology/upload.html",{"request":request,"user":user,"error":str(exc)},status_code=400)

    return RedirectResponse("/meteorology",303)

@router.post("/{dataset_id}/delete")
def delete_dataset(
    dataset_id:int,
    user=Depends(get_current_user),
    db:Session=Depends(get_db)
):
    if not user:
        return JSONResponse(
            status_code=401,
            content={"success":False,"message":"Session login berakhir."}
        )

    dataset=db.get(MeteorologyDataset,dataset_id)

    if not dataset:
        return JSONResponse(
            status_code=404,
            content={"success":False,"message":"Dataset tidak ditemukan."}
        )

    if dataset.uploaded_by!=user.id and user.role!="admin":
        return JSONResponse(
            status_code=403,
            content={"success":False,"message":"Anda tidak memiliki izin menghapus dataset ini."}
        )
    used_count=db.query(Simulation).filter(
        Simulation.meteorology_dataset_id==dataset.id
    ).count()

    if used_count>0:
        return JSONResponse(
            status_code=409,
            content={
                "success":False,
                "message":f"Dataset tidak dapat dihapus karena masih digunakan oleh {used_count} simulasi."
            }
        )
    paths=[
        BASE_DIR/value
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

        return JSONResponse(
            status_code=200,
            content={"success":True,"message":"Dataset meteorologi berhasil dihapus."}
        )
    except Exception as e:
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"success":False,"message":str(e)}
        )