import math

class ConcentrationService:
    def __init__(self,earth_radius=6.371e6,default_dz=500.0):
        self.earth_radius=float(earth_radius); self.default_dz=float(default_dz)

    def cell_area(self,latitude,dlat=0.25,dlon=0.25):
        phi=math.radians(float(latitude)); dphi=math.radians(float(dlat)); dlambda=math.radians(float(dlon))
        return self.earth_radius**2*abs(math.cos(phi))*dphi*dlambda

    def cell_volume(self,latitude,dlat=0.25,dlon=0.25,dz=None):
        dz=self.default_dz if dz is None else float(dz)
        if dz<=0: raise ValueError("Vertical grid size must be positive.")
        return self.cell_area(latitude,dlat,dlon)*dz

    def grid_index(self,latitude,longitude,altitude,origin_latitude,origin_longitude,dlat=0.25,dlon=0.25,dz=None):
        dz=self.default_dz if dz is None else float(dz)
        return (math.floor((float(latitude)-float(origin_latitude))/dlat),math.floor((float(longitude)-float(origin_longitude))/dlon),math.floor(max(float(altitude),0.0)/dz))

    def accumulate(self,trajectories,origin_latitude=None,origin_longitude=None,dlat=0.25,dlon=0.25,dz=None,active_only=True):
        if not trajectories: return {"time_series":[],"grid":{},"metadata":{},"max_concentration":0.0,"cell_count":0,"grid_size":dlat}
        dz=self.default_dz if dz is None else float(dz)
        if min(float(dlat),float(dlon),dz)<=0: raise ValueError("Grid resolutions must be positive.")
        valid=[p for p in trajectories if p.get("trajectory")]
        if not valid: return {"time_series":[],"grid":{},"metadata":{},"max_concentration":0.0,"cell_count":0,"grid_size":dlat}
        origin_latitude=min(float(p["trajectory"][0]["latitude"]) for p in valid) if origin_latitude is None else float(origin_latitude)
        origin_longitude=min(float(p["trajectory"][0]["longitude"]) for p in valid) if origin_longitude is None else float(origin_longitude)
        times=sorted({float(s["time"]) for p in valid for s in p["trajectory"]}); series=[]; final_grid={}; max_c=0.0; max_cells=0
        for t in times:
            cells={}
            for p in valid:
                dep=p.get("deposition") or {}
                if active_only and p.get("status")=="deposited" and dep.get("time") is not None and float(dep["time"])<=t+1e-9: continue
                prior=[s for s in p["trajectory"] if float(s["time"])<=t+1e-9]
                if not prior: continue
                state=prior[-1]; key=self.grid_index(state["latitude"],state["longitude"],state["altitude"],origin_latitude,origin_longitude,dlat,dlon,dz)
                cells[key]=cells.get(key,0.0)+float(p.get("mass",0.0))
            output=[]; snapshot={}
            for (i,j,k),mass in cells.items():
                lat=origin_latitude+(i+0.5)*dlat; lon=origin_longitude+(j+0.5)*dlon
                volume=self.cell_volume(lat,dlat,dlon,dz); c=mass/volume if volume else 0.0
                item={"i":i,"j":j,"k":k,"latitude":lat,"longitude":lon,"altitude":(k+0.5)*dz,"mass":mass,"volume":volume,"concentration":c}
                output.append(item); snapshot[str((i,j,k))]={"mass":mass,"volume":volume,"concentration":c}; max_c=max(max_c,c)
            max_cells=max(max_cells,len(output)); series.append({"time":t,"cells":output}); final_grid=snapshot
        return {"time_series":series,"grid":final_grid,"max_concentration":max_c,"cell_count":len(final_grid),"grid_size":float(dlat),
                "metadata":{"origin_latitude":origin_latitude,"origin_longitude":origin_longitude,"dlat":float(dlat),"dlon":float(dlon),"dz":dz,"earth_radius":self.earth_radius,"units":"kg/m3","grid_represents":"last available time snapshot","max_cells_per_snapshot":max_cells}}

    def surface_deposition(self,deposited_particles,dlat=0.25,dlon=0.25,origin_latitude=0.0,origin_longitude=0.0):
        cells={}
        for p in deposited_particles:
            dep=p.get("deposition")
            if not dep: continue
            lat=float(dep["latitude"]); lon=float(dep["longitude"])
            i=math.floor((lat-float(origin_latitude))/dlat); j=math.floor((lon-float(origin_longitude))/dlon); key=(i,j)
            cells.setdefault(key,{"i":i,"j":j,"mass":0.0,"latitude":float(origin_latitude)+(i+0.5)*dlat,"longitude":float(origin_longitude)+(j+0.5)*dlon,"particle_count":0})
            cells[key]["mass"]+=float(dep.get("mass",p.get("mass",0.0))); cells[key]["particle_count"]+=1
        for cell in cells.values():
            cell["area"]=self.cell_area(cell["latitude"],dlat,dlon); cell["deposition"]=cell["mass"]/cell["area"] if cell["area"] else 0.0
        values=list(cells.values())
        return {"cells":values,"total_deposited_mass":sum(x["mass"] for x in values),"cell_count":len(values),"units":"kg/m2","dlat":dlat,"dlon":dlon}

    def plume_metrics(self,mean_wind_speed,duration,plume_width_km=20.0,dlat=0.25,dlon=0.25,latitude=0.0):
        length_km=max(float(mean_wind_speed),0.0)*max(float(duration),0.0)/1000.0
        footprint_km2=length_km*max(float(plume_width_km),0.0)
        cell_km2=self.cell_area(latitude,dlat,dlon)/1e6
        return {"plume_length_km":length_km,"plume_width_km":float(plume_width_km),"footprint_km2":footprint_km2,
                "cell_area_km2":cell_km2,"estimated_affected_cells":footprint_km2/cell_km2 if cell_km2 else 0.0}
