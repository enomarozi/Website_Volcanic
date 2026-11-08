class GeoJSONService:
    PARTICLE_COLORS={"fine":"#3388ff","small":"#3388ff","medium":"#ffc107","coarse":"#dc3545","large":"#dc3545"}
    def _validate_coordinate(self,longitude,latitude):
        lon,lat=float(longitude),float(latitude)
        if not -180<=lon<=180 or not -90<=lat<=90: raise ValueError("Invalid longitude/latitude.")
        return lon,lat
    @staticmethod
    def _feature(geometry,properties): return {"type":"Feature","geometry":geometry,"properties":properties}
    @staticmethod
    def _collection(features): return {"type":"FeatureCollection","features":features}
    def _particle_class(self,p): return str(p.get("particle_class",p.get("class","medium"))).lower()
    def _color(self,p): return self.PARTICLE_COLORS.get(self._particle_class(p),"#6c757d")
    def trajectory_to_feature_collection(self,dispersion):
        features=[]
        for p in dispersion.get("trajectories",[]):
            points=p.get("trajectory",[])
            if len(points)<2: continue
            coords=[]
            for s in points:
                lon,lat=self._validate_coordinate(s["longitude"],s["latitude"]); coords.append([lon,lat,max(0.0,float(s.get("altitude",0.0)))])
            features.append(self._feature({"type":"LineString","coordinates":coords},{"particle_id":p.get("particle_id",p.get("id")),"particle_class":self._particle_class(p),"color":self._color(p),"mass":float(p.get("mass",0.0)),"radius":float(p.get("radius",0.0)),"radius_um":float(p.get("radius",0.0))*1e6,"settling_velocity":float(p.get("settling_velocity",0.0)),"deposited":p.get("status")=="deposited"}))
        return self._collection(features)
    def points_to_feature_collection(self,dispersion):
        features=[]
        for p in dispersion.get("trajectories",[]):
            for s in p.get("trajectory",[]):
                lon,lat=self._validate_coordinate(s["longitude"],s["latitude"]); alt=max(0.0,float(s.get("altitude",0.0)))
                features.append(self._feature({"type":"Point","coordinates":[lon,lat,alt]},{"particle_id":p.get("particle_id",p.get("id")),"particle_class":self._particle_class(p),"color":self._color(p),"time":float(s.get("time",0.0)),"time_index":int(s.get("time_index",0)),"altitude":alt,"mass":float(p.get("mass",0.0)),"deposited":p.get("status")=="deposited","u":float(s.get("u",0.0)),"v":float(s.get("v",0.0))}))
        return self._collection(features)
    trajectory_points_to_feature_collection=points_to_feature_collection
    def deposition_to_feature_collection(self,dispersion):
        features=[]
        for p in dispersion.get("trajectories",[]):
            d=p.get("deposition")
            if not d: continue
            lon,lat=self._validate_coordinate(d["longitude"],d["latitude"])
            features.append(self._feature({"type":"Point","coordinates":[lon,lat,max(0.0,float(d.get("altitude",0.0)))]},{"particle_id":p.get("particle_id",p.get("id")),"particle_class":self._particle_class(p),"color":self._color(p),"mass":float(d.get("mass",p.get("mass",0.0))),"altitude":max(0.0,float(d.get("altitude",0.0))),"time":float(d.get("time",0.0)),"mechanism":d.get("mechanism"),"dry_probability":float(d.get("dry_probability",0.0)),"wet_probability":float(d.get("wet_probability",0.0)),"total_probability":float(d.get("total_probability",d.get("probability",0.0))),"deposited_fraction":1.0}))
        return self._collection(features)
    def grid_to_feature_collection(self,grid_result):
        if not isinstance(grid_result,dict): raise TypeError("grid_result must be a dictionary.")
        fc=grid_result.get("deposition_geojson",grid_result.get("geojson"))
        return fc if isinstance(fc,dict) and fc.get("type")=="FeatureCollection" else self._collection([])
    def surface_deposition_to_feature_collection(self,surface):
        features=[]
        for c in (surface or {}).get("cells",[]):
            lat=float(c["latitude"]); lon=float(c["longitude"]); dlat=float((surface or {}).get("dlat",0.1)); dlon=float((surface or {}).get("dlon",0.1))
            coords=[[[lon,lat],[lon+dlon,lat],[lon+dlon,lat+dlat],[lon,lat+dlat],[lon,lat]]]
            features.append(self._feature({"type":"Polygon","coordinates":coords},{"mass":float(c.get("mass",0.0)),"deposition_density":float(c.get("deposition",0.0)),"area":float(c.get("area",0.0)),"particle_count":int(c.get("particle_count",0))}))
        return self._collection(features)
