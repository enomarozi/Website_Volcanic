class GeoJSONService:
    def trajectory_to_feature_collection(self, simulation):
        features = []
        for particle in simulation["trajectories"]:
            coordinates = [[point["longitude"], point["latitude"], point["altitude"]] for point in particle["trajectory"]]
            if len(coordinates) < 2:
                continue
            features.append({
                "type": "Feature",
                "properties": {
                    "particle_id": particle["particle_id"],
                    "class": particle["class"],
                    "radius": particle["radius"],
                    "mass": particle["mass"],
                    "settling_velocity": particle["settling_velocity"]
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                }
            })
        return {"type": "FeatureCollection", "features": features}

    def trajectory_points_to_feature_collection(self, simulation):
        features = []
        for particle in simulation["trajectories"]:
            for point in particle["trajectory"]:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "particle_id": particle["particle_id"],
                        "class": particle["class"],
                        "radius": particle["radius"],
                        "mass": particle["mass"],
                        "settling_velocity": particle["settling_velocity"],
                        "time": point["time"],
                        "time_index": point["time_index"],
                        "altitude": point["altitude"],
                        "u": point["u"],
                        "v": point["v"],
                        "vertical": point["vertical"]
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [point["longitude"], point["latitude"]]
                    }
                })
        return {"type": "FeatureCollection", "features": features}