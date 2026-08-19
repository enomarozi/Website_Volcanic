class GeoJSONService:
    def trajectory_to_feature_collection(self, simulation):
        features = []

        for particle in simulation["trajectories"]:
            coordinates = [
                [
                    float(point["longitude"]),
                    float(point["latitude"]),
                    float(point["altitude"])
                ]
                for point in particle["trajectory"]
            ]

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
                        "time": float(point["time"]),
                        "time_index": int(point["time_index"]),
                        "altitude": float(point["altitude"]),
                        "u": float(point["u"]),
                        "v": float(point["v"]),
                        "vertical": float(point["vertical"])
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            float(point["longitude"]),
                            float(point["latitude"]),
                            float(point["altitude"])
                        ]
                    }
                })

        return {"type": "FeatureCollection", "features": features}

    def create(self, simulation):
        return {
            "trajectory": self.trajectory_to_feature_collection(simulation),
            "points": self.trajectory_points_to_feature_collection(simulation)
        }