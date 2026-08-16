class GeoJSONService:

    def trajectories_to_geojson(self, trajectories):
        features = []

        grouped = {}

        for point in trajectories:
            particle_id = point["particle_id"]

            if particle_id not in grouped:
                grouped[particle_id] = []

            grouped[particle_id].append(point)

        for particle_id, points in grouped.items():
            coordinates = []

            for point in points:
                coordinates.append([
                    point["longitude"],
                    point["latitude"],
                    point["altitude"]
                ])

            features.append({
                "type": "Feature",
                "properties": {
                    "particle_id": particle_id
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                }
            })

        return {
            "type": "FeatureCollection",
            "features": features
        }