class GeoJSONService:
    PARTICLE_COLORS = {
        "fine": "#dc3545",
        "small": "#dc3545",
        "medium": "#fd7e14",
        "coarse": "#ffc107",
        "large": "#ffc107"
    }

    def _validate_coordinate(self, longitude, latitude):
        lon = float(longitude)
        lat = float(latitude)

        if not -180 <= lon <= 180 or not -90 <= lat <= 90:
            raise ValueError("Invalid longitude/latitude.")

        return lon, lat

    @staticmethod
    def _feature(geometry, properties):
        return {
            "type": "Feature",
            "geometry": geometry,
            "properties": properties
        }

    @staticmethod
    def _collection(features):
        return {
            "type": "FeatureCollection",
            "features": features
        }

    def _particle_class(self, particle):
        return str(
            particle.get(
                "particle_class",
                particle.get("class", "medium")
            )
        ).lower()

    def _color(self, particle):
        return self.PARTICLE_COLORS.get(
            self._particle_class(particle),
            "#6c757d"
        )

    def trajectory_to_feature_collection(self, dispersion):
        features = []

        for particle in dispersion.get("trajectories", []):
            points = particle.get("trajectory", [])
            if len(points) < 2:
                continue

            coords = []

            for state in points:
                lon, lat = self._validate_coordinate(
                    state["longitude"],
                    state["latitude"]
                )
                coords.append([
                    lon,
                    lat,
                    max(0.0, float(state.get("altitude", 0.0)))
                ])

            features.append(self._feature(
                {
                    "type": "LineString",
                    "coordinates": coords
                },
                {
                    "particle_id": particle.get(
                        "particle_id",
                        particle.get("id")
                    ),
                    "particle_class": self._particle_class(particle),
                    "color": self._color(particle),
                    "mass": float(particle.get("mass", 0.0)),
                    "radius": float(particle.get("radius", 0.0)),
                    "radius_um": (
                        float(particle.get("radius", 0.0)) * 1e6
                    ),
                    "settling_velocity": float(
                        particle.get("settling_velocity", 0.0)
                    ),
                    "deposited": (
                        particle.get("status") == "deposited"
                    )
                }
            ))

        return self._collection(features)

    def points_to_feature_collection(self, dispersion):
        features = []

        for particle in dispersion.get("trajectories", []):
            trajectory = particle.get("trajectory", [])

            if not trajectory:
                continue

            state = trajectory[-1]

            lon, lat = self._validate_coordinate(
                state["longitude"],
                state["latitude"]
            )

            altitude = max(0.0, float(state.get("altitude", 0.0)))

            features.append(self._feature(
                {
                    "type": "Point",
                    "coordinates": [lon, lat, altitude]
                },
                {
                    "particle_id": particle.get(
                        "particle_id",
                        particle.get("id")
                    ),
                    "particle_class": self._particle_class(particle),
                    "color": self._color(particle),
                    "radius": float(particle.get("radius", 0.0)),
                    "radius_um": float(particle.get("radius", 0.0)) * 1e6,
                    "time": float(state.get("time", 0.0)),
                    "time_index": int(state.get("time_index", 0)),
                    "altitude": altitude,
                    "mass": float(particle.get("mass", 0.0)),
                    "deposited": particle.get("status") == "deposited",
                    "u": float(state.get("u", 0.0)),
                    "v": float(state.get("v", 0.0))
                }
            ))

        return self._collection(features)

    trajectory_points_to_feature_collection = (
        points_to_feature_collection
    )

    def deposition_to_feature_collection(self, dispersion):
        features = []

        for particle in dispersion.get("trajectories", []):
            deposition = particle.get("deposition")
            if not deposition:
                continue

            lon, lat = self._validate_coordinate(
                deposition["longitude"],
                deposition["latitude"]
            )

            altitude = max(
                0.0,
                float(deposition.get("altitude", 0.0))
            )

            features.append(self._feature(
                {
                    "type": "Point",
                    "coordinates": [lon, lat, altitude]
                },
                {
                    "particle_id": particle.get(
                        "particle_id",
                        particle.get("id")
                    ),
                    "particle_class": self._particle_class(particle),
                    "color": self._color(particle),
                    "mass": float(
                        deposition.get(
                            "mass",
                            particle.get("mass", 0.0)
                        )
                    ),
                    "altitude": altitude,
                    "time": float(deposition.get("time", 0.0)),
                    "mechanism": deposition.get("mechanism"),
                    "dry_probability": float(
                        deposition.get("dry_probability", 0.0)
                    ),
                    "wet_probability": float(
                        deposition.get("wet_probability", 0.0)
                    ),
                    "total_probability": float(
                        deposition.get(
                            "total_probability",
                            deposition.get("probability", 0.0)
                        )
                    ),
                    "deposited_fraction": 1.0
                }
            ))

        return self._collection(features)

    def grid_to_feature_collection(self, grid_result):
        if not isinstance(grid_result, dict):
            raise TypeError(
                "grid_result must be a dictionary."
            )

        feature_collection = grid_result.get(
            "deposition_geojson",
            grid_result.get("geojson")
        )

        if (
            isinstance(feature_collection, dict)
            and feature_collection.get("type") == "FeatureCollection"
        ):
            return feature_collection

        return self._collection([])

    def surface_deposition_to_feature_collection(self, surface):
        features = []

        if not surface:
            return self._collection(features)

        dlat = float(surface.get("dlat", 0.25))
        dlon = float(surface.get("dlon", 0.25))

        if dlat <= 0 or dlon <= 0:
            raise ValueError(
                "Grid resolution must be positive."
            )

        half_lat = dlat / 2.0
        half_lon = dlon / 2.0

        for cell in surface.get("cells", []):
            lat = float(cell["latitude"])
            lon = float(cell["longitude"])

            west = lon - half_lon
            east = lon + half_lon
            south = lat - half_lat
            north = lat + half_lat

            coords = [[
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south]
            ]]

            features.append(self._feature(
                {
                    "type": "Polygon",
                    "coordinates": coords
                },
                {
                    "mass": float(cell.get("mass", 0.0)),
                    "deposition_density": float(
                        cell.get("deposition", 0.0)
                    ),
                    "area": float(cell.get("area", 0.0)),
                    "particle_count": int(
                        cell.get("particle_count", 0)
                    )
                }
            ))

        return self._collection(features)