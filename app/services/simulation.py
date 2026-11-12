
from typing import Any
from app.services.deposition_grid import DepositionGridService
import math
import numpy as np


class SimulationService:
    def __init__(
        self,
        meteorology,
        eruption,
        particle,
        dispersion,
        geojson,
        concentration=None,
        landcover=None,
        surface_meteorology=None,
        terrain=None
    ):
        self.meteorology = meteorology
        self.eruption = eruption
        self.particle = particle
        self.concentration = concentration
        self.dispersion = dispersion
        self.geojson = geojson
        self.concentration = concentration
        self.landcover = landcover
        self.terrain = terrain

        if surface_meteorology is None:
            surface_meteorology = getattr(
                dispersion, "surface_meteorology", None
            )
        self.deposition_grid = DepositionGridService(cell_size=0.01)
        self.surface_meteorology = surface_meteorology
        self.dispersion.terrain = terrain
        self.dispersion.landcover = landcover
        self.dispersion.surface_meteorology = surface_meteorology

    def run(self, config) -> dict[str, Any]:
        info = self.meteorology.time_information()

        if not info.get("times"):
            raise ValueError(
                "Dataset meteorologi tidak memiliki waktu yang valid."
            )

        time_index = self._resolve_time_index(config, info)

        if self.terrain is None:
            raise RuntimeError("TerrainService wajib tersedia.")

        latitude = float(config.latitude)
        longitude = float(config.longitude)
        source_altitude = float(config.altitude)
        eruption_height = float(config.eruption_height)

        if not all(math.isfinite(v) for v in (
            latitude, longitude, source_altitude, eruption_height
        )):
            raise ValueError(
                "Parameter lokasi atau tinggi erupsi tidak valid."
            )

        if eruption_height <= 0:
            raise ValueError(
                "Tinggi erupsi harus lebih besar dari 0."
            )

        terrain_data = self.terrain.elevation_at(
            latitude, longitude
        )
        ground_elevation = float(terrain_data["elevation"])

        if not math.isfinite(ground_elevation):
            raise ValueError(
                "Elevasi DEM sumber erupsi tidak valid."
            )

        release_clearance = 1.0
        release_altitude = max(
            source_altitude,
            ground_elevation + release_clearance
        )

        altitude_correction = (
            release_altitude - source_altitude
        )
        column_top = release_altitude + eruption_height

        profile = self.meteorology.bilinear_profile(
            latitude,
            longitude,
            time_index
        )

        if not isinstance(profile, dict) or "altitude" not in profile:
            raise ValueError(
                "Profil GFS tidak menyediakan data altitude yang valid."
            )

        z = np.asarray(
            profile["altitude"], dtype=float
        )
        z = z[np.isfinite(z)]

        if z.size < 2:
            raise ValueError(
                "Profil GFS memiliki kurang dari dua level ketinggian valid."
            )

        gfs_min = float(np.min(z))
        gfs_max = float(np.max(z))
        maximum_eruption_height = (
            gfs_max - release_altitude
        )

        print("\n========== VERTICAL VALIDATION ==========")
        print(f"Latitude         : {latitude:.6f}")
        print(f"Longitude        : {longitude:.6f}")
        print(f"Time index       : {time_index}")
        print(f"Vent altitude    : {source_altitude:.2f} m AMSL")
        print(f"DEM elevation    : {ground_elevation:.2f} m AMSL")
        print(f"Release altitude : {release_altitude:.2f} m AMSL")
        print(f"Eruption height  : {eruption_height:.2f} m AGL")
        print(f"Column top       : {column_top:.2f} m AMSL")
        print(f"GFS minimum      : {gfs_min:.2f} m AMSL")
        print(f"GFS maximum      : {gfs_max:.2f} m AMSL")
        print(f"Maximum height   : {maximum_eruption_height:.2f} m AGL")
        print(f"Vertical margin  : {gfs_max - column_top:.2f} m")
        print("=========================================\n")

        if release_altitude < gfs_min:
            raise ValueError(
                f"Ketinggian pelepasan {release_altitude:.2f} m AMSL "
                f"berada di bawah batas profil GFS "
                f"{gfs_min:.2f} m AMSL."
            )

        if release_altitude >= gfs_max:
            raise ValueError(
                f"Ketinggian pelepasan {release_altitude:.2f} m AMSL "
                f"mencapai atau melebihi batas GFS "
                f"{gfs_max:.2f} m AMSL."
            )

        if column_top > gfs_max:
            raise ValueError(
                f"Puncak kolom erupsi {column_top:.2f} m AMSL "
                f"melebihi batas profil GFS {gfs_max:.2f} m AMSL. "
                f"Tinggi erupsi maksimum pada lokasi ini adalah "
                f"{maximum_eruption_height:.2f} m AGL. "
                "Kurangi tinggi erupsi atau gunakan dataset GFS "
                "dengan cakupan vertikal lebih tinggi."
            )

        atmospheric = self.meteorology.atmospheric_profile(
            latitude,
            longitude,
            time_index
        )

        eruption = self.eruption.calculate(
            atmospheric,
            eruption_height,
            config.eruption_duration,
            vent_altitude=source_altitude
        )

        summary = self.particle.create_particle_summary(
            eruption["total_mass"]
        )

        particles = self.particle.create_particles(
            eruption["total_mass"],
            latitude,
            longitude,
            release_altitude,
            eruption_height=eruption_height,
            eruption_duration=config.eruption_duration
        )

        if not particles:
            raise ValueError(
                "Tidak ada partikel yang berhasil dibuat."
            )

        altitudes = np.asarray(
            [float(p["altitude"]) for p in particles],
            dtype=float
        )

        if not np.all(np.isfinite(altitudes)):
            raise ValueError(
                "Terdapat ketinggian injeksi partikel yang tidak valid."
            )

        injection_min = float(np.min(altitudes))
        injection_max = float(np.max(altitudes))

        print("\n========== INJECTION DIAGNOSTIC ==========")
        print(f"Vent altitude    : {source_altitude:.2f} m")
        print(f"DEM elevation    : {ground_elevation:.2f} m")
        print(f"Release altitude : {release_altitude:.2f} m")
        print(f"Eruption height  : {eruption_height:.2f} m")
        print(f"Injection minimum: {injection_min:.2f} m")
        print(f"Injection maximum: {injection_max:.2f} m")
        print(f"GFS maximum      : {gfs_max:.2f} m")
        print(f"Particles        : {len(particles)}")
        print("==========================================\n")

        if injection_min < gfs_min or injection_max > gfs_max:
            raise ValueError(
                f"Ketinggian injeksi partikel "
                f"{injection_min:.2f}–{injection_max:.2f} m AMSL "
                f"berada di luar profil GFS "
                f"{gfs_min:.2f}–{gfs_max:.2f} m AMSL."
            )

        grid = self.meteorology.nearest_grid(
            latitude,
            longitude
        )

        turbulence = getattr(
            self.dispersion, "turbulence", None
        )

        if turbulence is None:
            raise RuntimeError(
                "TurbulenceService wajib tersedia."
            )

        turbulence_params = turbulence.parameters()
        effective_dt = float(
            turbulence_params["timestep"]
        )

        if not math.isfinite(effective_dt) or effective_dt <= 0:
            raise ValueError(
                "Timestep hasil perhitungan M6 tidak valid."
            )

        duration = float(config.duration)

        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                "Durasi simulasi harus lebih besar dari 0."
            )

        print("\n========== TIMESTEP M6 ==========")
        print(
            f"Mixing height : "
            f"{turbulence_params['mixing_height']:.2f} m"
        )
        print(
            f"Sigma u       : "
            f"{turbulence_params['sigma_u']:.2f} m/s"
        )
        print(
            f"Tau L         : "
            f"{turbulence_params['tau_l']:.2f} s"
        )
        print(f"Delta t       : {effective_dt:.2f} s")
        print("=================================\n")

        disp = self.dispersion.simulate(
            particles,
            duration,
            effective_dt,
            time_index
        )

        from collections import Counter, defaultdict

        print("\n========== DISPERSION STRUCTURE ==========")
        print("Result type:", type(disp).__name__)

        if isinstance(disp, dict):
            print("Result keys:", list(disp.keys()))
        else:
            print("Result is not a dictionary")

        print("==========================================\n")

        particle_data = None

        if isinstance(disp, list):
            particle_data = disp

        elif isinstance(disp, dict):
            for key in ("particles", "particle_results", "results", "trajectories"):
                value = disp.get(key)

                if isinstance(value, list):
                    particle_data = value
                    print("Particle collection key:", key)
                    break

        if particle_data is None:
            print("MASS PARTITION DIAGNOSTIC: particle collection not found")

        elif not all(isinstance(p, dict) for p in particle_data):
            print("MASS PARTITION DIAGNOSTIC: unexpected particle structure")
            print("First item type:", type(particle_data[0]).__name__ if particle_data else "empty")

        else:
            counts = Counter()
            masses = defaultdict(float)

            print("\n========== PARTICLE STRUCTURE ==========")

            if particle_data:
                print("Particle keys:", list(particle_data[0].keys()))
                print("Deposition sample:", particle_data[0].get("deposition"))

            print("========================================\n")

            for particle in particle_data:
                status = particle.get("status", "unknown")
                deposition = particle.get("deposition") or {}

                if not isinstance(deposition, dict):
                    deposition = {}

                if status == "deposited":
                    category = (
                        deposition.get("mechanism")
                        or "deposited_unknown"
                    )
                else:
                    category = status

                counts[category] += 1
                masses[category] += float(particle.get("mass", 0.0))

            total_mass = sum(masses.values())

            print("\n========== MASS PARTITION DIAGNOSTIC ==========")

            for category in sorted(masses):
                mass = masses[category]
                percentage = mass / total_mass * 100 if total_mass else 0.0

                print(
                    f"{category:20s} "
                    f"particles={counts[category]:4d} "
                    f"mass={mass:.6f} kg "
                    f"percentage={percentage:.4f}%"
                )

            print(f"TOTAL: {total_mass:.6f} kg")
            print("===============================================\n")


        
        from tests.validate_dispersion import validate_dispersion
        validate_dispersion(disp)

        trajectories = disp["trajectories"]

        deposited = [
            p for p in trajectories
            if p.get("status") == "deposited"
        ]
        active = [
            p for p in trajectories
            if p.get("status") == "active"
        ]
        unclassified = [
            p for p in trajectories
            if p.get("status") not in ("deposited", "active")
        ]

        source_mass = float(eruption["total_mass"])

        deposited_mass = sum(
            float(p.get("mass", 0.0))
            for p in deposited
        )
        airborne_mass = sum(
            float(p.get("mass", 0.0))
            for p in active
        )
        unclassified_mass = sum(
            float(p.get("mass", 0.0))
            for p in unclassified
        )

        mass_error = (
            source_mass - deposited_mass - airborne_mass
        )
        accounted_mass = (
            deposited_mass + airborne_mass + unclassified_mass
        )
        total_mass_residual = (
            source_mass - accounted_mass
        )

        mechanism_mass = {}
        scavenging_mass = {}
        scavenging_particles = {}

        for p in deposited:
            dep = p.get("deposition") or {}
            mass = float(p.get("mass", 0.0))
            mechanism = dep.get("mechanism") or "unknown"

            mechanism_mass[mechanism] = (
                mechanism_mass.get(mechanism, 0.0) + mass
            )

            kind = dep.get("scavenging_type")

            if kind:
                scavenging_mass[kind] = (
                    scavenging_mass.get(kind, 0.0) + mass
                )
                scavenging_particles[kind] = (
                    scavenging_particles.get(kind, 0) + 1
                )

        grid_size = 0.1
        vertical_resolution = 500.0

        concentration = (
            self.concentration.accumulate(
                trajectories,
                config.latitude,
                config.longitude,
                dlat=grid_size,
                dlon=grid_size,
                dz=vertical_resolution,
                active_only=True
            )
            if self.concentration else None
        )

        if concentration is not None:
            print("\n========== TEMPORAL CONCENTRATION VALIDATION ==========")

            time_series = concentration.get("time_series") or []

            temporal_errors = 0
            mass_errors = 0
            release_errors = 0
            deposition_errors = 0

            max_mass_difference = 0.0
            previous_time = -math.inf

            for snapshot in time_series:
                t = float(snapshot["time"])

                if not math.isfinite(t) or t <= previous_time:
                    temporal_errors += 1

                if t < 0 or t > duration + 1e-9:
                    temporal_errors += 1

                previous_time = t

                expected_mass = 0.0

                for particle in trajectories:
                    states = particle.get("trajectory") or []

                    if not states:
                        continue

                    release_time = float(
                        particle.get("release_time", states[0]["time"])
                    )

                    if t + 1e-9 < release_time:
                        continue

                    deposition = particle.get("deposition") or {}
                    deposition_time = deposition.get("time")

                    if (
                        particle.get("status") == "deposited"
                        and deposition_time is not None
                        and float(deposition_time) <= t + 1e-9
                    ):
                        continue

                    if not any(
                        float(state["time"]) <= t + 1e-9
                        for state in states
                    ):
                        continue

                    expected_mass += float(particle.get("mass", 0.0))

                actual_mass = math.fsum(
                    float(cell.get("mass", 0.0))
                    for cell in snapshot.get("cells", [])
                )

                difference = abs(expected_mass - actual_mass)

                max_mass_difference = max(
                    max_mass_difference,
                    difference
                )

                if not math.isclose(
                    expected_mass,
                    actual_mass,
                    rel_tol=1e-12,
                    abs_tol=1e-6
                ):
                    mass_errors += 1

            for particle in trajectories:
                states = particle.get("trajectory") or []

                if not states:
                    continue

                release_time = float(
                    particle.get("release_time", states[0]["time"])
                )

                if float(states[0]["time"]) + 1e-9 < release_time:
                    release_errors += 1

                deposition = particle.get("deposition") or {}

                if particle.get("status") == "deposited":
                    deposition_time = deposition.get("time")

                    if deposition_time is None:
                        deposition_errors += 1
                    elif float(deposition_time) < release_time:
                        deposition_errors += 1

            overall_valid = all((
                len(time_series) > 0,
                temporal_errors == 0,
                mass_errors == 0,
                release_errors == 0,
                deposition_errors == 0
            ))

            print(f"Total snapshots       : {len(time_series)}")
            print(f"Temporal errors       : {temporal_errors}")
            print(f"Snapshot mass errors  : {mass_errors}")
            print(f"Release errors        : {release_errors}")
            print(f"Deposition time errors: {deposition_errors}")
            print(f"Maximum mass difference: {max_mass_difference:.9f} kg")
            print(f"Overall temporal valid: {overall_valid}")
            print("=======================================================\n")

        if concentration:
            print("\n========== CONCENTRATION 3D VALIDATION ==========")

            metadata = concentration.get("metadata") or {}
            time_series = concentration.get("time_series") or []

            print(f"Horizontal latitude : {metadata.get('dlat')}")
            print(f"Horizontal longitude: {metadata.get('dlon')}")
            print(f"Vertical resolution : {metadata.get('dz')}")
            print(f"Concentration units : {metadata.get('units')}")
            print(f"Total snapshots     : {len(time_series)}")
            print(f"Maximum concentration: {concentration.get('max_concentration')}")

            resolution_valid = (
                abs(float(metadata.get("dlat", 0)) - 0.1) < 1e-12
                and abs(float(metadata.get("dlon", 0)) - 0.1) < 1e-12
                and abs(float(metadata.get("dz", 0)) - 500.0) < 1e-9
            )

            units_valid = metadata.get("units") == "kg/m3"

            print(f"Resolution valid    : {resolution_valid}")
            print(f"Units valid         : {units_valid}")

            if time_series:
                first = time_series[0]
                last = time_series[-1]

                first_cells = first.get("cells") or []
                last_cells = last.get("cells") or []

                first_mass = sum(
                    float(cell.get("mass", 0.0))
                    for cell in first_cells
                )

                last_mass = sum(
                    float(cell.get("mass", 0.0))
                    for cell in last_cells
                )

                print(f"First snapshot time : {first.get('time')}")
                print(f"First snapshot cells: {len(first_cells)}")
                print(f"First snapshot mass : {first_mass:.6f} kg")

                print(f"Last snapshot time  : {last.get('time')}")
                print(f"Last snapshot cells : {len(last_cells)}")
                print(f"Last snapshot mass  : {last_mass:.6f} kg")

                print(
                    "Last snapshot vs airborne:",
                    abs(last_mass - airborne_mass)
                    if "airborne_mass" in locals()
                    else "airborne_mass variable unavailable"
                )

            print("=================================================\n")

        
        if concentration is not None:
            from bisect import bisect_right
            from collections import defaultdict

            print("\n========== SPATIAL CONCENTRATION VALIDATION ==========")

            metadata = concentration["metadata"]
            origin_lat = float(metadata["origin_latitude"])
            origin_lon = float(metadata["origin_longitude"])
            dlat = float(metadata["dlat"])
            dlon = float(metadata["dlon"])
            dz = float(metadata["dz"])
            earth_radius = float(metadata["earth_radius"])

            spatial_errors = 0
            cell_mass_errors = 0
            volume_errors = 0
            concentration_errors = 0
            duplicate_cells = 0
            max_cell_mass_difference = 0.0
            checked_cells = 0

            prepared = []

            for particle in trajectories:
                states = sorted(
                    particle.get("trajectory") or [],
                    key=lambda state: float(state["time"])
                )

                if not states:
                    continue

                prepared.append({
                    "times": [float(state["time"]) for state in states],
                    "states": states,
                    "release_time": float(
                        particle.get("release_time", states[0]["time"])
                    ),
                    "deposition_time": (
                        float(particle["deposition"]["time"])
                        if particle.get("deposition")
                        and particle["deposition"].get("time") is not None
                        else None
                    ),
                    "status": particle.get("status"),
                    "mass": float(particle.get("mass", 0.0))
                })

            for snapshot in concentration["time_series"]:
                t = float(snapshot["time"])
                expected_cells = defaultdict(float)

                for particle in prepared:
                    if t + 1e-9 < particle["release_time"]:
                        continue

                    deposition_time = particle["deposition_time"]

                    if (
                        particle["status"] == "deposited"
                        and deposition_time is not None
                        and deposition_time <= t + 1e-9
                    ):
                        continue

                    index = bisect_right(
                        particle["times"], t + 1e-9
                    ) - 1

                    if index < 0:
                        continue

                    state = particle["states"][index]

                    lat = float(state["latitude"])
                    lon = float(state["longitude"])
                    alt = float(state["altitude"])

                    i = math.floor((lat - origin_lat) / dlat)
                    j = math.floor((lon - origin_lon) / dlon)
                    k = math.floor(max(alt, 0.0) / dz)

                    expected_cells[(i, j, k)] += particle["mass"]

                actual_cells = {}

                for cell in snapshot.get("cells", []):
                    key = (
                        int(cell["i"]),
                        int(cell["j"]),
                        int(cell["k"])
                    )

                    if key in actual_cells:
                        duplicate_cells += 1

                    actual_cells[key] = cell

                if set(expected_cells) != set(actual_cells):
                    spatial_errors += 1

                for key, cell in actual_cells.items():
                    checked_cells += 1

                    i, j, k = key

                    expected_lat = origin_lat + (i + 0.5) * dlat
                    expected_lon = origin_lon + (j + 0.5) * dlon
                    expected_alt = (k + 0.5) * dz

                    if not all((
                        math.isclose(cell["latitude"], expected_lat, abs_tol=1e-10),
                        math.isclose(cell["longitude"], expected_lon, abs_tol=1e-10),
                        math.isclose(cell["altitude"], expected_alt, abs_tol=1e-8)
                    )):
                        spatial_errors += 1

                    expected_mass = expected_cells.get(key, 0.0)
                    actual_mass = float(cell["mass"])

                    difference = abs(expected_mass - actual_mass)

                    max_cell_mass_difference = max(
                        max_cell_mass_difference,
                        difference
                    )

                    if not math.isclose(
                        expected_mass,
                        actual_mass,
                        rel_tol=1e-12,
                        abs_tol=1e-6
                    ):
                        cell_mass_errors += 1

                    phi = math.radians(expected_lat)

                    expected_volume = (
                        earth_radius ** 2
                        * abs(math.cos(phi))
                        * math.radians(dlat)
                        * math.radians(dlon)
                        * dz
                    )

                    actual_volume = float(cell["volume"])

                    if not (
                        math.isfinite(actual_volume)
                        and actual_volume > 0
                        and math.isclose(
                            actual_volume,
                            expected_volume,
                            rel_tol=1e-10,
                            abs_tol=1e-6
                        )
                    ):
                        volume_errors += 1

                    actual_concentration = float(cell["concentration"])
                    expected_concentration = (
                        actual_mass / actual_volume
                        if actual_volume > 0 else float("nan")
                    )

                    if not (
                        math.isfinite(actual_concentration)
                        and math.isclose(
                            actual_concentration,
                            expected_concentration,
                            rel_tol=1e-10,
                            abs_tol=1e-12
                        )
                    ):
                        concentration_errors += 1

            overall_valid = all((
                spatial_errors == 0,
                cell_mass_errors == 0,
                volume_errors == 0,
                concentration_errors == 0,
                duplicate_cells == 0,
                checked_cells > 0
            ))

            print(f"Total snapshots          : {len(concentration['time_series'])}")
            print(f"Checked cells            : {checked_cells}")
            print(f"Spatial errors           : {spatial_errors}")
            print(f"Cell mass errors         : {cell_mass_errors}")
            print(f"Volume errors            : {volume_errors}")
            print(f"Concentration errors     : {concentration_errors}")
            print(f"Duplicate cells          : {duplicate_cells}")
            print(f"Maximum cell mass diff   : {max_cell_mass_difference:.9f} kg")
            print(f"Overall spatial valid    : {overall_valid}")
            print("======================================================\n")

        surface = (
            self.concentration.surface_deposition(
                deposited,
                dlat=grid_size,
                dlon=grid_size,
                origin_latitude=config.latitude,
                origin_longitude=config.longitude
            )
            if self.concentration else None
        )
        deposition_records = []

        for particle in deposited:
            deposition = particle.get("deposition") or {}

            if not deposition:
                continue

            deposition_records.append({
                "latitude": float(deposition["latitude"]),
                "longitude": float(deposition["longitude"]),
                "altitude": float(deposition.get("altitude", 0.0)),
                "mass": float(particle.get("mass", 0.0))
            })

        deposition_grid = self.deposition_grid.accumulate(
            deposition_records
        )

        grid_mass = math.fsum(
            float(cell["mass"])
            for cell in deposition_grid["cells"]
        )

        expected_mass = math.fsum(
            float(p["mass"])
            for p in deposited
        )

        grid_valid = math.isclose(
            grid_mass,
            expected_mass,
            rel_tol=1e-12,
            abs_tol=1e-6
        )

        print("\n========== DEPOSITION GRID 0.01 VALIDATION ==========")
        print(f"Grid resolution     : {deposition_grid['cell_size']}")
        print(f"Total cells         : {len(deposition_grid['cells'])}")
        print(f"Total deposited mass: {expected_mass:.6f} kg")
        print(f"Grid deposited mass : {grid_mass:.6f} kg")
        print(f"Mass difference     : {expected_mass - grid_mass:.9f} kg")
        print(f"Grid mass valid     : {grid_valid}")
        print("===================================================\n")
        if surface is not None:

            print("\n========== SURFACE DEPOSITION VALIDATION ==========")

            cells = surface.get("cells") or []
            total_mass = float(surface.get("total_deposited_mass", 0.0))
            cell_mass = math.fsum(float(c.get("mass", 0.0)) for c in cells)

            expected_mass = math.fsum(
                float(p.get("mass", 0.0))
                for p in deposited
            )

            resolution_lat = float(surface.get("dlat", 0.0))
            resolution_lon = float(surface.get("dlon", 0.0))
            units = surface.get("units")

            print(f"Grid latitude resolution : {resolution_lat}")
            print(f"Grid longitude resolution: {resolution_lon}")
            print(f"Deposition units         : {units}")
            print(f"Total deposition cells   : {len(cells)}")
            print(f"Total deposited mass     : {total_mass:.6f} kg")
            print(f"Sum of grid cell mass    : {cell_mass:.6f} kg")
            print(f"Expected deposited mass : {expected_mass:.6f} kg")

            tolerance = max(1e-6, abs(expected_mass) * 1e-12)

            mass_valid = (
                math.isclose(total_mass, expected_mass, rel_tol=1e-12, abs_tol=1e-6)
                and math.isclose(cell_mass, expected_mass, rel_tol=1e-12, abs_tol=1e-6)
            )

            units_valid = units == "kg/m2"

            area_valid = all(
                math.isfinite(float(c.get("area", 0.0)))
                and float(c.get("area", 0.0)) > 0
                for c in cells
            )

            density_valid = all(
                math.isclose(
                    float(c.get("deposition", 0.0)),
                    float(c.get("mass", 0.0)) / float(c.get("area", 1.0)),
                    rel_tol=1e-10,
                    abs_tol=1e-12
                )
                for c in cells
                if float(c.get("area", 0.0)) > 0
            )

            max_deposition = max(
                (float(c.get("deposition", 0.0)) for c in cells),
                default=0.0
            )

            print(f"Maximum deposition       : {max_deposition:.9f} kg/m2")
            print(f"Mass tolerance           : {tolerance:.6f} kg")
            print(f"Mass conservation valid  : {mass_valid}")
            print(f"Area valid               : {area_valid}")
            print(f"Density valid            : {density_valid}")
            print(f"Units valid              : {units_valid}")
            print(f"Overall surface valid    : {mass_valid and area_valid and density_valid and units_valid}")

            print("===================================================\n")


        spatial = (
            self.concentration.plume_metrics(
                eruption["mean_wind_speed"],
                duration,
                dlat=grid_size,
                dlon=grid_size,
                latitude=config.latitude
            )
            if self.concentration else None
        )

        trajectory_geojson = (
            self.geojson.trajectory_to_feature_collection(disp)
        )
        point_geojson = (
            self.geojson.points_to_feature_collection(disp)
        )
        deposition_geojson = (
            self.geojson.deposition_to_feature_collection(disp)
        )

        grid_geojson = self.geojson.grid_to_feature_collection(
            deposition_grid
        )

        print("\n========== GEOJSON DEPOSITION GRID VALIDATION ==========")

        features = grid_geojson.get("features", [])
        cells = deposition_grid.get("cells", [])
        cell_size = float(deposition_grid["cell_size"])

        geometry_errors = 0
        coordinate_errors = 0
        resolution_errors = 0
        density_errors = 0
        invalid_mass = 0

        geojson_masses = []
        geojson_particles = 0

        for feature in features:
            geometry = feature.get("geometry") or {}
            properties = feature.get("properties") or {}

            if geometry.get("type") != "Polygon":
                geometry_errors += 1
                continue

            rings = geometry.get("coordinates") or []

            if len(rings) != 1 or len(rings[0]) != 5:
                geometry_errors += 1
                continue

            ring = rings[0]

            if ring[0] != ring[-1]:
                geometry_errors += 1

            try:
                west = float(ring[0][0])
                south = float(ring[0][1])
                east = float(ring[1][0])
                north = float(ring[2][1])

                valid_rectangle = (
                    ring[1] == [east, south]
                    and ring[2] == [east, north]
                    and ring[3] == [west, north]
                    and west < east
                    and south < north
                )

                if not valid_rectangle:
                    geometry_errors += 1

                if not all(
                    math.isfinite(value)
                    for value in (west, south, east, north)
                ):
                    coordinate_errors += 1

                if not (
                    -180 <= west <= 180
                    and -180 <= east <= 180
                    and -90 <= south <= 90
                    and -90 <= north <= 90
                ):
                    coordinate_errors += 1

                if not (
                    math.isclose(east - west, cell_size, abs_tol=1e-10)
                    and math.isclose(north - south, cell_size, abs_tol=1e-10)
                ):
                    resolution_errors += 1

                mass = float(properties.get("mass", 0.0))
                area = float(properties.get("area", 0.0))
                density = float(properties.get("deposition_density", 0.0))

                if not math.isfinite(mass) or mass < 0:
                    invalid_mass += 1
                else:
                    geojson_masses.append(mass)

                if not (
                    math.isfinite(area)
                    and area > 0
                    and math.isfinite(density)
                    and math.isclose(
                        density,
                        mass / area,
                        rel_tol=1e-10,
                        abs_tol=1e-12
                    )
                ):
                    density_errors += 1

                geojson_particles += int(
                    properties.get("particle_count", 0)
                )

            except (TypeError, ValueError, IndexError, KeyError):
                geometry_errors += 1

        geojson_mass = math.fsum(geojson_masses)

        expected_mass = math.fsum(
            float(p.get("mass", 0.0))
            for p in deposited
        )

        mass_valid = (
            invalid_mass == 0
            and math.isclose(
                geojson_mass,
                expected_mass,
                rel_tol=1e-12,
                abs_tol=1e-6
            )
        )

        count_valid = len(features) == len(cells)
        particle_count_valid = geojson_particles == len(deposited)

        overall_valid = all((
            grid_geojson.get("type") == "FeatureCollection",
            count_valid,
            particle_count_valid,
            mass_valid,
            geometry_errors == 0,
            coordinate_errors == 0,
            resolution_errors == 0,
            density_errors == 0
        ))

        print(f"Grid resolution      : {cell_size}")
        print(f"Total grid cells     : {len(cells)}")
        print(f"Total GeoJSON features: {len(features)}")
        print(f"Deposited particles  : {len(deposited)}")
        print(f"GeoJSON particle sum : {geojson_particles}")
        print(f"Expected mass        : {expected_mass:.6f} kg")
        print(f"GeoJSON mass         : {geojson_mass:.6f} kg")
        print(f"Mass difference      : {expected_mass - geojson_mass:.9f} kg")
        print(f"Geometry errors      : {geometry_errors}")
        print(f"Coordinate errors    : {coordinate_errors}")
        print(f"Resolution errors    : {resolution_errors}")
        print(f"Density errors       : {density_errors}")
        print(f"Invalid mass         : {invalid_mass}")
        print(f"Feature count valid  : {count_valid}")
        print(f"Particle count valid : {particle_count_valid}")
        print(f"Mass valid           : {mass_valid}")
        print(f"Overall GeoJSON valid: {overall_valid}")
        print("=====================================================\n")


        closure = (
            100.0 * (deposited_mass + airborne_mass) / source_mass
            if source_mass > 0 else None
        )
        total_closure = (
            100.0 * accounted_mass / source_mass
            if source_mass > 0 else None
        )

        serialized_config = self._serialize_config(config)
        serialized_config.pop("auto_timestep", None)
        serialized_config.pop("requested_timestep", None)

        serialized_config.update({
            "grid_size": grid_size,
            "vertical_resolution": vertical_resolution,
            "timestep": effective_dt,
            "timestep_source": "M6_LAGRANGIAN",
            "turbulence": turbulence_params,
            "release_altitude": release_altitude,
            "dem_elevation": ground_elevation,
            "altitude_correction": altitude_correction
        })

        return {
            "config": serialized_config,
            "meteorology": {
                **info,
                "time_index": time_index,
                "times": [str(t) for t in info["times"]]
            },
            "location": {
                "latitude": config.latitude,
                "longitude": config.longitude,
                "altitude": config.altitude,
                "grid": grid,
                "dem_elevation": ground_elevation,
                "release_altitude": release_altitude,
                "altitude_correction": altitude_correction,
                "release_clearance": release_clearance
            },
            "atmospheric": atmospheric,
            "eruption": eruption,
            "particle_summary": summary,
            "particles": {
                "total": len(particles),
                "groups": particles
            },
            "dispersion": {
                "particle_count": len(trajectories),
                "duration": duration,
                "dt": effective_dt,
                "steps": disp["steps"],
                "trajectories": trajectories
            },
            "concentration": concentration,
            "surface_deposition": surface,
            "deposition_grid": deposition_grid,
            "spatial_analysis": spatial,
            "mass_balance": {
                "source_mass_kg": source_mass,
                "deposited_mass_kg": deposited_mass,
                "airborne_mass_kg": airborne_mass,
                "unclassified_mass_kg": unclassified_mass,
                "residual_kg": mass_error,
                "total_residual_kg": total_mass_residual,
                "closure_percent": closure,
                "total_closure_percent": total_closure,
                "deposition_mechanism_mass_kg": mechanism_mass,
                "scavenging_mass_kg": scavenging_mass,
                "scavenging_particles": scavenging_particles,
                "total": len(trajectories),
                "deposited": len(deposited),
                "active": len(active),
                "unclassified": len(unclassified),
                "total_particles": len(trajectories),
                "deposited_particles": len(deposited),
                "active_particles": len(active),
                "unclassified_particles": len(unclassified)
            },
            "geojson": {
                "trajectory": trajectory_geojson,
                "points": point_geojson,
                "deposition": deposition_geojson,
                "deposition_grid": grid_geojson
            }
        }

    @staticmethod
    def _resolve_time_index(config, info):
        index = getattr(config, "time_index", 0)

        if index is None:
            index = 0

        index = int(index)
        count = int(
            info.get("count", len(info.get("times", [])))
        )

        if not 0 <= index < count:
            raise ValueError(
                "time_index exceeds meteorological dataset."
            )

        return index

    @staticmethod
    def _get(config, key, default=None):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    @staticmethod
    def _serialize_config(config):
        if hasattr(config, "model_dump"):
            return config.model_dump()
        if hasattr(config, "dict"):
            return config.dict()
        if isinstance(config, dict):
            return dict(config)
        return vars(config).copy()

    def close(self):
        if self.landcover:
            self.landcover.close()
        if self.terrain:
            self.terrain.close()
