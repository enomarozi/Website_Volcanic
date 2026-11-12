import math


def validate_dispersion(result):
    trajectories = result["trajectories"]
    duration = float(result["duration"])

    total_mass = 0.0
    deposited_mass = 0.0
    airborne_mass = 0.0
    release_errors = []
    time_errors = []
    altitude_errors = []

    for p in trajectories:
        mass = float(p["mass"])
        release = float(p.get("release_time", 0.0))
        points = p["trajectory"]

        total_mass += mass

        if not points:
            time_errors.append(p["id"])
            continue

        if not math.isclose(
            float(points[0]["time"]),
            release,
            abs_tol=1e-6
        ):
            release_errors.append(p["id"])

        previous = -math.inf

        for point in points:
            t = float(point["time"])

            if t < previous - 1e-6 or t > duration + 1e-6:
                time_errors.append(p["id"])
                break

            previous = t

            if not math.isfinite(float(point["altitude"])):
                altitude_errors.append(p["id"])
                break

        if p["status"] == "deposited":
            deposited_mass += mass
        elif p["status"] == "active":
            airborne_mass += mass

    error = total_mass - deposited_mass - airborne_mass

    print("========== DISPERSION VALIDATION ==========")
    print("Particles       :", len(trajectories))
    print("Total mass      :", total_mass)
    print("Deposited mass  :", deposited_mass)
    print("Airborne mass   :", airborne_mass)
    print("Mass error      :", error)
    print("Release errors  :", len(release_errors))
    print("Time errors     :", len(time_errors))
    print("Altitude errors :", len(altitude_errors))
    print("===========================================")

    assert not release_errors
    assert not time_errors
    assert not altitude_errors
    assert math.isclose(
        total_mass,
        deposited_mass + airborne_mass,
        rel_tol=1e-10
    )
