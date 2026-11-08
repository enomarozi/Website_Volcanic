import math

class MassBalanceService:
    def __init__(self, relative_tolerance=1e-6, absolute_tolerance=1e-12):
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)

    def validate(self, source_mass, deposited_mass, airborne_mass, grid_deposited_mass=None):
        values = [source_mass, deposited_mass, airborne_mass]
        if grid_deposited_mass is not None:
            values.append(grid_deposited_mass)
        values = [float(v) for v in values]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Mass values must be finite and non-negative.")
        source_mass, deposited_mass, airborne_mass = values[:3]
        if self.relative_tolerance < 0 or self.absolute_tolerance < 0:
            raise ValueError("Mass tolerances must be non-negative.")

        simulated_mass = deposited_mass + airborne_mass
        error = source_mass - simulated_mass
        tolerance = max(self.absolute_tolerance, source_mass * self.relative_tolerance)
        valid = abs(error) <= tolerance
        grid_error = grid_valid = None

        if grid_deposited_mass is not None:
            grid_deposited_mass = values[3]
            grid_error = deposited_mass - grid_deposited_mass
            grid_valid = abs(grid_error) <= tolerance

        return {
            "source_mass": source_mass,
            "deposited_mass": deposited_mass,
            "airborne_mass": airborne_mass,
            "simulated_mass": simulated_mass,
            "error": error,
            "relative_error": abs(error) / max(source_mass, self.absolute_tolerance),
            "tolerance": tolerance,
            "valid": valid,
            "grid_deposited_mass": grid_deposited_mass,
            "grid_mass_error": grid_error,
            "grid_valid": grid_valid,
            "valid_all": bool(valid and (grid_valid is None or grid_valid))
        }

    def validate_result(self, result):
        if not isinstance(result, dict):
            raise TypeError("result must be a dictionary.")

        dispersion = result.get("dispersion") or {}
        grid = result.get("deposition_grid") or {}
        balance = result.get("mass_balance") or {}
        eruption = result.get("eruption") or {}

        source = eruption.get("total_mass", balance.get("source_mass_kg", balance.get("source_mass", 0.0)))
        deposited = dispersion.get("deposited_mass", balance.get("deposited_mass_kg", balance.get("deposited_mass", 0.0)))
        airborne = dispersion.get("airborne_mass", balance.get("airborne_mass_kg", balance.get("airborne_mass", balance.get("active_mass", 0.0))))
        grid_mass = grid.get("total_deposited_mass", grid.get("total_grid_mass"))

        return self.validate(source, deposited, airborne, grid_mass)
