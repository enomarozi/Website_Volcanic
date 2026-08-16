from app.services.dispersion import DispersionService


service = DispersionService(
    "data/era_merapi_jan_2026.nc",
    "data/data_stream-oper_stepType-instant.nc",
    "data/data_stream-oper_stepType-accum.nc"
)

result = service.atmospheric_conditions(
    latitude=-7.54,
    longitude=110.44,
    time_index=0
)

print("\n=== LOCATION ===")
print(result["latitude"], result["longitude"])

print("\n=== SURFACE ===")
print(result["surface"])

print("\n=== CORIOLIS ===")
print(result["coriolis"])

print("\n=== MIXING HEIGHT ===")
print(result["mixing_height"])

print("\n=== ATMOSPHERIC PROFILE ===")
for level in result["atmospheric"]["profile"]:
    print(level)