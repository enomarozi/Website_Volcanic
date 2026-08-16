import cdsapi

client = cdsapi.Client()

client.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": ["reanalysis"],
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind",
            "temperature",
            "geopotential",
        ],
        "year": ["2026"],
        "month": ["01"],
        "day": [
            "01", "02", "03", "04", "05",
            "06", "07", "08", "09", "10",
            "11", "12", "13", "14", "15",
            "16", "17", "18", "19", "20",
            "21", "22", "23", "24", "25",
            "26", "27", "28", "29", "30", "31"
        ],
        "time": [
            "00:00",
            "06:00",
            "12:00",
            "18:00"
        ],
        "pressure_level": [
            "1000",
            "850",
            "700",
            "500",
            "300"
        ],
        "area": [
            -6.5, 109.5,
            -8.5, 111.5
        ],
        "format": "netcdf",
    },
    "../../data/era5_pressure_jan_2026.nc"
)