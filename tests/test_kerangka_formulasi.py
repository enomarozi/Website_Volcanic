import math
from app.services.eruption import EruptionService
from app.services.particle import ParticleService
from app.services.turbulence import TurbulenceService
from app.services.deposition import DepositionService
from app.services.concentration import ConcentrationService

Z=[117.4,339.5,565.9,798.6,1036.8,1529.3,2046,2590.7,3167.5,3779.1,4430.2,5130.4,5889.5,6710.9,7613.1,8610.1,9721.7,10982.1,12462.8,14269.1,16644.9,18715.6,20757.4]
P=[1000,975,950,925,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,70,50]
T=[298.03,295.95,295.23,295.48,293.98,290.81,288.34,286.37,283.82,279.66,276.07,273.55,269.76,263.61,258.98,251.23,241.26,231.89,221.1,207.75,197.24,202.38,214.27]
U=[-3.12,-3.15,-2.98,-1.93,-1.04,-1.89,-4.22,-5,-4.29,-2.77,-.97,-.54,-3.89,-1.81,-10.59,-12.85,-13.19,-25.01,-14,-11.76,5.56,-2.04,7.59]
V=[6.97,6.55,4.09,2.46,2.4,1.78,2.34,2.41,1.87,1.11,-1.71,-2.68,-3.73,.6,2.25,-3.19,3.48,-8.91,3.15,-4.41,-3.37,-2.06,.67]
PROFILE=[{"altitude":z,"pressure":p,"temperature":t,"u":u,"v":v} for z,p,t,u,v in zip(Z,P,T,U,V)]

def test_mer_matches_kerangka_case():
    r=EruptionService().calculate(PROFILE,18000,120)
    assert math.isclose(r["mean_wind_speed"],7.240,rel_tol=5e-4)
    assert math.isclose(r["theta_mean"],338.4,rel_tol=5e-4)
    assert math.isclose(r["brunt_vaisala"],0.01734,rel_tol=5e-4)
    assert math.isclose(r["reduced_gravity"],7.374,rel_tol=5e-4)
    assert math.isclose(r["mer"],5.387e8,rel_tol=1e-3)
    assert math.isclose(r["total_mass"],6.464e10,rel_tol=1e-3)

def test_psd_and_equal_particle_mass():
    service=ParticleService(total_particles=50000,seed=42)
    particles=service.create_particles(6.464e10,-6.102,105.423,18157)
    counts={name:sum(p["particle_class"]==name for p in particles) for name in ("fine","medium","coarse")}
    assert counts=={"fine":8000,"medium":34000,"coarse":8000}
    assert len({p["mass"] for p in particles})==1
    assert math.isclose(sum(p["mass"] for p in particles),6.464e10,rel_tol=1e-12)

def test_settling_reference_values():
    p=ParticleService()
    assert math.isclose(p.settling_velocity(5e-6),0.007524,rel_tol=5e-4)
    assert math.isclose(p.settling_velocity(50e-6),0.5493,rel_tol=5e-4)
    assert math.isclose(p.settling_velocity(500e-6),6.776,rel_tol=5e-4)

def test_pbl_timestep_and_deposition_reference_values():
    t=TurbulenceService(); params=t.parameters()
    assert params["mixing_height"]==569.96
    assert params["sigma_u"]==1.2
    assert params["tau_l"]==71.25
    assert params["timestep"]==71.25
    d=DepositionService(); ra=d.aerodynamic_resistance()
    vs=ParticleService().settling_velocity(50e-6)
    rs=d.surface_resistance(vs)["resistance"]
    vd=d.dry_deposition_velocity(vs,ra,rs)
    assert math.isclose(ra,60.87,rel_tol=5e-4)
    assert rs==2.53
    assert math.isclose(vd,0.5560,rel_tol=5e-4)
    assert math.isclose(d.dry_probability(vd,dt=71.25),0.3271,rel_tol=5e-4)
    assert math.isclose(d.scavenging_coefficient(2.5),1.732e-4,rel_tol=5e-4)
    assert math.isclose(d.wet_probability(2.5,71.25),0.01227,rel_tol=5e-4)

def test_grid_resolution_and_plume_metrics():
    c=ConcentrationService()
    metrics=c.plume_metrics(7.240,21600,dlat=.25,dlon=.25,latitude=-6.102)
    assert math.isclose(metrics["plume_length_km"],156.384,rel_tol=5e-4)
    assert math.isclose(metrics["footprint_km2"],3127.68,rel_tol=5e-4)
    assert metrics["cell_area_km2"]>760 and metrics["cell_area_km2"]<780
