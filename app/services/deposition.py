import math
import numpy as np

class DepositionService:
    def __init__(self, gravity=9.81, von_karman=0.4, air_density=1.225, viscosity=1.81e-5,
                 rain_a=8.4e-5, rain_b=0.79, rh_threshold=80.0, reference_height=10.0,
                 roughness_length=0.0133, friction_velocity=0.272, layer_height=100.0,
                 surface_resistance_value=2.53, default_rain_rate=2.5):
        self.g=float(gravity); self.kappa=float(von_karman); self.rho=float(air_density)
        self.viscosity=float(viscosity); self.rain_a=float(rain_a); self.rain_b=float(rain_b)
        self.rh_threshold=float(rh_threshold); self.reference_height=float(reference_height)
        self.roughness_length=float(roughness_length); self.friction_velocity=float(friction_velocity)
        self.layer_height=float(layer_height); self.surface_resistance_value=None if surface_resistance_value is None else float(surface_resistance_value); self.default_rain_rate=float(default_rain_rate)

    def aerodynamic_resistance(self,friction_velocity=None,reference_height=None,roughness_length=None):
        u=max(float(self.friction_velocity if friction_velocity is None else friction_velocity),1e-9)
        zr=float(self.reference_height if reference_height is None else reference_height)
        z0=float(self.roughness_length if roughness_length is None else roughness_length)
        if zr<=z0 or z0<=0: raise ValueError("Reference height must exceed positive roughness length.")
        return math.log(zr/z0)/(self.kappa*u)

    def surface_resistance(self,settling_velocity,friction_velocity=None,particle_diffusivity=1e-6):
        # Kerangka revisi menetapkan rs=2.53 s/m untuk kondisi pesisir contoh.
        if self.surface_resistance_value is not None:
            return {"schmidt":None,"stokes":None,"resistance":self.surface_resistance_value}
        u=max(float(self.friction_velocity if friction_velocity is None else friction_velocity),1e-9)
        nu=self.viscosity/self.rho; sc=nu/max(float(particle_diffusivity),1e-15)
        st=max(float(settling_velocity),0.0)*u*u/max(self.g*nu,1e-15)
        rs=1.0/max(u*(sc**(-2.0/3.0)+10.0**(-3.0/max(st,1e-12))),1e-15)
        return {"schmidt":sc,"stokes":st,"resistance":rs}

    def settling_velocity(self,radius,particle_density=2500.0,air_density=None):
        r=float(radius); rho=self.rho if air_density is None else float(air_density); d=2*r
        if r<=0: raise ValueError("Particle radius must be positive.")
        delta=max(float(particle_density)-rho,0.0)
        if delta<=0: return 0.0
        v=delta*self.g*d*d/(18*self.viscosity)
        re0=rho*v*d/self.viscosity
        if re0 < 1.0:
            return float(v)
        for _ in range(200):
            re=rho*max(v,1e-15)*d/self.viscosity
            cd=24/max(re,1e-15)*(1+0.15*re**0.687) if re<1000 else 0.44
            vn=math.sqrt(4*d*self.g*delta/(3*rho*max(cd,1e-12)))
            if abs(vn-v)<max(1e-10,1e-7*vn): v=vn; break
            v=0.5*v+0.5*vn
        return float(v)

    def dry_deposition_velocity(self,settling_velocity,ra,rs):
        vs=max(float(settling_velocity),0.0)
        return 1.0/max(float(ra)+float(rs)+float(ra)*float(rs)*vs,1e-15)+vs

    def scavenging_coefficient(self,rain_rate,rh=None):
        rain=max(float(rain_rate or 0.0),0.0)
        # The revision specifies A=8.4e-5 and B=0.79; separate RH-dependent
        # coefficients are not supplied, so these documented constants are used for either branch.
        return self.rain_a*rain**self.rain_b if rain>0 else 0.0

    @staticmethod
    def deposition_probability(rate_per_second,dt):
        return -math.expm1(-max(float(rate_per_second),0.0)*max(float(dt),0.0))

    def wet_probability(self,rain_rate,dt,rh=None):
        # Lambda is given in s^-1 in the revised framework; do not divide by 3600.
        return self.deposition_probability(self.scavenging_coefficient(rain_rate,rh),dt)

    def dry_probability(self,vd,layer_height=None,dt=300.0):
        height=self.layer_height if layer_height is None else float(layer_height)
        return self.deposition_probability(max(float(vd),0.0)/max(height,1e-9),dt)

    def decide(self,particle,altitude,dt,friction_velocity=None,roughness_length=None,rain_rate=None,rh=None,rng=None):
        rng=np.random.default_rng() if rng is None else rng
        if float(altitude)>self.layer_height:
            return {"deposited":False,"mechanism":None,"probability":0.0,"pdry":0.0,"pwet":0.0}
        vs=float(particle.get("settling_velocity",0.0))
        rain_rate=0.0 if rain_rate is None else float(rain_rate)
        ra=self.aerodynamic_resistance(friction_velocity,roughness_length=roughness_length)
        rs=self.surface_resistance(vs,friction_velocity)["resistance"]
        vd=self.dry_deposition_velocity(vs,ra,rs)
        pdry=self.dry_probability(vd,dt=dt); pwet=self.wet_probability(rain_rate,dt,rh)
        ptotal=1.0-(1.0-pdry)*(1.0-pwet)
        deposited=bool(rng.random()<ptotal)
        if deposited:
            wet_only=pwet*(1.0-pdry)
            mechanism="wet" if rng.random() < wet_only/max(ptotal,1e-15) else "dry"
        else:
            mechanism=None
        return {"deposited":deposited,"mechanism":mechanism,"probability":ptotal,
                "vd":vd,"pdry":pdry,"pwet":pwet,"ra":ra,"rs":rs,"vs":vs}
