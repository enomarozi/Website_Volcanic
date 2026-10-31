import math
import numpy as np

class ParticleService:
    def __init__(self, particle_density=2500.0, air_density=1.225, viscosity=1.81e-5,
                 gravity=9.81, total_particles=50000, psd_mean_radius=50e-6,
                 psd_sigma=0.7, seed=42):
        self.particle_density=float(particle_density); self.air_density=float(air_density)
        self.viscosity=float(viscosity); self.gravity=float(gravity); self.total_particles=int(total_particles)
        self.psd_mean_radius=float(psd_mean_radius); self.psd_sigma=float(psd_sigma)
        self.rng=np.random.default_rng(seed)
        if min(self.particle_density,self.viscosity,self.gravity,self.total_particles,self.psd_mean_radius)<=0 or self.psd_sigma<0:
            raise ValueError("Invalid particle parameters.")

    def settling_velocity(self, radius, air_density=None):
        r=float(radius); rho=self.air_density if air_density is None else float(air_density); d=2*r
        if r<=0 or rho<=0: raise ValueError("Particle radius and air density must be positive.")
        delta=max(self.particle_density-rho,0.0)
        if delta==0: return 0.0
        # Stokes initial estimate follows the framework's rho_p approximation.
        v=delta*self.gravity*d*d/(18*self.viscosity)
        re0=rho*v*d/self.viscosity
        if re0 < 1.0:
            return float(v)
        for _ in range(200):
            re=rho*max(v,1e-15)*d/self.viscosity
            cd=24/max(re,1e-15)*(1+0.15*re**0.687) if re<1000 else 0.44
            vn=math.sqrt(4*d*self.gravity*delta/(3*rho*max(cd,1e-12)))
            if abs(vn-v)<=max(1e-10,1e-7*vn): v=vn; break
            v=0.5*v+0.5*vn
        return float(v)

    def particle_mass(self, radius): return self.particle_density*(4.0/3.0)*math.pi*float(radius)**3

    @staticmethod
    def classify_radius(radius):
        r=float(radius)
        return "fine" if r<=5e-6 else "medium" if r<=50e-6 else "coarse"

    def sample_radii(self,count):
        # The revised framework uses a three-class approximation to the log-normal PSD.
        n=int(count); counts=[round(0.16*n),round(0.68*n),0]; counts[2]=n-counts[0]-counts[1]
        radii=np.concatenate((np.full(counts[0],5e-6),np.full(counts[1],50e-6),np.full(counts[2],500e-6)))
        self.rng.shuffle(radii)
        return radii

    def create_particles(self,total_mass,latitude,longitude,altitude):
        total_mass=float(total_mass); n=self.total_particles
        if not math.isfinite(total_mass) or total_mass<=0 or n<=0: raise ValueError("Total mass and particle count must be positive.")
        radii=np.asarray(self.sample_radii(n),dtype=float)
        if not np.all(np.isfinite(radii)) or np.any(radii<=0): raise ValueError("Invalid particle radius samples.")
        mass_each=total_mass/n
        return [{"id":i,"particle_id":i,"class":self.classify_radius(r),"particle_class":self.classify_radius(r),
                 "radius":float(r),"mass":mass_each,"latitude":float(latitude),"longitude":float(longitude),
                 "altitude":float(altitude),"settling_velocity":self.settling_velocity(r),"status":"active"}
                for i,r in enumerate(radii)]

    def create_particle_summary(self,total_mass):
        n=self.total_particles
        return {"total_particles":n,"total_mass":float(total_mass),"mass_per_particle":float(total_mass)/n,
                "distribution":{"type":"three_class_lognormal_approximation","mean_radius_m":self.psd_mean_radius,
                                "sigma":self.psd_sigma,"classes":[{"name":"fine","radius_m":5e-6,"fraction":0.16},
                                {"name":"medium","radius_m":50e-6,"fraction":0.68},
                                {"name":"coarse","radius_m":500e-6,"fraction":0.16}]}}
