import math
import numpy as np

class ParticleSimulationService:
    def __init__(self,meteorology,turbulence=None,particle=None,seed=42):
        self.meteorology=meteorology; self.turbulence=turbulence; self.particle=particle; self.rng=np.random.default_rng(seed)

    def velocity(self,latitude,longitude,altitude,time_index):
        w=self.meteorology.wind_at_altitude(float(latitude),float(longitude),max(0.0,float(altitude)),int(time_index))
        return {"u":float(w.get("u",0.0)),"v":float(w.get("v",0.0)),"w":float(w.get("w",w.get("vertical_wind",0.0)))}

    @staticmethod
    def _derivative(position,wind,settling_velocity,turbulence_velocity):
        lat,lon,alt=position; u=wind["u"]+turbulence_velocity[0]; v=wind["v"]+turbulence_velocity[1]
        meters_lat=111320.0; meters_lon=111320.0*max(abs(math.cos(math.radians(lat))),1e-8)
        return np.array([v/meters_lat,u/meters_lon,wind["w"]+turbulence_velocity[2]-settling_velocity],dtype=float)

    def step(self,latitude,longitude,altitude,dt,time_index,settling_velocity,turbulence_velocity=None):
        dt=float(dt); pos=np.array([float(latitude),float(longitude),float(altitude)],dtype=float)
        turb=np.zeros(3) if turbulence_velocity is None else np.asarray(turbulence_velocity,dtype=float)
        def f(p,idx):
            wind=self.velocity(p[0],p[1],p[2],idx)
            return self._derivative(p,wind,settling_velocity,turb),wind
        k1,w1=f(pos,time_index); k2,_=f(pos+k1*dt/2,time_index); k3,_=f(pos+k2*dt/2,time_index); k4,_=f(pos+k3*dt,time_index)
        out=pos+dt*(k1+2*k2+2*k3+k4)/6
        out[2]=max(0.0,out[2]); out[1]=((out[1]+180)%360)-180
        return {"latitude":float(out[0]),"longitude":float(out[1]),"altitude":float(out[2]),"u":w1["u"]+turb[0],"v":w1["v"]+turb[1],"vertical":w1["w"]+turb[2]-float(settling_velocity),"time_index":int(time_index)}
