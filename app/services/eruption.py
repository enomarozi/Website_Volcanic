import math

class EruptionService:
    def __init__(self, rho_air=1.225, gravity=9.81, alpha=0.1, beta=0.5, z1=2.8,
                 mer_method="degruyter_bonadonna", source_temperature=1200.0,
                 gamma=1.41, reference_pressure=1000.0, default_n=0.011):
        self.rho_air=float(rho_air); self.gravity=float(gravity); self.alpha=float(alpha)
        self.beta=float(beta); self.z1=float(z1); self.mer_method=str(mer_method)
        self.source_temperature=float(source_temperature); self.gamma=float(gamma)
        self.reference_pressure=float(reference_pressure); self.default_n=float(default_n)
        if min(self.rho_air,self.gravity,self.z1,self.source_temperature,self.gamma,self.reference_pressure)<=0:
            raise ValueError("Invalid eruption constants.")

    @staticmethod
    def _levels(atmospheric):
        if isinstance(atmospheric,list): return atmospheric
        if isinstance(atmospheric,dict):
            for key in ("profile","levels","atmospheric_profile","data"):
                if isinstance(atmospheric.get(key),list): return atmospheric[key]
        return []

    @staticmethod
    def _num(item,*keys):
        for key in keys:
            try:
                value=float(item[key])
                if math.isfinite(value): return value
            except (KeyError,TypeError,ValueError): pass
        return None

    def _theta(self, item):
        theta=self._num(item,"potential_temperature","theta","theta_k")
        if theta is not None: return theta
        temp=self._num(item,"temperature","temp","t","T")
        pressure=self._num(item,"pressure","pressure_hpa","pressure_level","p","level_pressure")
        if temp is None or pressure is None or pressure<=0: return None
        if pressure > 2000: pressure /= 100.0
        return temp*(self.reference_pressure/pressure)**((self.gamma-1.0)/self.gamma)

    def calculate(self, atmospheric_profile, eruption_height, eruption_duration):
        h=float(eruption_height); duration=float(eruption_duration)
        if not math.isfinite(h) or h<=0 or not math.isfinite(duration) or duration<=0:
            raise ValueError("Eruption height and duration must be positive.")
        levels=[]
        for item in self._levels(atmospheric_profile):
            if not isinstance(item,dict): continue
            z=self._num(item,"altitude","height","z","level_height")
            u=self._num(item,"u","u_wind","wind_u"); v=self._num(item,"v","v_wind","wind_v")
            theta=self._theta(item)
            temp=self._num(item,"temperature","temp","t","T")
            if z is not None and u is not None and v is not None:
                levels.append({"z":z,"u":u,"v":v,"theta":theta,"temperature":temp})
        levels.sort(key=lambda x:x["z"])
        if len(levels)<2: raise ValueError("At least two levels with altitude and wind are required for D&B MER.")
        # The framework averages wind speed over all available GFS levels, not only levels below H.
        wind_speeds=[math.hypot(x["u"],x["v"]) for x in levels]
        mean_wind=sum(wind_speeds)/len(wind_speeds)
        # Direction uses mean horizontal components through the eruption column (z <= H).
        direction_levels=[x for x in levels if x["z"]<=h]
        if not direction_levels: direction_levels=levels
        mean_u=sum(x["u"] for x in direction_levels)/len(direction_levels)
        mean_v=sum(x["v"] for x in direction_levels)/len(direction_levels)
        bearing=math.degrees(math.atan2(mean_u,mean_v))%360.0
        theta_levels=[x for x in levels if x["theta"] is not None and x["theta"]>0]
        n=self.default_n; dtheta_dz=0.0; theta_mean=None
        if len(theta_levels)>=2 and theta_levels[-1]["z"]>theta_levels[0]["z"]:
            theta_mean=sum(x["theta"] for x in theta_levels)/len(theta_levels)
            dtheta_dz=(theta_levels[-1]["theta"]-theta_levels[0]["theta"])/(theta_levels[-1]["z"]-theta_levels[0]["z"])
            n2=self.gravity/max(theta_mean,1e-12)*dtheta_dz
            if n2>0 and math.isfinite(n2): n=math.sqrt(n2)
        surface_temp=next((x["temperature"] for x in levels if x["temperature"] is not None),None)
        if surface_temp is None: surface_temp=theta_levels[0]["theta"] if theta_levels else 298.0
        rho_source=self.rho_air*float(surface_temp)/self.source_temperature
        reduced_gravity=self.gravity*(self.rho_air-rho_source)/self.rho_air
        if reduced_gravity<=0: raise ValueError("Reduced gravity must be positive; check source and ambient temperatures.")
        buoyancy=(2.0**2.5/self.z1**4)*self.alpha**2*n**3*h**4
        wind_term=self.beta**2*n**2*mean_wind*h**3/6.0
        mer=math.pi*self.rho_air/reduced_gravity*(buoyancy+wind_term)
        total_mass=mer*duration
        if not all(math.isfinite(x) and x>0 for x in (n,mer,total_mass)):
            raise ValueError("MER calculation returned an invalid result; check meteorological profile and parameters.")
        return {"eruption_height":h,"eruption_duration":duration,"mean_wind_speed":mean_wind,
                "mean_u_column":mean_u,"mean_v_column":mean_v,"wind_bearing_degrees":bearing,
                "theta_mean":theta_mean,"dtheta_dz":dtheta_dz,"brunt_vaisala":n,
                "source_density":rho_source,"reduced_gravity":reduced_gravity,"mer":mer,
                "total_mass":total_mass,"buoyancy_term":buoyancy,"wind_term":wind_term,
                "mer_method":self.mer_method,"profile_levels":len(levels)}
