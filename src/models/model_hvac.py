import numpy as np
   
class Heatpump:
    """ Heat pump super class
    """
    def __init__(self, mdot_HP = 0.25):       # heatpump nominal mass flow 
        self.mdot_HP = mdot_HP
        self.c_water = 4181      # heat capacity of water [J/kg/K]
    
    def check_hp(self,T_HP,T_RL):
        """ Ensures that selected HP power is within controller bounds in order to 
        exclude invalid choices (which would result in system errors).
        """    
        Q_HP = self.mdot_HP*self.c_water*(T_HP-T_RL)
        #if Q_HP<0: print('WARNING in building class: negative Q_HP=%.2f, T_HP=%.2f, T_RL=%.2f, T_amb=%.2f'%(Q_HP,T_HP,T_RL,T_amb))
        
        Q_HP_min = 2000 # minimum power in W
        Q_HP_max = 60000 # maximum power in W
              
        if hasattr(Q_HP, "__len__")>0:
            Q_HP[Q_HP<Q_HP_min] = 0
            Q_HP[Q_HP>Q_HP_max] = Q_HP_max
        else:
            if Q_HP<Q_HP_min: Q_HP = 0 
            if Q_HP>Q_HP_max: Q_HP = Q_HP_max
            
        T_HP_new = Q_HP/self.mdot_HP/self.c_water + T_RL
      
        return T_HP_new


    def calc(self, T_hp_ret, T_hp_sup, T_amb, timestep):
        """ Calculate electrical and thermal performance of the heat pump for a given timestep. 

        Parameters
        ----------
        T_hp_ret : array like
            :math:`T_{hp,ret}` : Intermediate states of return temperature as returned by solve_ivp() [°C]
        T_hp_sup : float
             :math:`T_{hp,sup}` : Supply temperature of the hp system [°C]
        T_amb : float
             :math:`T_{src}` : Ambient temperature [°C]
        timestep : integer
            Duration of one timestep [s]

        Returns
        -------
        dict
            - E_el    (float) - :math:`E_{el}` : Electrical energy demand [kWh]
            - P_el    (float) - :math:`P_{el}` : Electric power [W]
            - COP     (float) - :math:`COP` : Coefficient of performance [-]
            - Qdot_th (float) - :math:`\dot{Q}_{el}` : Thermal power [W]

        Notes
        -----
        The electrical power is calculated with the following equations:
        
        :math:`COP = f(T_{hp,sup}, T_{amb})` - implemented per heat pump model  

        :math:`\overline{\dot{Q}}_{th} = \sum_{}^{i} \dot{m}_{hp} \cdot c_{p,water} \cdot (T_{hp,sup} - T_{hp,ret,i})`  
        
        :math:`P_{el} = \overline{\dot{Q}}_{th} / COP` 
        
        :math:`E_{el} = P_{el} \cdot \Delta t` 
        """
        
        isteps = len(T_hp_ret) # number of intermediate integration steps
        
        # Calculate COP based on individual efficiency curve
        COP = self.COP(T_hp_sup,T_amb)
         
        # Calculate average of thermal power [W] 
        Qdot_th = sum(self.mdot_HP * self.c_water*(T_hp_sup - T_hp_ret[i]) \
                         for i in range(isteps)) / isteps
        if Qdot_th < 0:
            Qdot_th = 0
       
        P_el =  Qdot_th / COP #  Electrical power [W]
               
        E_el = P_el * timestep / 3600 # Electrical energy demand [Wh]
        
        return {'P_el' : P_el, 'E_el' : E_el,
                'COP' : COP, 'Qdot_th' : Qdot_th}


    def calc_cost(self,x,u,p): 
      """
       Calculate electrical energy consumption in one time step
      """      
      T_RL = x[-1]
      T_HP = u[0]
      T_amb = p[0]
      #T_HP_mean = (T_HP+T_RL)/2. # -> smaller COP
      
      COP = self.COP(T_HP,T_amb)
      #Qth_nom = self.Qth(T_HP,T_amb)
       
      Qth = self.mdot_HP*self.c_water*(T_HP-T_RL)/1000  
      
      #if PRINT: print('Qth_HP=%.2fkW, COP=%.2f, Qth_nom=%.2fkW' %(Qth,COP,Qth_nom))
      
      #if Qth<-1: print('WARNING in Heatpump: Qth NEGATIVE',Qth,Qth_nom,COP); #1/0
    
      return Qth/(COP * 100)

    
class Heatpump_Vitocal(Heatpump):
    """ A model class for Vitocal ground collector water-water heatpump,
       performance based on fit with manufacturer data.
    """
      
    def __init__(self, 
                mdot_HP = 0.25,       # heatpump nominal mass flow 
                verbose = False,
                ): 
        print('Vitocal heat pump')
        super(Heatpump_Vitocal,self).__init__(mdot_HP=mdot_HP) 

        if verbose == True:
            self.plot_cop()     

    def COP(self, T_sink,T_amb=0):
        """ Calculates coefficient of performance (COP) from polynomial curve fit.

        Parameters
        ----------
        T_sink : float
            sink / supply flow temperature [degC]

        T_amb : float
            ambient temperature [degC], default = 0 degC (used to estimate ground temperature)

        Returns
        -------
        float
            COP
        """
        # estimation of ground temperature based on ambient temperature
        from disturbances import t_ground_from_amb
        T_source = t_ground_from_amb(T_amb)
    
        z0 = 10.893436
        a	 = -0.228602
        b	 = 0.266006
        c	 = 0.001461
        d	 = 0.000501
        f	 = -0.003546
        g	 = 0.0
        COP  = z0 + (a*T_sink) + (b*T_source) + (c*T_sink**2) + (d*T_source**2) + (f*T_sink*T_source) + (g*T_sink**2*T_source**2)
           
        return COP
       
    def Qth(self, T_sink,T_amb=0):
        """ Calculates thermal power from polynomial curve fit.

        Parameters
        ----------
        T_sink : float
            sink / supply flow temperature [degC]

        T_amb : float
            ambient temperature [degC], default = 0 degC (used to estimate ground temperature)

        Returns
        -------
        float
            thermal power
        """
        # estimation of ground temperature based on ambient temperature
        from disturbances import t_ground_from_amb
        T_source = t_ground_from_amb(T_amb)
    
        z0	 = 6.366547
        a	 = -0.013634
        b	 = 0.231815
        c	 = -0.000132
        d	 = 0.001413
        f	 = -0.001614
        g	 = 0.0
        Qdot_th_nom  = z0 + (a*T_sink) + (b*T_source) + (c*T_sink**2) + (d*T_source**2) + (f*T_sink*T_source) + (g*T_sink**2*T_source**2)
         
        return Qdot_th_nom

    def plot_cop(self):
        """ Plots COP vs source temperature.
        """
        import matplotlib.pyplot as plt
        T_amb = np.linspace(-20,30,51)
        from disturbances import t_ground_from_amb
        T_ground = t_ground_from_amb(T_amb)
        T_supply = np.linspace(35,60,6) # degC
        fig, ax = plt.subplots()
        for temp in T_supply:
            cop = self.COP(temp, T_ground)
            ax.plot(T_ground, cop, label = f'{temp}')
        fig.legend(title='T_sup [degC]', bbox_to_anchor=[0.5, 0.95], loc = 'center', ncol=3)
        ax.set_ylabel('Coefficient of performance (COP)')
        ax.set_xlabel('Ground temperature [degC]')
        

class Heatpump_AW(Heatpump):
    """ A model class for Dimplex LA 6TU air-water heatpump,
        performance based on fit with manufacturer data.

        Notes
        -----
        The parameterization of the heat pump model was based on the manufacturer data of the Dimplex LA 6TU air-water heat pump.
        This model was selected because it is listed in the BAFA list and ranks in the middle range regarding COP.
            - Return temperature min. / Supply temperature max. 18 / 60 °C +/- 2
            - Lower operating limit heat source (heating mode) / Upper operating limit heat source (heating mode) -22 / 35 °C
            - Heating capacity from 2.28..8.04 kW
        
        Original German text (for reference):
        Die Parametrierung des Wärmepumpenmodells erfolgte anhand der Herstellerdaten der Dimplex LA 6TU Luft/Wasser-Wärmepumpe.
        Dieses Modell wurde ausgewählt, weil es zum einen in der BAFA-Liste aufgeführt wird und dort bezüglich des COPs im Mittelfeld rangiert.
            - Rücklauftemperatur min. / Vorlauftemperatur max. 7) 18 / 60 °C +/- 2
            - Untere Einsatzgrenze Wärmequelle (Heizbetrieb) / Obere Einsatzgrenze Wärmequelle (Heizbetrieb) -22 / 35 °C
            - Heizleistung von 2.28..8.04
    """
    def __init__(self, 
                 mdot_HP = 0.25,      # heatpump nominal mass flow 
                 verbose = False,     # Set to true to plot performance curve of HP
                 ):
        #print('Air-water heat pump (Dimplex LA 6TU)')
        super(Heatpump_AW,self).__init__(mdot_HP=mdot_HP)
        if verbose == True:
            self.plot_cop()

    def COP(self,T_hp,T_amb):
        """ Calculates coefficient of performance (COP) from polynomial curve fit.

        Parameters
        ----------
        T_hp : float
            supply flow temperature [degC]

        T_amb : float
            ambient temperature [degC]

        Returns
        -------
        float
            COP
        """
        a = np.array((8.2553, -0.17068, 0.16176, 0.00108, 0.00022, -0.00186))
        COP = a[0] + a[1]*T_hp + a[2]*T_amb + a[3]*T_hp**2 + a[4]*T_amb**2 + a[5]*T_hp*T_amb
        return COP
        
    def Qth(self,T_hp,T_amb):
        """ Calculates thermal power from polynomial curve fit.

        Parameters
        ----------
        T_sink : float
            supply flow temperature [degC]

        T_amb : float
            ambient temperature [degC]

        Returns
        -------
        float
            thermal power [kW]
        """
        b = np.array((6154,-20.920,189.00,-0.13444,0.70513,-0.87076))
        Qth_nom = b[0] + b[1]*T_hp + b[2]*T_amb + b[3]*T_hp**2 + b[4]*T_amb**2 + b[5]*T_hp*T_amb 
        return Qth_nom/1000

    def plot_cop(self):
        """ Plots COP vs source temperature.
        """
        import matplotlib.pyplot as plt
        T_amb = np.linspace(-20,30,51)
        T_supply = np.linspace(35,60,6) # degC
        fig, ax = plt.subplots()
        for temp in T_supply:
            cop = self.COP(temp, T_amb)
            ax.plot(T_amb, cop, label = f'{temp}')
        fig.legend(title='T_sup [degC]', bbox_to_anchor=[0.5, 0.95], loc = 'center', ncol=3)
        ax.set_ylabel('Coefficient of performance (COP)')
        ax.set_xlabel('Ambient temperature [degC]')

if __name__ == '__main__':
    #hp_model = Heatpump_AW(mdot_HP = 0.25, verbose=True)
    hp_model = Heatpump_Vitocal(mdot_HP = 0.25, verbose=True)