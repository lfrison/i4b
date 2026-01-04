class PID:
    ''' Simple pid controller
    
    Attributes
    ----------
    timestep : int
        Duration of one simulation step [sec]

    KP : float
        Propotional gain

    KI : float
        Integration gain

    KD : float
        Differential gain 
    '''
    
    def __init__(self, timestep, KP, KI, KD):
        self.timestep = timestep   # Duration of one simulation step [sec]
        self.KP = KP               # Propotional gain 
        self.KI = KI               # Intgration gain
        self.KD = KD               # Differential gain
        self.last_error = 0
        self.sum_error = 0


    def calc(self, set_point, actual_value):
        ''' Calculate the control variable

        Parameters
        ----------
        set_point : float
            Set point (w, Sollwert)
        
        actual_value : float
            Actual value (x, Istwert)

        Returns
        -------
        float
            control variable (u, Stellgröße)

        '''

        error = set_point - actual_value      # error [degC] | Reglerabweichung (e = w - x)
        de = (error - self.last_error) / self.timestep       # difference of the error de/dt [K/s]
        self.last_error = error
        self.sum_error = self.sum_error + error
        uk = self.KP * error + self.KI * self.sum_error + self.KD * de # Stellgröße (u)
        return uk
    
