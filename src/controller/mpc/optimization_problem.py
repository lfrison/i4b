# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 17:58:34 2021

@author: lfrison
"""
from src.constants import C_WATER_SPEC
import casadi as cas

class optimization_problem:
   """Layered model of storage tank based on Eicker."""
   def __init__(self, dim,hp_model,building_model,ws):   
      # Declare variables (use scalar graph)
      p  = cas.SX.sym("p",dim['npar'])    # parameters
      u  = cas.SX.sym("u",dim['nu'])    # control
      s  = cas.SX.sym("s",dim['ns'])  # state
      x  = cas.SX.sym("x",dim['nx'])  # state
      #xa  = cas.SX.sym("xa",dim['nxa'])  # state
   
      self.hp_model = hp_model
      self.building_model = building_model
      # ODE rhs function and quadratures
      rhs = self.building_model.calc_casadi(x,u,p)
      cost = self.cost(x,u,p)
      cost_soft_constraint = self.cost_soft_constraint(x,u,p,s,ws)
      self.f = cas.Function('f', [x,u,p,s],[rhs, cost+cost_soft_constraint])
      self.m = cas.Function('m', [x,u,p,s],[cost_soft_constraint])
   
      self.constraint_array = dim['nc']*[None]
      self.constraint_array[0] = cas.Function('c_slack', [x,u,p,s],[self.constraint_slack(x,u,p,s)]) # soft comstraint
      self.constraint_array[1] = cas.Function('c_HP_low', [x,u,p,s],[self.constraint_HP_low(x,u,p,s)]) # (Hard) control constraint
      self.constraint_array[2] = cas.Function('c_HP_up', [x,u,p,s],[self.constraint_HP_up(x,u,p,s)]) # (Hard) control constraint
      self.constraint_array[3] = cas.Function('c_slack_upper', [x,u,p,s],[self.constraint_slack_upper(x,u,p,s)]) # soft comstraint
      
      # bounds on state and controls
      self.bounds_states = self.bounds_states(dim['nx'])
      self.bounds_controls = self.bounds_controls(dim['nu'])
      
      self.ws = ws # weighting factor for temperature slack variables
      

   ### Cost function for optimization problem, have to be defined as min c(x,u,p,s)  ###
   def cost(self,x,u,p):
      """
      Cost function for optimization problem   
      """
      return  self.hp_model.calc_cost(x,u,p)*p[-1]


   ### Constraints for optimization problem, have to be defined as c(x,u,p,s)>=0  ###
   def cost_soft_constraint(self,x,u,p,s,ws):
      """
      Soft constraints for optimization problem   
      ws : weighting factor for temperature slack variables
      """
      return ws*s[0]*s[0] + ws*s[1]*s[1]  #+ ws*s[0]
   
   def constraint_slack(self,x,u,p,s):
      T_room = x[0]
      T_set_low = p[-2]
      return T_room - T_set_low + s[0] 

   def constraint_slack_upper(self,x,u,p,s):
      T_room = x[0]
      T_set_upper = self.building_model.T_room_set_upper
      return T_set_upper - T_room + s[1] 
   
   # bounds on Qhp are temperature dependent: Qhp<=32.875-0.225Tsink , Qhp>=8.75-0.05Tsink
   def constraint_HP_low(self,x,u,p,s):
      T_RL = x[-1]
      return C_WATER_SPEC*self.hp_model.mdot_HP/1000*(u[0] - T_RL) 
   
   def constraint_HP_up(self,x,u,p,s):
      T_RL = x[-1]
      return -(C_WATER_SPEC*self.hp_model.mdot_HP/1000*(u[0] - T_RL)  - 26)# (20 - 0.225*u[0])) # check if HP specific
   


   def bounds_controls(self,nu):
      """
      Bounds for control   
      """
      u_min = [0]
      u_max = [65.]
      u_init = [35.]   
      return u_min[:nu], u_max[:nu], u_init[:nu]
      
   def bounds_states(self,nx):
      """
      Bounds for states
      """
      x_min = [0.0 for i in range(nx)]
      x_max = [100.0 for i in range(nx)]
      x_init = [35.0 for i in range(nx)]
      return x_min, x_max, x_init