# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 17:58:34 2021

@author: lfrison
"""
from src.constants import C_WATER_SPEC
import casadi as cas


class optimization_problem:
   """Layered model of storage tank based on Eicker."""
   def __init__(self, dim, hp_model, building_model, ws, chance_epsilon=None, soft_lower_constraint=False):
      # Declare variables (use scalar graph)
      p  = cas.SX.sym("p",dim['npar'])    # parameters
      u  = cas.SX.sym("u",dim['nu'])      # control
      s  = cas.SX.sym("s",dim['ns'])      # slack variables
      x  = cas.SX.sym("x",dim['nx'])      # state
      #xa  = cas.SX.sym("xa",dim['nxa'])  # state
   
      self.hp_model = hp_model
      self.building_model = building_model
      self.chance_epsilon = chance_epsilon
      self.soft_lower_constraint = soft_lower_constraint

      # ODE rhs function and quadratures
      rhs = self.building_model.calc_casadi(x,u,p)
      cost = self.cost(x,u,p)
      cost_soft_constraint = self.cost_soft_constraint(x,u,p,s,ws)
      self.f = cas.Function('f', [x,u,p,s],[rhs, cost+cost_soft_constraint])
      self.m = cas.Function('m', [x,u,p,s],[cost_soft_constraint])
   
      # Prefer model-provided constraints when available (used by newer PCE/BESS models).
      if hasattr(self.building_model, "constraint_exprs"):
         constraint_exprs = self.building_model.constraint_exprs(
            x, u, p, s, self.hp_model,
            chance_epsilon=self.chance_epsilon,
            soft_lower_constraint=self.soft_lower_constraint,
         )
         self.constraint_array = [
            cas.Function(f'c_{idx}', [x,u,p,s], [expr])
            for idx, expr in enumerate(constraint_exprs)
         ]
      else:
         self.constraint_array = []
         self.constraint_array.append(cas.Function('c_slack', [x,u,p,s],[self.constraint_slack(x,u,p,s)]))
         self.constraint_array.append(cas.Function('c_HP_low', [x,u,p,s],[self.constraint_HP_low(x,u,p,s)]))
         self.constraint_array.append(cas.Function('c_HP_up', [x,u,p,s],[self.constraint_HP_up(x,u,p,s)]))
         if hasattr(self.building_model, "T_room_set_upper") and dim['ns'] > 1:
            self.constraint_array.append(cas.Function('c_slack_upper', [x,u,p,s],[self.constraint_slack_upper(x,u,p,s)]))
      
      # bounds on state and controls
      self.bounds_states = self.bounds_states(dim['nx'])
      self.bounds_controls = self.bounds_controls(dim['nu'])
      
      self.ws = ws # weighting factor for temperature slack variables
      

   ### Cost function for optimization problem, have to be defined as min c(x,u,p,s)  ###
   def cost(self,x,u,p):
      """
      Cost function for optimization problem   
      """
      if hasattr(self.building_model, "stage_cost_expr"):
         return self.building_model.stage_cost_expr(x,u,p,self.hp_model)

      T_RL = self.building_model.return_temperature_expr(x)
      T_HP = u[0]
      T_amb = p[0]
      COP = self.hp_model.COP(T_HP,T_amb)
      Qth = self.hp_model.mdot_HP*self.hp_model.c_water*(T_HP-T_RL)/1000
      return Qth/(COP * 100)*p[-1]


   ### Constraints for optimization problem, have to be defined as c(x,u,p,s)>=0  ###
   def cost_soft_constraint(self,x,u,p,s,ws):
      """
      Soft constraints for optimization problem   
      ws : weighting factor for temperature slack variables
      """
      if not self.soft_lower_constraint:
         return 0
      return ws*cas.dot(s,s)
   
   def constraint_slack(self,x,u,p,s):
      slack_vec = s if self.soft_lower_constraint else 0*s
      return self.building_model.lower_comfort_constraint_expr(
         x, p, slack_vec, epsilon=self.chance_epsilon
      )

   def constraint_slack_upper(self,x,u,p,s):
      T_room = x[0]
      T_set_upper = self.building_model.T_room_set_upper
      return T_set_upper - T_room + s[1]
   
   # bounds on Qhp are temperature dependent: Qhp<=32.875-0.225Tsink , Qhp>=8.75-0.05Tsink
   def constraint_HP_low(self,x,u,p,s):
      T_RL = self.building_model.return_temperature_expr(x)
      return C_WATER_SPEC*self.hp_model.mdot_HP/1000*(u[0] - T_RL) 
   
   def constraint_HP_up(self,x,u,p,s):
      T_RL = self.building_model.return_temperature_expr(x)
      return -(C_WATER_SPEC*self.hp_model.mdot_HP/1000*(u[0] - T_RL)  - 26)# (20 - 0.225*u[0])) # check if HP specific
   


   def bounds_controls(self,nu):
      """
      Bounds for control   
      """
      u_min, u_max, u_init = self.building_model.control_bounds()
      return u_min[:nu], u_max[:nu], u_init[:nu]

   def bounds_slacks(self,ns):
      """Return slack bounds while keeping the original hard-slack default."""
      if self.soft_lower_constraint:
         s_min = [0.0 for i in range(ns)]
         s_max = [float("inf") for i in range(ns)]
         s_init = [0.0 for i in range(ns)]
         return s_min, s_max, s_init

      s_min = [0.0 for i in range(ns)]
      s_max = [0.0 for i in range(ns)]
      s_init = [0.0 for i in range(ns)]
      return s_min, s_max, s_init
      
   def bounds_states(self,nx):
      """
      Bounds for states
      """
      x_min, x_max, x_init = self.building_model.state_bounds()
      return x_min[:nx], x_max[:nx], x_init[:nx]
