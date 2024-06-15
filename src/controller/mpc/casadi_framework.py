import casadi as cas
import numpy as np


import src.controller.mpc.mpc_utility as util
import src.controller.mpc.optimization_problem as optimization_problem


class MPC_solver:
      
   def __init__(self,resultdir,resultfile,hp_model,building_model,nx=4,nu=1,npar=5,ns=1,nc=3,h = 900,nk = 96,ws=1):
      self.resultdir = resultdir
      self.resultfile = resultfile
        
      self.h = h # sampling length (in s)
      self.nk = nk # prediction horizon (number of steps)

      # Problem dimension and collocation points d
      self.dim = {'d':3,'nx':nx,'nxa':0,'nu':nu,'np':5,'ns':ns,'npar':npar,'nc':nc} 

      # Create instance of optimization problem class
      self.OP = optimization_problem.optimization_problem(self.dim,hp_model,building_model,ws)
      self.hp_model = hp_model
      self.building_model = building_model
               
      # Set up NLP with initial states xk
      self.collocation_setup()
      self.NLPvariable_setup()

      
   def update_NLP(self,xk_next):
      nx = xk_next.shape[0]
      # Update NLP for next iteration
      self.vars_init[:nx] =np.reshape(xk_next,self.vars_init[:nx].shape)
      self.vars_lb = np.concatenate((np.reshape(xk_next,(nx,)),self.vars_lb[nx:]))
      self.vars_ub = np.concatenate((np.reshape(xk_next,(nx,)),self.vars_ub[nx:]))


   def solve_NLP(self,P_opt,PRINT=False):
      dim= self.dim

      # Update & solve NLP     
      g,lbg,ubg,J = self.NLPfunction_setup(P_opt)
      nlp_opts = self.NLP_options()
      nlp = {'x':self.V, 'f':J, 'g':g}
      solver = cas.nlpsol("solver", "ipopt", nlp, nlp_opts)
      arg = {"x0":self.vars_init, "lbg":np.concatenate(lbg), "ubg":np.concatenate(ubg), "lbx":self.vars_lb, "ubx":self.vars_ub}
      
      res = solver(**arg)
      if res["f"] > 1e7: print("WARNING: ipopt solution very large obj=%s." %res["f"]); 
      #else: print("ipopt solution obj=%s." %res["f"]); 
      
      if PRINT:
         cost_opt, dev_opt = util.evaluate_ocp(res,self.dim,self.nk,self.h,P_opt,self.hp_model,self.building_model,mfig=(12,4))   
         
      # Extract first component of optimal control --> current control u_k
      d = dim['d']
      nx = dim['nx']
      nu = dim['nu']
      ns = dim['ns']
      uk = res["x"][(d+1)*nx+(d+1)*ns:(d+1)*nx+(d+1)*ns+nu]
      #uk_next = res["x"][(d+1)*nx+(d+1)*ns+(d+1)*nx+nu+(d+1)*ns:(d+1)*nx+(d+1)*ns+(d+1)*nx+nu+(d+1)*ns+nu]   
      #xk = res["x"][:nx]   
      xk_next = res["x"][(d+1)*nx+(d+1)*ns+nu:(d+1)*nx+(d+1)*ns+nu+nx]   
      
        # Update NLP for next iteration
      self.vars_init = np.concatenate((res["x"][((d+1)*(nx+ns)+nu):],res["x"][-((d+1)*(nx+ns)+nu):]))
      
      return uk, xk_next
      
   
   def NLP_options(self):
      opts = {}
      opts["expand"] = True
      #opts["ipopt.linear_solver"] = 'ma27'
      opts["ipopt.print_frequency_iter"] = 10
      opts["ipopt.print_level"] = 0
      opts["ipopt.file_print_level"] = 4
      opts["ipopt.output_file"] = 'ipopt.log'
      opts["ipopt.sb"] = "yes";
      opts["print_time"] = 0;
      #opts["ipopt.tol"] = 1e-8
      opts["ipopt.print_timing_statistics"] = 'no'
      opts["ipopt.max_iter"] = 1500
      
      return opts


   def NLPvariable_setup(self):   
      dim=self.dim
      
      # Bounds and initial guess
      u_min, u_max, u_init = self.OP.bounds_controls
      x_min, x_max, x_init = self.OP.bounds_states 
      
      # Dimensions
      NX = self.nk*(dim['d']+1)*dim['nx']+dim['nx'] # Collocated states
      NU = self.nk*dim['nu'] # Parametrized controls
      NS = self.nk*(dim['d']+1)*dim['ns']+dim['ns'] # Slack variables
      NVar = NX+NU+NS
   
      # NLP variable vector
      self.V = cas.MX.sym("V",NVar)  
   
      # All variables with bounds and initial guess
      self.vars_lb = np.zeros(NVar)
      self.vars_ub = np.zeros(NVar)
      self.vars_init = np.zeros(NVar)
   
      # Get collocated states and parametrized control
      self.X = np.resize(np.array([],dtype=cas.MX),(self.nk+1,dim['d']+1))
      self.U = np.resize(np.array([],dtype=cas.MX),self.nk)
      self.S = np.resize(np.array([],dtype=cas.MX),(self.nk+1 ,dim['d']+1))
   
      offset = 0
      for k in range(self.nk):
         for j in range(dim['d']+1):
           # Get the expression for the state vector
            self.X[k,j] =self.V[offset:offset+dim['nx']]
            # Add the initial condition & bounds
            self.vars_init[offset:offset+dim['nx']] = x_init
            self.vars_lb[offset:offset+dim['nx']] = x_min
            self.vars_ub[offset:offset+dim['nx']] = x_max
            offset += dim['nx']
            
            # Slack variable
            self.S[k,j] = self.V[offset:offset+dim['ns']]
            self.vars_init[offset:offset+dim['ns']] = [0 for i in range(dim['ns'])]
            self.vars_lb[offset:offset+dim['ns']] = [0 for i in range(dim['ns'])]
            self.vars_ub[offset:offset+dim['ns']] = [np.inf for i in range(dim['ns'])]
            offset+=dim['ns']
   
         # Parametrized controls
         self.U[k] = self.V[offset:offset+dim['nu']]
         self.vars_lb[offset:offset+dim['nu']] = u_min
         self.vars_ub[offset:offset+dim['nu']] = u_max
         self.vars_init[offset:offset+dim['nu']] = u_init
         offset += dim['nu']
        
      # State at end time
      self.X[self.nk,0] = self.V[offset:offset+dim['nx']]
      self.vars_lb[offset:offset+dim['nx']] = x_min #xf_min
      self.vars_ub[offset:offset+dim['nx']] = x_max
      self.vars_init[offset:offset+dim['nx']] = x_init
      offset += dim['nx']
   
      # Slack variable at end time
      self.S[self.nk,0] = self.V[offset:offset+dim['ns']]
      self.vars_lb[offset:offset+dim['ns']] = [0 for i in range(dim['ns'])]
      self.vars_ub[offset:offset+dim['ns']] = [np.inf for i in range(dim['ns'])]
      self.vars_init[offset:offset+dim['ns']] = [0 for i in range(dim['ns'])]
      offset+=dim['ns']
      
      
   def NLPfunction_setup(self,P):  
      dim=self.dim

      # Constraint & Objective function for the NLP
      g = []
      lbg = []
      ubg = []
      J = 0
      for k in range(self.nk):
         # Constraints at start point
         for i in range(dim['nc']):
            g.append(self.OP.constraint_array[i](self.X[k,0],self.U[k],P[k,:],self.S[k,0]))
            lbg.append(np.zeros(1))
            ubg.append(np.ones(1)*np.inf)
         
            
         for j in range(1,dim['d']+1):
            # Get an expression for the state derivative at the collocation point
            xp_jk = 0
            for r in range (dim['d']+1):
               xp_jk += self.C[r,j]*self.X[k,r]
            # Add collocation equations to the NLP
            fk,qk = self.OP.f(self.X[k,j],self.U[k],P[k,:],self.S[k,j])
            g.append(self.h*fk - xp_jk)
            lbg.append(np.zeros(dim['nx'])) # equality constraints
            ubg.append(np.zeros(dim['nx'])) # equality constraints
            
            # Add contribution to objective
            J += self.F[j]*self.h*qk
            
            # Constraints at the collocation point
            for i in range(dim['nc']):
               g.append(self.OP.constraint_array[i](self.X[k,j],self.U[k],P[k,:],self.S[k,j]))
               lbg.append(np.zeros(1))
               ubg.append(np.ones(1)*np.inf)
            
            # Add contribution to objective
            J += self.OP.m(self.X[k,j],self.U[k],P[k,:],self.S[k,j])

         # Get an expression for the state at the end of the finite element
         xf_k = 0
         for r in range(dim['d']+1):
            xf_k += self.D[r]*self.X[k,r]
            
         # Add continuity equation to NLP
         g.append(self.X[k+1,0] - xf_k)
         lbg.append(np.zeros(dim['nx']))
         ubg.append(np.zeros(dim['nx']))           

      # Constraints at the end of horizon (not necessary)
      for i in range(dim['nc']):
         g.append(self.OP.constraint_array[i](self.X[k,j],self.U[k],P[k,:],self.S[k,j]))
         lbg.append(np.zeros(1))
         ubg.append(np.ones(1)*np.inf)     

      # Add contribution to objective
      J += self.OP.m(self.X[self.nk-1,0],self.U[self.nk-1],P[k,:],self.S[self.nk,0])
          
      # Concatenate constraints
      g = cas.vertcat(*g)
      
      return g,lbg,ubg,J
   
   
     
   def collocation_setup(self):
      d = self.dim['d']
      tau_root = [0] + cas.collocation_points(d, "radau") 
      # Coefficients of the collocation equation, continuity equation, quadrature function
      self.C = np.zeros((d+1,d+1))
      self.D = np.zeros(d+1)
      self.F = np.zeros(d+1)
      # Construct polynomial basis
      for j in range(d+1):
         # Construct Lagrange polynomials to get the polynomial basis at the collocation point
         poly = np.poly1d([1])
         for r in range(d+1):
            if r != j:
               poly *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
         # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
         self.D[j] = poly(1.0)
         # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
         pder = np.polyder(poly)
         for r in range(d+1):
            self.C[j,r] = pder(tau_root[r])
         # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
         polyint = np.polyint(poly)
         self.F[j] = polyint(1.0)