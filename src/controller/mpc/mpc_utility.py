import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from shutil import copyfile

   
def evaluate_mpc(hp_model,building_model,resultfile='results_mpc',resultdir=None,nsamples=0,offset=0,h=900,mfig=(12,4)):
   df_data = pd.read_csv('%s/%s.csv'%(resultdir,resultfile), sep=',', header='infer')
   if nsamples==0: nsamples = len(df_data)
   time = range(0,nsamples*h,h)
   print ('\nEvaluate_mpc %s: # samples=%d, days=%d'%(resultfile,nsamples,nsamples/int(24*3600/h)))
   
   gains = df_data['Qdot_gains'].values[offset:offset+nsamples]/1000
   grid = df_data['grid_signal'].values[offset:offset+nsamples]
   T_amb = df_data['T_amb'].values[offset:offset+nsamples]
   T_room = df_data['T_room'].values[offset:offset+nsamples]
   if building_model.method == '4R3C' or building_model.method == '5R4C': 
      T_wall = df_data['T_wall'].values[offset:offset+nsamples]
      C_sto_wall = building_model.params['C_wall']*((T_wall-T_amb))/1000/3600
   if building_model.method == '5R4C': 
      T_int = df_data['T_int'].values[offset:offset+nsamples]
   T_return = df_data['T_return'].values[offset:offset+nsamples]
   T_HP = df_data['T_HP'].values[offset:offset+nsamples]

   ## Evaluate results
   cop = hp_model.COP(T_HP,T_amb)
   Pth_HP = hp_model.mdot_HP*hp_model.c_water*(T_HP-T_return)/1000
   Pel_HP = Pth_HP/cop
      
   T_lower = df_data['T_room_set_lower'] #building_model.T_room_set_lower
   arr = np.max(np.column_stack((T_lower*np.ones(nsamples)-T_room,np.zeros(nsamples))),1) 
   arrsum = np.sum(arr)

   ## Evaluate results
   print("Qth_HP=%.2fkWh (av. per day=%.2fkWh), Qel_HP=%.2fkWh."         
         % (np.nansum(Pth_HP/(3600/h)),np.nansum(Pth_HP/(3600/h))/(nsamples/(24*3600/h)),
            np.nansum(Pel_HP/(3600/h))))
   print("Total cost (grid impact): %.2f" % (np.sum(Pel_HP/(3600/h)*grid)))
         
   print("Consumer comfort deviation:av.=%.4fKh, max=%.4fK" 
         % (arrsum/(3600/h)/(nsamples*3600/h),max(arr)))

   if mfig==0: return 

   # Plot results

   fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=mfig)
   plt.plot(time,T_return,'b',label='T_return',drawstyle='steps-post') 
   plt.plot(time,T_room,'r',label='T_room',drawstyle='steps-post')
   if building_model.method == '4R3C' or building_model.method == '5R4C':
      plt.plot(time,T_wall,color='orange',label='T_wall',drawstyle='steps-post')
   if building_model.method == '5R4C': 
      plt.plot(time,T_int,'y',label='T_int',drawstyle='steps-post')
   plt.plot(time,T_HP,'g',label='T_HP',drawstyle='steps-post')
   plt.plot(time,T_amb,'k',label='T_amb',drawstyle='steps-post')
   if np.mean(grid)!=1:
      plt.plot(time,grid*100,color='black',linestyle='dashed',label='grid',drawstyle='steps-post')

   ax0 = ax.twinx()
   ax0.plot(time,Pth_HP,color='darkred',linestyle='dashed',label='Pth_HP',drawstyle='steps-post')
   ax0.plot(time,gains,color='darkgreen',linestyle='dashed',label='gains',drawstyle='steps-post')
   ax0.set_ylabel('Pth [kW]', color='k')
   #if building_model.method == '4R3C' or building_model.method == '5R4C':
   #   ax0.plot(time,C_sto_wall,color='magenta',label='C_sto_wall',drawstyle='steps-post')
   ax0.tick_params('y', colors='k')
   #ax0.set_ylim([0,30])
   ax.set_ylabel('temperature [°C]', color='k')
   ax.grid()
   ax.legend()
   ax0.legend()
   plot(fig,ax,plt,h*nsamples)

   
 
    
def evaluate_ocp(res,dim,nk,h,P,hp_model,building_model,mfig=0):
   d = dim['d']
   nx = dim['nx']
   nxa = dim['nxa']
   nu = dim['nu']
   ns = dim['ns']
   
   T_amb = P[:nk,0]
   
   ## Get values at the beginning of each finite element
   x0_opt = res["x"][0::(d+1)*nx+nu+(d+1)*ns+d*nxa]
   xn_opt = res["x"][nx-1::(d+1)*nx+nu+(d+1)*ns+d*nxa]
   u0_opt = res["x"][(d+1)*nx+(d+1)*ns+d*nxa::(d+1)*nx+nu+(d+1)*ns+d*nxa]
   tgrid = np.linspace(0,h*nk,nk+1)
   
   T_HP = u0_opt
   Pth_HP = hp_model.mdot_HP*hp_model.c_water*np.reshape((T_HP-xn_opt[:nk]),(nk,))/1000
   #Pth_load = 0.583*4181*np.reshape((xn_opt[:nk]-x0_opt[:nk]),(nk,))/1000
   cop = hp_model.COP(T_HP,T_amb)
   grid = P[:,-1]

    
   ## Evaluate results

   T_room_set = P[:-1,-2] #building_model.T_room_set_lower
   arr = np.max(np.column_stack((T_room_set*np.ones(nk)-x0_opt[:nk],np.zeros(nk))),1) 
   print("\nEvaluate_ocp:\n- Total consumption Qth_HP=%.2fkWh (av. per day %.2fkWh), Qel_HP=%.2fkWh." % (np.nansum(Pth_HP/(3600/h)),np.nansum(Pth_HP/(3600/h))/(nk/(24*3600/h)),np.nansum(Pth_HP/cop/(3600/h))))
   #print("\nPth_load=%.2fkWh." % (np.nansum(Pth_load/(3600/h))))
   print("- Consumer comfort deviation: av.=%.3fK, max=%.2f" % (np.sum(arr/(3600/h))/(nk*3600/h), max(arr)))
    
   if mfig==0: return Pth_HP/cop, arr

   # Plot results
   fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=mfig)
   plt.plot(tgrid,xn_opt,'b',label='T_return',drawstyle='steps-post') 
   plt.plot(tgrid,x0_opt,'r',label='T_room',drawstyle='steps-post')
   plt.plot(tgrid[:-1],T_HP,'g',label='T_HP',drawstyle='steps-post')
   plt.plot(tgrid[:-1],T_amb,'k',label='T_amb',drawstyle='steps-post')
   if np.mean(grid)!=1:
      plt.plot(tgrid,grid*100,color='black',linestyle='dashed',label='grid',drawstyle='steps-post')
   ax0 = ax.twinx()
   ax0.plot(tgrid[:-1],Pth_HP,color='darkred',label='Pth_HP',drawstyle='steps-post')
   ax0.set_ylabel('Pth_HP', color='k')
   ax0.tick_params('y', colors='k')
   ax.grid()
   ax.legend()
   ax0.legend()
   plot(fig,ax,plt,h*nk)
   
   return Pth_HP/cop, arr


def init_file(columns,resultdir = "results_mpc",resultfile=None):
   # MPC result data frame
   df_mpc = pd.DataFrame(columns=columns)
   if not resultfile==None: 
      # MPC result file initialization
      resultdir = resultdir
      if not os.path.exists(resultdir): os.makedirs(resultdir)
      resultfile = resultfile
      if os.path.exists("%s/%s.csv" % (resultdir,resultfile)): os.remove('%s/%s.csv' % (resultdir,resultfile))  
      df_mpc.to_csv('%s/%s.csv' % (resultdir,resultfile), header=True,index=None)
      print ('\n****** Result file: %s/%s.csv.' %(resultdir,resultfile))
      
      
def update_file(time,resultfile,resultdir,P,uk,xk):
   nx = xk.shape[0]
   res_arr = np.column_stack((time,np.reshape(xk[:nx],(1,nx)),uk,np.reshape(P[0,:],(1,P.shape[1]))))
   df_mpc = pd.DataFrame(data=res_arr)
   df_mpc.to_csv('%s/%s.csv' % (resultdir,resultfile), header=False,mode='a',index=None) 
  

def storage_capacity(P,P_ref,nk):
   s=sum(x for x in P-P_ref if x>0)/(nk/24)
   print("- Storage capacity: %.2fkWh, %.2fkWh/day" % (s,s/float(P.shape[0]/nk)))
   return s


def read_Tamb(name,Plant_model,col='Dry_bulb',folder='..\inputdata'):
   read_data(name,col,res=0,folder=folder)
   df_data = pd.read_csv(Path(folder) / ('%s.csv'%name),header='infer')
   input_data = df_data[col].values
   data = np.zeros((input_data.shape[0],3))   
   data[:,0] = input_data # ambient temperature 
   T_sup_set, T_ret_set = Plant_model.Heatingcurve().calc(data[:,0])
   data[:,1] = T_sup_set
   data[:,2] = T_ret_set
   return data


def read_data(name,col='Grid',scale=1.,res_old=0.25,res_new=0.25,days=50,SCALE=False,NORM=False,folder='data'):
   prevwd = os.getcwd(); os.chdir(folder)
   df_data = pd.read_csv('%s.csv'%name, sep=',',header='infer')
   os.chdir(prevwd)
   data = np.zeros(days*int(24/res_new))

   #\
   if res_old==res_new: 
      data=df_data[col].values*scale
   elif res_old<res_new: #wrong
      for i in range(int(24/res_new)*days): data[i] = df_data[col].values[int(i/res_old*res_new)]*scale
   elif res_old>res_new: 
      for i in range(int(24/res_old)*days): data[i*int(res_old/res_new):(i+1)*int(res_old/res_new)] = df_data[col].values[i]*scale
   
   ## scale grid signal  
   if NORM: data-= np.nanmin(data)
   if SCALE==1: data /= np.nanmax(data)
   elif SCALE==2: data[:96] /= np.nanmax(data[:96])
      #for i in range(2*days): data[i*int(12/res):(i+1)*int(12/res)] /= np.nanmax(data[i*int(12/res):(i+1)*int(12/res)])
   
     #data = np.max(np.column_stack((data,np.ones(nk*days)*0)),1)
      #for i in range(days): 
      #   data[i*nk:(i+1)*nk] -= np.min(data[i*nk:(i+1)*nk])
      #   data[i*nk:(i+1)*nk] /= np.max(data[i*nk:(i+1)*nk])
   
   return data

def read_data_Strabu(inputfile,mpc_steps=8760):
   
   # read data file 
   df_data = pd.read_csv('data\data_Strabu\%s.ISE'%inputfile, sep='\t',header='infer',index_col=0,skiprows=1)
   
   # read ambient temperature and generate 
   Tamb = df_data[ 'DRY_BULB                 '].values
   T_sup_set, T_ret_set = Plant_model.Heatingcurve().calc(Tamb,rr=True)
   data_Tamb = np.column_stack((Tamb,T_sup_set,T_ret_set))
   
   # read building load data
   data_load = df_data['QHEAT_SUM_kW             '].values*1000
   
   # read grid data
   data_grid = pd.read_csv('data\grid_signals.csv',sep=',',header='infer')['EEX2020'].values

   print ('Input data shape',data_grid.shape,data_load.shape,data_Tamb.shape)
   #print (data_Tamb[:int(24/res_new),:])
   data = np.column_stack((data_load[:mpc_steps],data_grid[:mpc_steps],data_Tamb[:mpc_steps]))
   
   plot_data(data)
   
   return data


def read_data_csv(mpc_steps,inputfile,res_new=.25,res_old=0.25,scale=1,scale_grid=1):
   days=int(mpc_steps/24*res_new)+1
   #print('Load data for %d days.' %days)
   
   # read ambient temperature and generate 
   Tamb = read_data(inputfile,'T_amb',res_new=res_new,res_old=res_old,days=days)
   T_sup_set, T_ret_set = Plant_model.Heatingcurve().calc(Tamb,rr=True)
   data_Tamb = np.column_stack((Tamb,T_sup_set,T_ret_set))
   
   # read building load data
   #data_load = util.read_data('Pth_load_meas','Pth_load',res_new=res_new,res_old=.125,days=days)
   data_load = read_data(inputfile,'P_htg',scale=scale,res_new=res_new,res_old=res_old,days=days)
   
   # read grid data
   data_grid = read_data('grid_signals','EEX2020',scale_grid,SCALE=0,res_new=res_new,res_old=.25,days=days) #np.ones(data_load.shape)

   print ('Input data shape',data_grid.shape,data_load.shape,data_Tamb.shape)
   #print (data_Tamb[:int(24/res_new),:])
   data = np.column_stack((data_load[:int(mpc_steps+24/res_new)],data_grid[:int(mpc_steps+24/res_new)],data_Tamb[:int(mpc_steps+24/res_new),:]))
   
   plot_data(data)
   return data


def read_data_hil(mpc_steps,inputfile,res_new=.25,res_old=0.25,scale=1,scale_grid=1):
   days=int(mpc_steps/24*res_new)+1
   #print('Load data for %d days.' %days)
   Tamb = read_data(inputfile,'T_amb',res_new=res_new,res_old=res_old,days=days)
   T_sup_set, T_ret_set = Plant_model.Heatingcurve().calc(Tamb,rr=True)
   data_Tamb = np.column_stack((Tamb,T_sup_set,T_ret_set))
   #data_load = util.read_data('Pth_load_meas','Pth_load',res_new=res_new,res_old=.125,days=days)
   data_load = read_data(inputfile,'Pth_load',scale=scale,res_new=res_new,res_old=res_old,days=days)
   data_grid = read_data(inputfile,'Grid',scale_grid,SCALE=0,res_new=res_new,res_old=res_old,days=days) #np.ones(data_load.shape)
   print (data_grid.shape,data_load.shape,data_Tamb.shape,days)
   #print (data_Tamb[:int(24/res_new),:])
   data = np.column_stack((data_load[:],data_grid[:],data_Tamb[:,:]))
   
   plot_data(data)
   return data
   """
   def add_noise(self,data):
      num = data.shape[0]
      #print ('Data before noise:',data[:5],data[-5:])  
      noise = np.random.standard_normal(size=(num))*.1
      factor = [i**1.5 for i in range(num)]
      #print ('Data after noise:',data[:5],data[-5:])   

         self.grid.append(noise_vec)
         df = pd.DataFrame(self.grid)
         df.to_csv('%s/noise_%s.csv' 
                   %(self.resultdir,self.resultfile), index=False, header=False)
         
   return noise*factor*np.mean(data)
   """
   
def plot_data(data,nk=24,h=3600,mfig=10):      
    print ('Load: mean=%.3fkW, max=%.3fkW, min=%.3fkW.' 
           %(np.mean(data[:,0]/1000),np.max(data[:,0]/1000),np.min(data[:,0]/1000)))
    print ('Grid: signal mean=%.3f, dev=%.3f, cov=%.3f, max=%.3f, min=%.3f,' 
           %(np.mean(data[:,1]),np.std(data[:,1]),np.std(data[:,1])/np.mean(data[:,1]),np.max(data[:,1]),np.min(data[:,1])))

    fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(mfig,2))
    ax.plot(data[:nk*int(3600/h),0]/1000.,'r',label='Qload_kW',drawstyle='steps-post')
    ax.plot(data[:nk*int(3600/h),2],'b',label='T_amb',drawstyle='steps-post')
    ax0 = ax.twinx()
    ax0.plot(data[:nk*int(3600/h),1],'k',label='grid',drawstyle='steps-post')
    ax0.set_ylabel('grid signal', color='k')
    ax0.tick_params('y', colors='k')
    ax.grid()
    ax.legend()
    plt.show()   

   
def plot(fig,ax,plt,tf,savename=None, folder="figures"):
   if tf>86400:
      step=86400
      ax.set_xlabel('time [d]')
   else:
      step=3600
      ax.set_xlabel('Zeit in h')
      #ax.set_xlabel('time [h]')
   #ax.legend();
   ax.set_xlim([0,tf])
   n=int(tf/step)
   plt.xticks([scale*step for scale in range(n+1)],['%i'%scale for scale in range(n+1)])
   plt.show()   
   
   if savename is not None:
      if folder is not None and not os.path.exists(folder):
         os.mkdir(folder)
      fig.savefig('%s/%s.png' % (folder,savename),bbox_inches='tight')
      
       
