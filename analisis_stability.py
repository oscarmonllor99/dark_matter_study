#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:18:45 2021

@author: oscar
"""

import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import variation

sim_parameters = np.loadtxt('parameters.dat')

M = 3245*2.325*1e7 #masa total de las particulas q van a interactuar
N = int(sim_parameters[0]) #Número de partículas
m = M / N #masa de las particulas en Msolares
G = 4.518 * 1e-12

n = int(sim_parameters[2])
div_v = int(sim_parameters[4])
n_v = int(n / div_v) #numero de pasos de tiempo guardados para E y v
dt = sim_parameters[5]
t = np.linspace(0, n*dt, n_v)

lim = sim_parameters[1]
Np = 100
h = lim/Np

tray = np.loadtxt('trayectorias.dat', dtype = float)
tray_3D = tray.reshape(n_v, N, 3)
tray_CM = np.loadtxt('trayectoria_CM.dat', dtype = float)
R_CM = tray_CM.reshape(n_v,3)

vels = np.loadtxt('velocidades.dat', dtype = float)
vels_3D = vels.reshape(n_v, N, 3)

eners_pot =  np.loadtxt('energia_pot.dat', dtype = float)

@jit
def dispersion(array):
    root_square_v = np.zeros(len(array))
    mean = np.mean(array)
    for i in range(len(array)):
        root_square_v[i] = (array[i] - mean)**2
    return np.sqrt(np.mean(root_square_v))

@njit
def Q_calculator(eners_pot, vels, m, N, n_v, rho):
    
    Q_list = np.empty(n_v)
    
    for k in range(n_v):
        
        v_list_r = []
        v_list_c = []
        
        for i in range(N):
                
                R_centro = np.zeros(3)
                R_centro[0] = tray_3D[k, i, 0] - R_CM[k, 0]
                R_centro[1] = tray_3D[k, i, 1] - R_CM[k, 1]
                R_centro[2] = 0
                R_norm = np.linalg.norm(R_centro)
                if 4 <= R_norm <= 5:
                    ur = R_centro / R_norm
                    vr = abs(np.dot(vels_3D[k, i], ur))
                    vz = vels_3D[k, i, 2]
                    v = np.linalg.norm(vels_3D[k, i])
                    v_list_r.append(vr)
                    v_list_c.append(np.sqrt(v**2 - vr**2 - vz**2))
        
        rho_sum = 0
        N_rho = 0
        for x in range(Np):
            for y in range(Np):
                if 4 <= np.sqrt((x - Np/2)**2 + (y - Np/2)**2) <= 5:
                    if rho[k, Np, Np] > 0:
                        rho_sum += rho[k, x, y]
                        N_rho += 1
                    
        v_list_r = np.array(v_list_r)
        v_list_c = np.array(v_list_c)
        
        sigma_r = dispersion(v_list_r)
        sigma_phi = dispersion(v_list_c)
        rho_mean = rho_sum/N_rho
        vc_mean = np.mean(v_list_c)
        kappa = (2*vc_mean*sigma_phi/sigma_r)*(1/4.5)
        sigma_min = 3.36*G*rho_mean/kappa
        
        if sigma_r/sigma_min < 100:
            Q_list[k] = sigma_r/sigma_min
        else:
            Q_list[k] = Q_list[k-1]
        
    return Q_list

@njit
def t_calculator(eners_pot, vels, m, N, n_v):
    
    t_list = np.empty(n_v)
    
    for k in range(n_v):
        
        ener_pot = 0.
        ener_cin = 0.
        
        for i in range(N):
            
            vx = vels[k, i, 0]
            vy = vels[k, i, 1]
            
            x = tray_3D[k, i, 0]
            y = tray_3D[k, i, 1]
            
            if (x - R_CM[k, 0]) != 0.:
                
                theta_g = np.arctan((y - R_CM[k, 1]) / (x - R_CM[k, 0])) 
                
                if (y  - R_CM[k, 1]) < 0 or (x - R_CM[k, 0]) < 0: 
                    theta_g = np.pi + theta_g
            else:
                theta_g = 0.
                
            v_ang = -vx*np.sin(theta_g) + vy*np.cos(theta_g)
            
            ener_cin += 0.5*m*v_ang**2 
            ener_pot += 0.5*eners_pot[k, i]

        t_list[k] = abs(ener_cin/ener_pot)
    
    return t_list

#posiciones iniciales de las partículas
#tray[paso de tiempo, particula, eje]
@jit(nopython=True, fastmath = True)
def densidad(r_list_0):
    rho = np.ones((Np + 1, Np + 1))
    for i in range(N):
        x_pos = int(r_list_0[i,0] / h)
        y_pos = int(r_list_0[i,1] / h)
        if x_pos <= 0 or x_pos >= Np or y_pos <= 0 or y_pos >= Np:
            
            pass
        
        else:
        
            rho[y_pos+1, x_pos+1] += m / h**2

    return rho

@jit(nopython=True, fastmath = True)
def iterator_rho(rho):
    for k in range(n_v):
        rho[k, :, :] = densidad(tray_3D[k, :, :])
    return rho

rho = iterator_rho(np.empty((n_v, Np+1, Np+1)))
  
t_list = t_calculator(eners_pot, vels_3D, m, N, n_v)
Q_list = Q_calculator(eners_pot, vels_3D, m, N, n_v, rho)


guess = np.array([1, 1])
def c(x, a, b):
    return a + b*x

param_t, cov_t = curve_fit(c, t, t_list, guess, sigma = None, absolute_sigma = True)    
c_fit_t = c(t, param_t[0], param_t[1]) 

param_Q, cov_Q = curve_fit(c, t, Q_list, guess, sigma = None, absolute_sigma = True)    
c_fit_Q = c(t, param_Q[0], param_Q[1],) 

fig, axs = plt.subplots(1, 2, figsize = (9, 5))
axs[0].scatter(t, t_list, c = 'red', s = 5, marker = '*', label = "$t$")
#axs[0].plot(t, c_fit_t, c = 'black')
axs[0].set_xlabel('Tiempo (My)')
axs[0].set_ylabel('t (Ostriker-Peebels)')
axs[1].scatter(t, Q_list, c = 'blue', s = 5, marker = '*', label = "$Q$")
#axs[1].plot(t, c_fit_Q, c = 'black')
axs[1].set_xlabel('Tiempo (My)')
axs[1].set_ylabel('Q (Toomre)')
plt.savefig('curva_de_estabilidad.png')
plt.show()


