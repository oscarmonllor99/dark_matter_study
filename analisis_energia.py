#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:18:45 2021

@author: oscar
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

sim_parameters = np.loadtxt('parameters.dat')

M = 3245*2.325*1e7 #masa total de las particulas q van a interactuar
N = int(sim_parameters[0]) #Número de partículas
m = M / N #masa de las particulas en Msolares
G = 4.518 * 1e-12

lim = sim_parameters[2]

n = int(sim_parameters[2])
div_v = int(sim_parameters[4])
n_v = int(n / div_v) #numero de pasos de tiempo guardados para E y v
dt = sim_parameters[5]
t = np.linspace(0, n*dt, n_v)

tray = np.loadtxt('trayectorias.dat', dtype = float)
tray_3D = tray.reshape(n_v, N, 3)

vels = np.loadtxt('velocidades.dat', dtype = float)
vels_3D = vels.reshape(n_v, N, 3)

eners_pot =  np.loadtxt('energia_pot.dat', dtype = float)

@njit
def energy(eners_pot, vels, tray, m, N, n_v):
    
    ener_pot_list = np.empty(n_v)
    ener_cin_list = np.empty(n_v)
    ener_mec_list = np.empty(n_v)
    virial_list = np.empty(n_v)
    
    for k in range(n_v):
        
        ener_pot = 0.
        ener_cin = 0.
        
        for i in range(N):

            ener_cin += 0.5*m*(vels[k, i, 0]**2 + vels[k, i, 1]**2 + vels[k, i, 2]**2)
            ener_pot += 0.5*eners_pot[k, i]
        
        ener_pot_list[k] = ener_pot
        ener_cin_list[k] = ener_cin
        ener_mec_list[k] = ener_pot + ener_cin
        virial_list[k] = ener_cin + 0.5*ener_pot
        
    return ener_pot_list, ener_cin_list, ener_mec_list, virial_list

ener_pot_list, ener_cin_list, ener_mec_list, virial_list = energy(eners_pot, vels_3D, tray_3D, m, N, n_v)

fig, ax = plt.subplots()
ax.scatter(t, ener_pot_list, c = 'red', s = 5, marker = '*', label = "$U$")
ax.scatter(t, ener_cin_list, c = 'blue', s = 5, marker = '*', label = "$T$")
ax.scatter(t, ener_mec_list, c = 'black', s = 5, marker = '*', label = "$T+U$")
ax.scatter(t, virial_list, c = 'green', s = 5, marker = '*', label = "$T+0.5·U$")
plt.title('Curva de energia')
plt.xlabel('t (My)')
plt.ylabel('Energia (u)')
plt.legend()
plt.savefig('curva_de_energia.png')
plt.show()


    


