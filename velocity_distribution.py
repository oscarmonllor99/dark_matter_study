#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:16:48 2021

@author: oscar
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.neighbors import KDTree
from matplotlib.animation import FuncAnimation

sim_parameters = np.loadtxt('parameters.dat')

M = 3245*2.325*1e7 #masa total de las particulas q van a interactuar
N = int(sim_parameters[0]) #Número de partículas
m = M / N #masa de las particulas en Msolares
G = 4.518 * 1e-12
dt = sim_parameters[5]

lim = sim_parameters[1]

ntot = int(sim_parameters[2])
div_v = int(sim_parameters[4])
n = int(ntot / div_v) #numero de pasos de tiempo guardados para r

vels = np.loadtxt('velocidades.dat', dtype = float)
vels_3D = vels.reshape(n, N, 3)

vels_abs = np.zeros((n, N))
vels_abs[:, :] = np.sqrt(vels_3D[:,:,0]**2 + vels_3D[:,:,1]**2 + vels_3D[:,:,2]**2)

Nv = 200
lim_v = np.max(vels_abs[n-1,:])
hv = lim_v/Nv
Vs = np.linspace(0, lim_v, Nv)
f_v = np.zeros((Nv))

@njit
def v_distribution(f_v, vels_abs):
    for i in range(N):
        v_index = int(vels_abs[n-1, i]/hv)
        f_v[v_index] += 1
    return f_v

guess1 = [0.1, 1, 0.1]
def gauss(v, sigma, a, mu):
    return a * np.exp(-(v-mu)**2/sigma**2)

guess2 = [0.3, 1]
def maxwell(v, sigma, a):
    return a * v**2 * np.exp(-v**2/sigma**2)

#def cauchi

points_v = v_distribution(f_v, vels_abs)
param1, cov1 = curve_fit(gauss, Vs, points_v/N, guess1, sigma = None, absolute_sigma = True)
param2, cov2 = curve_fit(maxwell, Vs, points_v/N, guess2, sigma = None, absolute_sigma = True)
fit1 = gauss(Vs, param1[0], param1[1], param1[2])
fit2 = maxwell(Vs, param2[0], param2[1])
R21 = r2_score(fit1, points_v/N)
R22 = r2_score(fit2, points_v/N)
print(param2, param1)
print(R22, R21)

fit_v_info = np.empty((2, 4))
fit_v_info[0,0] = param1[0]
fit_v_info[0,1] = param1[1]
fit_v_info[0,2] = param1[2]
fit_v_info[0,3] = R21
fit_v_info[1,0] = param2[0]
fit_v_info[1,1] = param2[1]
fit_v_info[1,2] = R22
np.savetxt('fit_v_info.txt', fit_v_info, fmt = '%.2e')

plt.figure()
plt.plot(Vs/(1.022*1e-3), 100*points_v/N , c='black', marker='*', label = 'Datos')
plt.plot(Vs/(1.022*1e-3), 100*fit1, c='blue', label = 'Gaussiana')
plt.plot(Vs/(1.022*1e-3), 100*fit2, c='red', label='Maxwelliana')
plt.vlines(Vs[np.argmax(points_v)]/(1.022*1e-3), 0, np.max(100*points_v/N), 
           color="black", linestyle="dashed", alpha = 0.7, label='Máx.')
plt.annotate('$v_0$ = {} (km/s)'.format(int(Vs[np.argmax(points_v)]/(1.022*1e-3))), 
             xy=(Vs[np.argmax(points_v)]/(1.022*1e-3), np.max(100*points_v/N)), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.1, headwidth = 0.5, width = 0.1),
            )
plt.xlabel('$v$ (km/s)')
plt.ylabel('$N$ %')
plt.legend()
plt.savefig('velocity_distribution.png')
plt.show()
