#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:03:50 2021

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
div_r = int(sim_parameters[3])
n = int(ntot / div_r) #numero de pasos de tiempo guardados para r

tray = np.loadtxt('trayectorias.dat', dtype = float)
tray_3D = tray.reshape(n, N, 3)

@njit
def partir_tray(tray):
    if N>10000:
        N_reduce = 4000
        div = int(N/N_reduce)
        tray_reduced = np.zeros((n, N_reduce, 3))
        for i in range(N_reduce):
            tray_reduced[:, i, :] = tray[:, i*div, :]
        return tray_reduced
    else:
        return tray

tray_reduced = partir_tray(tray_3D)

r = np.linspace(0, 20, 200)


r_list_0 = tray_reduced[0]
tree = KDTree(r_list_0, leaf_size=int(0.01*N))
correlacion = tree.two_point_correlation(r_list_0, r, dualtree = True)
correl_norm = correlacion/np.max(correlacion)

fig, ax = plt.subplots()
ax.set_xlabel('R (kpc)')
ax.set_ylabel('Función de correlación normalizada')
ax.plot(r, correl_norm, c='black')
txt = fig.suptitle('{} millones de años'.format(0))

guess = [10]
def ajuste(x, a):
    return np.tanh(x/a)

"""
r_fit = np.linspace(0, lim/2, 100)
param, cov = curve_fit(ajuste, r, correl_norm, guess, sigma = None, absolute_sigma = True)    
fit = ajuste(r_fit, param[0]) #curve fit 
ax.plot(r_fit, fit, label = '$tanh(r/b)$', c = 'black', alpha = 0.5)
plt.legend()
"""
plt.savefig('two_point_correlation_init.png')

def animation_frame(k):
    ax.clear()
    
    txt.set_text('{} millones de años'.format(k*div_r*dt))
    r_list = tray_reduced[k]
    tree = KDTree(r_list, leaf_size=40)
    correlacion = tree.two_point_correlation(r_list, r, dualtree = True)
    correl_norm = correlacion/np.max(correlacion)
    ax.plot(r, correl_norm, c='black')
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Función de correlación normalizada')
    
    """
    r_fit = np.linspace(0, lim/2, 100)
    param, cov = curve_fit(ajuste, r, correl_norm, guess, sigma = None, absolute_sigma = True)    
    fit = ajuste(r_fit, param[0]) #curve fit 
    ax.plot(r_fit, fit, label = '$tanh(r/b)$', c = 'black', alpha = 0.5)
    plt.legend()
    """
    
    if k == n-5:
        plt.savefig('two_point_correlation_final.png')
    return txt
            
    
movimiento = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, n, 5),repeat=True,
                interval = 10, blit=False) #blit sirve para dibujar o no las trayectorias que siguen
