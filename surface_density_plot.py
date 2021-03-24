# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 17:58:16 2020

@author: Oscar
"""

from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib import colors
import matplotlib

###########################################
##########################################
#GUARDAR LA PELÍCULA SI/NO
###########################################
##########################################
save = True
###########################################
##########################################

sim_parameters = np.loadtxt('parameters.dat')

##############################################
######### PARÁMETROS FÍSICOS  ################
##############################################
N = int(sim_parameters[0]) #Número de partículas
Nb = int(0.14 * N) #el 14% de la materia ordinaria es del bulbo
Nd = N-Nb
M = 3245*2.325*1e7 #masa total de las particulas q van a interactuar
m = M / N #masa de las particulas en Msolares
G = 4.518 * 1e-12 #constante de gravitación universal en Kpc, Msolares y Millones de años
##############################################
##############################################

##############################################
######### PARÁMETROS DE SIMULACION ################
##############################################
lim = sim_parameters[1]

ntot = int(sim_parameters[2])
div_r = int(sim_parameters[3])
n = int(ntot / div_r) #numero de pasos de tiempo guardados para r
n_graf = 6

dt = sim_parameters[5]

Np = 2*int(lim)

h = lim/Np

tray = np.loadtxt('trayectorias.dat', dtype = float)
tray_3D = tray.reshape(n, N, 3)

tray_CM = np.loadtxt('trayectoria_CM.dat', dtype = float)
R_CM = tray_CM.reshape(n,3)



#posiciones iniciales de las partículas
#tray[paso de tiempo, particula, eje]
@jit(nopython=True, fastmath = True)
def densidad(r_list_0):
    rho = np.ones((Np + 1, Np + 1))
    for i in range(Nd, N):
        x_pos = int(r_list_0[i,0] / h)
        y_pos = int(r_list_0[i,1] / h)
        
        if x_pos <= 0 or x_pos >= Np or y_pos <= 0 or y_pos >= Np:
            
            pass
        
        else:
        
            rho[y_pos+1, x_pos+1] += m / h**2

    return rho


x = np.linspace(-lim/2, lim/2, Np+1)
y = np.linspace(-lim/2, lim/2, Np+1)

X, Y = np.meshgrid(x,y)

@jit(nopython=True, fastmath = True)
def iterator_rho(rho):
    for k in range(n_graf):
        k_graf = int(n/n_graf)*k
        rho[k, :, :] = densidad(tray_3D[k_graf, :, :])
    return rho

rho = iterator_rho(np.empty((n_graf, Np+1, Np+1)))


for k in range(n_graf):
    k_graf = int(n/n_graf)*k
    #creamos la figura
    fig, ax1 = plt.subplots(1, 1, figsize = (5,4), dpi=400)
    ax1.set_xlabel('Plano galáctico')

    im1 = ax1.contourf(X[int(Np/4):int(3*Np/4), int(Np/4):int(3*Np/4)], 
                       Y[int(Np/4):int(3*Np/4), int(Np/4):int(3*Np/4)], 
                       rho[k,int(Np/4):int(3*Np/4), int(Np/4):int(3*Np/4)], cmap='inferno', 
                      levels = 999, norm=colors.PowerNorm(gamma=0.3)) 
    scatter_CM1 = ax1.scatter(R_CM[k_graf,0] -lim/2, R_CM[k_graf,1]-lim/2, c = 'white', s=1000000, 
                              marker = '+', linewidths=1, alpha = 0.2)

    txt = fig.suptitle('{} millones de años'.format(int((k_graf*div_r*dt))))
    colorbar = fig.colorbar(im1, fraction = 0.1)
    colorbar.ax.set_xlabel('$M_0 · kpc^{-3}$', loc = 'left')
    plt.tight_layout()
    plt.savefig('Gráfico de densidad para {} millones de años'.format(int((k_graf*div_r*dt))))
    



