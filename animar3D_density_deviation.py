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
salt = 1

lim = sim_parameters[1]

ntot = int(sim_parameters[2])
div_r = int(sim_parameters[3])
n = int(ntot / div_r) #numero de pasos de tiempo guardados para r

dt = sim_parameters[5]

Np = int(lim)

h = lim/Np

tray = np.loadtxt('trayectorias.dat', dtype = float)
tray_3D = tray.reshape(n, N, 3)

tray_CM = np.loadtxt('trayectoria_CM.dat', dtype = float)
R_CM = tray_CM.reshape(n,3)

#creamos la figura
fig, ax1 = plt.subplots()
ax1.set_xlabel('XY')


#posiciones iniciales de las partículas
#tray[paso de tiempo, particula, eje]
@jit(nopython=True, fastmath = True)
def densidad(r_list_0):
    rho = np.ones((Np + 1, Np + 1))
    for i in range(Nd,N):
        x_pos = int(r_list_0[i,0] / h)
        y_pos = int(r_list_0[i,1] / h)
        
        if x_pos <= 0 or x_pos >= Np or y_pos <= 0 or y_pos >= Np:
            
            pass
        
        else:
        
            rho[y_pos+1, x_pos+1] += m / h**2

    return rho

x = np.arange(0, lim + h, h)
y = np.arange(0, lim + h, h)
z = np.arange(0, lim + h, h)

X, Y = np.meshgrid(x,y)

@jit(nopython=True, fastmath = True)
def iterator_rho(rho):
    for k in range(n):
        if k == 0:
            rho[k, :, :] = densidad(tray_3D[k, :, :])
        else:
            rho[k, :, :] = (densidad(tray_3D[k, :, :]) - rho[0, :, :])
    return rho

rho = iterator_rho(np.empty((n, Np+1, Np+1)))

im1 = ax1.imshow(rho[1,:,:], cmap = 'seismic', interpolation = 'gaussian', norm=colors.DivergingNorm(vmin=np.min(rho[1]), vcenter=0., vmax=np.max(rho[1])))
scatter_CM1 = ax1.scatter(R_CM[0,0], R_CM[0,1], c = 'black', s=1000000, 
                          marker = '+', linewidths=1, alpha = 0.5)


txt = fig.suptitle('{:d} millones de años'.format(int(0*div_r*dt)))
colorbar = fig.colorbar(im1, fraction = 0.15)
colorbar.ax.set_xlabel('Dev.D')

def animation_frame(k):
    
    txt.set_text('{:d} millones de años'.format(int(k*div_r*dt)))
    scatter_CM1.set_offsets([R_CM[k,0], R_CM[k,1]])
    im1.set_array(rho[k,:,:] + 0.1)
    
    return txt
            
    
movimiento = FuncAnimation(fig, func=animation_frame, frames=np.arange(1, n, salt),repeat=True,
                interval = 1, blit=False) #blit sirve para dibujar o no las trayectorias que siguen
    

if save:
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        movimiento.save('density_deviation.mp4', writer=writer)

