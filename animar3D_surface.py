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
from matplotlib import *

###########################################
##########################################
#GUARDAR LA PELÍCULA SI/NO
###########################################
##########################################

save = True
###########################################
##########################################


"""""""""""""""
Parámetros físicos

"""""""""""""""
M = 3245*2.325*1e7#masa total de las particulas q van a interactuar

N = 20000 #Número de partículas

m = M / N #masa de las particulas en Msolares

T_sol = 225 #periodo del Sol alrededor de la galaxia

"""""""""""""""
Parámetros de simulación

"""""""""""""""
salt = 10

lim = 100 

x_lim, y_lim, z_lim = lim, lim, lim

ntot = 10000 #número de pasos totales de tiempo
div_r = 25
n = int(ntot / div_r) #numero de pasos de tiempo guardados para r

dt = (T_sol / 2000) 
N_p = 100
n_p = N_p 
m_p = N_p 
l_p = 100

hx = x_lim / n_p #distancia entre puntos de red eje X
hy = y_lim / m_p #distancia entre puntos de red eje Y
hz = z_lim / l_p #distancia entre puntos de red eje Y

tray = np.loadtxt('trayectorias.dat', dtype = float)
tray_3D = tray.reshape(n, N, 3)

#tray = np.loadtxt('trayectorias.txt', dtype = float).reshape(n, N, 3)


#creamos la figura
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.tight_layout()
ax1.set_xlabel('Above')
ax2.set_xlabel('Side')


#posiciones iniciales de las partículas
#tray[paso de tiempo, particula, eje]
@jit(nopython=True, fastmath = True)
def densidad(r_list_0):
    rho = np.ones((n_p + 1, m_p + 1))
    for i in range(N):
        x_pos = int(r_list_0[i,0] / hx)
        y_pos = int(r_list_0[i,1] / hy)
        
        if x_pos <= 0 or x_pos >= N_p or y_pos <= 0 or y_pos >= N_p:
            
            pass
        
        else:
        
            rho[y_pos, x_pos] += m / (hx*hy)

    return rho

@jit(nopython=True, fastmath = True)
def densidad_z(r_list_0):
    rho = np.ones((m_p + 1, l_p + 1))
    for i in range(N):
        y_pos = int(r_list_0[i,1] / hy)
        z_pos = int(r_list_0[i,2] / hz)
        
        if y_pos <= 0 or y_pos >= m_p or z_pos <= 0 or z_pos >= l_p:
            
            pass
        
        else:
            rho[z_pos, y_pos] += m / (hy*hz)
    return rho


x = np.arange(0, x_lim + hx, hx)
y = np.arange(0, y_lim + hy, hy)
z = np.arange(0, z_lim + hz, hz)

X, Y = np.meshgrid(x,y)
Y_z, Z_z = np.meshgrid(y, z)

@jit(nopython=True, fastmath = True)
def iterator_rho(rho):
    for k in range(n):
        rho[k, :, :] = densidad(tray_3D[k, :, :])
    return rho

@jit(nopython=True, fastmath = True)
def iterator_rho_z(rho_z):
    for k in range(n):
        rho_z[k, :, :] = densidad_z(tray_3D[k, :, :])
    return rho_z

rho = iterator_rho(np.empty((n, n_p+1, m_p+1)))
rho_z = iterator_rho_z(np.empty((n, m_p+1, l_p+1)))


contour = ax1.imshow(rho[0,:,:], cmap = 'nipy_spectral', norm=colors.LogNorm(), interpolation = 'gaussian')
ax2.imshow(rho_z[0,:,:], cmap = 'nipy_spectral', norm=colors.LogNorm(), interpolation = 'gaussian')
txt = fig.suptitle('{:d} millones de años'.format(int(0*div_r*dt)))
colorbar = fig.colorbar(contour, fraction = 0.15)
colorbar.ax.set_xlabel('$M_0$')

def animation_frame(k):
        
    txt.set_text('{:d} millones de años'.format(int(k*div_r*dt)))
    
    ax1.imshow(rho[k,:,:], cmap = 'nipy_spectral', norm=colors.LogNorm(), interpolation = 'gaussian')
    ax2.imshow(rho_z[k,:,:], cmap = 'nipy_spectral', norm=colors.LogNorm(), interpolation = 'gaussian')
    
    return txt
            
    
movimiento = FuncAnimation(fig, func=animation_frame, frames=np.arange(1, n, salt),repeat=True,
                interval = 1, blit=False) #blit sirve para dibujar o no las trayectorias que siguen
    

if save:
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        movimiento.save('movimiento_surface.mp4', writer=writer)

