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
N_b = int(0.14 * N) #el 14% de la materia ordinaria es del bulbo
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

Np = 100

h = lim/Np

tray = np.loadtxt('trayectorias.dat', dtype = float)
tray_3D = tray.reshape(n, N, 3)

tray_CM = np.loadtxt('trayectoria_CM.dat', dtype = float)
R_CM = tray_CM.reshape(n,3)

#creamos la figura
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.tight_layout()
ax1.set_xlabel('Above')
ax2.set_xlabel('Side')


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
        
            rho[y_pos, x_pos] += m / h**2

    return rho

@jit(nopython=True, fastmath = True)
def densidad_z(r_list_0):
    rho = np.ones((Np + 1, Np+ 1))
    for i in range(N):
        y_pos = int(r_list_0[i,1] / h)
        z_pos = int(r_list_0[i,2] / h)
        
        if y_pos <= 0 or y_pos >= Np or z_pos <= 0 or z_pos >= Np:
            
            pass
        
        else:
            rho[z_pos, y_pos] += m / h**2
    return rho


x = np.arange(0, lim + h, h)
y = np.arange(0, lim + h, h)
z = np.arange(0, lim + h, h)

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

rho = iterator_rho(np.empty((n, Np+1, Np+1)))
rho_z = iterator_rho_z(np.empty((n, Np+1, Np+1)))


im1 = ax1.imshow(rho[0,:,:], cmap = 'nipy_spectral', norm=colors.LogNorm(), interpolation = 'gaussian')
scatter_CM1 = ax1.scatter(R_CM[0,0], R_CM[0,1], c = 'white', s=1000000, 
                          marker = '+', linewidths=1, alpha = 0.5)
im2 = ax2.imshow(rho_z[0,:,:], cmap = 'nipy_spectral', norm=colors.LogNorm(), interpolation = 'gaussian')
scatter_CM2 = ax2.scatter(R_CM[0,1], R_CM[0,2], c = 'white', s=1000000, 
                          marker = '+', linewidths=1, alpha = 0.5)
txt = fig.suptitle('{:d} millones de años'.format(int(0*div_r*dt)))
colorbar = fig.colorbar(im1, fraction = 0.15)
colorbar.ax.set_xlabel('$M_0$')

def animation_frame(k):
        
    txt.set_text('{:d} millones de años'.format(int(k*div_r*dt)))
    scatter_CM1.set_offsets([R_CM[k,0], R_CM[k,1]])
    im1.set_array(rho[k,:,:])
    scatter_CM2.set_offsets([R_CM[k,1], R_CM[k,2]])
    im2.set_array(rho_z[k,:,:])
    
    return txt
            
    
movimiento = FuncAnimation(fig, func=animation_frame, frames=np.arange(1, n, salt),repeat=True,
                interval = 1, blit=False) #blit sirve para dibujar o no las trayectorias que siguen
    

if save:
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        movimiento.save('movimiento_surface.mp4', writer=writer)

