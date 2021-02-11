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

N = 30000 #Número de partículas

m = M / N #masa de las particulas en Msolares

T_sol = 225 #periodo del Sol alrededor de la galaxia

"""""""""""""""
Parámetros de simulación

"""""""""""""""
salt = 4 

lim = 100 

x_lim, y_lim, z_lim = lim, lim, lim

n = 400
dt = (T_sol / 2000) 
N_p = 200
n_p = N_p 
m_p = N_p 

hx = x_lim / n_p #distancia entre puntos de red eje X
hy = y_lim / m_p #distancia entre puntos de red eje Y

tray = np.loadtxt('trayectorias.dat', dtype = float)
tray_3D = tray.reshape(n, N, 3)

#tray = np.loadtxt('trayectorias.txt', dtype = float).reshape(n, N, 3)


#creamos la figura
fig = plt.figure()
ax = plt.axes(xlim = (0, x_lim), ylim = (0, y_lim))

#posiciones iniciales de las partículas
#tray[paso de tiempo, particula, eje]
@jit(nopython=True, fastmath = True)
def densidad(r_list_0):
    rho = np.zeros((n_p + 1, m_p + 1))
    for i in range(N):
        x_pos = int(r_list_0[i,0] / hx)
        y_pos = int(r_list_0[i,1] / hy)
        
        if x_pos <= 0 or x_pos >= N_p or y_pos <= 0 or y_pos >= N_p:
            
            pass
        
        else:
        
            rho[x_pos, y_pos] += m / (hx*hy)

    return rho

x = np.arange(0, x_lim + hx, hx)
y = np.arange(0, y_lim + hy, hy)

X, Y = np.meshgrid(x,y)

@jit(nopython=True, fastmath = True)
def iterator_rho(rho):
    for k in range(n):
        rho[k, :, :] = densidad(tray_3D[k, :, :])
    return rho

rho = iterator_rho(np.empty((n, n_p+1, m_p+1)))
    

    
contour = ax.contourf(X,Y, rho[0,:,:], cmap = 'nipy_spectral', levels = 300)
txt = fig.suptitle('')
colorbar = fig.colorbar(contour)
colorbar.ax.set_xlabel('$M_0$')

def animation_frame(k):
        
    txt.set_text('{:f} millones de años'.format(k*25*dt))
    
    ax.contourf(X,Y, rho[k,:,:], cmap = 'nipy_spectral', levels = 60)
    
    return txt
            
    
movimiento = FuncAnimation(fig, func=animation_frame, frames=np.arange(1, n, salt),repeat=True,
                interval = 1, blit=False) #blit sirve para dibujar o no las trayectorias que siguen
    

if save:
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        movimiento.save('movimiento_surface.mp4', writer=writer)

