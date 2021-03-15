# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 17:58:16 2020

@author: Oscar
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
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

"""""""""""""""
Parámetros físicos

"""""""""""""""

N = int(sim_parameters[0]) #Número de partículas

T_sol = 225 #periodo del Sol alrededor de la galaxia

"""""""""""""""
Parámetros de simulación

"""""""""""""""
salt = 1 #salto en los pasos, en vez de ir de 1 paso en 1 paso, se va de salt en salt para agilizar y ahorrar espacio

lim = sim_parameters[1]

x_lim, y_lim, z_lim = lim, lim, lim

ntot = int(sim_parameters[2])
div_r = int(sim_parameters[3])
n = int(ntot / div_r) #numero de pasos de tiempo guardados para r

dt = sim_parameters[5]

r = 0.01 #en kpc

tray = np.loadtxt('trayectorias.dat', dtype = float)
tray_3D = tray.reshape(n, N, 3)

#tray = np.loadtxt('trayectorias.txt', dtype = float).reshape(n, N, 3)


#creamos la figura
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 5))
#fig.tight_layout()
ax1.set_xlabel('XY')
ax2.set_xlabel('YZ')

ax1.set_xlim([-x_lim/2, x_lim/2])
    
ax1.set_ylim([-y_lim/2, y_lim/2])
    
ax2.set_xlim([-y_lim/2, y_lim/2])
    
ax2.set_ylim([-z_lim/2, z_lim/2])

ax1.set_facecolor('white')
ax2.set_facecolor('white')

ax1.set_axis_off()
ax2.set_axis_off()

#dibujamos las esferas que representan a cada partícula
    
X = np.empty(N)  
Y = np.copy(X)
Z = np.copy(X)

#posiciones iniciales de las partículas
#tray[paso de tiempo, particula, eje]
X[:], Y[:], Z[:] = tray_3D[0, :, 0], tray_3D[0, :, 1], tray_3D[0, :, 2],

plot_XY, = ax1.plot(X, Y, marker = 'd', c = 'black', markersize = 10*r, linestyle='None')
plot_YZ, = ax2.plot(Y, Z, marker = 'd', c = 'black', markersize = 10*r, linestyle='None')
plt.grid()
txt = fig.suptitle('')
       
def animation_frame(k):
        
        txt.set_text('{:f} millones de años'.format(k*div_r*dt))
        
        #se crean x_data e y_data donde se almacenan las posiciones para cada paso k de tiempo
        
        X = tray_3D[k, :, 0]- x_lim/2
        Y = tray_3D[k, :, 1] - y_lim/2
        Z = tray_3D[k, :, 2] - z_lim/2

        plot_XY.set_data(X, Y)
        plot_YZ.set_data(Y, Z)

        return txt
            
    
movimiento = FuncAnimation(fig, func=animation_frame, frames=np.arange(1, n, salt),repeat=True,
                interval = 1, blit=False) #blit sirve para dibujar o no las trayectorias que siguen
    

if save:
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)
        movimiento.save('movimiento.mp4', writer=writer)

