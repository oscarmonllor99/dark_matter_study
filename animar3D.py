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


"""""""""""""""
Parámetros físicos

"""""""""""""""

N = 100 #Número de partículas

T_sol = 225 #periodo del Sol alrededor de la galaxia

"""""""""""""""
Parámetros de simulación

"""""""""""""""
salt = 1 #salto en los pasos, en vez de ir de 1 paso en 1 paso, se va de salt en salt para agilizar y ahorrar espacio

lim = 100 #en kpc

x_lim, y_lim, z_lim = lim, lim, lim

n = 400
dt = (T_sol / 2000) #intervalo de tiempo entre cada paso

r = 0.01 #en kpc

tray = np.loadtxt('trayectorias.dat', dtype = float)
tray_3D = tray.reshape(n, N, 3)

#tray = np.loadtxt('trayectorias.txt', dtype = float).reshape(n, N, 3)


#creamos la figura
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.azim = 180
ax.elev = 50

ax.set_xlim([-x_lim/2, x_lim/2])
ax.set_xlabel('X (kpc)')
    
ax.set_ylim([-y_lim/2, y_lim/2])
ax.set_ylabel('Y (kpc)')
    
ax.set_zlim([-z_lim/2 , z_lim/2])
ax.set_zlabel('Z (kpc)')

ax.set_facecolor('white')

ax.set_axis_off()

#dibujamos las esferas que representan a cada partícula
    
X = np.empty(N)  
Y = np.copy(X)
Z = np.copy(X)

#posiciones iniciales de las partículas
#tray[paso de tiempo, particula, eje]
X[:], Y[:], Z[:] = tray_3D[0, :, 0], tray_3D[0, :, 1], tray_3D[0, :, 2],

particulas, = ax.plot(X, Y, Z, marker = 'd', c = 'black', markersize = 10*r, linestyle='None')
plt.grid()
txt = fig.suptitle('')
       
def animation_frame(k):
        
        txt.set_text('{:f} millones de años'.format(k*25*dt))
        
        #se crean x_data e y_data donde se almacenan las posiciones para cada paso k de tiempo
        
        X = tray_3D[k, :, 0]- x_lim/2
        Y = tray_3D[k, :, 1] - y_lim/2
        Z = tray_3D[k, :, 2] - z_lim/2

        particulas.set_data(X, Y)
        particulas.set_3d_properties(Z, 'z')

        return particulas, txt
            
    
movimiento = FuncAnimation(fig, func=animation_frame, frames=np.arange(1, n, salt),repeat=True,
                interval = 1, blit=False) #blit sirve para dibujar o no las trayectorias que siguen
    

if save:
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        movimiento.save('movimiento.mp4', writer=writer)
