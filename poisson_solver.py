# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 17:23:39 2020

@author: Oscar
"""
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import random 
import time
##############################################
######### PARÁMETROS FÍSICO  ################
##############################################

N = 20000  #Número de partículas

M = 1e11 #masa total de las particulas q van a interactuar: fraccion no significativa de la total galactica.

m = M/N #masa de las particulas en Msolares, consideramos la masa solar

G = 4.518 * 1e-12 #constante de gravitación universal en Kpc, Msolares y Millones de años

##############################################
### PARÁMETROS DE SIMULACIÓN
##############################################

lim = 100 #en kpc

x_lim, y_lim, z_lim = lim, lim, lim #Tamaño de la caja en 

N_p = 100 #número de pasos común a los dos ejes, es decir, número de celdas para calcular
#el gradiente de la gravedad

n_p = N_p #número de celdas en el eje X
m_p = N_p #número de celdas en el eje Y
l_p = 50 #número de celdas en el eje Z

hx = x_lim / n_p #distancia entre puntos de red eje X
hy = y_lim / m_p #distancia entre puntos de red eje Y
hz = z_lim / l_p #distancia entre puntos de red eje Y

x = np.arange(0, x_lim, hx)
y = np.arange(0, y_lim, hy)
z = np.arange(0, z_lim, hz)

X, Y = np.meshgrid(x,y)


r0 = np.zeros((n_p, m_p, l_p))

def bordes(r):
    for i in range(n_p):
        for j in range(m_p):
            for k in range(l_p):
                r[i, j, k] = np.sqrt(((i-n_p/2)*hx)**2 + ((j-m_p/2)*hy)**2 + ((k-l_p/2)*hz)**2)
    return r

r = bordes(r0)

plt.figure()
plt.imshow(r[:,:, int(l_p/2)])
plt.show()

##############################################
### FUNCIONES A UTILIZAR ###
##############################################

#vector con componentes nulas
def zero_vect(N):
    v = []
    for i in range(N):
        v.append(0)
    return v

##############################################
#DISTRIBUCIÓN DE MASA INICIAL
##############################################


m_list_0 =  zero_vect(N)
    
r_list_0 = zero_vect(N)
    
for i in range(N):
        
        #posició de la masa central galáctica si no hay potencial galactico
        if i == 0:
            
            m_cent = 1*m
            
            m_list_0[i] = m_cent #[m_cent, m_cent, m_cent] #pongo masa en x,y,z para poder vectorizarlo
            
            r_list_0[i] = [(n_p/2) * hx, (m_p/2)*hy, (l_p/2)*hz]
            
        else:
            #Distribucion uniforme  
            R = 30*random.expovariate(5) #valor aleatorio de la distancia radial
            theta = random.uniform(0, 2*np.pi) #valor aleatorio del ángulo polar
            
            #Distribucion gaussiana
            #R = 0.1 + abs(random.gauss(1, 6)) #valor aleatorio de la distancia radial según la distribucion gaussiana
            z = random.uniform(-0.5, 0.5)

            
            r_i = [(n_p/2) * hx + R*np.cos(theta), (m_p/2) * hy + R*np.sin(theta), (l_p/2) * hz + z]
            
            r_list_0[i] = r_i
            
            m_list_0[i] = m #[m, m, m]
            

def densidad(r_list_0, m_list_0, rho):
    for i in range(len(r_list_0)):
        
        x_pos = int(r_list_0[i][0] // hx)
        y_pos = int(r_list_0[i][1] // hy)
        z_pos = int(r_list_0[i][2] // hz)
        
        if x_pos < 0 or x_pos >= N_p or y_pos < 0 or y_pos >= N_p or z_pos < 0 or z_pos >= l_p:
            
            pass
        
        else:
        
            rho[x_pos, y_pos, z_pos] += m_list_0[i] / (hx*hy*hz)

    return rho

rho0 = np.zeros((n_p, m_p, l_p), dtype = float)
rho = densidad(r_list_0, m_list_0, rho0)



plt.figure()
plt.imshow(rho[:,:, int(l_p/2)])
plt.show()


print('Se ha calculado la densidad, ahora el potencial')


phi0 =np.zeros((n_p, m_p, l_p), dtype = float) 

#PLANOS
phi0[:, :, 0] = -G*M/(r[:, :, 0])
phi0[:, :, l_p-1] = -G*M/(r[:, :, l_p-1])
phi0[:, 0, :] = -G*M/(r[:, 0, :])
phi0[:, N_p-1, :] = -G*M/(r[:, N_p-1, :])
phi0[0, :, :] = -G*M/(r[0, :, :])
phi0[N_p-1, :, :] = -G*M/(r[N_p-1, :, :])

plt.figure()
plt.imshow(phi0[:,:, int(l_p/2)])
plt.show()

@jit(nopython=True, fastmath = True)
def poisson(rho, phi):
    w = 0.95
    pasos = 100
    for u in range(pasos):
        for i in range(1, n_p-1):
            for j in range(1, m_p-1):
                for k in range(1, l_p-1):
                    phi[i, j, k] = (1+w)*(1/(4/hx**2 + 2/hz**2))*((1/hx**2)*(phi[i+1, j, k] + phi[i-1, j, k]
                                                                         + phi[i, j+1, k] + phi[i, j-1, k])
                                                                         + (1/hz**2)*(phi[i, j, k+1] + phi[i, j, k-1])
                                                                         - 4*np.pi*G*rho[i, j, k]) - w*phi[i, j, k]



    return phi 
  
t0 = time.time()
phi = poisson(rho, phi0)
tf = time.time()

print(tf-t0)

fig, ax = plt.subplots()
cmap = plt.cm.get_cmap("inferno")
cs = ax.contourf(X, Y, phi[:,:, int(l_p/2)], cmap = cmap, levels = 75) #pasamos el potencial a km**2 / s**2
fig.colorbar(cs)
plt.show()

pot = (phi[int(N_p/2):, int(N_p/2), int(l_p/2)])

x = np.linspace(hx, (N_p/2)*hx, int(N_p/2))

fig1, ax1 = plt.subplots()
ax1.plot(x, pot, label='Aproximación')
ax1.plot(x, -(G*M/(x)), label = 'analítico')
plt.legend()
plt.show()

    