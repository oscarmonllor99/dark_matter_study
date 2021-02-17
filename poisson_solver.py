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
from matplotlib import colors
##############################################
######### PARÁMETROS FÍSICOS  ################
##############################################
NUM_PARTICLES = 200000 #Número de partículas
NUM_PARTICLES_BULGE = int(0.14 * NUM_PARTICLES) #el 14% de la materia ordinaria es del bulbo
M_TOTAL = 3245*2.325*1e7 #masa total de las particulas q van a interactuar
M_PARTICLE = M_TOTAL / NUM_PARTICLES #masa de las particulas en Msolares
G = 4.518 * 1e-12 #constante de gravitación universal en Kpc, Msolares y Millones de años
T_SOL = 225 #periodo del Sol alrededor de la galaxia en Millones de años

##############################################
### PARÁMETROS DE SIMULACIÓN
##############################################

lim = 100 #tamaño de la caja en kpc

NP = 100 #número de celdas

H = lim / NP #lado de una celda

##############################################
##############################################

x = np.arange(0, lim, H)
y = np.arange(0, lim, H)
z = np.arange(0, lim, H)

X, Y = np.meshgrid(x,y)

R0 = np.zeros((NP, NP, NP)) #para las condiciones de contorno del potencial
@jit(nopython=True, fastmath = True)
def bordes(r):
        for i in range(NP):
            for j in range(NP):
                for k in range(NP):
                    r[i, j, k] = np.sqrt(((i-NP/2)*H)**2 + ((j-NP/2)*H)**2 + ((k-NP/2)*H)**2)
        return r
R0 = bordes(R0)

##############################################
#DISTRIBUCIÓN DE MASA INICIAL
##############################################
r_list_0 = np.zeros((NUM_PARTICLES, 3), dtype = float)
    
for i in range(NUM_PARTICLES):
             
            if i < NUM_PARTICLES_BULGE:
                
                R = random.uniform(0.1, 3)
                while R>49:
                    R = random.uniform(0.1, 3)
                
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2*np.pi)

                
                r_list_0[i, 0] = lim/2 + R*np.cos(phi)*np.sin(theta)
                r_list_0[i, 1] = lim/2 + R*np.sin(phi)*np.sin(theta)
                r_list_0[i, 2] = lim/2 + R*np.cos(theta)
                
                
            else:
                
                R = 10*random.expovariate(1)
                while R>49:
                    R = 10*random.expovariate(1)
                    
                phi = random.uniform(0, 2*np.pi) 
                z = random.uniform(-0.5, 0.5)

                r_list_0[i, 0] = lim/2 + R*np.cos(phi)
                r_list_0[i, 1] = lim/2 + R*np.sin(phi)
                r_list_0[i, 2] = lim/2 + z


@jit(nopython=True, fastmath = True, parallel = False)
def densidad(r_list, Np, h):
    rho = np.zeros((Np, Np, Np))
    for i in range(NUM_PARTICLES):
        
        x_pos = int(r_list[i,0] / h)
        y_pos = int(r_list[i,1] / h)
        z_pos = int(r_list[i,2] / h)
        
        if x_pos <= 0 or x_pos >= Np or y_pos <= 0 or y_pos >= Np or z_pos <= 0 or z_pos >= Np:
            
            pass
        
        else:
        
            rho[x_pos, y_pos, z_pos] += M_PARTICLE / h**3

    return rho


RHO0 = densidad(r_list_0, NP, H)

plt.figure()
plt.imshow(RHO0[:,:,int(NP/2)] + 0.01, norm=colors.LogNorm())
plt.show()

PHI0 = np.zeros((NP, NP, NP)) 
#PLANOS DE CONDICIONES DE FRONTERA
PHI0[:, :, 0] = -G*M_TOTAL/(R0[:, :, 0])
PHI0[:, :, NP-1] = -G*M_TOTAL/(R0[:, :, NP-1])
PHI0[:, 0, :] = -G*M_TOTAL/(R0[:, 0, :])
PHI0[:, NP-1, :] = -G*M_TOTAL/(R0[:, NP-1, :])
PHI0[0, :, :] = -G*M_TOTAL/(R0[0, :, :])
PHI0[NP-1, :, :] = -G*M_TOTAL/(R0[NP-1, :, :])

@jit(nopython=True, fastmath = True)
def poisson(rho, phi, Np, h):
    w = 0.95
    pasos = 100
    for u in range(pasos):
        for i in range(1, Np-1):
            for j in range(1, Np-1):
                for k in range(1, Np-1):
                    phi[i, j, k] = (1.+w)*(1/6)*(phi[i+1, j, k] + phi[i-1, j, k]
                                         + phi[i, j+1, k] + phi[i, j-1, k]
                                         + phi[i, j, k+1] + phi[i, j, k-1]
                                         - h**2 * 4.*np.pi*G*rho[i, j, k]) - w*phi[i, j, k]
    return phi
  
t0 = time.time()
phi = poisson(RHO0, PHI0, NP, H)
tf = time.time()

print(tf-t0)

fig, ax = plt.subplots()
cmap = plt.cm.get_cmap("inferno")
cs = ax.contourf(X, Y, phi[:,:, int(NP/2)], cmap = cmap, levels = 75) #pasamos el potencial a km**2 / s**2
fig.colorbar(cs)
plt.show()

pot = (phi[int(NP/2):, int(NP/2), int(NP/2)])

x = np.linspace(H, int(NP/2)*H, int(NP/2))

fig1, ax1 = plt.subplots()
ax1.plot(x, pot, label='Aproximación')
ax1.plot(x, -(G*M_TOTAL/(x)), label = 'analítico')
plt.legend()
plt.show()

    