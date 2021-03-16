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
from scipy.optimize import curve_fit
##############################################
######### PARÁMETROS FÍSICOS  ################
##############################################
NUM_PARTICLES = 2000000 #Número de partículas
NUM_PARTICLES_BULGE = int(0.14 * NUM_PARTICLES) #el 14% de la materia ordinaria es del bulbo
M_TOTAL = 3245*2.325*1e7 #masa total de las particulas q van a interactuar
M_PARTICLE = M_TOTAL / NUM_PARTICLES #masa de las particulas en Msolares
G = 4.518 * 1e-12 #constante de gravitación universal en Kpc, Msolares y Millones de años
T_SOL = 225 #periodo del Sol alrededor de la galaxia en Millones de años

##############################################
### PARÁMETROS DE SIMULACIÓN
##############################################

dark = True

@jit(nopython=True, fastmath = True, parallel = False)
def pot_dark(r_vec, lim):
    r_dark = np.zeros(3)
    r_dark[0] = r_vec[0] - lim/2
    r_dark[1] = r_vec[1] - lim/2
    r_dark[2] = r_vec[2] - lim/2
    #MODELO 3: Navarro-Frenk-White
    Mh = 12474 * 2.325*1e7
    ah = 7.7
    r = np.sqrt(np.dot(r_dark, r_dark))
    if r > 0.0:
        return (-G*Mh/r)*np.log(1 + r/ah)
    else:
        return 0.0
    
lim = 100 #tamaño de la caja en kpc

NP = 100 #número de celdas
NC = int(NP/2)

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
    
    #Montecarlo para obtener las densidades iniciales
    ###################################################
    ###################################################
@jit(fastmath = True, nopython = True)
def bulge(r):
        b = 0.267
        return  1/(r**2 + b**2)**(5/2)
    
max_bulge = bulge(0)
    
@jit(fastmath = True, nopython = True)
def MN(R, z):
        a = 4.4
        b = 0.308
        return ( (a*R**2 + (a + 3*(z**2 + b**2)**(1/2))*(a + (z**2+b**2)**(1/2))**2 ) 
                / (( R**2 + (a + (z**2 + b**2)**(1/2))**2)**(5/2) * (z**2 + b**2)**(3/2)) )
    
max_disk = MN(0, 0)
    
@jit(fastmath = True, nopython = True)
def get_random_bulge(max_bulge):
        R = random.uniform(0,3)
        y = random.uniform(0, max_bulge)
        while y > bulge(R):
            R = random.uniform(0,3)
            y = random.uniform(0, max_bulge)
        return R
    
@jit(fastmath = True, nopython = True)
def get_random_disk(max_disk):
        R = random.uniform(0, 50)
        z = random.uniform(-2, 2)
        y = random.uniform(0, max_disk)
        while y > MN(R,z):
            R = random.uniform(0, 50)
            z = random.uniform(-2, 2)
            y = random.uniform(0, max_disk)
        return R,z
    ###################################################
    ###################################################
r_list_0 = np.zeros((NUM_PARTICLES, 3), dtype = float)

for i in range(NUM_PARTICLES):
             
            if i < NUM_PARTICLES_BULGE:

                R = get_random_bulge(max_bulge)
                while R>49:
                    R = get_random_bulge(max_bulge)
                    
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2*np.pi)
                
                r_list_0[i, 0] = lim/2 + R*np.cos(phi)*np.sin(theta)
                r_list_0[i, 1] = lim/2 + R*np.sin(phi)*np.sin(theta)
                r_list_0[i, 2] = lim/2 + R*np.cos(theta)

            else:
        
                R, z = get_random_disk(max_disk)
                while R>49 or z>49:
                    R, z = get_random_disk(max_disk)
                    
                phi = random.uniform(0, 2*np.pi) 
                
                r_list_0[i, 0] = lim/2 + R*np.cos(phi)
                r_list_0[i, 1] = lim/2 + R*np.sin(phi)
                r_list_0[i, 2] = lim/2 + z

@jit(nopython=True, fastmath = True, parallel = False)
def sign(x):
    if x >= 0.:
        return 1
    if x < 0.:
        return -1
    return

"""
@jit(nopython=True, fastmath = True, parallel = False)
def W(d, h):
    if d <= h/2:
        return 3/4 - (d/h)**2
    elif h/2 <= d <= 3*h/2:
        return 0.5*(3/2 - d/h)**2
    else:
        return 0
    
"""

@jit(nopython=True, fastmath = True, parallel = False)
def W(d, h):
    if d <= h:
        return 1-d/h
    else:
        return 0

@jit(nopython=True, fastmath = True, parallel = False)
def densidad(r_list, Np, h):
    
    rho = np.zeros((Np, Np, Np))
    
    for i in range(NUM_PARTICLES):
        
        x_pos = int(r_list[i,0] // h)
        y_pos = int(r_list[i,1] // h)
        z_pos = int(r_list[i,2] // h)

        if (x_pos <= 1 or x_pos >= Np-1 or y_pos  <= 1
        or y_pos  >= Np-1 or z_pos <= 1 or z_pos  >= Np-1):
            
            pass
        
        else:
            
            for x in range(-1, 2):
                for y in range(-1, 2):
                    for z in range(-1, 2):
                        dx = abs((x_pos + x + 0.5)*h - r_list[i,0])
                        dy = abs((y_pos + y + 0.5)*h - r_list[i,1])
                        dz = abs((z_pos + z + 0.5)*h - r_list[i,2])
                        rho[x_pos + x, y_pos + y, z_pos + z] += M_PARTICLE * W(dx, h) * W(dy, h) * W(dz, h)/h**3

    return rho

t0 = time.time()
RHO0 = densidad(r_list_0, NP, H)

plt.figure()
cmap = plt.cm.get_cmap("gray")
plt.imshow(RHO0[:,:,int(NP/2)] + 0.01, norm=colors.LogNorm(), cmap = cmap)
plt.show()

plt.figure()
cmap = plt.cm.get_cmap("gray")
plt.imshow(np.transpose(RHO0[int(NP/2),:,:]) + 0.01, norm=colors.LogNorm(), cmap = cmap)
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
    tol = 1e-5
    acabar = False
    iterations = 0
    while not acabar:
        max_diff = 0
        for i in range(1, Np-1):
            for j in range(1, Np-1):
                for k in range(1, Np-1):
                    phi_0 = phi[i, j, k] 
                    phi[i, j, k] = (1.+w)*(1/6)*(phi[i+1, j, k] + phi[i-1, j, k]
                                         + phi[i, j+1, k] + phi[i, j-1, k]
                                         + phi[i, j, k+1] + phi[i, j, k-1]
                                         - h**2 * 4.*np.pi*G*rho[i, j, k]) - w*phi[i, j, k]
                    diff = abs(phi_0 - phi[i,j,k])
                    if diff > max_diff:
                        max_diff = diff
        iterations += 1
        if max_diff < tol:
            acabar = True
    print('Number of iterations:', iterations)
    return phi

t0 = time.time()
phi = poisson(RHO0, PHI0, NP, H)
tf = time.time()

print(tf-t0)

fig, ax = plt.subplots()
cmap = plt.cm.get_cmap("gray")
cs = ax.contourf(phi[:,:, int(NP/2)], cmap = cmap, levels = 70) #pasamos el potencial a km**2 / s**2
colorbar = fig.colorbar(cs)
colorbar.ax.set_xlabel('$\phi (kpc^2/My^2)$')
plt.show()
