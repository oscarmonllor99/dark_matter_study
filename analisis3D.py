#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:16:38 2021

@author: oscar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.interpolate import *

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

n = int(sim_parameters[2])
div_r = int(sim_parameters[3])
div_v = int(sim_parameters[4])
n_r = int(n / div_r) #numero de pasos de tiempo guardados para r
n_v = int(n / div_v) #numero de pasos de tiempo guardados para v
n_graf = 6

dt = sim_parameters[5]

Np = 100

h = lim/Np

tray = np.loadtxt('trayectorias.dat', dtype = float)
tray_3D = tray.reshape(n_r, N, 3)

vels = np.loadtxt('velocidades.dat', dtype = float)
vels_3D = vels.reshape(n_v, N, 3)

tray_CM = np.loadtxt('trayectoria_CM.dat', dtype = float)
R_CM = tray_CM.reshape(n_r, 3)

#ESTO ES PARA LAS REPRESENTACIONES DE LAS CURVAS DE VELOCIDAD
R_lim =  np.sqrt((lim/2)**2 + (lim/2)**2)
Rs = np.linspace(0.001, R_lim, 100) 

def empty_list_of_lists(N):
    lista = []
    for i in range(N):
        lista.append([])
    return lista


guess = np.array([462, -1])
def c(r, a, b):
    #f_conv = 978.5
    return (a*r**(b))

fit_info = np.empty((n_graf, 2*len(guess) + 2))


for k in range(n_graf):
    #PARA REPRESENTAR
    X = []
    Y = []
    Z = []
    V = []
    V_ANG = []
    V_RAD = []
    k_r = int(n_r/n_graf)*k
    k_v = int(n_v/n_graf)*k
    for i in range(Nd, N):    
                
        v = np.linalg.norm(vels_3D[k_v, i, :])

        vx = vels_3D[k_v, i, 0]
        vy = vels_3D[k_v, i, 1]
        vz = vels_3D[k_v, i, 2]
        
        ##########################################
        #velocidad radial y velocidad angular
        #########################################
        
        theta_g = np.arctan((tray_3D[k_r, i, 1] - R_CM[k_r, 1]) / (tray_3D[k_r, i, 0] - R_CM[k_r, 0])) 
        
        ###################################
        #arreglo a la función arctan, para que se ajuste a nuestras necesidades
        ###################################
                
        if (tray_3D[k_r, i, 1]  - R_CM[k_r, 1]) < 0 or (tray_3D[k_r, i, 0] - R_CM[k_r, 0]) < 0:
                    
            theta_g = np.pi + theta_g
                    
        ###################################
        ###################################
     
        v_rad = (vx*np.cos(theta_g) + vy*np.sin(theta_g))
        v_ang = abs(-vx*np.sin(theta_g) + vy*np.cos(theta_g))
        
        ###################################
        ###################################
        #PARA REPRESENTAR EN GRÁFICAS
        ###################################
        ###################################
        X.append(tray_3D[k_r, i, 0]  - R_CM[k_r, 0])
        Y.append(tray_3D[k_r, i, 1]  - R_CM[k_r, 1])
        Z.append(tray_3D[k_r, i, 2]  - R_CM[k_r, 2])
                   
        V.append(v/(1.022*1e-3)) #pasamos a km/s
        V_ANG.append(v_ang/(1.022*1e-3))
        V_RAD.append(v_rad/(1.022*1e-3))

    ########################################
    ########################################
    #Representación de la curva de velocidades
    ########################################
                
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
                
    V = np.array(V)
    V_ANG = np.array(V_ANG)
    V_RAD = np.array(V_RAD) 
                
    ######################################################
    ######################################################
    #SE VA A SEPARAR LA CURVA DE VELOCIDADES POR BANDAS
    #POSTERIORMENTE SE REALIZA LA MEDIA DE CADA BANDA.
    V_med = empty_list_of_lists(len(Rs))
    V_ANG_med = empty_list_of_lists(len(Rs))
    V_RAD_med = empty_list_of_lists(len(Rs))
    
    sigma_ANG = empty_list_of_lists(len(Rs))
    
    for i in range(N-Nd):
                    
        Ri = np.sqrt(X[i]**2 + Y[i]**2) 
                    
        for j in range(len(Rs)):
                        
            if Rs[j-1] <= Ri <= Rs[j]:
                
                if abs(V_ANG[i])>0 and abs(V[i])>0 and abs(V_RAD[i])>0:
                    
                    V_med[j].append(V[i])
                    V_ANG_med[j].append(V_ANG[i])
                    V_RAD_med[j].append(V_RAD[i])
                    
                break
                            
            if j == 0:
                            
                pass

    #AHORA SE REALIZAN LAS MEDIAS, SI LA LISTA NO ES DE LONGITUD 0:
                    
    for j in range(len(Rs)):
                    
        if len(V_med[j]) != 0:
                    
            V_med[j] = sum(V_med[j])/len(V_med[j])
                        
        else:
                        
            V_med[j] = 0
                        
        if len(V_ANG_med[j]) != 0:
            std = np.std(np.array(V_ANG_med[j]))
            sigma_ANG[j] = std/np.sqrt(len(V_ANG_med[j]))
            V_ANG_med[j] = abs(sum(V_ANG_med[j])/len(V_ANG_med[j]))
                        
        else:
                        
            V_ANG_med[j] = 0
                        
        if len(V_RAD_med[j]) != 0:
                    
            V_RAD_med[j] = sum(V_RAD_med[j])/len(V_RAD_med[j])
                        
        else:
                        
            V_RAD_med[j] = 0
            
    ######################################################     
    ######################################################
    ### FITTING ########################################3
    ######################################################     
    ######################################################
    v_tan_data = []
    r_data = []
    sigma_data = []
    for u in range(len(V_ANG_med)):
        if V_ANG_med[u] < 25: #km/s
            pass
        else:
            v_tan_data.append(V_ANG_med[u])
            r_data.append(Rs[u])
            sigma_data.append(sigma_ANG[u])
            
    r_data = np.array(r_data)
    v_tan_data = np.array(v_tan_data)
    sigma_data = np.array(sigma_data)
    
  
    corte = 10
    param, cov = curve_fit(c, r_data[corte:-1], v_tan_data[corte:-1], guess, sigma = None, absolute_sigma = True)    
    c_fit = c(r_data[corte:-1], param[0], param[1]) #curve fit 
    R2 = r2_score(c_fit, v_tan_data[corte:-1])
    err = np.sqrt(np.diag(cov)) 
    #Si las componentes de cov no diagonales son del orden o mayores 
    #que las diagonales el modelo está mal
    print("######################################")
    print("a*r^b: ", param)
    print("R² =", R2,", paso", k)
    print("Standard Deviation of fitted param =", err)
    print("Covariance =", cov-np.diag(cov))
    print("######################################")
    
    
    fit_info[k, 0] = param[0]
    fit_info[k, 1] = err[0]
    fit_info[k, 2] = param[1]
    fit_info[k, 3] = err[1]
    fit_info[k, 4] = R2
    fit_info[k, 5] = k
    np.savetxt('fit_info.txt', fit_info, fmt = '%.2e')
    ######################################################     
    ######################################################
    ######################################################     
    ######################################################
    """
    fig1, ax1 = plt.subplots()
    ax1.scatter(Rs, V_med, color = 'red', s = 0.5, marker = '*')
    plt.title('Curva de módulo velocidad para el paso {}'.format(k))
    plt.xlabel('R (kpc)')
    plt.ylabel('v (km/s)')
    plt.ylim(0, 400)
    plt.xlim(0, x_lim/2)
    plt.savefig('Curva de módulo velocidad paso {}.png'.format(k))
    plt.show()
    """
    fig2, ax2 = plt.subplots()
    plt.errorbar(r_data, v_tan_data, sigma_data, fmt='r.')
    ax2.scatter(Rs, V_ANG_med, c = 'red', s = 0.5, marker = '*', label = "$v_c$")
    plt.plot(r_data[corte:-1], c_fit, color = 'black', label = "$v_c (r) = a · r^b$")
    plt.title('Curva de velocidad circular para {} millones de años'.format(int((k*n/n_graf)*dt)))
    plt.xlabel('$R (kpc)$')
    plt.ylabel('$v_c (km/s)$')
    plt.ylim(0, 400)
    plt.xlim(0, lim/2)
    plt.legend()
    plt.savefig('Curva de velocidad circular para {} millones de años'.format(int((k*n/n_graf)*dt)))
    plt.show()
    """
    fig3, ax3 = plt.subplots()
    ax3.scatter(Rs, V_RAD_med, color = 'black', s = 0.5, marker = '*')
    plt.title('Curva de velocidad radial para el paso {}'.format(k))
    plt.xlabel('R (kpc)')
    plt.ylabel('v_r (km/s)')
    plt.ylim(-300, 300)
    plt.xlim(0, x_lim/2)
    plt.savefig('Curva de velocidad radial paso {}.png'.format(k))
    plt.show()
    """


