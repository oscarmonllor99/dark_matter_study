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

"""""""""""""""
Parámetros físicos

"""""""""""""""

N = 30000 #Número de partículas

T_sol = 225 #periodo del Sol alrededor de la galaxia

G = 4.518 * 1e-12#en km²/kg*s²

M = 1e11 #en kg

"""""""""""""""
Parámetros de simulación

"""""""""""""""
n = 10000 #número de pasos totales de tiempo
div_r = 25
div_v = 100
n_r = int(n / div_r) #numero de pasos de tiempo guardados para r
n_v = int(n / div_v) #numero de pasos de tiempo guardados para v
n_graf = 5

salt = 1 #salto en los pasos, en vez de ir de 1 paso en 1 paso, se va de salt en salt para agilizar y ahorrar espacio

lim = 100 #en kpc

x_lim, y_lim, z_lim = lim, lim, lim

dt = (T_sol / 2000) #intervalo de tiempo entre cada paso

tray = np.loadtxt('trayectorias.dat', dtype = float)
tray_3D = tray.reshape(n_r, N, 3)

vels = np.loadtxt('velocidades.dat', dtype = float)
vels_3D = vels.reshape(n_v, N, 3)


#ESTO ES PARA LAS REPRESENTACIONES DE LAS CURVAS DE VELOCIDAD
R_lim =  np.sqrt((x_lim/2)**2 + (y_lim/2)**2)
Rs = np.linspace(0.001, R_lim, 200) 

def empty_list_of_lists(N):
    lista = []
    for i in range(N):
        lista.append([])
    return lista


guess = np.array([120, -1, 0.1])
def c(r, a, b, c):
    #f_conv = 978.5
    return (a + b*r + c*r**2)

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
    for i in range(N):    
                
        v = np.linalg.norm(vels_3D[k_v, i, :])

        vx = vels_3D[k_v, i, 0]
        vy = vels_3D[k_v, i, 1]
        vz = vels_3D[k_v, i, 2]

        ##########################################
        #velocidad radial y velocidad angular
        #########################################

        theta_g = np.arctan((tray_3D[k_r, i, 1] - y_lim/2) / (tray_3D[k_r, i, 0] - x_lim/2))  
   
        ###################################
        #arreglo a la función arctan, para que se ajuste a nuestras necesidades
        ###################################
                
        if (tray_3D[k_r, i, 1]  - y_lim/2) < 0 or (tray_3D[k_r, i, 0] - x_lim/2) < 0:
                    
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
        X.append(tray_3D[k_r, i, 0]  - x_lim/2)
        Y.append(tray_3D[k_r, i, 1]  - y_lim/2)
        Z.append(tray_3D[k_r, i, 2]  - z_lim/2)
                   
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
    
    for i in range(N):
                    
        Ri = np.sqrt(X[i]**2 + Y[i]**2) 
                    
        for j in range(len(Rs)):
                        
            if Rs[j-1] <= Ri <= Rs[j]:
                            
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
            V_ANG_med[j] = sum(V_ANG_med[j])/len(V_ANG_med[j])
                        
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
        if V_ANG_med[u] < 20: #km/s
            pass
        else:
            v_tan_data.append(V_ANG_med[u])
            r_data.append(Rs[u])
            sigma_data.append(sigma_ANG[u])
            
    r_data = np.array(r_data)
    v_tan_data = np.array(v_tan_data)
    sigma_data = np.array(sigma_data)
    
    corte = int(len(Rs)/5)
    param, cov = curve_fit(c, r_data[corte:-1], v_tan_data[corte:-1], guess, sigma = None, absolute_sigma = True)    
    c_fit = c(r_data[corte:-1], param[0], param[1], param[2]) #curve fit 
    R2 = r2_score(c_fit, v_tan_data[corte:-1])
    err = np.sqrt(np.diag(cov)) 
    #Si las componentes de cov no diagonales son del orden o mayores 
    #que las diagonales el modelo está mal
    print("######################################")
    print("a + br + cr²: ", param)
    print("R² =", R2,", paso", k)
    print("Standard Deviation of fitted param =", err)
    print("Covariance =", cov-np.diag(cov))
    print("######################################")
    
    
    fit_info[k, 0] = param[0]
    fit_info[k, 1] = err[0]
    fit_info[k, 2] = param[1]
    fit_info[k, 3] = err[1]
    fit_info[k, 4] = param[2]
    fit_info[k, 5] = err[2]
    fit_info[k, 6] = R2
    fit_info[k, 7] = k
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
    plt.errorbar(r_data, v_tan_data, sigma_data, fmt='b.')
    
    ax2.scatter(Rs, V_ANG_med, color = 'blue', s = 0.5, marker = '*', label = "$v_circ$")
    plt.plot(r_data[corte:-1], c_fit, color = 'black', label = "$v(r) = a + b·r + c·r^2$")
    plt.title('Curva de velocidad circular para el paso {}'.format(k))
    plt.xlabel('R (kpc)')
    plt.ylabel('v_ang (km/s)')
    plt.ylabel('v (km/s)')
    plt.ylim(0, 400)
    plt.xlim(0, x_lim/2)
    plt.savefig('Curva de velocidad tangencial paso {}.png'.format(k))
    plt.legend()
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




