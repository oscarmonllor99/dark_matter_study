# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:05:09 2020

@author: Oscar
"""
from numba import jit
import numba
import numpy as np
import random 
import time

##############################################
######### PARÁMETROS FÍSICOS  ################
##############################################

N = 20000 #Número de partículas

N_b = int(0.14*N) #el 30% de la materia ordinaria es del bulbo

M = 3245*2.325*1e7 #masa total de las particulas q van a interactuar: fraccion no significativa de la total galactica.

m = M / N #masa de las particulas en Msolares

G = 4.518 * 1e-12 #constante de gravitación universal en Kpc, Msolares y Millones de años

T_sol = 225 #periodo del Sol alrededor de la galaxia en Millones de años

##############################################
##############################################

##############################################
### PARÁMETROS DE SIMULACIÓN
##############################################

lim = 100 #en kpc

x_lim, y_lim, z_lim = lim, lim, lim 

n = 40000 #número de pasos totales de tiempo 
div_r = 100
div_v = 100
n_r = int(n // div_r) #numero de pasos de tiempo guardados para r
n_v = int(n // div_v) #numero de pasos de tiempo guardados para v

dt = T_sol / 2000 #intervalo de tiempo entre cada paso

#Parámetros para calcular el potencial y su gradiente

N_p = 10000 #número de pasos común a los dos ejes, es decir, número de celdas para calcular
#el gradiente de la gravedad

d_min = 0.05 #distancia minima a la cual se debe calcular la fuerza en kpc
eps = np.sqrt(2*d_min / 3**(3/2)) 

k_vel=0.95#parámetro de control de momento angular inicial (0--> velocidad angular inicial 0
#                                                          1--> velocidad angular inicial máxima)
##############################################
##############################################



##############################################
### FUNCIONES A UTILIZAR ###
##############################################
@jit(nopython=True, fastmath = True, parallel = False)
def interpolation(y):
    return 0.5*( 1 + (1+4*y**(-1))**(1/2) )


@jit(nopython=True, fastmath = True, parallel = False)
def CM(r_list):
    
    M = N*m
    
    #listas que tiene la posicion de cada particula por su masa
    Xm = np.empty((N))
    Ym = np.empty((N))
    Zm = np.empty((N))
    
    Xm[:] = r_list[:, 0] * m
    Ym[:] = r_list[:, 1] * m
    Zm[:] = r_list[:, 2] * m
    
    X_CM = np.sum(Xm) / M
    Y_CM = np.sum(Ym) / M
    Z_CM = np.sum(Zm) / M

    return np.array([X_CM, Y_CM, Z_CM])

@jit(nopython=True, fastmath = True, parallel = False)
def fuerza_part(i, r_list, sumatorio_f):

    for j in range(N):
        if j != i:
            
            rx = r_list[i, 0] - r_list[j, 0] 
            ry = r_list[i, 1] - r_list[j, 1] 
            rz = r_list[i, 2] - r_list[j, 2] 

            rij2 = rx*rx + ry*ry + rz*rz

            sumatorio_f[0] = sumatorio_f[0] + rx * m * (1 / (eps**2 + rij2))**(3 / 2)
            sumatorio_f[1] = sumatorio_f[1] + ry * m * (1 / (eps**2 + rij2))**(3 / 2)
            sumatorio_f[2] = sumatorio_f[2] + rz * m * (1 / (eps**2 + rij2))**(3 / 2)

    return sumatorio_f * (-G)
    
@jit(nopython=True, fastmath = True, parallel = False)
def ener_pot(i, r_list, sumatorio_E):

    for j in range(N):
        if j != i:

            rx = r_list[i, 0] - r_list[j, 0] 
            ry = r_list[i, 1] - r_list[j, 1] 
            rz = r_list[i, 2] - r_list[j, 2] 

            rij2 = rx*rx + ry*ry + rz*rz

            sumatorio_E = sumatorio_E +  m**2 * (1 / (eps**2 + rij2))**(1 / 2)

    return sumatorio_E * (-G)


###########################################################################
###########################################################################


#################################################################
# CONDICIONES INICIALES DE LAS PARTÍCULAS ###
#################################################################

def cond_inicial():
    
    #Posición inicial de las partículas
    #r_list es una matriz con cada fila el vector posición de una partícula   
    #distribución aleatoria de las partículas, con ciertas restricciones
    
    r_list_0 = np.zeros((N, 3))
    r_esf_tot = np.zeros((N, 3))
    
    for i in range(N):
             
            if i < N_b:
                
                R = random.uniform(0.1, 3)
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2*np.pi)
                
                r_esf = [R, phi, theta]
                r_esf_tot[i] = r_esf 
                
                r_list_0[i, 0] = x_lim/2 + R*np.cos(phi)*np.sin(theta)
                r_list_0[i, 1] = y_lim/2 + R*np.sin(phi)*np.sin(theta)
                r_list_0[i, 2] = z_lim/2 + R*np.cos(theta)
                
                
            else:
        
                R = 10*random.expovariate(1)
                phi = random.uniform(0, 2*np.pi) 
                z = random.uniform(-0.5, 0.5)
                
                r_esf = [R, phi, z]
                r_esf_tot[i] = r_esf 
                
                r_list_0[i, 0] = x_lim/2 + R*np.cos(phi)
                r_list_0[i, 1] = y_lim/2 + R*np.sin(phi)
                r_list_0[i, 2] = z_lim/2 + z
            

    f_list_0 = np.zeros((N, 3))
 
    
    for u in range(N):
        a_0 = 3.87*1e-3
        sumatorio_f = np.array([0,0,0])
        fuerza = fuerza_part(u, r_list_0, sumatorio_f)
        f_mod = np.sqrt(fuerza[0]**2 + fuerza[1]**2 + fuerza[2]**2)
        nu = interpolation(f_mod/a_0)
        f_list_0[u, :] = fuerza[:]*nu
            

    #distribución aleatoria de las velocidades, teniendo siempre velocidad tangencial
    #se distribuyen las velocidades según la velocidad de escape dada para cada posición   

    v_list_0 = np.zeros((N, 3))
    
    for i in range(N):

            
            ###################################################
            #Cálculo de la energía potencial para v escape
            ###################################################
            sumatorio_E = 0.

            Ui = ener_pot(i, r_list_0, sumatorio_E)  
           
            v_esc = np.sqrt(-2*Ui/m)
            
            """
            if 3 <= np.linalg.norm(r_list_0[i]) <= 5:
                v_R = random.uniform(-0.4*v_esc, 0.4*v_esc)
            else:
                v_R = random.uniform(-0.1*v_esc, 0.1*v_esc)
            """
            
            #producto vectorial de g con ur (vector unitario radial)
            g_vec = f_list_0[i]
            
            R_centro = np.zeros(3)
            R_centro[0] = r_list_0[i, 0] - x_lim/2
            R_centro[1] = r_list_0[i, 1] - y_lim/2
            R_centro[2] = r_list_0[i, 2] - z_lim/2
            
            R_norm =  np.linalg.norm(R_centro)
            ur = R_centro / R_norm
            prod =  abs(np.inner(g_vec, ur))
            
            v_circ = np.sqrt(R_norm * prod)
            
            phi_g = r_esf_tot[i, 1]

            if N < N_b:
                
                theta_g = r_esf_tot[i, 2]    
                v_r = random.uniform(-0.1*v_esc, 0.1*v_esc)   
                v_z = v_circ*np.cos(theta_g)
                v_tan = v_circ*np.sin(theta_g)
                
                v_list_0[i, 0] = k_vel * (-v_tan * np.sin(phi_g) + v_r * np.cos(phi_g))
                v_list_0[i, 1] = k_vel * (v_tan * np.cos(phi_g) + v_r * np.sin(phi_g))
                v_list_0[i, 2] = k_vel * v_z
                
            else:
                
                v_tan = v_circ
                v_R = random.uniform(-0.1*v_esc, 0.1*v_esc)   
                v_z = random.uniform(-0.05*v_esc, 0.05*v_esc)
                
                v_list_0[i, 0] = k_vel * (-v_tan * np.sin(phi_g) + v_R * np.cos(phi_g))
                v_list_0[i, 1] = k_vel * (v_tan * np.cos(phi_g) + v_R * np.sin(phi_g))
                v_list_0[i, 2] = k_vel * v_z

    return r_list_0, v_list_0, f_list_0
    
#################################################################
#################################################################

#################################################################
#Se aplica el algoritmo de Leapfrog para resolver los primeros pasos de tiempo
#################################################################

@jit(nopython=True, fastmath = True, parallel = False)
def paso(r_list, v_list, f_list):
    
    a_0 = 3.87*1e-3
    
    r_list_new = r_list + dt*v_list + 0.5 * dt**2 * f_list
    
    f_list_new = np.empty((N, 3))
    
    for u in range(N):
        sumatorio_f = np.array([0,0,0])
        fuerza = fuerza_part(u, r_list_new, sumatorio_f)
        f_mod = np.sqrt(fuerza[0]**2 + fuerza[1]**2 + fuerza[2]**2)
        nu = interpolation(f_mod/a_0)
        f_list_new[u, :] = fuerza[:]*nu
        
    v_list_new = v_list + 0.5*dt*(f_list_new + f_list)
    
    return r_list_new, v_list_new, f_list_new
 
################################################################# 
#################################################################


#####################################################################################################
#####################################################################################################  
# FUNCIÓN QUE VA A OBTENER LAS TRAYECTORIAS Y LAS CARACTERÍSTICAS PARA CADA PASO DE TIEMPO
# LLAMANDO A LOS MÉTODOS DE RESOLUCIÓN PASO
#####################################################################################################
#####################################################################################################

def tiempo(r_list, v_list, f_list):
    #Lista de trayectorias de todas las partículas
    tray = np.empty((n_r, N, 3))
    tray_CM = np.empty((n_r, 3))
    vels = np.empty((n_v, N, 3))

    for k in range(n):
        #Estos son los indices de paso de tiempo para guardar r y v
        k_r = int(k // div_r)
        k_v = int(k // div_v)

        R_CM = CM(r_list)
        
        if k == 1:
            t0 = time.time()
            r_list, v_list, f_list = paso(r_list, v_list, f_list)
            tf = time.time()
            print("El programa va a tardar:", (n*(tf-t0)/60),"minutos")
        else:
            r_list, v_list, f_list= paso(r_list, v_list, f_list)
            
        if k%div_r == 0.0:
 
            tray[k_r, :, :] = r_list[:, :]

            tray_CM[k_r, :] = R_CM[:]

        if k%div_v == 0.0:
            print((k/n)*100, "%")
            vels[k_v, :, :] = v_list[:, :]

    return tray, tray_CM, vels
      
####################################################################################
####################################################################################
####################################################################################

###########################################################################
# Guaradamos todas las trayectorias obtenidas con el tiempo en tray
###########################################################################

r_list_0, v_list_0, f_list_0 = cond_inicial()

t0 = time.time()
trayectorias, trayectoria_CM, velocidades = tiempo(r_list_0, v_list_0, f_list_0)    
tf = time.time()

print('El programa ha tardado: ', int(tf-t0), 'segundos en completar las trayectorias.')
trayectorias3D = trayectorias.reshape(trayectorias.shape[0], -1) 
velocidades3D = velocidades.reshape(velocidades.shape[0], -1) 

np.savetxt('trayectorias.dat', trayectorias3D, fmt = '%.6e') #fmt es cuantas cifras decimales
np.savetxt('trayectoria_CM.dat', trayectoria_CM, fmt = '%.6e') #fmt es cuantas cifras decimales
np.savetxt('velocidades.dat', velocidades3D, fmt = '%.6e') #fmt es cuantas cifras decimales


