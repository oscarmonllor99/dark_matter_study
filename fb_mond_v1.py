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
import argparse

##############################################
######### PARÁMETROS FÍSICOS  ################
##############################################

NUM_PARTICLES = 2000 #Número de partículas
NUM_PARTICLES_BULB = int(0.14 * NUM_PARTICLES) #el 30% de la materia ordinaria es del bulbo
M_TOTAL = 3245*2.325*1e7 #masa total de las particulas q van a interactuar: fraccion no significativa de la total galactica.
M_PARTICLE = M_TOTAL / NUM_PARTICLES #masa de las particulas en Msolares
G = 4.518 * 1e-12 #constante de gravitación universal en Kpc, Msolares y Millones de años
T_SOL = 225 #periodo del Sol alrededor de la galaxia en Millones de años

##############################################
### FUNCIONES A UTILIZAR ###
##############################################
@jit(nopython=True, fastmath = True, parallel = False)
def interpolation(y):
    return 0.5*( 1 + (1+4*y**(-1))**(1/2) )


@jit(nopython=True, fastmath = True, parallel = False)
def CM(r_list):
    
    M = NUM_PARTICLES * M_TOTAL
    
    #listas que tiene la posicion de cada particula por su masa
    Xm = np.empty((NUM_PARTICLES))
    Ym = np.empty((NUM_PARTICLES))
    Zm = np.empty((NUM_PARTICLES))
    
    Xm[:] = r_list[:, 0] * M_TOTAL
    Ym[:] = r_list[:, 1] * M_TOTAL
    Zm[:] = r_list[:, 2] * M_TOTAL
    
    X_CM = np.sum(Xm) / M
    Y_CM = np.sum(Ym) / M
    Z_CM = np.sum(Zm) / M

    return np.array([X_CM, Y_CM, Z_CM])

@jit(nopython=True, fastmath = True, parallel = False)
def fuerza_part(i, r_list, sumatorio_f, eps):

    for j in range(NUM_PARTICLES):
        if j != i:
            
            rx = r_list[i, 0] - r_list[j, 0] 
            ry = r_list[i, 1] - r_list[j, 1] 
            rz = r_list[i, 2] - r_list[j, 2] 

            rij2 = rx*rx + ry*ry + rz*rz

            sumatorio_f[0] = sumatorio_f[0] + rx * M_TOTAL * (1 / (eps**2 + rij2))**(3 / 2)
            sumatorio_f[1] = sumatorio_f[1] + ry * M_TOTAL * (1 / (eps**2 + rij2))**(3 / 2)
            sumatorio_f[2] = sumatorio_f[2] + rz * M_TOTAL * (1 / (eps**2 + rij2))**(3 / 2)

    return sumatorio_f * (-G)
    
@jit(nopython=True, fastmath = True, parallel = False)
def ener_pot(i, r_list, sumatorio_E, eps):

    for j in range(NUM_PARTICLES):
        if j != i:

            rx = r_list[i, 0] - r_list[j, 0] 
            ry = r_list[i, 1] - r_list[j, 1] 
            rz = r_list[i, 2] - r_list[j, 2] 

            rij2 = rx*rx + ry*ry + rz*rz

            sumatorio_E = sumatorio_E +  M_TOTAL**2 * (1 / (eps**2 + rij2))**(1 / 2)

    return sumatorio_E * (-G)

###########################################################################
###########################################################################

#################################################################
# CONDICIONES INICIALES DE LAS PARTÍCULAS ###
#################################################################

def cond_inicial(lim, k_vel, eps):

    #Posición inicial de las partículas
    #r_list es una matriz con cada fila el vector posición de una partícula   
    #distribución aleatoria de las partículas, con ciertas restricciones

    r_list_0 = np.zeros((NUM_PARTICLES, 3))
    r_esf_tot = np.zeros((NUM_PARTICLES, 3))
    E_rot = 0
    E_pot = 0
    for i in range(NUM_PARTICLES):

        if i < NUM_PARTICLES_BULB:

            R = random.uniform(0.1, 3)
            theta = random.uniform(0, np.pi)
            phi = random.uniform(0, 2*np.pi)
            
            r_esf = [R, phi, theta]
            r_esf_tot[i] = r_esf 
            
            r_list_0[i, 0] = lim[0] / 2 + R*np.cos(phi)*np.sin(theta)
            r_list_0[i, 1] = lim[1] / 2 + R*np.sin(phi)*np.sin(theta)
            r_list_0[i, 2] = lim[2] / 2 + R*np.cos(theta)

        else:

            R = 10*random.expovariate(1)
            phi = random.uniform(0, 2*np.pi) 
            z = random.uniform(-0.5, 0.5)
            
            r_esf = [R, phi, z]
            r_esf_tot[i] = r_esf 
            
            r_list_0[i, 0] = lim[0] / 2 + R*np.cos(phi)
            r_list_0[i, 1] = lim[1] / 2 + R*np.sin(phi)
            r_list_0[i, 2] = lim[2] / 2 + z

    f_list_0 = np.zeros((NUM_PARTICLES, 3))

    for u in range(NUM_PARTICLES):
        a_0 = 3.87*1e-3
        sumatorio_f = np.array([0,0,0])
        fuerza = fuerza_part(u, r_list_0, sumatorio_f, eps)
        f_mod = np.sqrt(fuerza[0]**2 + fuerza[1]**2 + fuerza[2]**2)
        nu = interpolation(f_mod/a_0)
        f_list_0[u, :] = fuerza[:]*nu

    #distribución aleatoria de las velocidades, teniendo siempre velocidad tangencial
    #se distribuyen las velocidades según la velocidad de escape dada para cada posición   

    v_list_0 = np.zeros((NUM_PARTICLES, 3))
    
    for i in range(NUM_PARTICLES):
        ###################################################
        #Cálculo de la energía potencial para v escape
        ###################################################
        sumatorio_E = 0.0

        Ui = ener_pot(i, r_list_0, sumatorio_E, eps)
        E_pot += Ui
        
        v_esc = np.sqrt(-2*Ui / M_TOTAL)
        
        """
        if 3 <= np.linalg.norm(r_list_0[i]) <= 5:
            v_R = random.uniform(-0.4*v_esc, 0.4*v_esc)
        else:
            v_R = random.uniform(-0.1*v_esc, 0.1*v_esc)
        """
        
        #producto vectorial de g con ur (vector unitario radial)
        g_vec = f_list_0[i]
        
        R_centro = np.zeros(3)
        R_centro[0] = r_list_0[i, 0] - lim[0] / 2
        R_centro[1] = r_list_0[i, 1] - lim[1] / 2
        R_centro[2] = r_list_0[i, 2] - lim[2] / 2
        
        R_norm =  np.linalg.norm(R_centro)
        ur = R_centro / R_norm
        prod =  abs(np.inner(g_vec, ur))
        
        v_circ = k_vel*np.sqrt(R_norm * prod)
        E_rot += 0.5 * M_TOTAL * v_circ**2 
        
        phi_g = r_esf_tot[i, 1]

        if NUM_PARTICLES < NUM_PARTICLES_BULB:
            
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
            v_z = random.uniform(-0.01*v_esc, 0.01*v_esc)
            
            v_list_0[i, 0] = (-v_tan * np.sin(phi_g) + v_R * np.cos(phi_g))
            v_list_0[i, 1] = (v_tan * np.cos(phi_g) + v_R * np.sin(phi_g))
            v_list_0[i, 2] = v_z

    print('Ostriker-Peebles criterion (t ~< 0.14): ', E_rot/abs(E_pot))
    return r_list_0, v_list_0, f_list_0

#################################################################
#################################################################

#################################################################
#Se aplica el algoritmo de Leapfrog para resolver los primeros pasos de tiempo
#################################################################

@jit(nopython=True, fastmath = True, parallel = False)
def paso(r_list, v_list, f_list, dt, eps):
    
    a_0 = 3.87*1e-3
    
    r_list_new = r_list + dt*v_list + 0.5 * dt**2 * f_list
    
    f_list_new = np.empty((NUM_PARTICLES, 3))
    
    for u in range(NUM_PARTICLES):
        sumatorio_f = np.array([0,0,0])
        fuerza = fuerza_part(u, r_list_new, sumatorio_f, eps)
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

def tiempo(r_list, v_list, f_list, n_r, n_v, div_r, div_v, dt, eps):
    #Lista de trayectorias de todas las partículas
    tray = np.empty((n_r, NUM_PARTICLES, 3))
    tray_CM = np.empty((n_r, 3))
    vels = np.empty((n_v, NUM_PARTICLES, 3))

    for k in range(NUM_PARTICLES):
        #Estos son los indices de paso de tiempo para guardar r y v
        k_r = int(k // div_r)
        k_v = int(k // div_v)

        R_CM = CM(r_list)
        
        if k == 1:
            t0 = time.time()
            r_list, v_list, f_list = paso(r_list, v_list, f_list, dt, eps)
            tf = time.time()
            print("El programa va a tardar:", int(NUM_PARTICLES*(tf-t0)/60),"minutos")
        else:
            r_list, v_list, f_list= paso(r_list, v_list, f_list, dt, eps)
            
        if k%div_r == 0.0:
 
            tray[k_r, :, :] = r_list[:, :]

            tray_CM[k_r, :] = R_CM[:]

        if k%div_v == 0.0:
            print((k/NUM_PARTICLES)*100, "%")
            vels[k_v, :, :] = v_list[:, :]

    return tray, tray_CM, vels

####################################################################################
####################################################################################
####################################################################################

###########################################################################
# Guaradamos todas las trayectorias obtenidas con el tiempo en tray
###########################################################################

def main(args):


    ##############################################
    ### PARÁMETROS DE SIMULACIÓN
    ##############################################

    LIM = [args.lim, args.lim, args.lim] #en kpc
    TIME_STEPS = args.time_steps #número de pasos totales de tiempo 
    DIV_R = args.div_r
    DIV_V = args.div_v
    N_R = int(TIME_STEPS // DIV_R) #numero de pasos de tiempo guardados para r
    N_V = int(TIME_STEPS // DIV_V) #numero de pasos de tiempo guardados para v
    DT = T_SOL / 2000 #intervalo de tiempo entre cada paso
    D_MIN = args.dmin #distancia minima a la cual se debe calcular la fuerza en kpc
    EPS = np.sqrt(2*D_MIN / 3**(3/2)) 
    K_VEL=0.6#parámetro de control de momento angular inicial (0--> velocidad angular inicial 0
    #                                                          1--> velocidad angular inicial máxima)
    ##############################################
    ##############################################

    r_list_0, v_list_0, f_list_0 = cond_inicial(LIM, K_VEL, EPS)

    t0 = time.time()
    trayectorias, trayectoria_CM, velocidades = tiempo(r_list_0, v_list_0, f_list_0, N_R, N_V, DIV_R, DIV_V, DT, EPS)
    tf = time.time()

    print('El programa ha tardado: ', ((tf-t0)/60), 'minutos en completar las trayectorias.')
    trayectorias3D = trayectorias.reshape(trayectorias.shape[0], -1) 
    velocidades3D = velocidades.reshape(velocidades.shape[0], -1) 

    np.savetxt('trayectorias.dat', trayectorias3D, fmt = '%.6e') #fmt es cuantas cifras decimales
    np.savetxt('trayectoria_CM.dat', trayectoria_CM, fmt = '%.6e') #fmt es cuantas cifras decimales
    np.savetxt('velocidades.dat', velocidades3D, fmt = '%.6e') #fmt es cuantas cifras decimales


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FB MOND V1")

    parser.add_argument(
        "--lim",
        type=int,
        default=100,
        help="em kpc.",
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        default=40000,
        help="timesteps.",
    )
    parser.add_argument(
        "--div_r",
        type=int,
        default=100,
        help="divr.",
    )
    parser.add_argument(
        "--div_v",
        type=int,
        default=100,
        help="divv.",
    )
    parser.add_argument(
        "--dmin",
        type=float,
        default=0.05,
        help="dmin.",
    )
    parser.add_argument(
        "--k_vel",
        type=float,
        default=0.6,
        help="kvel.",
    )
   
    args = parser.parse_args()
    main(args)