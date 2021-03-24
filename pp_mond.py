# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:05:09 2020

@author: Oscar
"""
from numba import jit, prange
import numpy as np
import random 
import time
import argparse

##############################################
######### PARÁMETROS FÍSICOS  ################
##############################################
Q = 1.5
NUM_PARTICLES = 4000 #Número de partículas
NUM_PARTICLES_BULGE = int(0.14 * NUM_PARTICLES) #el 14% de la materia ordinaria es del bulbo
M_TOTAL = 3245*2.325*1e7 #masa total de las particulas q van a interactuar
M_PARTICLE = M_TOTAL / NUM_PARTICLES #masa de las particulas en Msolares
G = 4.518 * 1e-12 #constante de gravitación universal en Kpc, Msolares y Millones de años
T_SOL = 225 #periodo del Sol alrededor de la galaxia en Millones de años
##############################################
##############################################

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

@jit(nopython=True, fastmath = True, parallel = False)   
def pot_bulge(r_vec, lim):
    r_dark = np.zeros(3)
    r_dark[0] = r_vec[0] - lim/2
    r_dark[1] = r_vec[1] - lim/2
    r_dark[2] = r_vec[2] - lim/2
    #MODELO 3: Navarro-Frenk-White
    Mb = 443 * 2.325*1e7
    bb = 0.267
    r = np.sqrt(np.dot(r_dark, r_dark))
    return  - G*Mb / (r**2 + bb**2)**(1/2)

@jit(nopython=True, fastmath = True, parallel = False)   
def pot_disk(r_vec, lim):
    r_dark = np.zeros(3)
    r_dark[0] = r_vec[0] - lim/2
    r_dark[1] = r_vec[1] - lim/2
    r_dark[2] = r_vec[2] - lim/2
    #MODELO 3: Navarro-Frenk-White
    Md = 2798 * 2.325*1e7
    bd = 0.308
    ad = 4.4
    r = np.sqrt(np.dot(r_dark, r_dark))
    return  -G*Md/(r**2 + (ad+bd)**2)**(1/2)

##############################################
### FUNCIONES A UTILIZAR ###
##############################################
@jit(nopython=True, fastmath = True, parallel = False)
def interpolation(y):
    return 0.5*( 1 + (1+4*y**(-1))**(1/2) )


@jit(nopython=True, fastmath = True, parallel = False)
def CM(r_list):
    return np.sum(r_list * M_PARTICLE, axis=0) / M_TOTAL

@jit(nopython=True, fastmath = True, parallel = False)
def fuerza_part(i, r_list, sumatorio_f, eps):
    for j in range(NUM_PARTICLES):
        if j != i:
            
            rx = r_list[i, 0] - r_list[j, 0] 
            ry = r_list[i, 1] - r_list[j, 1] 
            rz = r_list[i, 2] - r_list[j, 2] 

            rij2 = rx*rx + ry*ry + rz*rz

            sumatorio_f[0] += rx * M_PARTICLE * (1 / (eps**2 + rij2))**(3 / 2)
            sumatorio_f[1] += ry * M_PARTICLE * (1 / (eps**2 + rij2))**(3 / 2)
            sumatorio_f[2] += rz * M_PARTICLE * (1 / (eps**2 + rij2))**(3 / 2)

    return sumatorio_f * (-G)
    
@jit(nopython=True, fastmath = False, parallel = False)
def ener_pot(i, r_list, sumatorio_E, eps):

    for j in range(NUM_PARTICLES):
        if j != i:

            rx = r_list[i, 0] - r_list[j, 0] 
            ry = r_list[i, 1] - r_list[j, 1] 
            rz = r_list[i, 2] - r_list[j, 2] 

            rij2 = rx*rx + ry*ry + rz*rz

            sumatorio_E += M_PARTICLE**2 * (1 / (eps**2 + rij2))**(1 / 2)

    return sumatorio_E * (-G)

@jit(nopython = True, fastmath = True, parallel = False)
def ener_pot_calculator(r_list, eps):
    ener_list = np.zeros(NUM_PARTICLES)
    for i in range(NUM_PARTICLES):
        ener_list[i] = ener_pot(i, r_list, 0.0, eps)  
    return ener_list

###########################################################################
###########################################################################


#################################################################
# CONDICIONES INICIALES DE LAS PARTÍCULAS ###
#################################################################

def cond_inicial(lim, k_vel, eps):

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
        return ( (a*R**2 + (a + 3*(z**2 + b**2)**(1/2))
                  *(a + (z**2+b**2)**(1/2))**2 ) 
                / (( R**2 + (a + (z**2 + b**2)**(1/2))**2)**(5/2) 
                   * (z**2 + b**2)**(3/2)) )
    
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
    r_esf_tot = np.zeros((NUM_PARTICLES, 3), dtype = float)
    E_rot = 0
    E_pot = 0

    for i in range(NUM_PARTICLES):
             
            if i < NUM_PARTICLES_BULGE:

                R = get_random_bulge(max_bulge)
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2*np.pi)
                
                r_esf = [R, phi, theta]
                r_esf_tot[i] = r_esf 
                
                r_list_0[i, 0] = lim/2 + R*np.cos(phi)*np.sin(theta)
                r_list_0[i, 1] = lim/2 + R*np.sin(phi)*np.sin(theta)
                r_list_0[i, 2] = lim/2 + R*np.cos(theta)
                
                
            else:
        
                R, z = get_random_disk(max_disk)
                phi = random.uniform(0, 2*np.pi) 
                
                r_esf = [R, phi, z]
                r_esf_tot[i] = r_esf 
                
                r_list_0[i, 0] = lim/2 + R*np.cos(phi)
                r_list_0[i, 1] = lim/2 + R*np.sin(phi)
                r_list_0[i, 2] = lim/2 + z
            

    f_list_0 = np.zeros((NUM_PARTICLES, 3), dtype = float)
 
    
    for u in range(NUM_PARTICLES):
        a_0 = 3.87*1e-3
        sumatorio_f = np.array([0.,0.,0.])
        fuerza = fuerza_part(u, r_list_0, sumatorio_f, eps)
        f_mod = np.sqrt(fuerza[0]**2 + fuerza[1]**2 + fuerza[2]**2)
        nu = interpolation(f_mod/a_0)
        f_list_0[u, :] = fuerza[:]*nu
            

    #distribución aleatoria de las velocidades, teniendo siempre velocidad tangencial
    #se distribuyen las velocidades según la velocidad de escape dada para cada posición   

    v_list_0 = np.zeros((NUM_PARTICLES, 3))
    
    Rs = np.linspace(0, lim/2, 10000)
    NR = len(Rs)
    HR = lim / NR #lado de una celda
    
    phi_data = np.zeros(len(Rs))
    for i in range(len(phi_data)):
        r_vec = np.array([HR*(i+0.5) + lim/2, lim/2, lim/2])
        phi_data[i] += pot_dark(r_vec, lim) + pot_bulge(r_vec, lim) + pot_disk(r_vec, lim)

    
    phi1_data = np.gradient(phi_data, HR)
    phi2_data = np.gradient(phi1_data, HR)
    
    def k_ep(R):
            R_pos = int(R/HR)
            phi1_k = phi1_data[R_pos]
            phi2_k = phi2_data[R_pos]
            return np.sqrt(((3/R) * phi1_k) + (phi2_k))
    
    
    def v_circular(R):
            R_pos = int(R/HR)
            phi1_k = phi1_data[R_pos]
            return np.sqrt(R*phi1_k)
    
    
    def omega(R):
            R_pos = int(R/HR)
            phi1_k = phi1_data[R_pos]
            return np.sqrt(abs((1/R)*phi1_k))
    
    
    
    def sigma(R):
        Md = 2798 * 2.325*1e7
        hs = 2.43
        return Md/(2*np.pi*hs**2) * np.exp(-R/hs)
    
    def sigma2_R(R):
        return (Q*3.36*G*sigma(R) / k_ep(R))**2
    
        
    def sigma2_phi(R):
        return  sigma2_R(R) * (k_ep(R)/(2*omega(R)))**2
    
    def sigma2_z(R):
        z0  = 0.26
        return  np.pi*G*sigma(R)*z0
    
    def v2_phi(R):
        hs = 2.43
        return  v_circular(R)**2 + sigma2_R(R) * (1 - (k_ep(R)/(2*omega(R)))**2 - 2*R/hs)
    
    for i in range(NUM_PARTICLES):

            
            ###################################################
            #Cálculo de la energía potencial para v escape
            ###################################################
            sumatorio_E = 0.

            Ui = ener_pot(i, r_list_0, sumatorio_E, eps)  
            E_pot += Ui
            
            v_esc = np.sqrt(-2*Ui/M_PARTICLE)

            #producto vectorial de g con ur (vector unitario radial)
            g_vec = f_list_0[i]
            
            R_centro = np.zeros(3)
            R_centro[0] = r_list_0[i, 0] - lim/2
            R_centro[1] = r_list_0[i, 1] - lim/2
            R_centro[2] = r_list_0[i, 2] - lim/2
            
            R_norm =  np.linalg.norm(R_centro)
            ur = R_centro / R_norm
            prod =  abs(np.inner(g_vec, ur))
            
            v_circ = k_vel*np.sqrt(R_norm * prod)
            E_rot += 0.5*M_PARTICLE*v_circ**2 
            
            phi_g = r_esf_tot[i, 1]
    
            if i < NUM_PARTICLES_BULGE:
                
                theta_g = r_esf_tot[i, 2]    
                v_r = random.uniform(-0.1*v_esc, 0.1*v_esc)   
                v_z = v_circ*np.cos(theta_g)
                v_tan = v_circ*np.sin(theta_g)
                
                v_list_0[i, 0] = k_vel * (-v_tan * np.sin(phi_g) + v_r * np.cos(phi_g))
                v_list_0[i, 1] = k_vel * (v_tan * np.cos(phi_g) + v_r * np.sin(phi_g))
                v_list_0[i, 2] = k_vel * v_z
                
            else:
                
                sigma_R = np.sqrt(sigma2_R(R_norm))
                v_phi_med = np.sqrt(v2_phi(R_norm))
                sigma_phi = np.sqrt(sigma2_phi(R_norm))
                sigma_z = np.sqrt(sigma2_z(R_norm))
                
                v_tan = random.gauss(v_phi_med, sigma_phi)  
                v_R = random.gauss(0, sigma_R)   
                v_z = random.gauss(0, sigma_z) 

                v_list_0[i, 0] = (-v_tan * np.sin(phi_g) + v_R * np.cos(phi_g))
                v_list_0[i, 1] = (v_tan * np.cos(phi_g) + v_R * np.sin(phi_g))
                v_list_0[i, 2] = v_z

    print('Ostriker-Peebles criterion (t ~< 0.14): ', E_rot/abs(0.5*E_pot))
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
    
    f_list_new = np.zeros((NUM_PARTICLES, 3))
    
    for u in range(NUM_PARTICLES):
        sumatorio_f = np.array([0.,0.,0.])
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


def tiempo(r_list, v_list, f_list, n, n_r, n_v, div_r, div_v, dt, eps):
    #Lista de trayectorias de todas las partículas
    tray = np.empty((n_r, NUM_PARTICLES, 3))
    tray_CM = np.empty((n_r, 3))
    vels = np.empty((n_v, NUM_PARTICLES, 3))
    eners = np.empty((n_v, NUM_PARTICLES))

    for k in range(n):
        #Estos son los indices de paso de tiempo para guardar r y v

        R_CM = CM(r_list)
        
        if k == 1:
            t0 = time.time()
            r_list, v_list, f_list = paso(r_list, v_list, f_list, dt, eps)
            tf = time.time()
            print("El programa va a tardar:", int(n*(tf-t0)/60),"minutos")
        else:
            r_list, v_list, f_list= paso(r_list, v_list, f_list, dt, eps)
            
        if k%div_r == 0.0:
            k_r = int(k // div_r)
            tray[k_r, :, :] = r_list[:, :]
            tray_CM[k_r, :] = R_CM[:]

        if k%div_v == 0.0:
            k_v = int(k // div_v)
            print((k/n)*100, "%")
            vels[k_v, :, :] = v_list[:, :]
            ener_list = ener_pot_calculator(r_list, eps)
            eners[k_v, :] = ener_list[:]

    return tray, tray_CM, vels, eners

####################################################################################
####################################################################################
####################################################################################

###########################################################################
# MAIN
###########################################################################

def main(args):
    ##############################################
    ### PARÁMETROS DE SIMULACIÓN
    ##############################################
    LIM = args.lim #en kpc
    TIME_STEPS = args.time_step #número de pasos totales de tiempo 
    DIV_R = args.div_r
    DIV_V = args.div_v
    N_R = int(TIME_STEPS // DIV_R) #numero de pasos de tiempo guardados para r
    N_V = int(TIME_STEPS // DIV_V) #numero de pasos de tiempo guardados para v
    DT = T_SOL / 200 #intervalo de tiempo entre cada paso
    D_MIN = args.dmin #distancia minima a la cual se debe calcular la fuerza en kpc
    EPS = np.sqrt(2*D_MIN / 3**(3/2)) 
    K_VEL= args.k_vel#parámetro de control de momento angular inicial (0--> velocidad angular inicial 0
    #                                                          1--> velocidad angular inicial máxima)
    simulation_parameters = np.array([NUM_PARTICLES, LIM, TIME_STEPS, DIV_R, DIV_V, DT, EPS, K_VEL])
    np.savetxt('parameters.dat', simulation_parameters, fmt = '%.5e')
    ##############################################
    ##############################################

    r_list_0, v_list_0, f_list_0 = cond_inicial(LIM, K_VEL, EPS)

    t0 = time.time()
    trayectorias, trayectoria_CM, velocidades, energia_pot = tiempo(r_list_0, v_list_0, f_list_0, TIME_STEPS, N_R, N_V, DIV_R, DIV_V, DT, EPS)
    tf = time.time()

    print('El programa ha tardado: ', ((tf-t0)/60), 'minutos en completar las trayectorias.')
    trayectorias3D = trayectorias.reshape(trayectorias.shape[0], -1) 
    velocidades3D = velocidades.reshape(velocidades.shape[0], -1) 

    np.savetxt('trayectorias.dat', trayectorias3D, fmt = '%.3e') #fmt es cuantas cifras decimales
    np.savetxt('trayectoria_CM.dat', trayectoria_CM, fmt = '%.3e') #fmt es cuantas cifras decimales
    np.savetxt('velocidades.dat', velocidades3D, fmt = '%.3e') #fmt es cuantas cifras decimales
    np.savetxt('energia_pot.dat', energia_pot, fmt = '%.3e')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FB MOND")

    parser.add_argument(
        "--lim",
        type=float,
        default=100.0,
        help="en kpc.",
    )
    parser.add_argument(
        "--time_step",
        type=int,
        default=2000,
        help="timesteps.",
    )
    parser.add_argument(
        "--div_r",
        type=int,
        default=25,
        help="divr.",
    )
    parser.add_argument(
        "--div_v",
        type=int,
        default=25,
        help="divv.",
    )
    parser.add_argument(
        "--dmin",
        type=float,
        default=1,
        help="dmin.",
    )
    parser.add_argument(
        "--k_vel",
        type=float,
        default=1.0,
        help="kvel.",
    )
   
    args = parser.parse_args()
    main(args)


