# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:05:09 2020

@author: Oscar
"""
from numba import jit, cuda, prange
import numpy as np
import random 
import time
import argparse
##############################################
######### PARÁMETROS FÍSICOS  ################
##############################################
Q = 1
NUM_PARTICLES = 100000 #Número de partículas
NUM_PARTICLES_BULGE = int(0.14 * NUM_PARTICLES) #el 14% de la materia ordinaria es del bulbo
M_TOTAL = 3245*2.325*1e7 #masa total de las particulas q van a interactuar
M_PARTICLE = M_TOTAL / NUM_PARTICLES #masa de las particulas en Msolares
G = 4.518 * 1e-12 #constante de gravitación universal en Kpc, Msolares y Millones de años
T_SOL = 225 #periodo del Sol alrededor de la galaxia en Millones de años
##############################################
##############################################

##############################################
### FUNCIONES A UTILIZAR ###
##############################################
@jit(nopython=True, fastmath = True, parallel = False)
def factor_r(r):
    ah = 7.7
    return (1/r**3)*np.log(1+r/ah) - 1/(ah*r**2 + r**3)

@jit(nopython=True, fastmath = True, parallel = False)
def f_dark(x_vec, y_vec, z_vec, lim):
    r_dark = np.zeros(3)
    r_dark[0] = x_vec - lim/2
    r_dark[1] = y_vec - lim/2
    r_dark[2] = z_vec - lim/2
    #MODELO 3: Navarro-Frenk-White
    Mh = 12474 * 2.325*1e7
    r = np.sqrt(np.dot(r_dark, r_dark))
    g = np.zeros(3)
    if r > 0.0:
        factor = factor_r(r)
        g = -G*Mh*factor*r_dark
    return g

@jit(nopython=True, fastmath = True, parallel = False)
def pot_dark(x_vec, y_vec, z_vec, lim):
    r_dark = np.zeros(3)
    r_dark[0] = x_vec - lim/2
    r_dark[1] = y_vec - lim/2
    r_dark[2] = z_vec - lim/2
    #MODELO 3: Navarro-Frenk-White
    Mh = 12474 * 2.325*1e7
    ah = 7.7
    r = np.sqrt(np.dot(r_dark, r_dark))
    if r > 0.0:
        return (-G*Mh/r)*np.log(1 + r/ah)
    else:
        return 0.
    
@jit(nopython=True, fastmath = True, parallel = False)   
def pot_bulge(r_vec, lim):
    r_dark = np.zeros(3)
    r_dark[0] = r_vec[0] - lim/2
    r_dark[1] = r_vec[1] - lim/2
    r_dark[2] = r_vec[2] - lim/2
    #MODELO 3: Navarro-Frenk-White
    Mb = 1.05*1e10
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
    Md = 6.5*1e10
    bd = 0.308
    ad = 4.4
    r = np.sqrt(np.dot(r_dark, r_dark))
    return  -G*Md/(r**2 + (ad+bd)**2)**(1/2)

@jit(nopython=True, fastmath = True, parallel = False)
def CM(x_list, y_list, z_list):
    X_CM = np.sum(x_list * M_PARTICLE) / M_TOTAL
    Y_CM = np.sum(y_list * M_PARTICLE) / M_TOTAL
    Z_CM = np.sum(z_list * M_PARTICLE) / M_TOTAL
    return np.array([X_CM, Y_CM, Z_CM])

@jit(nopython=True, fastmath = True, parallel = False)
def fuerza_part(i, x_list, y_list, z_list, sumatorio_f, eps):
    fxi = np.zeros(NUM_PARTICLES)
    fyi = np.zeros(NUM_PARTICLES)
    fzi = np.zeros(NUM_PARTICLES)
    
    for j in range(NUM_PARTICLES):
        if j != i:
            rx = x_list[i] - x_list[j] 
            ry = y_list[i] - y_list[j] 
            rz = z_list[i] - z_list[j] 

            rij2 = rx*rx + ry*ry + rz*rz

            fxi[j] = rx * M_PARTICLE * (1 / (eps**2 + rij2))**(3 / 2)
            fyi[j] = ry * M_PARTICLE * (1 / (eps**2 + rij2))**(3 / 2)
            fzi[j] = rz * M_PARTICLE * (1 / (eps**2 + rij2))**(3 / 2)
            
    sumatorio_f[0] = np.sum(fxi)
    sumatorio_f[1] = np.sum(fyi)
    sumatorio_f[2] = np.sum(fzi)       
    return sumatorio_f * (-G)
    
@jit(nopython=True, fastmath = True, parallel = False)
def ener_pot(i, x_list, y_list, z_list, sumatorio_E, eps):

    for j in range(NUM_PARTICLES):
        if j != i:

            rx = x_list[i] - x_list[j] 
            ry = y_list[i] - y_list[j] 
            rz = z_list[i] - z_list[j] 
            
            rij2 = rx*rx + ry*ry + rz*rz

            sumatorio_E += M_PARTICLE**2 * (1 / (eps**2 + rij2))**(1 / 2)

    return sumatorio_E * (-G)

@jit(nopython = True, fastmath = True, parallel = False)
def ener_pot_calculator(x_list, y_list, z_list, eps, dark, lim):
    ener_list = np.zeros(NUM_PARTICLES)
    for i in range(NUM_PARTICLES):
        if dark:
            ener_list[i] = ener_pot(i, x_list, y_list, z_list, 0.0, eps) + M_PARTICLE*pot_dark(x_list[i], y_list[i], z_list[i], lim)
        else:
            ener_list[i] = ener_pot(i, x_list, y_list, z_list, 0.0, eps)  
    return ener_list

###########################################################################

#################################################################
# CONDICIONES INICIALES DE LAS PARTÍCULAS ###
#################################################################

def cond_inicial(lim, k_vel, eps, dark):
    
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
    x_list_0 = np.zeros((NUM_PARTICLES), dtype = float)
    y_list_0 = np.zeros((NUM_PARTICLES), dtype = float)
    z_list_0 = np.zeros((NUM_PARTICLES), dtype = float)

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
                
                x_list_0[i] = lim/2 + R*np.cos(phi)*np.sin(theta)
                y_list_0[i] = lim/2 + R*np.sin(phi)*np.sin(theta)
                z_list_0[i] = lim/2 + R*np.cos(theta)
                
                
            else:
        
                R, z = get_random_disk(max_disk)
                phi = random.uniform(0, 2*np.pi) 
                
                r_esf = [R, phi, z]
                r_esf_tot[i] = r_esf 
                
                x_list_0[i] = lim/2 + R*np.cos(phi)
                y_list_0[i] = lim/2 + R*np.sin(phi)
                z_list_0[i] = lim/2 + z

    
    fx_list_0 = np.zeros((NUM_PARTICLES), dtype = float) 
    fy_list_0 = np.zeros((NUM_PARTICLES), dtype = float) 
    fz_list_0 = np.zeros((NUM_PARTICLES), dtype = float) 
    
    for u in range(NUM_PARTICLES):
        sumatorio_f = np.array([0.,0.,0.])
        if dark:
            force = (fuerza_part(u, x_list_0, y_list_0, z_list_0, sumatorio_f, eps) 
                    + f_dark(x_list_0[u], y_list_0[u], z_list_0[u], lim))
            fx_list_0[u] = force[0]
            fy_list_0[u] = force[1]
            fz_list_0[u] = force[2]
        else:
            force = fuerza_part(u, x_list_0, y_list_0, z_list_0, sumatorio_f, eps) 
            fx_list_0[u] = force[0]
            fy_list_0[u] = force[1]
            fz_list_0[u] = force[2]
            

    vx_list_0 = np.zeros((NUM_PARTICLES), dtype = float)
    vy_list_0 = np.zeros((NUM_PARTICLES), dtype = float)
    vz_list_0 = np.zeros((NUM_PARTICLES), dtype = float)
    
   
    Rs = np.linspace(0, lim/2, 10000)
    NR = len(Rs)
    HR = lim / NR #lado de una celda
    
    phi_data = np.zeros(len(Rs))
    for i in range(len(phi_data)):
        r_vec = np.array([HR*(i+0.5) + lim/2, lim/2, lim/2])
        if dark:
            phi_data[i] += pot_dark(HR*(i+0.5) + lim/2, lim/2, lim/2, lim) + pot_bulge(r_vec, lim) + pot_disk(r_vec, lim)
        else:
            phi_data[i] += pot_bulge(r_vec, lim) + pot_disk(r_vec, lim)
    
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
            
            if dark:
                Ui = ( ener_pot(i, x_list_0, y_list_0, z_list_0, sumatorio_E, eps) 
                    + M_PARTICLE*pot_dark(x_list_0[i], y_list_0[i], z_list_0[i], lim) )
                E_pot += Ui
            else:
                Ui = ener_pot(i, x_list_0, y_list_0, z_list_0, sumatorio_E, eps) 
                E_pot += Ui
           

            v_esc = np.sqrt(-2*Ui/M_PARTICLE)

            #producto vectorial de g con ur (vector unitario radial)
            g_vec = np.array([fx_list_0[i], fy_list_0[i], fz_list_0[i]])
            
            R_centro = np.zeros(3)
            R_centro[0] = x_list_0[i] - lim/2
            R_centro[1] = y_list_0[i] - lim/2
            R_centro[2] = z_list_0[i] - lim/2
            
            R_norm = np.sqrt(np.dot(R_centro, R_centro))
            ur = R_centro / R_norm
            prod = abs(np.dot(g_vec, ur))
            
            v_circ = k_vel*np.sqrt(R_norm * prod)
            E_rot += 0.5*M_PARTICLE*v_circ**2 
            
            phi_g = r_esf_tot[i, 1]
    
            if i < NUM_PARTICLES_BULGE:
                
                theta_g = r_esf_tot[i, 2]    
                v_r = random.uniform(-0.3*v_esc, 0.3*v_esc)   
                v_z = v_circ*np.cos(theta_g)
                v_tan = v_circ*np.sin(theta_g)
                
                vx_list_0[i] = k_vel * (-v_tan * np.sin(phi_g) + v_r * np.cos(phi_g))
                vy_list_0[i] = k_vel * (v_tan * np.cos(phi_g) + v_r * np.sin(phi_g))
                vz_list_0[i] = k_vel * v_z
                
            else:
                
                if dark:
                    v_phi_med = np.sqrt(v2_phi(R_norm))
                    
                else:
                    v_phi_med = v_circular(R_norm)
                    
                sigma_R = np.sqrt(sigma2_R(R_norm))
                sigma_phi = np.sqrt(sigma2_phi(R_norm))
                sigma_z = np.sqrt(sigma2_z(R_norm))
                
                v_tan = random.gauss(v_phi_med, sigma_phi)  
                v_R = random.gauss(0, sigma_R)   
                v_z = random.gauss(0, sigma_z) 
                
                vx_list_0[i] = (-v_tan * np.sin(phi_g) + v_R * np.cos(phi_g))
                vy_list_0[i] = (v_tan * np.cos(phi_g) + v_R * np.sin(phi_g))
                vz_list_0[i] = v_z
                    
    print('Ostriker-Peebles criterion (t ~< 0.14): ', E_rot/abs(0.5*E_pot))
    
    return (x_list_0, y_list_0, z_list_0, vx_list_0, vy_list_0, vz_list_0, 
            fx_list_0, fy_list_0, fz_list_0)
    
#################################################################
#################################################################

#################################################################
#Se aplica el algoritmo de Leapfrog para resolver los primeros pasos de tiempo
#################################################################

@jit(nopython=True, fastmath = True, parallel = False)
def paso(x_list, y_list, z_list, vx_list, vy_list, vz_list, fx_list, fy_list, fz_list, dt, eps, dark, lim):
    
    x_list_new = x_list + dt*vx_list + 0.5 * dt**2 * fx_list
    y_list_new = y_list + dt*vy_list + 0.5 * dt**2 * fy_list
    z_list_new = z_list + dt*vz_list + 0.5 * dt**2 * fz_list
    
    fx_list_new = np.zeros((NUM_PARTICLES)) 
    fy_list_new = np.zeros((NUM_PARTICLES)) 
    fz_list_new = np.zeros((NUM_PARTICLES)) 
    
    for u in range(NUM_PARTICLES):
        sumatorio_f = np.array([0.,0.,0.])
        if dark:
            force = ( fuerza_part(u, x_list_new, y_list_new, z_list_new, sumatorio_f, eps)  
                     + f_dark(x_list_new[u], y_list_new[u], z_list_new[u], lim) )
        else:
            force = fuerza_part(u, x_list_new, y_list_new, z_list_new, sumatorio_f, eps)  
        
        fx_list_new[u] = force[0]
        fy_list_new[u] = force[1]
        fz_list_new[u] = force[2]
        
    vx_list_new = vx_list + 0.5*dt*(fx_list_new + fx_list)
    vy_list_new = vy_list + 0.5*dt*(fy_list_new + fy_list)
    vz_list_new = vz_list + 0.5*dt*(fz_list_new + fz_list)
    
    return (x_list_new, y_list_new, z_list_new, vx_list_new, vy_list_new, 
            vz_list_new, fx_list_new, fy_list_new, fz_list_new)
 
################################################################# 
#################################################################


#####################################################################################################
#####################################################################################################  
# FUNCIÓN QUE VA A OBTENER LAS TRAYECTORIAS Y LAS CARACTERÍSTICAS PARA CADA PASO DE TIEMPO
# LLAMANDO A LOS MÉTODOS DE RESOLUCIÓN PASO
#####################################################################################################
#####################################################################################################

def tiempo(x_list, y_list, z_list, vx_list, vy_list, vz_list, fx_list, fy_list, fz_list, 
           n, n_r, n_v, div_r, div_v, dt, eps, dark, lim):
    #Lista de trayectorias de todas las partículas
    tray = np.empty((n_r, NUM_PARTICLES, 3))
    tray_CM = np.empty((n_r, 3))
    vels = np.empty((n_v, NUM_PARTICLES, 3))
    eners = np.empty((n_v, NUM_PARTICLES))

    for k in range(n):
        
        R_CM = CM(x_list, y_list, z_list)
        
        if k == 1:
            t0 = time.time()
            x_list, y_list, z_list, vx_list, vy_list, vz_list, fx_list, fy_list, fz_list = paso(x_list, y_list, z_list, vx_list, vy_list, vz_list, fx_list, fy_list, fz_list, dt, eps, dark, lim)
            tf = time.time()
            print("El programa va a tardar:", int(n*(tf-t0)/60),"minutos")
        else:
            x_list, y_list, z_list, vx_list, vy_list, vz_list, fx_list, fy_list, fz_list = paso(x_list, y_list, z_list, vx_list, vy_list, vz_list, fx_list, fy_list, fz_list, dt, eps, dark, lim)
            
        if k%div_r == 0.0:
            k_r = int(k // div_r)
            tray[k_r, :, 0] = x_list[:]
            tray[k_r, :, 1] = y_list[:]
            tray[k_r, :, 2] = z_list[:]
            tray_CM[k_r, :] = R_CM[:]

        if k%div_v == 0.0:
            k_v = int(k // div_v)
            print((k/n)*100, "%")
            vels[k_v, :, 0] = vx_list[:]
            vels[k_v, :, 1] = vy_list[:]
            vels[k_v, :, 2] = vz_list[:]
            ener_list = ener_pot_calculator(x_list, y_list, z_list, eps, dark, lim)
            eners[k_v, :] = ener_list[:]
            
    return tray, tray_CM, vels, eners
      
####################################################################################
####################################################################################
####################################################################################


#############################
# MAIN
###########################

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
    DARK = args.dark
    
    simulation_parameters = np.array([NUM_PARTICLES, LIM, TIME_STEPS, DIV_R, DIV_V, DT, EPS, K_VEL, DARK])
    np.savetxt('parameters.dat', simulation_parameters)
    ##############################################
    ##############################################
    x_list_0, y_list_0, z_list_0, vx_list_0, vy_list_0, vz_list_0, fx_list_0, fy_list_0, fz_list_0 = cond_inicial(LIM, K_VEL, EPS, DARK)

    t0 = time.time()
    trayectorias, trayectoria_CM, velocidades, energia_pot = tiempo(x_list_0, y_list_0, z_list_0, vx_list_0, 
                                                                    vy_list_0, vz_list_0, fx_list_0, fy_list_0, 
                                                                    fz_list_0, TIME_STEPS, N_R, N_V, DIV_R, 
                                                                    DIV_V, DT, EPS, DARK, LIM)  
    tf = time.time()

    print('El programa ha tardado: ', ((tf-t0)/60), 'minutos en completar las trayectorias.')
    trayectorias3D = trayectorias.reshape(trayectorias.shape[0], -1) 
    velocidades3D = velocidades.reshape(velocidades.shape[0], -1) 

    np.savetxt('trayectorias.dat', trayectorias3D, fmt = '%.3e') #fmt es cuantas cifras decimales
    np.savetxt('trayectoria_CM.dat', trayectoria_CM, fmt = '%.3e') #fmt es cuantas cifras decimales
    np.savetxt('velocidades.dat', velocidades3D, fmt = '%.3e') #fmt es cuantas cifras decimales
    np.savetxt('energia_pot.dat', energia_pot, fmt = '%.3e')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FB")

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
    parser.add_argument(
        "--dark",
        type=bool,
        default=False,
        help="dark_matter.",
    )
   
    args = parser.parse_args()
    main(args)




