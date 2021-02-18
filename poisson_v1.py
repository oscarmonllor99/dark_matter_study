import numpy as np
from numba import jit
import random
import time
import argparse
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
##############################################


##############################################
### FUNCIONES A UTILIZAR ###
##############################################

@jit(nopython=True, fastmath = True, parallel = False)
def factor_r(r):
    ah = 7.7
    return (1/r**3)*np.log(1+r/ah) - 1/(ah*r**2 + r**3)

@jit(nopython=True, fastmath = True, parallel = False)
def f_dark(r_vec, lim):
    r_dark = np.zeros(3)
    r_dark[0] = r_vec[0] - lim/2
    r_dark[1] = r_vec[1] - lim/2
    r_dark[2] = r_vec[2] - lim/2
    #MODELO 3: Navarro-Frenk-White
    Mh = 12474 * 2.325*1e7
    r = np.sqrt(np.dot(r_dark, r_dark))
    g = np.zeros(3)
    if r > 0.0:
        factor = factor_r(r)
        g = -G*Mh*factor*r_dark
    return g

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
def CM(r_list):
    return np.sum(r_list * M_PARTICLE, axis=0) / M_TOTAL

@jit(nopython=True, fastmath = True)
def fuerza(r_list, g, i, dark, Np, h, lim):

    r_list_i = r_list[i, :]
    #identificamos en qué celda está la partícula
    x_pos =  int(r_list_i[0]/h)
    y_pos =  int(r_list_i[1]/h)
    z_pos =  int(r_list_i[2]/h)

    if x_pos <= 0 or x_pos >= Np or y_pos <= 0 or y_pos >= Np or z_pos <= 0 or z_pos >= Np:

        return np.zeros(3)

    else:

        if dark:
            return g[:, x_pos, y_pos, z_pos] + f_dark(r_list_i, lim)
        else:
            return g[:, x_pos, y_pos, z_pos]
        
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

def gradiente(phi, h, lim):
    
    x = np.arange(0, lim, h) 
    y = np.arange(0, lim, h)
    z = np.arange(0, lim, h)
    
    g = np.array(np.gradient(phi, x, y, z))
    #El array es del tipo g[componente][NpuntosX][NpuntosY][NpuntosZ]
    return -g
###########################################################################
###########################################################################

def cond_inicial(lim, k_vel, eps, dark, Np, h, phi0):

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
    r_esf_tot = np.zeros((NUM_PARTICLES, 3), dtype = float)
    E_rot = 0
    E_pot = 0

    for i in range(NUM_PARTICLES):
             
            if i < NUM_PARTICLES_BULGE:

                R = get_random_bulge(max_bulge)
                while R>49:
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
                while R>49 or z>49:
                    R, z = get_random_disk(max_disk)
                    
                phi = random.uniform(0, 2*np.pi) 
                
                r_esf = [R, phi, z]
                r_esf_tot[i] = r_esf 
                
                r_list_0[i, 0] = lim/2 + R*np.cos(phi)
                r_list_0[i, 1] = lim/2 + R*np.sin(phi)
                r_list_0[i, 2] = lim/2 + z

            
    rho = densidad(r_list_0, Np, h)
    
    phi = poisson(rho, phi0, Np, h)
    g = gradiente(phi, h, lim)

    v_list_0 = np.zeros((NUM_PARTICLES, 3), dtype = float)
    
    for i in range(NUM_PARTICLES):

        x_pos =  int(r_list_0[i,0] // h)
        y_pos =  int(r_list_0[i,1] // h)
        z_pos =  int(r_list_0[i,2] // h)
        
        if dark:
            phii = phi[x_pos, y_pos, z_pos] + pot_dark(r_list_0[i, :], lim)
            E_pot += M_PARTICLE*phii
        else:
            phii = phi[x_pos, y_pos, z_pos]
            E_pot += M_PARTICLE*phii
            
        v_esc = np.sqrt(-2 * phii)

        #producto de g con ur (vector unitario radial)
        if dark:
            g_vec = g[:, x_pos, y_pos, z_pos] + f_dark(r_list_0[i], lim)
            
        else:
            g_vec = g[:, x_pos, y_pos, z_pos]
        

        R_centro = np.zeros(3)
        R_centro[0] = r_list_0[i, 0] - lim/2
        R_centro[1] = r_list_0[i, 1] - lim/2
        R_centro[2] = r_list_0[i, 2] - lim/2

        R_norm = np.sqrt(np.dot(R_centro, R_centro))
        ur = R_centro / R_norm
        prod = abs(np.dot(g_vec, ur))
        
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
                
                v_tan = v_circ
                v_R = random.uniform(-0.1*v_esc, 0.1*v_esc)   
                v_z = random.uniform(-0.01*v_esc, 0.01*v_esc)
                
                v_list_0[i, 0] = (-v_tan * np.sin(phi_g) + v_R * np.cos(phi_g))
                v_list_0[i, 1] = (v_tan * np.cos(phi_g) + v_R * np.sin(phi_g))
                v_list_0[i, 2] = v_z

    f_list_0 = np.zeros((NUM_PARTICLES, 3))
    for i in range(NUM_PARTICLES):
        force = fuerza(r_list_0, g, i, dark, Np, h, lim)
        f_list_0[i, 0] = force[0]
        f_list_0[i, 1] = force[1]
        f_list_0[i, 2] = force[2]

    print('Ostriker-Peebles criterion (t ~< 0.14): ', E_rot/abs(E_pot))
    return r_list_0, v_list_0, f_list_0


#################################################################
#################################################################

#################################################################
#Se aplica el algoritmo de Leapfrog para resolver los primeros pasos de tiempo
#################################################################

@jit(nopython=True, fastmath = True)
def force_update(r_list_new, g, dark, Np, h, lim):
    f_list_new = np.zeros((NUM_PARTICLES, 3))
    for i in range(NUM_PARTICLES):
        force = fuerza(r_list_new, g, i, dark, Np, h, lim)
        f_list_new[i, 0] = force[0]
        f_list_new[i, 1] = force[1]
        f_list_new[i, 2] = force[2]
    return f_list_new 


def paso(r_list, v_list, f_list, dt, eps, dark, lim, 
          Np, h, phi0):

    r_list_new = r_list + dt*v_list + 0.5 * dt**2 * f_list
 
    rho = densidad(r_list_new, Np, h)
    phi = poisson(rho, phi0, Np, h)
    g = gradiente(phi, h, lim)
    
    f_list_new = force_update(r_list_new, g, dark, Np, h, lim)

    v_list_new = v_list + 0.5*dt*(f_list_new + f_list)

    return r_list_new, v_list_new, f_list_new

#################################################################
#################################################################

#####################################################################################################
#####################################################################################################
#####################################################################################################

#####################################################################################################
# FUNCIÓN QUE VA A OBTENER LAS TRAYECTORIAS Y LAS CARACTERÍSTICAS PARA CADA PASO DE TIEMPO
# LLAMANDO A LOS MÉTODOS DE RESOLUCIÓN PASO
#####################################################################################################
#####################################################################################################


def tiempo(r_list, v_list, f_list, n, n_r, 
           n_v, div_r, div_v, dt, eps, dark, lim,
            Np, h, phi0):
    
    #Lista de trayectorias de todas las partículas
    tray = np.empty((n_r, NUM_PARTICLES, 3))
    tray_CM = np.empty((n_r, 3))
    vels = np.empty((n_v, NUM_PARTICLES, 3))

    for k in range(n):
        #Estos son los indices de paso de tiempo para guardar r y v

        R_CM = CM(r_list)
        
        if k == 1:
            t0 = time.time()
            r_list, v_list, f_list = paso(r_list, v_list, f_list, dt, eps, dark, lim,
                                          Np, h, phi0)
            tf = time.time()
            print("El programa va a tardar:", int(n*(tf-t0)/60),"minutos")
        else:
            r_list, v_list, f_list= paso(r_list, v_list, f_list, dt, eps, dark, lim,
                                         Np, h, phi0)
            
        if k%div_r == 0.0:
            k_r = int(k // div_r)
            tray[k_r, :, :] = r_list[:, :]
            tray_CM[k_r, :] = R_CM[:]

        if k%div_v == 0.0:
            k_v = int(k // div_v)
            print((k/n)*100, "%")
            vels[k_v, :, :] = v_list[:, :]

    return tray, tray_CM, vels
      
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
    DT = T_SOL / 2000 #intervalo de tiempo entre cada paso
    D_MIN = args.dmin #distancia minima a la cual se debe calcular la fuerza en kpc
    EPS = np.sqrt(2*D_MIN / 3**(3/2)) 
    K_VEL= args.k_vel#parámetro de control de momento angular inicial (0--> velocidad angular inicial 0
    #                                                          1--> velocidad angular inicial máxima)
    DARK = args.dark #hay materia oscura o no
    NP = args.Nc #celdas en cada eje
    H = LIM / NP #distancia entre puntos de red eje X
    
    simulation_parameters = np.array([NUM_PARTICLES, LIM, TIME_STEPS, DIV_R, DIV_V, DT, NP, EPS, K_VEL, DARK])
    np.savetxt('parameters.dat', simulation_parameters, fmt = '%.5e')
    ##############################################
    ##############################################
    
    R0 = np.zeros((NP, NP, NP)) #para las condiciones de contorno del potencial
    @jit(nopython=True, fastmath = True)
    def bordes(r):
        for i in range(NP):
            for j in range(NP):
                for k in range(NP):
                    r[i, j, k] = np.sqrt(((i-NP/2)*H)**2 + ((j-NP/2)*H)**2 + ((k-NP/2)*H)**2)
        return r
    R = bordes(R0)
    
    PHI0 = np.zeros((NP, NP, NP)) 
    #PLANOS DE CONDICIONES DE FRONTERA
    PHI0[:, :, 0] = -G*M_TOTAL/(R[:, :, 0])
    PHI0[:, :, NP-1] = -G*M_TOTAL/(R[:, :, NP-1])
    PHI0[:, 0, :] = -G*M_TOTAL/(R[:, 0, :])
    PHI0[:, NP-1, :] = -G*M_TOTAL/(R[:, NP-1, :])
    PHI0[0, :, :] = -G*M_TOTAL/(R[0, :, :])
    PHI0[NP-1, :, :] = -G*M_TOTAL/(R[NP-1, :, :])
    
    
    r_list_0, v_list_0, f_list_0 = cond_inicial(LIM, K_VEL, EPS, DARK, NP, H, PHI0)

    t0 = time.time()

    trayectorias, trayectoria_CM, velocidades = tiempo(r_list_0, v_list_0, f_list_0, TIME_STEPS, 
                                                       N_R, N_V, DIV_R, DIV_V, DT, EPS, DARK, LIM,
                                                       NP, H, PHI0)  
    tf = time.time()

    print('El programa ha tardado: ', ((tf-t0)/60), 'minutos en completar las trayectorias.')
    trayectorias3D = trayectorias.reshape(trayectorias.shape[0], -1) 
    velocidades3D = velocidades.reshape(velocidades.shape[0], -1) 

    np.savetxt('trayectorias.dat', trayectorias3D, fmt = '%.3e') #fmt es cuantas cifras decimales
    np.savetxt('trayectoria_CM.dat', trayectoria_CM, fmt = '%.3e') #fmt es cuantas cifras decimales
    np.savetxt('velocidades.dat', velocidades3D, fmt = '%.3e') #fmt es cuantas cifras decimales


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
        default=1.0,
        help="kvel.",
    )
    parser.add_argument(
        "--dark",
        type=bool,
        default=True,
        help="dark_matter.",
    )
    parser.add_argument(
        "--Nc",
        type=int,
        default=100,
        help="x_cells.",
    )
   
    args = parser.parse_args()
    main(args)




