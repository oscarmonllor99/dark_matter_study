import numpy as np
from numba import jit
import random
import time
import argparse

##############################################
######### PARÁMETROS FÍSICOS  ################
##############################################
Q = 1.3
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
"""
@jit(nopython=True, fastmath = True, parallel = False)
def W(d, h):
    if d <= h:
        return 1 - d/h
    else:
        return 0
"""
@jit(nopython=True, fastmath = True, parallel = False)
def W(d, h):
    if d <= h/2:
        return 3/4 - (d/h)**2
    elif h/2 <= d <= 3*h/2:
        return 0.5*(3/2 - d/h)**2
    else:
        return 0


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
def CM(r_list):
    return np.sum(r_list * M_PARTICLE, axis=0) / M_TOTAL

@jit(nopython=True, fastmath = True)
def fuerza(r_list, g, i, dark, Np, h, lim):

    r_list_i = r_list[i, :]
    #identificamos en qué celda está la partícula
    x_pos =  int(r_list_i[0]/h)
    y_pos =  int(r_list_i[1]/h)
    z_pos =  int(r_list_i[2]/h)


    if x_pos <= 1 or x_pos >= Np-1 or y_pos <= 1 or y_pos >= Np-1 or z_pos <= 1 or z_pos >= Np-1:

        if dark:
            r_centro = np.array([lim/2, lim/2, lim/2])
            return ( -G*M_TOTAL*(r_list_i - r_centro)/np.linalg.norm(r_list_i - r_centro)**3 
                    + f_dark(r_list_i, lim) )
        else:
            r_centro = np.array([lim/2, lim/2, lim/2])
            return -G*M_TOTAL*(r_list[i] - r_centro)/np.linalg.norm(r_list[i] - r_centro)**3 
        
        
    else:
        
        fuerza_i = np.array([0., 0., 0.])
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                        dx = abs((x_pos + x + 0.5)*h - r_list[i,0])
                        dy = abs((y_pos + y + 0.5)*h - r_list[i,1])
                        dz = abs((z_pos + z + 0.5)*h - r_list[i,2])
                        fuerza_i += ( g[:, x_pos + x, y_pos + y, z_pos + z] 
                                      * W(dx, h) * W(dy, h) * W(dz, h) )
        #r_vec = r_list_i - lim/2
        
        if dark:
            return fuerza_i + f_dark(r_list_i, lim)
        else:
            return fuerza_i
        


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

@jit(nopython=True, fastmath = True)
def poisson(rho, phi, Np, h):
    w = 0.95
    tol = 1e-4
    acabar = False
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
        if max_diff < tol:
            acabar = True
    return phi

@jit(nopython=True, fastmath = True)
def gradiente(phi, Np, h):
    g = np.zeros((3, Np, Np, Np))
    for i in range(1, Np-1):
        for j in range(1, Np-1):
            for k in range(1, Np-1):
                g[0,i,j,k] = - (phi[i+1, j, k] - phi[i-1, j, k])/(2*h)
                g[1,i,j,k] = - (phi[i, j+1, k] - phi[i, j-1, k])/(2*h)
                g[2,i,j,k] = - (phi[i, j, k+1] - phi[i, j, k-1])/(2*h)
    return g
    
@jit(nopython = True, fastmath = True, parallel = False)
def ener_pot_calculator(r_list, phi, dark, lim, Np, h):
    ener_list = np.zeros(NUM_PARTICLES)
    for i in range(NUM_PARTICLES):
        
        x_pos = int(r_list[i,0] // h)
        y_pos = int(r_list[i,1] // h)
        z_pos = int(r_list[i,2] // h)
        
        if (x_pos <= 0 or x_pos >= Np-1 or y_pos <= 0 or y_pos >= Np-1 or 
        z_pos <= 0 or z_pos >= Np-1):
            
            if dark:
                r_centro = np.array([lim/2, lim/2, lim/2])
                ener_list[i] = (-G*M_PARTICLE*M_TOTAL/np.linalg.norm(r_list[i]-r_centro) 
                                + M_PARTICLE*pot_dark(r_list[i], lim))
            else:
                r_centro = np.array([lim/2, lim/2, lim/2])
                ener_list[i] = -G*M_PARTICLE*M_TOTAL/np.linalg.norm(r_list[i]-r_centro)
        
        else:
            
            pot_i = 0.
            for x in range(-1, 2):
                for y in range(-1, 2):
                    for z in range(-1, 2):
                            dx = abs((x_pos + x + 0.5)*h - r_list[i,0])
                            dy = abs((y_pos + y + 0.5)*h - r_list[i,1])
                            dz = abs((z_pos + z + 0.5)*h - r_list[i,2])
                            pot_i += (phi[x_pos + x, y_pos + y, z_pos + z] 
                                          * W(dx, h) * W(dy, h) * W(dz, h) )
            
            if dark:
                
                ener_list[i] = (M_PARTICLE*pot_i + M_PARTICLE*pot_dark(r_list[i], lim))
            else:
                
                ener_list[i] = M_PARTICLE*pot_i 
                
    return ener_list

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

    for i in range(NUM_PARTICLES):
             
            if i < NUM_PARTICLES_BULGE:

                R = get_random_bulge(max_bulge)
                while R>lim/2-1:
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
                while R>lim/2-1 or z>lim/2-1:
                    R, z = get_random_disk(max_disk)
                    
                phi = random.uniform(0, 2*np.pi) 
                
                r_esf = [R, phi, z]
                r_esf_tot[i] = r_esf 
                
                r_list_0[i, 0] = lim/2 + R*np.cos(phi)
                r_list_0[i, 1] = lim/2 + R*np.sin(phi)
                r_list_0[i, 2] = lim/2 + z

            
    rho = densidad(r_list_0, Np, h)
    phi = poisson(rho, phi0, Np, h)
    g = gradiente(phi, Np, h)
    
    Rs = np.linspace(0, lim/2, 10000)
    NR = len(Rs)
    HR = lim / NR #lado de una celda
    
    phi_data = np.zeros(len(Rs))
    for i in range(len(phi_data)):
                r_vec = np.array([HR*(i+0.5) + lim/2, lim/2, lim/2])

                if dark:
                    phi_data[i] += pot_dark(r_vec, lim) + pot_bulge(r_vec, lim) + pot_disk(r_vec, lim)
                else:
                    phi_data[i] += pot_bulge(r_vec, lim) + pot_disk(r_vec, lim)

                
    phi1_data = np.gradient(phi_data, HR)
    phi2_data = np.gradient(phi1_data, HR)
    

    v_list_0 = np.zeros((NUM_PARTICLES, 3), dtype = float)
    
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
    
    E_rot = 0
    E_pot = 0
    for i in range(NUM_PARTICLES):

        x_pos =  int(r_list_0[i,0] // h)
        y_pos =  int(r_list_0[i,1] // h)
        z_pos =  int(r_list_0[i,2] // h)

        if dark:
            phii = ( pot_dark(r_list_0[i,:], lim) + pot_bulge(r_list_0[i,:], lim) 
                    + pot_disk(r_list_0[i,:], lim) )
            E_pot += M_PARTICLE*phii
        else:
            phii = pot_bulge(r_list_0[i,:], lim) + pot_disk(r_list_0[i,:], lim)
            E_pot += M_PARTICLE*phii


        E_pot += M_PARTICLE*phii
        
        v_esc = np.sqrt(-2 * phii)
        
        R_norm = r_esf_tot[i,0]
        v_circ = k_vel*v_circular(R_norm)
        
        phi_g = r_esf_tot[i, 1]

        if i < NUM_PARTICLES_BULGE:
                
                theta_g = r_esf_tot[i, 2]    
                v_r = random.uniform(-0.3*v_esc, 0.3*v_esc)   
                v_z = v_circ*np.cos(theta_g)
                v_tan = v_circ*np.sin(theta_g)
                E_rot += 0.5*M_PARTICLE*v_tan**2 
                
                v_list_0[i, 0] = k_vel * (-v_tan * np.sin(phi_g) + v_r * np.cos(phi_g))
                v_list_0[i, 1] = k_vel * (v_tan * np.cos(phi_g) + v_r * np.sin(phi_g))
                v_list_0[i, 2] = k_vel * v_z
                
        else:

                v_phi_med = np.sqrt(v2_phi(R_norm))

                sigma_R = np.sqrt(sigma2_R(R_norm))
                sigma_phi = np.sqrt(sigma2_phi(R_norm))
                sigma_z = np.sqrt(sigma2_z(R_norm))
                
                v_tan = random.gauss(v_phi_med, sigma_phi)  
                v_R = random.gauss(0, sigma_R)   
                v_z = random.gauss(0, sigma_z) 
                
                E_rot += 0.5*M_PARTICLE*v_tan**2 
                
                v_list_0[i, 0] = (-v_tan * np.sin(phi_g) + v_R * np.cos(phi_g))
                v_list_0[i, 1] = (v_tan * np.cos(phi_g) + v_R * np.sin(phi_g))
                v_list_0[i, 2] = v_z

    f_list_0 = np.zeros((NUM_PARTICLES, 3))
    for i in range(NUM_PARTICLES):
        force = fuerza(r_list_0, g, i, dark, Np, h, lim)
        f_list_0[i, 0] = force[0]
        f_list_0[i, 1] = force[1]
        f_list_0[i, 2] = force[2]

    print('Ostriker-Peebles criterion (t ~< 0.14): ', E_rot/abs(0.5*E_pot))
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
    g = gradiente(phi, Np, h)
    
    f_list_new = force_update(r_list_new, g, dark, Np, h, lim)

    v_list_new = v_list + 0.5*dt*(f_list_new + f_list)

    return r_list_new, v_list_new, f_list_new, phi

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
    eners = np.empty((n_v, NUM_PARTICLES))
    
    for k in range(n):
        #Estos son los indices de paso de tiempo para guardar r y v

        R_CM = CM(r_list)
        
        if k == 1:
            t0 = time.time()
            r_list, v_list, f_list, phi = paso(r_list, v_list, f_list, dt, eps, dark, lim,
                                          Np, h, phi0)
            tf = time.time()
            print("El programa va a tardar:", int(n*(tf-t0)/60),"minutos")
        else:
            r_list, v_list, f_list, phi = paso(r_list, v_list, f_list, dt, eps, dark, lim,
                                         Np, h, phi0)
            
        if k%div_r == 0.0:
            k_r = int(k // div_r)
            tray[k_r, :, :] = r_list[:, :]
            tray_CM[k_r, :] = R_CM[:]

        if k%div_v == 0.0:
            k_v = int(k // div_v)
            print((k/n)*100, "%")
            vels[k_v, :, :] = v_list[:, :]
            ener_list = ener_pot_calculator(r_list, phi, dark, lim, Np, h)
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
    DT = T_SOL / 100 #intervalo de tiempo entre cada paso
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

    trayectorias, trayectoria_CM, velocidades, energia_pot = tiempo(r_list_0, v_list_0, f_list_0, TIME_STEPS, 
                                                       N_R, N_V, DIV_R, DIV_V, DT, EPS, DARK, LIM,
                                                       NP, H, PHI0)  
    tf = time.time()

    print('El programa ha tardado: ', ((tf-t0)/60), 'minutos en completar las trayectorias.')
    trayectorias3D = trayectorias.reshape(trayectorias.shape[0], -1) 
    velocidades3D = velocidades.reshape(velocidades.shape[0], -1) 

    np.savetxt('trayectorias.dat', trayectorias3D, fmt = '%.3e') #fmt es cuantas cifras decimales
    np.savetxt('trayectoria_CM.dat', trayectoria_CM, fmt = '%.3e') 
    np.savetxt('velocidades.dat', velocidades3D, fmt = '%.3e') 
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
        default=1000,
        help="timesteps.",
    )
    parser.add_argument(
        "--div_r",
        type=int,
        default=50,
        help="divr.",
    )
    parser.add_argument(
        "--div_v",
        type=int,
        default=50,
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
        default=1,
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




