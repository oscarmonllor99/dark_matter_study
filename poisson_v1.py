import numpy as np
from numba import jit
import random
import time
##############################################
######### PARÁMETROS FÍSICOS  ################
##############################################

N = 20000 #Número de partículas

M = 3245*2.325*1e7#masa total de las particulas q van a interactuar: fraccion no significativa de la total galactica.

m = M / N #masa de las particulas en Msolares

G = 4.518 * 1e-12 #constante de gravitación universal en Kpc, Msolares y Millones de años

T_sol = 225 #periodo del Sol alrededor de la galaxia en Millones de años

##############################################
##############################################

##############################################
### PARÁMETROS DE SIMULACIÓN
##############################################

lim = 100 #en kpc

x_lim = lim
y_lim = lim
z_lim = lim

n = 10000  #número de pasos totales de tiempo
div_r = 100
div_v = 100
n_r = int(n / div_r) #numero de pasos de tiempo guardados para r
n_v = int(n / div_v) #numero de pasos de tiempo guardados para v
dt = T_sol / 2000 #intervalo de tiempo entre cada paso

#Parámetros para calcular el potencial y su gradiente

N_p = 100 #número de pasos común a los dos ejes, es decir, número de celdas para calcular
#el gradiente de la gravedad
n_p = N_p #número de celdas en el eje X
m_p = N_p #número de celdas en el eje Y
l_p = 50 #número de celdas en el eje Z

hx = x_lim / n_p #distancia entre puntos de red eje X
hy = y_lim / m_p #distancia entre puntos de red eje Y
hz = z_lim / l_p #distancia entre puntos de red eje Z

dark = False
print("Se consdiera la materia oscura?: ", dark)
k_vel = 0.95 #parámetro de control de momento angular inicial (0--> velocidad angular inicial 0
#                                                          1--> velocidad angular inicial máxima)
##############################################
##############################################
r0 = np.zeros((n_p, m_p, l_p))

@jit(nopython=True, fastmath = True)
def bordes(r):
    for i in range(n_p):
        for j in range(m_p):
            for k in range(l_p):
                r[i, j, k] = np.sqrt(((i-n_p/2)*hx)**2 + ((j-m_p/2)*hy)**2 + ((k-l_p/2)*hz)**2)
    return r

r = bordes(r0)
##############################################
### FUNCIONES A UTILIZAR ###
##############################################

@jit(nopython=True, fastmath = True, parallel = False)
def factor_r(r):
    ah = 7.7
    return (1/r**3)*np.log(1+r/ah) - 1/(ah*r**2 + r**3)

@jit(nopython=True, fastmath = True, parallel = False)
def f_dark(r_vec):
    r_dark = np.zeros(3)
    r_dark[0] = r_vec[0] - x_lim/2
    r_dark[1] = r_vec[1] - y_lim/2
    r_dark[2] = r_vec[2] - z_lim/2
    #MODELO 3: Navarro-Frenk-White
    Mh = 12474 * 2.325*1e7
    r = np.sqrt(np.dot(r_dark, r_dark))
    g = np.zeros(3)
    if r > 0.0:
        factor = factor_r(r)
        g = -G*Mh*factor*r_dark
    return g

@jit(nopython=True, fastmath = True, parallel = False)
def pot_dark(r_vec):
    r_dark = np.zeros(3)
    r_dark[0] = r_vec[0] - x_lim/2
    r_dark[1] = r_vec[1] - y_lim/2
    r_dark[2] = r_vec[2] - z_lim/2
    #MODELO 3: Navarro-Frenk-White
    Mh = 12474 * 2.325*1e7
    ah = 7.7
    r = np.sqrt(np.dot(r_dark, r_dark))
    if r > 0.0:
        return (-G*Mh/r)*np.log(1 + r/ah)
    else:
        return 0

def CM(r_list):
    M = N*m

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

def fuerza(r_list, g, i):

    r_list_i = r_list[i, :]
    
    #identificamos en qué celda está la partícula
    x_pos =  int(r_list_i[0]/hx)
    y_pos =  int(r_list_i[1]/hy)
    z_pos =  int(r_list_i[2]/hz)

    if x_pos <= 0 or x_pos >= N_p or y_pos <= 0 or y_pos >= N_p or z_pos <= 0 or z_pos >= l_p:

        return np.array([0,0,0])

    else:

        if dark:
            return g[:, x_pos, y_pos, z_pos] + f_dark(r_list_i)
        else:
            return g[:, x_pos, y_pos, z_pos]

def densidad(r_list_0):
    rho = np.zeros((n_p, m_p, l_p), dtype = float)
    for i in range(len(r_list_0)):
        
        x_pos = int(r_list_0[i,0] / hx)
        y_pos = int(r_list_0[i,1] / hy)
        z_pos = int(r_list_0[i,2] / hz)
        
        if x_pos <= 0 or x_pos >= N_p or y_pos <= 0 or y_pos >= N_p or z_pos <= 0 or z_pos >= l_p:
            
            pass
        
        else:
        
            rho[x_pos, y_pos, z_pos] += m / (hx*hy*hz)

    return rho

@jit(nopython=True, fastmath = True)
def poisson(rho, phi):
    w = 0.95
    pasos = 100
    for u in range(pasos):
        for i in range(1, n_p-1):
            for j in range(1, m_p-1):
                for k in range(1, l_p-1):
                    phi[i, j, k] = (1+w)*(1/(4/hx**2 + 2/hz**2))*((1/hx**2)*(phi[i+1, j, k] + phi[i-1, j, k]
                                                                         + phi[i, j+1, k] + phi[i, j-1, k])
                                                                         + (1/hz**2)*(phi[i, j, k+1] + phi[i, j, k-1])
                                                                         - 4*np.pi*G*rho[i, j, k]) - w*phi[i, j, k]

    return phi

def gradiente(phi):
    
    x = np.arange(0, x_lim, hx) 
    y = np.arange(0, y_lim, hy)
    z = np.arange(0, z_lim, hz)
    
    #GRAVEDAD DEBIDA AL POTENCIAL
    g = np.gradient(phi, x, y, z)
    g = - np.array(g)  #el menos viene de que g = - grad (phi)
    #El array es del tipo g[componente][NpuntosX][NpuntosY][NpuntosZ]
    
    return g
###########################################################################
###########################################################################

def cond_inicial():

    r_list_0 = np.zeros((N, 3))

    for i in range(N):

        R = 0.3 + 10*random.expovariate(1)
        while R > 40:
            R = 0.3 + 10*random.expovariate(1)
            
        
        theta = 2*np.pi*random.random()
        z = 0.5 - random.random()
        
        r_list_0[i, 0] = x_lim/2 + R * np.cos(theta)
        r_list_0[i, 1] = y_lim/2 + R * np.sin(theta)
        r_list_0[i, 2] = z_lim/2 + z

    rho = densidad(r_list_0)
    
    phi0 =np.zeros((n_p, m_p, l_p), dtype = float) 
    #PLANOS DE CONDICIONES DE FRONTERA
    phi0[:, :, 0] = -G*M/(r[:, :, 0])
    phi0[:, :, l_p-1] = -G*M/(r[:, :, l_p-1])
    phi0[:, 0, :] = -G*M/(r[:, 0, :])
    phi0[:, N_p-1, :] = -G*M/(r[:, N_p-1, :])
    phi0[0, :, :] = -G*M/(r[0, :, :])
    phi0[N_p-1, :, :] = -G*M/(r[N_p-1, :, :])
    
    phi = poisson(rho, phi0)
    g = gradiente(phi)

    v_list_0 = np.zeros((N, 3))
    
    for i in range(N):

        x_pos =  int(r_list_0[i,0] // hx)
        y_pos =  int(r_list_0[i,1] // hy)
        z_pos =  int(r_list_0[i,2] // hz)
        
        if dark:
            phii = phi[x_pos, y_pos, z_pos] + pot_dark(r_list_0[i, :])
            
        else:
            phii = phi[x_pos, y_pos, z_pos]
            
        v_esc = np.sqrt(-2 * phii)
        
        if 8 <= np.linalg.norm(r_list_0[i]) <= 10:
            v_R = random.uniform(-0.4*v_esc, 0.4*v_esc)
        else:
            v_R = random.uniform(-0.1*v_esc, 0.1*v_esc)
            
        v_z = random.uniform(-0.05*v_esc, 0.05*v_esc) 


        #producto de g con ur (vector unitario radial)
        if dark:
            g_vec = g[:, x_pos, y_pos, z_pos] + f_dark(r_list_0[i])
            
        else:
            g_vec = g[:, x_pos, y_pos, z_pos]
        

        R_centro = np.zeros(3)
        R_centro[0] = r_list_0[i, 0] - x_lim/2
        R_centro[1] = r_list_0[i, 1] - y_lim/2
        R_centro[2] = r_list_0[i, 2] - z_lim/2

        R_norm = np.sqrt(np.dot(R_centro, R_centro))
        ur = R_centro / R_norm
        prod = abs(np.dot(g_vec, ur))

        v_tan = np.sqrt(R_norm * prod)

        theta_g = np.arctan((r_list_0[i][1] - y_lim/2) / (r_list_0[i][0] - x_lim/2))

        ###################################
        #arregle a la función arctan, para que se ajuste a nuestras necesidades
        ###################################
        if (r_list_0[i][1] - y_lim/2) < 0:
                
                if (r_list_0[i][0] - x_lim/2) < 0:
                    
                    theta_g = np.pi + theta_g
                    
                elif (r_list_0[i][0] - x_lim/2) < 0: 
                    
                    theta_g = np.pi + theta_g
                
        elif (r_list_0[i][1] - y_lim/2) > 0:
                
                if (r_list_0[i][0]  - x_lim/2) < 0:
                    
                    theta_g = np.pi + theta_g

        v_list_0[i, 0] = k_vel * (-v_tan * np.sin(theta_g) + v_R * np.cos(theta_g))
        v_list_0[i, 1] = k_vel * (v_tan * np.cos(theta_g) + v_R * np.sin(theta_g))
        v_list_0[i, 2] = k_vel * v_z


    f_list_0 = np.zeros((N, 3))
    for i in range(N):
        force = fuerza(r_list_0, g, i)
        f_list_0[i, :] = force[:]


    return r_list_0, v_list_0, f_list_0


#################################################################
#################################################################

#################################################################
#Se aplica el algoritmo de Leapfrog para resolver los primeros pasos de tiempo
#################################################################

def paso(r_list, v_list, f_list):

    r_list_new = r_list + dt*v_list + 0.5 * dt**2 * f_list
 
    rho = densidad(r_list_new)
    
    phi0 =np.zeros((n_p, m_p, l_p), dtype = float) 
    #PLANOS DE CONDICIONES DE FRONTERA
    phi0[:, :, 0] = -G*M/(r[:, :, 0])
    phi0[:, :, l_p-1] = -G*M/(r[:, :, l_p-1])
    phi0[:, 0, :] = -G*M/(r[:, 0, :])
    phi0[:, N_p-1, :] = -G*M/(r[:, N_p-1, :])
    phi0[0, :, :] = -G*M/(r[0, :, :])
    phi0[N_p-1, :, :] = -G*M/(r[N_p-1, :, :])
    
    phi = poisson(rho, phi0)

    g = gradiente(phi)

    f_list_new = np.zeros((N, 3))

    for i in range(N):
        force = fuerza(r_list_new, g, i)
        f_list_new[i, 0] = force[0]
        f_list_new[i, 1] = force[1]
        f_list_new[i, 2] = force[2]

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

def tiempo():

    tray = np.empty((n_r, N, 3), dtype = float)
    tray_CM = np.empty((n_r, 3), dtype = float)
    vels = np.empty((n_v, N, 3), dtype = float)

    r_list, v_list, f_list = cond_inicial()

    for k in range(n):
        #Estos son los indices de paso de tiempo para guardar r y v
        k_r = int(k // div_r)
        k_v = int(k // div_v)

        R_CM = CM(r_list)

        if k == 1:
            t0 = time.time()
            r_list, v_list, f_list= paso(r_list, v_list, f_list)
            tf = time.time()
            print("El programa va a tardar:", int(n*(tf-t0)/60),"minutos")
        else:
            r_list, v_list, f_list= paso(r_list, v_list, f_list)

        if k%div_r == 0.0:
 
            tray[k_r, :, :] = r_list[:, :]

            tray_CM[k_r, :] = R_CM[:]

        if k%div_v == 0.0:
            print((k/n)*100, "%")

            vels[k_v, :, :] = v_list[:, :]

    return tray, tray_CM, vels


###########################################################################
# Guaradamos todas las trayectorias obtenidas con el tiempo en tray
###########################################################################

###########################################################################
# Guaradamos todas las trayectorias obtenidas con el tiempo en tray
###########################################################################
t0 = time.time()
trayectorias, trayectoria_CM, velocidades = tiempo()    
tf = time.time()

print('El programa ha tardado: ', (tf-t0)/60, 'minutos en completar las trayectorias.')
trayectorias3D = trayectorias.reshape(trayectorias.shape[0], -1) 
velocidades3D = velocidades.reshape(velocidades.shape[0], -1) 

np.savetxt('trayectorias.dat', trayectorias3D, fmt = '%.3e') #fmt es cuantas cifras decimales
np.savetxt('trayectoria_CM.dat', trayectoria_CM, fmt = '%.3e') #fmt es cuantas cifras decimales
np.savetxt('velocidades.dat', velocidades3D, fmt = '%.3e') #fmt es cuantas cifras decimales


