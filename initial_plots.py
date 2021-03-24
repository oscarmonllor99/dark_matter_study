#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 14:57:22 2021

@author: oscar
"""
from numba import jit
import numpy as np
import matplotlib.pyplot as plt


G = 4.518 * 1e-12 #constante de gravitación universal en Kpc, Msolares y Millones de años
dark = False
Q = 1


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

lim = 50 #tamaño de la caja en kpc


Rs = np.linspace(0.001, lim/2, 10000)
NP = len(Rs)
H = lim / NP #lado de una celda

def k_ep(R):
        R_pos = int(R/H)
        phi1_k = phi1_data[R_pos]
        phi2_k = phi2_data[R_pos]
        return np.sqrt(((3/R) * phi1_k) + (phi2_k))


def v_circular(R):
        R_pos = int(R/H)
        phi1_k = phi1_data[R_pos]
        return np.sqrt(R*phi1_k)


def omega(R):
        R_pos = int(R/H)
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

def v_phi(R):
    hs = 2.43
    return  v_circular(R)**2 + sigma2_R(R) * (1 - (k_ep(R)/(2*omega(R)))**2 - 2*R/hs)
    


phi_data = np.zeros(len(Rs))
for i in range(NP):
            r_vec = np.array([H*(i+0.5) + lim/2, lim/2, lim/2])
            if dark:
                phi_data[i] += pot_dark(r_vec, lim) + pot_bulge(r_vec, lim) + pot_disk(r_vec, lim)
            else:
                phi_data[i] += pot_bulge(r_vec, lim) + pot_disk(r_vec, lim)

phi1_data = np.gradient(phi_data, H)
phi2_data = np.gradient(phi1_data, H)
plt.plot(Rs, phi1_data)
plt.plot(Rs, phi2_data)


k_ep_res = np.zeros(len(Rs))
sigma_res = np.zeros(len(Rs))
sigma2_R_res = np.zeros(len(Rs))
sigma2_z_res = np.zeros(len(Rs))
sigma2_phi_res = np.zeros(len(Rs))
vcirc_res = np.zeros(len(Rs))
omega_res = np.zeros(len(Rs))
v_phi_res = np.zeros(len(Rs))
for i in range(len(Rs)):
    R = Rs[i]
    k_ep_res[i] = k_ep(R)
    sigma_res[i] = sigma(R)
    sigma2_R_res[i] = sigma2_R(R)
    sigma2_z_res[i] = sigma2_z(R)
    sigma2_phi_res[i] = sigma2_phi(R)
    vcirc_res[i] = v_circular(R)
    omega_res[i] = omega(R)
    v_phi_res[i] = np.sqrt(v_phi(R))

plt.figure()
plt.plot(Rs, k_ep_res)
plt.title('k')

plt.figure()
plt.plot(Rs, sigma_res)
plt.title('dens_suf')

plt.figure()
plt.plot(Rs, sigma2_R_res)
plt.title('sigmaR')

plt.figure()
plt.plot(Rs, sigma2_phi_res)
plt.title('sigmaphi')

plt.figure()
plt.plot(Rs, sigma2_z_res)
plt.title('sigmaz')


plt.figure()
plt.plot(Rs, v_phi_res/(1.022*1e-3))
plt.title('v_phi_res')

plt.figure()
plt.plot(Rs, vcirc_res/(1.022*1e-3))
plt.title('v')

plt.figure()
plt.plot(Rs, omega_res)
plt.title('omega')