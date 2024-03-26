# -*- coding: utf-8 -*-
'''
RBD model for reproducing the Dzhanibekov effect also call The Tennis-Racket Paradox or intermediate axis theorem

Algorithm coupled with the Journal's paper: DOI:.......

@autor: ntrivisonno, lgarelli, mstorti
@date: APR2024
'''

import math as math
import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt

import modelo
import utiles
import os

from parameters import parameters
from fluid_prop import fluid_prop
from initial_cond import initial_cond
import plot_data as plt_data
import save_data as sv

def main():
    '''
    aca va le prog principal
    '''
    print "#########################"
    print "Reading parameters."
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    m, diam, xcg, ycg, zcg, Ixx, Iyy, Izz, steps, dt = parameters('./Data/data.dat')
    print "Mass=", m, "[kg]"
    print "Diam=", diam, "[m]"
    print "xcg, ycg, zcg=", xcg, ycg, zcg, "[m]"
    print "Ixx, Iyy, Izz=", Ixx, Iyy, Izz, "[m^4]"
    print "Steps, Dt=", steps, dt

    print "#########################"
    print "Reading fluid properties."
    rho, mu, c = fluid_prop(0, 0)
    print "Density=", rho, "[kg/m3]"
    print "mu=", mu, "[kg/m s]"
    print "c=", c, "[m/s]"

    print "#########################"
    print "Reading Initial Conditions."
    V, alpha, beta, p, q, r, phi, theta, psi, XE, YE, ZE = initial_cond('./Data/Initial_cond.dat')
    print "Velocity=", V, "[m/s]"
    print "alpha, beta=", alpha, beta, "[deg]"
    print "p,q,r=", p, q, r, "[RPM]"
    print "phi, theta, psi=", phi, theta, psi, "[deg]"
    print "XE, YE, ZE=", XE, YE, ZE, "[m]"

    MYDIR = ("Results")
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("Result'sFolder created: ", MYDIR)

    else:
        print(MYDIR, "Result folder already created.")


    # Transform deg->rad and RPM->rad/s
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)

    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    psi = np.deg2rad(psi)

    p = p * 2 * math.pi / 60
    q = q * 2 * math.pi / 60
    r = r * 2 * math.pi / 60

    Ts = dt  # Temporal interval
    N = steps # Total numerical simulation time is Ttot = N*Ts

    # Inital State
    x0 = np.zeros(13)
    q0 = utiles.Quaternion.fromEulerAngles(roll=phi, pitch=theta, yaw=psi)
    x0[6] = q0.d
    x0[7:10] = q0.v
    x0[0:3] += [XE, YE, ZE]
    x0[3] = V*math.cos(alpha)*math.cos(beta)
    x0[4] = V*math.sin(beta)
    x0[5] = V*math.sin(alpha)*math.cos(beta)
    x0[10] = p
    x0[11] = q
    x0[12] = r

    x = np.zeros((N+1,13))
    x[0] = x0
    u = np.zeros((N,4)) 
    t = np.arange(0, N+1) * Ts
    for k in range(N):
        x[k + 1] = sp.integrate.odeint(lambda _x, _t: modelo.ED_cuaterniones(_x, u[k], k, _t), x[k], [k*Ts, (k+1)*Ts],
                                       rtol=1e-6, atol=1e-6)[-1]
        t_N = k+1


        if x[t_N,2] < -1.0:
            print "#########################"
            print "Cut-off, floor impact, t=",t[t_N],'[s]'
            break
        if (k % 5000) == 0:
            sv.save_data(t_N,t[0:t_N+1],x[0:t_N+1,:],Ixx,Iyy,Izz)

    print "#########################"
    print "General Results"
    print "Maximum Height =", np.amax(x[:,2])  , "[m]"
    print "Distance =", math.sqrt((x[t_N,0]**2+x[t_N,1]**2)) , "[m]"
    print "Last state Height =", t[t_N], "[s]", (x[t_N,2]) , "[m]"

    quat = utiles.Quaternion(x[t_N,6], x[t_N,7:10])
    velocidad_ned = quat.rotate_vector(x[t_N,3:6])

    print "Last state velocity Vx,Vy,Vz=", velocidad_ned , "[m/s]"
    print "Last state distance Xned,Yned,Zned=", x[t_N,0:3] , "[m]"

    sv.save_data(t_N,t[0:t_N+1],x[0:t_N+1,:],Ixx,Iyy,Izz)

    plt_data.plot_data(t_N, t[0:t_N+1], x[0:t_N+1,:], phi, theta, psi)

    plt.show()
    print "#--------------------------------------------'"
    print "\n FINISHED, OK!"
    
if __name__ == "__main__":
    main()
