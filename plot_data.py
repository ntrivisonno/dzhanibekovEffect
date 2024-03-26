# -*- coding: utf-8 -*-
'''
@autor: ntrivisonno, lgarelli, mstorti
@date: APR2024
'''

import numpy as np
import matplotlib.pyplot as plt
import utiles
import math as math

# Evolution of the states

def plot_data(N, t, x, phi, theta, psi):

    fig_size = (12,4)

    fig, axs = plt.subplots(1,3, figsize=fig_size)
    fig.canvas.set_window_title('Distance Inertial Frame')
    fig.suptitle('Distance Inertial Frame (z=height)')
    for k in range(3):
        axs[k].plot(t, x[:,k])
        axs[k].grid()
    axs[0].set_title('x')
    axs[1].set_title('y')
    axs[2].set_title('z')

    fig, axs = plt.subplots(1,3, figsize=fig_size)
    fig.canvas.set_window_title('Velocity Body Frame')
    fig.suptitle('Velocity Body Frame')
    for k in range(3):
        axs[k].plot(t, x[:,k+3])
        axs[k].grid()
    axs[0].set_title('u')
    axs[1].set_title('v')
    axs[2].set_title('w')

    velocidad_ned = np.empty((N+1,3))
    for k, xk in enumerate(x):
        quat = utiles.Quaternion(xk[6], xk[7:10])
        velocidad_ned[k] = quat.rotate_vector(xk[3:6])

    fig, axs = plt.subplots(1,4, figsize=fig_size)
    fig.canvas.set_window_title('Velocity NED Frame')
    fig.suptitle('Velocity NED Frame')
    for k in range(3):
        axs[k].plot(t, velocidad_ned[:,k])
        axs[k].grid()
    axs[3].plot(t, np.linalg.norm(velocidad_ned,axis=1))
    axs[3].grid()
    axs[0].set_title('vx')
    axs[1].set_title('vy')
    axs[2].set_title('vz')
    axs[3].set_title('mag(V)')

    if 1:
        fig, axs = plt.subplots(1, 4, figsize=fig_size)
        fig.canvas.set_window_title(u'Orientation quaternion')
        fig.suptitle(u'Orientation quaternion')
        for k in range(4):
            axs[k].plot(t, x[:,k+6])
            axs[k].grid()
        axs[0].set_title('qe')
        axs[1].set_title('qv1')
        axs[2].set_title('qv2')
        axs[3].set_title('qv3')

    if 1:
        fig, axs = plt.subplots(1, 3, figsize=fig_size)
        fig.canvas.set_window_title('Angular velocity Body Frame')
        fig.suptitle('Angular velocity Body Frame')
        for k in range(3):
            axs[k].plot(t, x[:,k+10])
            axs[k].grid()
        axs[0].set_title('p')
        axs[1].set_title('q')
        axs[2].set_title('r')

    # Euler Angles as a transformation of the quaternions
    euler_angles = np.empty((N+1,3))
    for k, xk in enumerate(x):
        quat = utiles.Quaternion(xk[6], xk[7:10])
        euler_angles[k] = quat.get_euler_anles()

    if 1:
        fig, axs = plt.subplots(1, 3, figsize=fig_size)
        fig.canvas.set_window_title(u'Euler angles')
        fig.suptitle(u'Euler angles (Deg)')
        for k in range(3):
            axs[k].plot(t, euler_angles[:,k]*180/math.pi)
            axs[k].grid()
        axs[0].set_title('roll (phi)')
        axs[1].set_title('pitch (theta)')
        axs[2].set_title('yaw (psi)')

    if 1:
        fig, axs = plt.subplots(1, 3, figsize=fig_size)
        fig.canvas.set_window_title('Angle of Attack, yaw and total angle')
        fig.suptitle('Angles')
        vt = np.linalg.norm(velocidad_ned, axis=1)
        # AoA
        alfa = np.arctan2(x[:, 5], x[:, 3])
        alfad = alfa * 180 / math.pi
        axs[0].plot(t, alfad)
        axs[0].grid()
        # Yaw
        beta = np.arcsin(x[:, 4]/vt)
        betad = beta * 180 / math.pi
        axs[1].plot(t, betad)
        axs[1].grid()
        sin_alfa_t = np.sqrt(((np.sin(beta)) ** 2 + (np.cos(beta)) ** 2 * (np.sin(alfa)) ** 2))
        alfa_t = np.arcsin(sin_alfa_t)
        alfa_t_d = alfa_t * 180 / math.pi
        axs[2].plot(t, alfa_t_d)
        axs[2].grid()
        axs[0].set_title('Alpha')
        axs[1].set_title('Beta')
        axs[2].set_title('Alpha total')

    plt.show(block=False)

    #plt.show()

    return

