# -*- coding: utf-8 -*-
'''
@autor: ntrivisonno, lgarelli, mstorti
'''
import numpy as np
import scipy as sp
import utiles
import math as math

from parameters import parameters
from fluid_prop import fluid_prop


m, diam, xcg, ycg, zcg, Ixx, Iyy, Izz, steps, dt = parameters('./Data/data.dat')

inertia_tensor = np.array([[Ixx, 0., 0.],
                           [0., Iyy, 0.],
                           [0., 0., Izz]])

# Superficie Ref
S = math.pi * (0.5 * diam) ** 2

def ED_cuaterniones(x, u, k, t):
    '''
    :param x: Vector de estados
    :param u: Vector de acciones de control => podria incidir incidir sobre las fins
    :return: Vector de derivadas de los estados
    x[0] -> x_ned (north)
    x[1] -> y_ned (east)
    x[2] -> h_ned = -z_ned (z_ned = down)
    x[3] -> u (vel x en marco body)
    x[4] -> v (vel y en marco body)
    x[5] -> w (vel z en marco body)
    x[6] -> q_e (parte escalar del cuaternión de orientación)
    x[7] -> q_v1 (primera componente de la parte vectorial del cuaternión de orientación)
    x[8] -> q_v2 (segunda componente de la parte vectorial del cuaternión de orientación)
    x[9] -> q_v3 (tercera componente de la parte vectorial del cuaternión de orientación)
    x[10] -> p (1era componente de la velocidad angular en marco body)
    x[11] -> q (2da componente de la velocidad angular en marco body)
    x[12] -> r (3era componente de la velocidad angular en marco body)
    x[13] -> alfa
    x[14] -> beta
    '''
    # t = absolute time
    x_prima = np.zeros_like(x)

    q_body2ned = utiles.Quaternion(x[6], x[7:10])
    Q_body2ned = q_body2ned.calc_rot_matrix()

    #--- Kinematic Eq---#

    x_prima[0:3] = Q_body2ned.dot(x[3:6])  
    x_prima[2] *= -1. # Compute Height  instead of 'depth'

    [phi, theta, psi] = q_body2ned.get_euler_anles()
    q_prima = q_body2ned.mult_cuat_times_vec(np.array([-x[12]*np.tan(theta),x[11],x[12]]) * .5)
    x_prima[6] = q_prima.d
    x_prima[7:10] = q_prima.v

    # Fluid properties
    rho, mu, c = fluid_prop(x[2], 0)

    # --- Dynamic Eq ---#

    # por si consideramos viento no nulo. Son las componentes del vector viento en el marco fijo.
    velWind_ned = np.zeros(3)
    # vector vel viento en marco body
    velWind_body = Q_body2ned.T.dot(velWind_ned)
    # velocidad relativa en marco body
    vel_rel_body = x[3:6] - velWind_body
     vt = np.linalg.norm(vel_rel_body)
    # AoA
    alpha = np.arctan2(vel_rel_body[2], vel_rel_body[0])
    alphad = alphaa * 180 / math.pi
    # Yaw
    beta = np.arctan2(vel_rel_body[1], np.sqrt(vel_rel_body[0]**2 + vel_rel_body[2]**2))
    betad = beta * 180 / math.pi

    sin_alpha_t = math.sqrt(((math.sin(beta)) ** 2 + (math.cos(beta)) ** 2 * (math.sin(alpha)) ** 2))
    alpha_t = math.asin(math.sqrt(((math.sin(beta)) ** 2 + (math.cos(beta)) ** 2 * (math.sin(alpha)) ** 2)))

    ca = np.cos(alpha)
    cb = np.cos(beta)
    sa = np.sin(alpha)
    sb = np.sin(beta)

    mach = vt / c

    x_prima[3] = x[12] * x[4] - x[11] * x[5]
    x_prima[4] = -x[12]*np.tan(theta) * x[5]
    x_prima[5] = x[11] * x[3] - x[10] * x[4]

    x_prima[10:13] = sp.linalg.solve(inertia_tensor, np.cross(inertia_tensor.dot(x[10:13]),
                                                              np.array([-x[12]*np.tan(theta),x[11],x[12]]))
                                     , sym_pos=True)




    return x_prima

