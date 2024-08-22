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
    :param x: states vector
    :param u: control actions
    :return: states derivatives
    x[0] -> x_ned (north)
    x[1] -> y_ned (east)
    x[2] -> h_ned = -z_ned (z_ned = down)
    x[3] -> u (vel x body frame)
    x[4] -> v (vel y body frame)
    x[5] -> w (vel z body frame)
    x[6] -> q_e (quaternion scalar component)
    x[7] -> q_v1 (quaternion first vetorial component)
    x[8] -> q_v2 (quaternion second vetorial component
    x[9] -> q_v3 (quaternion third vetorial component)
    x[10] -> p (first component angular velocity body frame)
    x[11] -> q (second component angular velocity body frame)
    x[12] -> r (third component angular velocity body frame)
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

    q_prima = q_body2ned.mult_cuat_times_vec(x[10:13]*.5)
    x_prima[6] = q_prima.d
    x_prima[7:10] = q_prima.v

    # Fluid properties
    rho, mu, c = fluid_prop(x[2], 0)

    # --- Dynamic Eq ---#

    velWind_ned = np.zeros(3)
    # vector wind vel body frame
    velWind_body = Q_body2ned.T.dot(velWind_ned)
    # relative vel body frame
    vel_rel_body = x[3:6] - velWind_body
    vt = np.linalg.norm(vel_rel_body)
    # AoA
    alpha = np.arctan2(vel_rel_body[2], vel_rel_body[0])
    alphad = alpha * 180 / math.pi
    # Yaw
    beta = np.arctan2(vel_rel_body[1], np.sqrt(vel_rel_body[0]**2 + \
                                               vel_rel_body[2]**2))
    betad = beta * 180 / math.pi

    sin_alpha_t = math.sqrt(((math.sin(beta)) ** 2 + \
                             (math.cos(beta)) ** 2 * (math.sin(alpha)) ** 2))
    alpha_t = math.asin(math.sqrt(((math.sin(beta)) ** 2 + \
                                   (math.cos(beta)) ** 2 * \
                                   (math.sin(alpha)) ** 2)))

    ca = np.cos(alpha)
    cb = np.cos(beta)
    sa = np.sin(alpha)
    sb = np.sin(beta)

    mach = vt / c

    x_prima[3] = x[12] * x[4] - x[11] * x[5] + 1.0 * 0.0
    x_prima[4] = x[10] * x[5] - x[12] * x[3] + 1.0 * 0.0
    x_prima[5] = x[11] * x[3] - x[10] * x[4] + 1.0 * 0.0

    x_prima[10:13] = sp.linalg.solve(inertia_tensor, \
                                     np.cross(inertia_tensor.dot(x[10:13]), \
                                              x[10:13]), sym_pos=True)
    

    return x_prima

