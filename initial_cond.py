# -*- coding: utf-8 -*-
'''
@autor: ntrivisonno, lgarelli, mstorti
@date: APR2024
'''

def initial_cond(file_in):

    f = open(file_in, 'r')
    # Read comments
    line = f.readline()
    line = f.readline()
    frags = line.split()
    V = float(frags[1])

    line = f.readline()
    frags = line.split()
    alpha = float(frags[1])

    line = f.readline()
    frags = line.split()
    beta = float(frags[1])

    line = f.readline()
    frags = line.split()
    p = float(frags[1])

    line = f.readline()
    frags = line.split()
    q = float(frags[1])

    line = f.readline()
    frags = line.split()
    r = float(frags[1])

    line = f.readline()
    frags = line.split()
    phi = float(frags[1])

    line = f.readline()
    frags = line.split()
    theta = float(frags[1])

    line = f.readline()
    frags = line.split()
    psi = float(frags[1])

    line = f.readline()
    frags = line.split()
    XE = float(frags[1])
    YE = float(frags[3])
    ZE = float(frags[5])

    return [V, alpha, beta, p, q, r, phi, theta, psi, XE, YE, ZE]
