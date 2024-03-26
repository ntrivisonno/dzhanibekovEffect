# -*- coding: utf-8 -*-
'''
@autor: ntrivisonno, lgarelli, mstorti
@date: APR2024
'''

def parameters(file_in):
    f = open(file_in, 'r')
    # Read comments
    line = f.readline()
    line = f.readline()
    frags = line.split()
    mass = float(frags[1])

    line = f.readline()
    frags = line.split()
    diam = float(frags[1])

    line = f.readline()
    frags = line.split()
    xcg = float(frags[1])
    ycg = float(frags[3])
    zcg = float(frags[5])

    line = f.readline()
    frags = line.split()
    Ixx = float(frags[1])
    Iyy = float(frags[3])
    Izz = float(frags[5])

    line = f.readline()
    frags = line.split()
    steps = int(frags[1])

    line = f.readline()
    frags = line.split()
    dt = float(frags[1])

    return [mass, diam, xcg, ycg, zcg, Ixx, Iyy, Izz, steps, dt]
