# -*- coding: utf-8 -*-
'''
@autor: ntrivisonno, lgarelli, mstorti
'''

import numpy as np

class Quaternion:

        def __init__(self, d, v):
            """
            Create the quaternion from the scalar part and the vector part
            """
            self.d = d
            self.v = np.asarray(v)

        def __str__(self):
            return '<{0}, ({1}, {2}, {3})>'.format(self.d, self.v[0], \
                                                   self.v[1], self.v[2])

        __repr__ = __str__

        @classmethod
        def fromAngleAndDirection(cls, alpha, u):
            """
            Create the quaternion from an angle and a unit direction vector
            """
            return Quaternion(np.cos(alpha/2.), np.asarray(u) * \
                              np.sin(alpha/2.))

        @classmethod
        def fromRotationVector(cls, v):
            """
            Create the quaternion from a rotation vector
            """
            alpha = np.linalg.norm(v)
            if alfa == 0:
                u = np.zeros(3)
            else:
                u = v/alpha
            return cls.fromAngleAndDirection(alpha, u)

        @classmethod
        def fromEulerAngles(cls, roll, pitch, yaw):
            """
            Create the quaternion from Euler angles (or more precisely, 
            Tait-Bryan angles). It is assumed that the rotation corresponding to
            the Euler angles is performed in the zyx order. In this way, the 
            evolutions of this models using quaternions and Euler angles are 
            equivalent if the quaternions are obtained from the Euler angles 
            using this constructor.
            """
            q_z_conj = Quaternion.fromAngleAndDirection(-yaw, [0, 0, -1])
            q_y_conj = Quaternion.fromAngleAndDirection(-pitch, [0, -1, 0])
            q_x_conj = Quaternion.fromAngleAndDirection(-roll, [-1, 0, 0])
            return q_z_conj.mult(q_y_conj).mult(q_x_conj)

        def mult(self, q, normalize=False):
            """
            Returns the multiplication of the quaternion by the quaternion q.
            :param q: Quaternion to post-multiply by.
            :param normalize: Whether to normalize the result. If the quaternions
            are unitary, it shouldn't be necessary (in theory), but for numerical
            reasons, it might be advisable.
            :return: The multiplication.
            """
            qr = Quaternion(self.d*q.d - np.dot(self.v, q.v), self.d*q.v + \
                            q.d*self.v + np.cross(self.v, q.v))
            if normalize:
                qr.normalize()
            return qr

        def modulus(self):
            return np.sqrt(self.d*self.d + np.dot(self.v, self.v))

        def normalize(self):
            modul = self.modulus()
            assert modul > 1e-8
            self.d /= modul
            self.v /= modul

        def get_conjugate(self):
            return Quaternion(self.d, -self.v)

        def calc_rot_matrix(self, unit_quat=True):
            """
            Compute the rotation matrix M such that, given a vector v and the
            quaternion q, it's the same to perform Mv as qvq_inv.
            :param unit_quat: Indicates whether the quaternion is unitary or not
            """
            assert unit_quat
            # e1q = Quaternion(0., [1.,0,0])
            # e2q = Quaternion(0., [0,1.,0])
            # e3q = Quaternion(0., [0,0,1.])
            # c1 = self.mult(e1q).mult(self.get_conjugate())  # col 1 of matriz
            # c2 = self.mult(e2q).mult(self.get_conjugate())  # col 2 of matriz
            # c3 = self.mult(e3q).mult(self.get_conjugate())  # col 3 of matriz
            # Q = np.array([c1.v, c2.v, c3.v]).T

            #  de "The quaternions with an application to rigid body dynamics"
            q0 = self.d
            q1, q2, q3 = self.v
            q0_2 = q0**2
            q1_2 = q1**2
            q2_2 = q2**2
            q3_2 = q3**2
            q0q1 = q0*q1
            q0q2 = q0*q2
            q0q3 = q0*q3
            q1q2 = q1*q2
            q1q3 = q1*q3
            q2q3 = q2*q3
            # Compute Q cost 16 multiplications
            Q = np.array([[q0_2 + q1_2 - q2_2 - q3_2,
                           2.*(q1q2 - q0q3),
                           2.*(q1q3 + q0q2)],
                         [2.*(q1q2 + q0q3),
                          q0_2 - q1_2 + q2_2 - q3_2,
                          2.*(q2q3 - q0q1)],
                         [2.*(q1q3 - q0q2),
                          2.*(q2q3 + q0q1),
                          q0_2 - q1_2 - q2_2 + q3_2]])

            return Q

        def rotate_vector(self, vec, unit_quat=True):
            """
            Compute the rotation of the vector v using the quaternion q through
            the formula qvq_inv.
            :param unit_quat: Indicates whether the quaternion is unitary or not
            """
            assert unit_quat
            q_vec = Quaternion(0, vec)
            return self.mult(q_vec).mult(self.get_conjugate()).v

        def get_angle_and_axys(self, unit_quat=True):
            assert unit_quat
            angle = np.arccos(self.d) * 2.  # this is done between 0 and 2pi
            if angle > np.pi:
                angle -= 2*np.pi  # this is done between -pi and pi
            axys = self.v/np.linalg.norm(self.v)
            return angle, axys

        def get_euler_anles(self, order='zyx'):
            """
            Compute the Euler angles such that, if a rotation is performed with
            these angles, it gives the same result as if the rotation is 
            performed with the quaternion. To ensure uniqueness of the result,
            it is assumed that the roll and yaw are in the range [-pi, pi], and
            pitch is in the range [-pi/2, pi/2]. If the quaternion is obtained 
            using Euler angles outside these ranges, and then this function is 
            called, the resulting angles will not match the original ones.
            :param order: The order in which the rotation would be performed 
            with the Euler angles (to give the same result as performing the 
            rotation with the quaternion). It can be 'zyx' or 'xyz'.
            :return: The Euler angles (or more precisely, Tait-Bryan angles):
            roll (x), pitch (y), yaw (z).
            """
            if order == 'zyx':
                q0 = self.d
                q1, q2, q3 = self.v
                roll = np.arctan2(q2*q3+q0*q1, .5 - (q1**2+q2**2))
                pitch = np.arcsin(-2.*(q1*q3-q0*q2))
                yaw = np.arctan2(q1*q2+q0*q3, .5 - (q2**2+q3**2))
            elif order == 'xyz':
                q0 = self.d
                q1, q2, q3 = -self.v
                roll = -np.arctan2(q2*q3+q0*q1, .5 - (q1**2+q2**2))
                pitch = -np.arcsin(-2.*(q1*q3-q0*q2))
                yaw = -np.arctan2(q1*q2+q0*q3, .5 - (q2**2+q3**2))
            else:
                raise Exception('The order "{0}" is unknown'.format(order))
            return roll, pitch, yaw

        def __mul__(self, escalar):
            assert isinstance(escalar, (int,float))
            return Quaternion(self.d*escalar, self.v*escalar)

        __rmul__ = __mul__

        def mult_cuat_times_vec(self, v):
            """
            Multiply the quaternion by the vector v (in that order). It is
            equivalent to self.mult(Quaternion(0, v)), but faster.
            :param v: Vector in R3.
            :return: The quaternion resulting from multiplying this quaternion
            by the quaternion <0, v>.
            """
            q0 = self.d
            q1, q2, q3 = self.v
            q0_r = -v[0]*q1 - v[1]*q2 - v[2]*q3
            q1_r = v[0]*q0 + v[2]*q2 - v[1]*q3
            q2_r = v[1]*q0 - v[2]*q1 + v[0]*q3
            q3_r = v[2]*q0 + v[1]*q1 - v[0]*q2
            return Quaternion(q0_r, [q1_r, q2_r, q3_r])

