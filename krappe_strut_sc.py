"""Program to find binding energy using Krappe formula"""
# Ref.: Krappe, H. J., Physical Review C 59, no. 5 (1999): 2640.

import numpy as np
import pandas as pd
class B_E:
    def __init__(self, A, Z, T):
        self.A = A
        self.Z = Z
        self.N = self.A - self.Z
        self.I = (self.N - self.Z) / (self.N + self.Z)
        self.T = T
        #self.R = R
        #self.B = B
    def term1(self):
        MH = 7.28903
        return MH * self.Z

    def term2(self):
        Mn = 8.07143
        return Mn * self.N

    def term3(self):
        # Liquid drop energy
        """
        kv = 1.911
        ks = 2.3
        r0 = 1.16
        a = 0.68
        aden = 0.7
        av = 16
        a_s = 21.13

        """
        xkv = 5.61 * 10**(-3)
        kv = 1.911 * (1-xkv*(self.T**2))
        xks = -14.79 * 10**(-3)
        ks = 2.3 * (1-xks*(self.T**2))
        xr0 = -0.763 * 10**(-3)
        r0 = 1.16 * (1-xr0*(self.T**2))
        xa = -7.37 * 10**(-3)
        a = 0.68 * (1-xa*(self.T**2))
        xaden = xa
        aden = 0.7 * (1-xaden*(self.T**2))
        xav = -3.22 * 10**(-3)
        av = 16 * (1-xav*(self.T**2))
        xas = 4.81 * 10**(-3)
        a_s = 21.13 * (1-xas*(self.T**2))
        e2 = 1.439976
        C0 = 5.8
        C1 = (3*e2)/(5*r0)
        C4 = C1 * (5/4) * (3/(2*np.pi))**(2/3)
        x0 = (r0 * self.A**(1/3))/a
        y0 = (r0 * self.A**(1/3))/aden
        B1 = 1 - (3/(x0**2)) + (1+x0)*(2+(3/x0)+(3/(x0**2)))* np.exp(-2*x0)
        B3 = 1 - (5/(y0**2)) * (1-(15/(8*y0))+(21/(8*(y0**3)))
                              -(3/4)*(1+(9/(2*y0))+(7/(y0**2))+(7/(2*(y0**3))))*np.exp(-2*y0))
        E1 = -av*(1-(kv*(self.I**2)))*self.A
        E2 = a_s*(1-(ks*(self.I**2)))*B1*(self.A**(2/3))
        E3 = C0*self.A**0
        E4 = C1*((self.Z**2)/(self.A**(1/3)))*B3
        E5 = -C4*((self.Z**(4/3))/(self.A**(1/3)))
        return E1 + E2 + E3 + E4 +E5

    def term4(self):
        # Wigner term
        I = abs(self.I)
        if self.A > 11:
            W = 35
        else:
            W = 0
        modz = self.Z % 2
        modn = self.N % 2
        if (self.Z==self.N) and (modz!=0 and modn!=0):
            Ew = W * (I + (1/self.A))
        else:
            Ew = W * (I + 0)
        return Ew

    def term5(self):
        # Pairing term
        Bsurf = 1
        r = 5.72
        h = 6.82
        s = 0.118
        t = 8.12
        modz = self.Z % 2
        modn = self.N % 2
        if modz==0 and modn==0:
            return 0
        elif modz!=0 and modn!=0:
            deln = ((r * Bsurf) / (self.N ** (1 / 3))) * np.exp(-s * self.I - (t * (self.I ** 2)))
            delp = ((r * Bsurf) / (self.Z ** (1 / 3))) * np.exp(-s * self.I - (t * (self.I ** 2)))
            delnp = h / (Bsurf * (self.A ** (2 / 3)))
            return delp + deln - delnp
        elif (modz!= 0) and (modn == 0):
            delp = ((r * Bsurf) / (self.Z ** (1 / 3))) * np.exp(-s * self.I - (t * (self.I ** 2)))
            return delp
        elif (modz== 0)and (modn != 0):
            deln = ((r * Bsurf) / (self.N ** (1 / 3))) * np.exp(-s * self.I - (t * (self.I ** 2)))
            return deln

    def term6(self):
        # Proton form factor
        e2 = 1.439976
        xr0 = -0.763 * 10 ** (-3)
        r0 = 1.16 * (1 - xr0 * (self.T ** 2))
        rp = 0.8
        kf = (((9*np.pi*self.Z)/(4*self.A))**(1/3))*(r0**(-1))
        F1 = - ((rp**2)*e2)/(8*(r0**3))
        F2 = 145/48
        F3 = - (327/2880)*((kf*rp)**2)
        F4 = (1527/1209600)*((kf*rp)**4)
        F5 = (self.Z**2)/self.A
        F = F1*(F2+F3+F4)*F5
        return F

    def term7(self):
        ca = 0.145
        return -ca*(self.N-self.Z)

    def term8(self):
        ael=1.433*(10**(-5))
        return -ael*(self.Z**2.39)

    def sc(self):
        infile = 'sc_ame_strt.txt'
        df = pd.read_csv(infile, sep='\t', skiprows=1, names=('Z', 'N', 'sc', 'nan'))
        cond = df[((df['Z']+df['N']) == self.A) & (df['Z'] == self.Z)]
        sc = cond['sc'].values[0]
        return sc

    def bek(self):
        t1 = self.term1()
        t2 = self.term2()
        t3 = self.term3()
        t4 = self.term4()
        t5 = self.term5()
        t6 = self.term6()
        t7 = self.term7()
        t8 = self.term8()
        sc = self.sc()
        sc = sc * np.exp((-self.T ** 2) / (1.5 ** 2))
        t = t3 + t4 + t5 + t6 + t7 + t8
        return -(t + sc)

