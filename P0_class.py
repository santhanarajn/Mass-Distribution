"""Program for performation probability P0 (for both odd and even mass nuclei) with window restriction"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


class P0calculator:

    def __init__(self, pot_file, mass_file, out_file, T, A_start, A_end,ODD):
        self.pot_file = pot_file
        self.mass_file = mass_file
        self.out_file = out_file
        self.T = T
        self.A_start = A_start
        self.A_end = A_end-1
        self.odd= ODD

    @staticmethod
    def Yield(B, dB, E, h):
        hbar2 = 197.32 ** 2 / 938
        t1 = (hbar2 / (B * h ** 2)) + E
        t21 = -hbar2 / (2 * B * h ** 2)
        t22 = (dB * h) / (4 * B)
        t2 = t21 * (1 - t22)
        t3 = t21 * (1 + t22)
        return t1, t2, t3

    @staticmethod
    def spline(xi, y):
        bc_type = ((2, 0.0), (2, 0.0))
        pb = CubicSpline(xi, y, bc_type=bc_type)
        v1 = xi
        v2 = pb(v1)
        v3 = pb(v1, 1)  # First derivative of the spline
        v4 = pb(v1, 2)  # Second derivative of the spline
        return v2, v3, v4

    def run(self):
        with open(self.mass_file, 'r') as file2:
            f2 = file2.readlines()
        xi1 = []
        BB1 = []
        for _ in f2:
            f2 = _.strip().split()
            xi1.append(float(f2[0]))
            BB1.append(float(f2[1]))
        rxi1 = -np.array(xi1[::-1])
        xi1 = np.array(xi1)

        if self.odd:
            xi = np.concatenate((rxi1[:], xi1))
            BB = np.array(BB1[::-1] + BB1[:])
        else:
            xi = np.concatenate((rxi1[:-1], xi1))
            BB = np.array(BB1[::-1] + BB1[1:])
        Nx = len(xi) - 1
        ee = []
        with (open(self.pot_file, 'r') as f1):
            f1 = f1.readlines()
        fir_line = f1[0].strip().split()
        sec_line = f1[1].strip().split()
        thir_line = f1[2].strip().split()
        fir_line1 = f1[0]
        sec_line1 = f1[1]
        NA = float(fir_line[0])
        NA2 = float(fir_line[0]) / 2
        AN2 = int(NA2)
        #N = int(sec_line[0])
        Diff = float(thir_line[0])
        #N2 = int(N / 2)
        #h = xi[Nx] / float(N2)
        h = abs(xi[0]-xi[1])

        neig = int(sec_line[2])
        f1 = f1[3:]
        e = []
        for _ in f1:
            f1 = _.strip().split()
            e.append(float(f1[0]))
        EE0 = e[0]
        if self.odd:
            ee1 = np.array(e[::-1] + e[:])
        else:
            ee1 = np.array(e[::-1] + e[1:])

        for i in range(len(ee1)):
            ee1[i] = ee1[i] - EE0 + Diff
            i += 1
            ee.append(ee1)
        ee = np.array(ee[0])
        v2, v3, v4 = self.spline(xi, ee)
        E = v2  # Energies
        V2, V3, V4 = self.spline(xi, BB)
        B = abs(V2)
        dB = V3
        yt1, yt2, yt3 = self.Yield(B, dB, E, h)
        yt1 = yt1[:AN2]
        yt2 = yt2[:AN2]
        yt3 = yt3[:AN2]
        yt = sp.diags([yt1, yt2[:-1], yt3[1:]], [0, 1, -1])
        v0 = np.exp(ee[AN2 - 1] - ee[np.arange(0, AN2)])  # initial guess for the wavefunction
        eig, eigvec = eigsh(yt, which='SM', k=neig, v0=v0, tol=5e-6)
        eigvec, eig = np.abs(eigvec.T), np.abs(eig)
        m_p = pd.DataFrame({'xi': xi1, 'V': e, 'B': BB1})
        energy = pd.DataFrame({'xi': xi, 'E': E, 'B': BB})
        with open(self.out_file, 'w') as f_out:
            f_out.write(f'\tDifference = {Diff:.5f}\n\n\n')
            f_out.write(f"\t{fir_line1}\t{sec_line1}\n")
            f_out.write(f'{m_p.to_string(index=False, col_space=15)}\n')
            f_out.write(f'\nENERGIES AND MASSES:\n{energy.to_string(index=False, col_space=15)}\n')
            P0_T = 0
            for j in range(len(eig)):
                P0_T = P0_T
                eigv = eigvec[j]
                if self.odd:
                    eigv = np.concatenate((eigv[:], eigv[::-1]))
                else:
                    eigv = np.concatenate((eigv[:-1], eigv[::-1]))
                A_Start = self.A_start - 1
                A_End = self.A_end
                eigv = eigv[A_Start:A_End]
                sqb = np.sqrt(B[A_Start:A_End])
                A_End = A_End + 1
                aa = np.sum(eigv ** 2 * sqb)
                P0 = (eigv ** 2) / (h * aa * NA2) * 200 * sqb
                P0 = pd.DataFrame({'A': np.arange(self.A_start, A_End), 'P0': P0})
                f_out.write(f'\nEigenvalue\t\t\t{eig[j]:.8f}\t{j + 1}\n')
                for _, row in P0.iterrows():
                    f_out.write(f"{row['A']:>6.1f}\t\t{row['P0']:.9e}\n")
                f_out.write('\n')
                if self.T!=0:
                    AF = np.exp(-eig[j] / self.T)
                    ANOR = sum(np.exp(-eig / self.T))
                    P0_T = P0_T + P0 * AF
            if self.T!=0:
                EMEV = ((float(NA) / 9) * (self.T ** 2)) - self.T
                P0_T = P0_T / ANOR
                P0_Temp = pd.DataFrame({'A': np.arange(self.A_start, A_End), 'P0': P0_T['P0']})
                f_out.write(f'TEMP  {self.T}\tE(MeV)  {EMEV:.2f}\n')
                for _, row in P0_Temp.iterrows():
                    f_out.write(f"{row['A']:>6.1f}\t\t{row['P0']:.9e}\n")
                f_out.write('\n')
