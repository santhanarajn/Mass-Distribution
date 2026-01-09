"""Program to find shell correction using Nillson single particle energies based on Strutinsky method"""
import numpy as np
from scipy.special import hermite, erf
from math import factorial
import pandas as pd

class ssc:
    def __init__(self, A, Z, B):
        self.A = A
        self.Z = Z
        self.N = self.A - self.Z
        self.I = (self.N - self.Z) / (self.N + self.Z)
        self.B = B


    def sc(self):
        p = 6
        hermite_cache = [hermite(n) for n in range(p + 1)]
        sqrt_pi = np.sqrt(np.pi)
        fmax = 30

        def hermite_polynomials(smp, u, p):
            # Generate Hermite polynomials up to degree p
            H = np.array([hermite_cache[n](u) for n in range(p + 1)])
            H_d = np.array([hermite_cache[n].deriv()(u/smp) for n in range(p + 1)])
            C = np.array([((-1) ** (m // 2)) / (2 ** m * factorial(m // 2)) if m % 2 == 0 else 0 for m in range(1, 7)])
            return H, H_d, C

        n2 = self.N / 2 + 0.5 if self.N != 0 else 0
        z2 = self.Z / 2 + 0.5 if self.Z != 0 else 0
        n2 = int(n2)  # level required for neutron to be occupied
        z2 = int(z2)  # level required for proton to be occupied
        # print('levels required for the calculation: n2',n2,'z2',z2)
        a = self.A
        hw0 = 41 * a ** (-1 / 3)
        smp = 1 * hw0
        formatted_B = f"d{self.B:.3f}"
        if formatted_B.startswith("d0."):  # Add space for numbers starting with 0
            formatted_B = formatted_B.replace("0.", " .", 1)
        elif formatted_B.startswith("d-0."):  # Add space for numbers starting with 0
            formatted_B = formatted_B.replace("-0.", "-.", 1)
        else:
            formatted_B = formatted_B.lstrip('0')

        with open(fr'single_particle_levels\{formatted_B}.spe') as f1:
            f11 = f1.readlines()
            nmax = int(f11[0].strip().split()[0])
            f1 = f11[1:nmax + 1]
            f2 = f11[nmax+2:]
            ez = [float(line.strip().split()[0]) * hw0 for line in f1]
            en = [float(line.strip().split()[0]) * hw0 for line in f2]
            ez = np.array(ez)
            en = np.array(en)
            ez0 = sum(ez[:z2]) if z2 != 0 else 0  # Sum total of single particle energies for proton
            en0 = sum(en[:n2]) if n2 != 0 else 0  # Sum total of single particle energies for neutron
            etot = ez0 + en0  # Sum total of single particle energies
            ap = ez[z2 - 1] if z2 != 0 else 0  # Fermi energy for proton
            an = en[n2 - 1] if n2 != 0 else 0  # Fermi energy for neutron
            #print(f'Fermi energy for proton {ap}\nFermi energy for neutron {an}\n'
                  #f'Sum total of single particle energies for proton {ez0}\n'
                  #f'Sum total of single particle energies for neutron {en0}\n'
                  #f'Sum total of single particle energies {etot}')

        # To evaluate the average Fermi energy for neutron and proton simultaneously

        for j in range(1, 101):
            un = (an - en) / smp
            uz = (ap - ez) / smp
            f = g = fa1 = ga2 = 0
            chk =0.001
            for i in range(len(un)+1):
                un2 = float(un[i] ** 2)
                uz2 = float(uz[i] ** 2)
                erfn, erfz = erf(float(un[i])), erf(float(uz[i]))
                Hn, H_dn, cn = hermite_polynomials(smp, float(un[i]), p)
                Hz, H_dz, cz = hermite_polynomials(smp, float(uz[i]), p)
                sumn = np.sum(cn[:p] * Hn[:p], axis=0)
                sumz = np.sum(cz[:p] * Hz[:p], axis=0)
                sumnd = np.sum(cn[:p] * H_dn[:p], axis=0)
                sumzd = np.sum(cz[:p] * H_dz[:p], axis=0)
                un2 = np.minimum(un2, fmax)
                uz2 = np.minimum(uz2, fmax)
                rn = 0.5 * (1 + erfn) - (1 / sqrt_pi) * np.exp(-un2) * sumn
                rz = 0.5 * (1 + erfz) - (1 / sqrt_pi) * np.exp(-uz2) * sumz
                if -un[i] > 1 and -uz[i] > 1:
                    break
                f += rn
                g += rz
                fa1 += 1 / (smp * sqrt_pi) * np.exp(-un2) * (1 + 2 * un[i] * sumn - sumnd)
                ga2 += 1 / (smp * sqrt_pi) * np.exp(-uz2) * (1 + 2 * uz[i] * sumz - sumzd)
            f -= n2
            g -= z2
            det = fa1 * ga2
            if det <= 0:
                det = 0.0001
            rh = (-f * ga2) / det
            rk = (-fa1 * g) / det
            an = an + rh
            ap = ap + rk
            if (abs(f)-chk)<=0:
                if(abs(g)-chk)<=0:
                    break

        #print(f'avg. fermi energy for proton {ap}\navg. fermi energy for neutron {an}')

    # To evaluate the average energy for neutron and proton
        entot = 0
        eztot = 0
        for i in range(nmax):
            unb = float((an - en[i]) / smp)
            uzb = float((ap - ez[i]) / smp)
            unb2 = unb ** 2
            uzb2 = uzb ** 2
            if -unb > 1 and -uzb > 1:
                break
            unb2 = min(unb2, fmax)
            uzb2 = min(uzb2, fmax)
            erfbn, erfbz = erf(unb), erf(uzb)
            Hbn, H_dbn, cbn = hermite_polynomials(smp, unb, p)
            Hbz, H_dbz, cbz = hermite_polynomials(smp, uzb, p)

            m_vals = np.arange(2, p + 1)

            # Precompute the terms for sumbn
            Hn_terms_bn = (
                    0.5 * smp * Hbn[m_vals] +
                    en[i] * Hbn[m_vals - 1] +
                    m_vals * smp * Hbn[m_vals - 2])
            sumbn = np.sum(cbn[m_vals - 1] * Hn_terms_bn)

            # Precompute the terms for sumbz
            Hn_terms_bz = (
                    0.5 * smp * Hbz[m_vals] +
                    ez[i] * Hbz[m_vals - 1] +
                    m_vals * smp * Hbz[m_vals - 2])
            sumbz = np.sum(cbz[m_vals - 1] * Hn_terms_bz)

            enb1 = 0.5 * en[i] * (1 + erfbn)
            enb2 = 1 / (2 * sqrt_pi) * smp * np.exp(-unb2)
            enb3 = 1 / sqrt_pi * np.exp(-unb2) * sumbn
            entot += enb1 - enb2 - enb3
            ezb1 = 0.5 * ez[i] * (1 + erfbz)
            ezb2 = 1 / (2 * sqrt_pi) * smp * np.exp(-uzb2)
            ezb3 = 1 / sqrt_pi * np.exp(-uzb2) * sumbz
            eztot += ezb1 - ezb2 - ezb3
        ebtot = entot + eztot

        # print(f'Sum total of avg. energy {ebtot}')
        sc = etot - ebtot
        #print(f'Shell correction {sc}')
        return sc

df = pd.read_excel(r'C:\Users\Nuclear Theory Lab\PycharmProjects\subprogram\Totel_Potential\ame2020.xlsx')
df_f = pd.read_excel(r'C:\Users\Nuclear Theory Lab\PycharmProjects\subprogram\Totel_Potential\frdm_table.xlsx')
df_r = pd.read_excel(r'C:\Users\Nuclear Theory Lab\PycharmProjects\subprogram\Totel_Potential\raman_table.xlsx')

'''
""" with deformation """

with open('sc_ame_strt_demo.txt','w') as f1:
    f1.write('Z\tN\tsc\n')
    for index, row in df.iterrows():
        A=row['A']
        Z=row['Z']
        cond1 = df_f[(df_f['A'] == A) & (df_f['Z'] == Z)]
        cond2 = df_r[(df_r['A'] == A) & (df_r['Z'] == Z)]
        if not cond1.empty:
            B2=cond1['B2'].values[0]
        elif not cond2.empty:
            B2=cond2['B2'].values[0]
        else:
            B2=0
        shell_correction = ssc(A, Z, B2).sc()
        f1.write(f'{Z}\t{A-Z}\t{shell_correction}\n')
'''
""" without deformation """
with open('sc_ame_strt_demo.txt','w') as f1:
    f1.write('Z\tN\tsc\n')
    for index, row in df.iterrows():
        A=row['A']
        Z=row['Z']
        B2=0
        shell_correction = ssc(A, Z, B2).sc()
        f1.write(f'{Z}\t{A-Z}\t{shell_correction}\n')

