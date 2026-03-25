"""Program for Dynamical Cluster-decay Model (DCM)"""

import pandas as pd
import numpy as np
from krappe_strut_sc import B_E
from P0_class import P0calculator

df = pd.read_excel(r'C:\Users\Nuclear Theory Lab\PyCharmMiscProject\PycharmProjects\subprogram\Totel_Potential\ame2020.xlsx')
df_f = pd.read_excel(r'C:\Users\Nuclear Theory Lab\PyCharmMiscProject\PycharmProjects\subprogram\Totel_Potential\frdm_table.xlsx')
df_r = pd.read_excel(r'C:\Users\Nuclear Theory Lab\PyCharmMiscProject\PycharmProjects\subprogram\Totel_Potential\raman_table.xlsx')


def Radius(a1, a2, a):
    """Radius expression"""
    # R1 = 1.2 * (a1 ** (1 / 3))
    # R2 = 1.2 * (a2 ** (1 / 3))
    # RA = 1.2 * (a ** (1. / 3.))
    # R1 = (1.2536 * (a1 ** (1 / 3))) - (0.80012 * (a1 ** (-1 / 3))) - (0.0021444 * (a1 ** (-1)))
    # R2 = (1.2536 * (a2 ** (1 / 3)))-(0.80012 * (a2 ** (-1 / 3)))-(0.0021444 * (a2 ** (-1)))
    # RA = (1.2536 * (a ** (1 / 3))) - (0.80012 * (a ** (-1 / 3))) - (0.0021444 * (a ** (-1)))
    #R1 = 1.28 * (a1 ** (1. / 3.)) - 0.76 + 0.8 / (a1 ** (1. / 3.))
    #R2 = 1.28 * (a2 ** (1. / 3.)) - 0.76 + 0.8 / (a2 ** (1. / 3.))
    #RA = 1.28 * (a ** (1. / 3.)) - 0.76 + 0.8 / (a ** (1. / 3.))
    TE = 1.16
    C1 = TE * a1 ** (1. / 3.)
    C2 = TE * a2 ** (1. / 3.)
    CA = TE * a ** (1. / 3.)
    #C1 = R1 - (1 / R1)
    #C2 = R2 - (1 / R2)
    #CA = RA - (1 / RA)
    ct = C1 + C2
    R = ct +0 # At touching configuration
    #b=1
    b = 0.68
    return C1, C2, CA, ct, R, b


def T_radius(a1, a2, a, T):
    """Temperature dependent radius using Krappe formula"""
    # Ref.: Krappe, H. J., Physical Review C 59, no. 5 (1999): 2640.
    TE = 1.16 * (1 + ((7.63 * 10 ** (-4)) * (T ** 2)))
    b = 0.68 * (1 + ((7.37 * 10 ** (-3)) * (T ** 2)))
    C1 = TE * a1 ** (1. / 3.)
    C2 = TE * a2 ** (1. / 3.)
    CA = TE * a ** (1. / 3.)
    ct = (C1 + C2)
    R = ct  # At touching configuration
    return C1, C2, CA, ct, R, b


def B_radius(B1, B31, Theta1, B2, B32, Theta2, a1, a2, a, T,delr):
    if temp_dependent:
        R1, R2, CA, Rt, R, b = T_radius(a1, a2, a, T)
    else:
        R1, R2, CA, Rt, R, b = Radius(a1, a2, a)
    Y20_1 = 0.5 * ((3 * np.cos(Theta1) ** 2) - 1)
    Y20_2 = 0.5 * ((3 * np.cos(Theta2) ** 2) - 1)
    Y30_1 = 0.5 * ((5 * np.cos(Theta1) ** 3) - (3 * np.cos(Theta1)))
    Y30_2 = 0.5 * ((5 * np.cos(Theta2) ** 3) - (3 * np.cos(Theta2)))

    C1 = R1 * (1 + B1 * Y20_1 + B31 * Y30_1)
    C2 = R2 * (1 + B2 * Y20_2 + B32 * Y30_2)
    ct = C1 + C2
    R = ct+delr
    return C1, C2, CA, ct, R, b


def masstrans(R1, R2, R, A, A1, A2, fr):
    """Mass Transfer formula from Hydrodynamical Approach"""
    # Ref.: J.Phys.G : Nucl. Phys. 6 (1980) L85-L88.
    V1 = (4 / 3) * np.pi * (R1 ** 3)
    V2 = (4 / 3) * np.pi * (R2 ** 3)
    alpha = 0.4
    Rc = alpha * R2 * fr
    V = V1 + V2
    Vc = np.pi * (Rc ** 2) * R
    BETA = (Rc / (4 * R)) * (2 - (Rc / R1) - (Rc / R2))
    BNN = ((A * R ** 2) / 4) * ((V * (1 + BETA)) / Vc - 1)
    ETA = (A1 - A2) / (A1 + A2)
    return ETA, BNN


def coulomb(Z1, Z2, R):
    """Coulomb Energy"""
    return (1.44 * Z1 * Z2) / R


def proximity(R, ct, R1, R2, A1, A2, Z1, Z2, b):
    """Proximity Potential"""
    S = (R - ct) / b
    phi = np.where(S <= 1.2511,
                   (-0.5 * ((S - 2.54) ** 2) - (0.0852 * ((S - 2.54) ** 3))),
                   -3.437 * np.exp(-(S / 0.75)))
    Vp = 4 * np.pi * ((R1 * R2) / ct) \
         * 0.9517 * (1 - 1.7826 * ((A1 + A2 - 2 * (Z1 + Z2)) / (A1 + A2)) ** 2) \
         * phi * b
    return Vp


def centrifugal(mu, Rt, A1, A2, R1, R2, hbarc, l):
    """Centrifugal Potential"""
    I = 931.5 * ((mu * (Rt ** 2)) + ((2 / 5) * A1 * (R1 ** 2)) + ((2 / 5) * A2 * (R2 ** 2)))
    VL = ((hbarc ** 2) * (l * (l + 1))) / (2 * I)
    return VL


def B_coulomb(B1, B31, B2, B32, Theta1, Theta2, Z1, Z2, R, C1, C2):
    """Deformation dependent Coulomb energy"""

    Y20_1 = 0.5 * (3 * np.cos(Theta1)**2 - 1)
    Y20_2 = 0.5 * (3 * np.cos(Theta2)**2 - 1)

    Y30_1 = 0.5 * (5 * np.cos(Theta1)**3 - (3 * np.cos(Theta1)))
    Y30_2 = 0.5 * (5 * np.cos(Theta2)**3 - (3 * np.cos(Theta2)))

    V0 = (1.44 * Z1 * Z2) / R

    V1 = (3/(5*R**2)) * (
            C1**2 * np.sqrt(5/(4*np.pi))*Y20_1 * (B1 + (4/7)*B1**2 * Y20_1)
          + C2**2 * np.sqrt(5/(4*np.pi))*Y20_2 * (B2 + (4/7)*B2**2 * Y20_2)
        )

    V2 = (3/(7*R**3)) * (
            C1**3 * np.sqrt(7/(4*np.pi))*Y30_1 * (B31 + (4/7)*B31**2 * Y30_1)
          + C2**3 * np.sqrt(7/(4*np.pi))*Y30_2 * (B32 + (4/7)*B32**2 * Y30_2)
        )

    Vc = V0 * (1 + V1 + V2)

    return Vc


def B_proximity(A1, A2, A, Z1, Z2, B1, B2, T):
    """Deformation dependent coulomb energy"""
    if temp_dependent:
        R1, R2, RA, Rt, R, b = T_radius(A1, A2, A, T)
    else:
        R1, R2, RA, Rt, R, b = Radius(A1, A2, A)
    R_p1, R_p2, _, _, _, _ = B_radius(B1, (np.pi / 2), B2, (np.pi / 2), a1, a2, a, T,delr)
    R_z1, R_z2, _, _, _, _ = B_radius(B1, 0, B2, 0, a1, a2, a, T,delr)
    Si = ((R_p1 ** 2) * (R_p2 ** 2)) / ((R_p1 ** 2) * R_z2 + (R_p2 ** 2) * R_z1)
    S0 = (R1 * R2) / Rt
    S = Si / S0
    Vn = proximity(R, Rt, R1, R2, A1, A2, Z1, Z2, b)
    Vp = S * Vn
    return Vp



def B_masstrans(R1, R2, R, A, A1, A2, fr, Theta1, Theta2):
    """Mass Transfer formula from Hydrodynamical Approach"""
    # Ref.: J.Phys.G : Nucl. Phys. 6 (1980) L85-L88.
    V1 = (4 / 3) * np.pi * (R1 ** 3)
    V2 = (4 / 3) * np.pi * (R2 ** 3)
    Rc = 0.4 * R2 * fr
    V = V1 + V2
    Vc = np.pi * (Rc ** 2) * R
    BETA = (Rc / (2 * R)) * (
            (1 / (1 + np.cos(Theta1))) * (1 - (Rc / R1)) + (1 / (1 + np.cos(Theta2))) * (1 - (Rc / R2)))
    gamma = (1 / (2 * R)) * ((1 - np.cos(Theta1))(R1 - Rc)) + (1 - np.cos(Theta2) * (R2 - Rc))
    BNN = ((A * R ** 2) / 4) * (((V * (1 + BETA)) / (Vc * (1 + gamma) ** 2)) - 1)
    ETA = (A1 - A2) / (A1 + A2)
    return ETA, BNN

def element():
    a = int(input('Enter the Mass Number = '))
    z = int(input('Enter the Charge Number = '))
    if z in df['Z'].values:
        e = df.loc[df['Z'] == z, 'EL'].values[0]
        return a, z, e


"""~~~~~~~~Main program starts here~~~~~~~"""
if __name__ == "__main__":
    a, z, e = element()
    for k in np.round(np.arange(0.0,0.8,0.4),2):
        delr=k
        choice = int(input('Enter 1 for spherical or 2 for deformed nuclei: '))
        if choice == 1:
            choice1 = int(input('Enter 3 for T-dependent or 4 for T-independent: '))
            if choice1 == 3:
                ame_sc_file = 'sc_ame_strt.txt'     # Shell corrections for ame2020 datas without deformations
                T = float(input('Enter the value of Temperature = '))
                temp_dependent = True
            elif choice1 == 4:
                T = None
                temp_dependent = False
            deformation = False
        elif choice == 2:
            condition_B = df_f[(df_f['A'] == a) & (df_f['Z'] == z)]
            condition_R = df_r[(df_r['A'] == a) & (df_r['Z'] == z)]
            if not condition_R.empty:
                B = condition_R['B2'].values[0]
            else:
                B = condition_B['B2'].values[0]
            Theta1 = 0
            Theta2 = 0
            choice2 = int(input('Enter 3 for T-dependent or 4 for T-independent: '))

            if choice2 == 3:
                ame_sc_file = 'sc_ame_def.txt'   # Shell corrections for ame2020 datas with deformations
                T = float(input('Enter the value of Temperature = '))

                temp_dependent = True
            elif choice2 == 4:
                T = None
                temp_dependent = False
            deformation = True
        condition = df[(df['A'] == a) & (df['Z'] == z)]
        if temp_dependent:
            if a - z == 0 or z == 0:
                BE = (a * condition['BEN'].values[0]) * (10 ** -3)
            else:
                BEA = B_E(a, z, T, ame_sc_file)
                BE = BEA.bek()
        else:
            BE = (a * condition['BEN'].values[0]) * (10 ** -3)

        print(f'The Parent nuclei is {e.strip()} with', f'\n Mass Number = {a}', f'\n Charge Number = {z}')
        n = int(a / 2)
        m = int(z)
        new_data = []

        for i in range(1, n + 1):
            A1 = a - i
            A2 = i
            for j in range(0, m):
                Z1 = z - j
                Z2 = j
                N1 = A1 - Z1
                N2 = A2 - Z2
                if A2 >= Z2 and A1 >= Z1:
                    new_data.append({'A1': A1, 'A2': A2, 'Z1': Z1, 'Z2': Z2, 'N1': N1, 'N2': N2})
        df1 = pd.DataFrame(new_data)

        m_df = pd.concat([df1, df['Z'], df['A'], df[['N', 'EL']],df['ME'], df[['BEN', 'M']]], axis=1, ignore_index=False)
        frag = []
        df2 = pd.DataFrame()

        ang_choice = int(input('Enter 7 for single value or 8 for range of angular momentum: '))

        if ang_choice == 7:
            l = int(input('Enter the value l: '))
            k = l
        elif ang_choice == 8:
            l, k = map(int, input("Enter the starting and ending values of l separated by space: ").split())

        if deformation:
            choice3 = int(input('Enter 5 for zero deformation or 6 for non-zero deformation: '))

        for i in range(l, k + 1, 1):
            hbarc = 197.32

            for index, row in m_df.iterrows():
                a1 = row['A1']
                a2 = row['A2']
                z1 = row['Z1']
                z2 = row['Z2']
                condition1 = m_df[(m_df['A'] == a2) & (m_df['Z'] == z2)]
                condition2 = m_df[(m_df['A'] == a1) & (m_df['Z'] == z1)]

                if not condition1.empty and not condition2.empty:
                    mu = (a1 * a2) / (a1 + a2)
                    if deformation:
                        condition_B1 = df_f[(df_f['A'] == a1) & (df_f['Z'] == z1)]
                        condition_B2 = df_f[(df_f['A'] == a2) & (df_f['Z'] == z2)]
                        condition_R1 = df_r[(df_r['A'] == a1) & (df_r['Z'] == z1)]
                        condition_R2 = df_r[(df_r['A'] == a2) & (df_r['Z'] == z2)]
                        if choice3 == 5:
                            B1 = 0
                            B2 = 0
                            B31 = 0
                            B32 = 0
                        elif choice3 == 6:
                            B1 = (condition_B1['B2'].iloc[0] if not condition_B1.empty
                                  else condition_R1['B2'].iloc[0] if not condition_R1.empty
                            else 0)

                            B2 = (condition_B2['B2'].iloc[0] if not condition_B2.empty
                                  else condition_R2['B2'].iloc[0] if not condition_R2.empty
                            else 0)

                            B31 = (condition_B1['B3'].iloc[0] if not condition_B1.empty
                                   else 0)

                            B32 = (condition_B2['B3'].iloc[0] if not condition_B2.empty
                                   else 0)

                        C1, C2, CA, ct, R, b = B_radius(B1, B31, Theta1, B2, B32, Theta2, a1, a2, a, T,delr)
                    else:
                        if temp_dependent:
                            C1, C2, CA, ct, R, b = T_radius(a1, a2, a, T)
                        else:
                            C1, C2, CA, ct, R, b = Radius(a1, a2, a)
                    if R / ct <= 1:
                        fr = 1
                    elif (R / ct > 1) and (R / ct <= 2):
                        fr = np.sin((np.pi / 2) * R / ct) ** 2

                    # At Touching Configuration
                    ETA,BNN = masstrans(C1, C2, R, a, a1, a2, fr)

                    if deformation:
                        Ec = B_coulomb(B1, B31, B2, B32, Theta1, Theta2, z1, z2, R, C1, C2)
                        Vp = proximity(R, ct, C1, C2, a1, a2, z1, z2, b)
                    else:
                        Ec = coulomb(z1, z2, R)
                        Vp = proximity(R, ct, C1, C2, a1, a2, z1, z2, b)
                    VL = centrifugal(mu, R, a1, a2, C1, C2, hbarc, l)

                    if temp_dependent:
                        '''Temperature dependent binding energies'''
                        if z2 == 0 or a2 - z2 == 0:
                            BEF2 = ((a2 * condition1['BEN'].values[0]) * (
                                    10 ** -3))
                        else:
                            BEFF2 = B_E(a2, z2, T, ame_sc_file)
                            BEF2 = BEFF2.bek()
                        if z1 == 0 or a1 - z1 == 0:
                            BEF1 = ((a1 * condition2['BEN'].values[0]) * (
                                    10 ** -3))
                        else:
                            BEFF1 = B_E(a1, z1, T, ame_sc_file)
                            BEF1 = BEFF1.bek()
                        Binding_Energy = BEF1 + BEF2
                    else:
                        T = 0
                        '''Ground state binding energies'''
                        BEF1 = (a1 * condition2['BEN'].values[0]) * (10 ** -3)
                        BEF2 = (a2 * condition1['BEN'].values[0]) * (10 ** -3)
                        Binding_Energy = BEF1 + BEF2

                    BEFG1 = (a1 * condition2['BEN'].values[0]) * (10 ** -3)
                    BEFG2 = (a2 * condition1['BEN'].values[0]) * (10 ** -3)
                    Vtot = -Binding_Energy + Ec + Vp + VL
                    Q_Value = (BEFG1 + BEFG2) - BE  # Effective Q_Value
                    if deformation:
                        frag.append({'A1': a1, 'Z1': z1, 'N1': abs(a1 - z1), 'EL1': condition2['EL'].values[0],
                                     'A2': a2, 'Z2': z2, 'N2': abs(a2 - z2), 'EL2': condition1['EL'].values[0],
                                     'Beta1': B1, 'Beta2': B2,'Beta31': B31, 'Beta32': B32, 'BE1': BEF1, 'BE2': BEF2, 'Ec': Ec, 'Vp': Vp,
                                     'Vtot': Vtot, 'L': l, 'VL': VL, 'Q_Value': Q_Value, 'ETA': ETA, 'BNN': BNN})
                    else:
                        frag.append({'A1': a1, 'Z1': z1, 'N1': abs(a1 - z1), 'EL1': condition2['EL'].values[0],
                                     'A2': a2, 'Z2': z2, 'N2': abs(a2 - z2), 'EL2': condition1['EL'].values[0],
                                     'BE1': BEF1, 'BE2': BEF2, 'Ec': Ec, 'Vp': Vp, 'Vtot': Vtot, 'L': l,
                                     'VL': VL, 'Q_Value': Q_Value, 'ETA': ETA, 'BNN': BNN})
            df2 = pd.DataFrame(frag)
            last_l_value = df2['L'].max()
            if deformation:
                df2_last_l = df2[df2['L'] == last_l_value][['A1', 'Z1', 'N1', 'EL1', 'A2', 'Z2', 'N2', 'EL2', 'Beta1',
                                                            'Beta2','Beta31','Beta32', 'BE1', 'BE2', 'Ec', 'Vp', 'Vtot', 'L', 'VL',
                                                            'Q_Value', 'ETA', 'BNN']]
            else:
                df2_last_l = df2[df2['L'] == last_l_value][['A1', 'Z1', 'N1', 'EL1', 'A2', 'Z2', 'N2', 'EL2', 'BE1',
                                                            'BE2', 'Ec', 'Vp', 'Vtot', 'L', 'VL', 'Q_Value', 'ETA',
                                                            'BNN']]

            min_vtot = df2_last_l.groupby('A2')['Vtot'].min().reset_index()
            min_vtot.columns = ['A', 'Vtot_min']
            min_vtot_rows = df2_last_l.loc[df2_last_l.groupby('A2')['Vtot'].idxmin()]
            min_vtot_1 = min_vtot['Vtot_min'].values
            min_vtot_r = min_vtot_rows[['ETA', 'BNN']].values
            beta = pd.DataFrame(min_vtot_r)
            v = pd.DataFrame(min_vtot_1)
            beta = beta.iloc[::-1]
            v = v.iloc[::-1]
            diff = (v.max() - v.min()).values[0]
            min_vtot1 = df2.groupby(['L', 'A2'], as_index=False)['Vtot'].min()
            min_vtot1.columns = ['L', 'A', 'Vtot_min']
            l_groups = min_vtot1.groupby('L', as_index=False)

            pot_file = fr'v{a}{e.strip()}_t{T}l{l:02d}def_delr{delr}reply.out'         # Fragmentation potential file
            out_file = f"P0_{a}{e.strip()}_t{T}l{l:02d}def_delr{delr}reply.out"        # Output file
            with open(pot_file, 'w') as file:
                file.write(f"{a}  {a}\n{a - 1}  {a - 1}  10\n{diff}\n")
                for index, row in v.iterrows():
                    file.write('\t'.join(map(str, row)) + '\n')
        mass_file = fr'm{a}{e.strip()}_t{T}def_delr{delr}reply.out'                    # Mass parameter file
        beta.to_csv(mass_file, mode='w', index=False, header=False, sep='\t')

        with pd.ExcelWriter(f'out_{a}_{e.strip()}l{l:02d}_t{T}def_delr{delr}reply.xlsx') as writer:
            df2_last_l.to_excel(writer, sheet_name='Sheet 1', index=False)
            min_vtot_rows.to_excel(writer, sheet_name='Sheet 2', index=False)
            min_vtot.to_excel(writer, sheet_name='Sheet 3', index=False)
        '''
        with pd.ExcelWriter(f'output_L_{a}_{e.strip()}_t{T}.xlsx') as writer1:
            for l_value, group in min_vtot1.groupby('L', as_index=False):
                sheet_name = f'L_{l_value}' if 'L' in group.columns else 'No_L'
                group.drop(columns='L', errors='ignore').to_excel(writer1, sheet_name=sheet_name, index=False)
        '''
        if a%2==0:
            choice_oe = False
        else:
            choice_oe = True

        """ ----Calling P0 code to execute---- """
        '''
         Input parameters for the class
         ******************************
            potential file, mass parameter file, output file name, temperature, starting and ending masses for window restriction,
            and choice for odd or even mass nuclei
        '''
        P0calculator(pot_file, mass_file, out_file, T, 1,a,ODD=choice_oe).run()