import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps, quad
from scipy.constants import hbar
from scipy.spatial.distance import cdist
from sympy.physics.quantum import Ket, Bra, Operator
from sympy import *
init_printing(use_unicode=True)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import sympy as sp
from matplotlib import cm
from scipy import diag
import subprocess
import math
import itertools
from numpy import linalg as LA
np.set_printoptions(suppress=True)

class wc1_oef1:
    @staticmethod
    def t_deel1(b):
        print('Tip:')
        print('Gebruik hiervoor np.arange(...) of np.linspace(...)')


    @staticmethod
    def a_deel1(b):
        print('Antwoord:')
        x = np.linspace(0, 10, 1000)
        print(x)


    @staticmethod
    def t_deel2(b):
        print('Tip:')
        print('Maak een functie die enkel het sinusgedeelte',
              'van de golfvergelijking teruggeeft. Hierbij maak je gebruik',
              'van numpy.sin(...) en numpy.pi.')

    @staticmethod
    def a_deel2(b):
        print('Antwoord:')
        print('De functie geeft np.sin( (n*np.pi*x)/l ) terug.')

    @staticmethod
    def t_deel3(b):
        print('Tip:')
        print('Gebruik plt.plot(...) om de drie toestanden te plotten.')

    @staticmethod
    def a_deel3(b):
        print('Antwoord:')
        def tijdsonafh(x, n, l):
            return np.sin((n * np.pi * x) / l)

        x = np.linspace(0, 10, 1000)
        toestanden = 3

        fig, ax = plt.subplots()
        ax.set_xlabel('positie')
        ax.set_ylabel('amplitude')

        for n in range(1, toestanden + 1):
            harm_osc = tijdsonafh(x, n, 10)

            ax.plot(x, harm_osc, label='n={}'.format(n))

        ax.axhline(linewidth=.5, color='k')
        ax.legend()
        fig.show()

class wc1_oef2:

    @staticmethod
    def t_deel1(b):
        print('Tip:')
        print('Deze deelvraag is eigenlijk hetzelfde als in oefening 1.',
              ' Gebruik np.sin(...) om deze vraag op te lossen.')

    @staticmethod
    def a_deel1(b):
        print('Antwoord:')
        print('np.sin( (n * np.pi * x) / a )')

    @staticmethod
    def t_deel2(b):
        print('Tip:')
        print('Gebruik de functie uit de vorige deelvraag om de 4 eigenfuncties',
              'terug te geven met n=0, n=1, n=2, n=3. Hierna sommeer je deze.')

    @staticmethod
    def t_deel3(b):
        print('Tip:')
        print('Gebruik vergelijking 1.9 om de probabiliteitsdensiteit te schrijven.',
              'Hierna gebruik je scipy.integrate.simps om te integreren.',
              'Vergeet niet de vierkantswortel te nemen van de bekomen integraal.')

    @staticmethod
    def a_deel3(b):
        print('Antwoord:')
        x = np.linspace(0, 10, 1000)
        a = 10
        n = 1

        def eigenfunctie(x, a, n):
            return np.sin((n * np.pi * x) / a)

        def golffunctie():
            psi_x = np.zeros(1000)
            for n in range(5):
                psi_x += eigenfunctie(x, a, n)
            return psi_x

        def norm(psi):
            pd = np.conjugate(psi) * psi
            return np.sqrt(simps(pd, x))

        psi_x = golffunctie()
        print('de norm is: ', norm(psi_x))

    @staticmethod
    def t_deel4(b):
        print('Tip:')
        print('Bereken hiervoor de norm van psi en deel de golffunctie',
              'psi door de berekende norm. Sla deze genormaliseerde',
              'golffunctie op in een nieuwe variabele.')

    @staticmethod
    def a_deel4(bg):
        print('Antwoord:')
        x = np.linspace(0, 10, 1000)
        a = 10
        n = 1

        def eigenfunctie(x, a, n):
            return np.sin((n * np.pi * x) / a)

        def golffunctie():
            psi_x = np.zeros(1000)
            for n in range(5):
                psi_x += eigenfunctie(x, a, n)
            return psi_x

        def norm(psi):
            pd = np.conjugate(psi) * psi
            return np.sqrt(simps(pd, x))

        def normaliseer(psi):
            norm_psi = norm(psi)
            return psi / norm_psi

        psi_x = golffunctie()
        psi_x_norm = normaliseer(psi_x)

        fig, ax = plt.subplots()
        ax.axhline(linewidth=.5, color='k')

        ax.plot(psi_x, label='niet-genormaliseerd')
        ax.plot(psi_x_norm, label='genormaliseerd')

        ax.legend()
        fig.show()

class wc1_oef3:

    @staticmethod
    def t_deel1(b):
        print('Tip:')
        print('Gebruik hiervoor np.exp(...) en np.sqrt(...). We gebruiken',
              'de NumPy modules omdat we hier met een grid zullen werken.')

    @staticmethod
    def a_deel1(b):
        print('Antwoord:')
        print('In de coordinatenruimte: np.exp(-(x**2/(2.0*sigma**2)))/np.sqrt(sigma*np.sqrt(np.pi))')
        print('In de momentenruimte: np.exp(-(((p**2)*(sigma**2))/(2.0)))/np.sqrt(np.sqrt(np.pi))')

    @staticmethod
    def t_deel2(b):
        print('Tip:')
        print('Maak een grid van x door gebruik te maken van np.linspace(...).',
              'Gebruik de vorige functies (coordinatenruimte en momentenruimte)',
              'met x als input en plot deze grafieken op 1 plot.')

    @staticmethod
    def a_deel2(b):
        print('Antwoord:')

        def gauss_x(x):
            return np.exp(-(x ** 2 / (2.0 * sigma ** 2))) / np.sqrt(sigma * np.sqrt(np.pi))

        def gauss_p(p):
            return np.sqrt(sigma) * np.exp(-(((p ** 2) * (sigma ** 2)) / (2.0))) / np.sqrt(np.sqrt(np.pi))

        sigma = 0.1
        grid = np.linspace(-10, 10, 1000)

        fig, ax = plt.subplots()

        ax.plot(grid, gauss_x(grid), label='positie')
        ax.plot(grid, gauss_p(grid), label='momentum')

        ax.legend()
        fig.show()

    @staticmethod
    def t_deel3(b):
        print('Tip:')
        print('Gebruik scipy.integrate.quad(...) om te integreren.')

    @staticmethod
    def a_deel3(b):
        print('Antwoord:')

        def gauss_x(x):
            return np.exp(-(x ** 2 / (2.0 * sigma ** 2))) / np.sqrt(sigma * np.sqrt(np.pi))

        def gauss_p(p):
            return np.sqrt(sigma) * np.exp(-(((p ** 2) * (sigma ** 2)) / (2.0))) / np.sqrt(np.sqrt(np.pi))

        sigma = 0.1
        grid = np.linspace(-10, 10, 1000)

        def x_int(x):
            return gauss_x(x) * x * gauss_x(x)

        def x2_int(x):
            return gauss_x(x) * (x ** 2) * gauss_x(x)

        def p_int(p):
            return gauss_p(p) * p * gauss_p(p)

        def p2_int(p):
            return gauss_p(p) * (p ** 2) * gauss_p(p)

        sigma = 0.1

        # positie
        exp_x = quad(x_int, -np.inf, np.inf)[0]
        exp_x2 = quad(x2_int, -np.inf, np.inf)[0]
        delta_x = np.sqrt(exp_x2 - exp_x ** 2)

        # momentum
        exp_p = quad(p_int, -np.inf, np.inf)[0]
        exp_p2 = quad(p2_int, -np.inf, np.inf)[0]
        delta_p = np.sqrt(exp_p2 - exp_p ** 2)

        onzekerheid = delta_x * delta_p
        print('De onzekerheid bedraagt ', onzekerheid)

class wc2_oef1:

    @staticmethod
    def t_deel1(b):
        print('Tip:')
        print('Gebruik np.arange(...) of np.linspace(...).')

    @staticmethod
    def a_deel1(b):
        print('Antwoord:')
        L = 10
        x = np.linspace(0, L, num=1000)
        print(x)

    @staticmethod
    def t_deel2(b):
        print('Tip:')
        print('Maak een vierkante matrix van (1000x1000) met np.zeros(...).',
              'Vul de diagonaal in met sin(x), gebruik np.sin(...).')

    @staticmethod
    def a_deel2(b):
        print('Antwoord:')

        L = 10
        x = np.linspace(0, L, num=1000)

        def pot_operator(x):
            n = x.size
            V = np.zeros((n, n))

            for i in range(n):
                V[i][i] = x[i]

            return np.sin(V)

        print('V \n', pot_operator(x))

    @staticmethod
    def t_deel3(b):
        print('Tip:')
        print('Stel de constante a op. Hiervoor ga je op dezelfde manier',
              'te werk als in de vorige deelvraag. Enkel worden nu ook de rijen',
              'boven en onder de diagonaal gevuld.')

    @staticmethod
    def a_deel3(b):
        print('Antwoord:')

        L = 10
        x = np.linspace(0, L, num=1000)

        def kin_operator(x, m=1, hbar=1):

            n = x.size

            delta_x = x[1] - x[0]
            a = -(hbar ** 2) / (2 * m * delta_x ** 2)

            T = np.zeros((n, n))

            for i in range(n):
                T[i][i] = -2 * a

                if i == 0:
                    T[i][i + 1] = a
                elif i == n - 1:
                    T[i][i - 1] = a
                else:
                    T[i][i - 1] = a
                    T[i][i + 1] = a

            return T

        print('T \n', kin_operator(x))

    @staticmethod
    def t_deel4(b):
        print('Tip:')
        print('De potentiele en kinetische energiematrices zijn beiden Numpy-',
              'arrays. Je kunt ze dus zonder problemen optellen d.m.v. +.')

    @staticmethod
    def a_deel4(b):
        print('Antwoord:')

        L = 10
        x = np.linspace(0, L, num=1000)
        def pot_operator(x):
            n = x.size
            V = np.zeros((n, n))

            for i in range(n):
                V[i][i] = x[i]

            return np.sin(V)

        def kin_operator(x, m=1, hbar=1):

            n = x.size

            delta_x = x[1] - x[0]
            a = -(hbar ** 2) / (2 * m * delta_x ** 2)

            T = np.zeros((n, n))

            for i in range(n):
                T[i][i] = -2 * a

                if i == 0:
                    T[i][i + 1] = a
                elif i == n - 1:
                    T[i][i - 1] = a
                else:
                    T[i][i - 1] = a
                    T[i][i + 1] = a

            return T

        H = kin_operator(x) + pot_operator(x)
        print('H = \n', H)

    @staticmethod
    def t_deel5(b):
        print('Tip:')
        print('Gebruik np.linalg.eigh(...) om de Hamtiltoniaan te diagonaliseren.')

    @staticmethod
    def t_deel6(b):
        print('Tip:')
        print('Gebruik je geschreven normaliseerfuncties van WC2_oef2.',
              'De grondtoestand is de eerste kolomvector van de eigenvectoren-',
              'matrix die je terugkrijgt bij np.linalg.eigh(...).')

    @staticmethod
    def a_deel6(b):
        print('Antwoord:')
        L = 10
        x = np.linspace(0, L, num=1000)
        def pot_operator(x):
            n = x.size
            V = np.zeros((n, n))

            for i in range(n):
                V[i][i] = x[i]

            return np.sin(V)

        def kin_operator(x, m=1, hbar=1):

            n = x.size

            delta_x = x[1] - x[0]
            a = -(hbar ** 2) / (2 * m * delta_x ** 2)

            T = np.zeros((n, n))

            for i in range(n):
                T[i][i] = -2 * a

                if i == 0:
                    T[i][i + 1] = a
                elif i == n - 1:
                    T[i][i - 1] = a
                else:
                    T[i][i - 1] = a
                    T[i][i + 1] = a

            return T
        H = kin_operator(x) + pot_operator(x)
        eigw, eigv = np.linalg.eigh(H)

        def prob_densiteit(psi):
            return np.conjugate(psi) * psi

        def norm(psi):
            pd = prob_densiteit(psi)
            return np.sqrt(simps(pd, x))

        def normaliseer(psi):
            norm_psi = norm(psi)

            return psi / norm_psi

        psi_0 = normaliseer(eigv[:, 0])

        pd = prob_densiteit(psi_0)

        fig, ax = plt.subplots()
        ax.axhline(color='k', linewidth=0.5)

        ax.plot(x, psi_0, label='psi_0')
        ax.plot(x, pd, label='probabiliteit', color='r')

        ax.legend()
        fig.show()

class wc2_oef2:

    @staticmethod
    def t_deel1(b):
        print('Tip:')
        print('Stel Sx, Sy en Sz op. Dit zijn NumPy-arrays waardoor het makkelijk',
              'wordt om hun commutator uit te rekenen. Indien ze commuteren, moet hun',
              'commutator gelijk zijn aan de nulmatrix. Gebruik hiervoor np.allclose(...).',
              'Gebruik ook scipy.constants.hbar.')

    @staticmethod
    def a_deel1(b):
        print('Antwoord:')
        s_x = np.array([[0, 1], [1, 0]])
        s_y = np.array([[0, -1j], [1j, 0]])
        s_z = np.array([[1, 0], [0, -1]])

        def commutator(A, B):
            return np.allclose(A @ B - B @ A, np.zeros((2, 2)))

        comm = commutator(s_x, s_y)
        print('Commuteren S_x en S_y? {}'.format(comm))

        comm = commutator(s_x, s_z)
        print('Commuteren S_x en S_z? {}'.format(comm))

        comm = commutator(s_y, s_z)
        print('Commuteren S_y en S_z? {}'.format(comm))

    @staticmethod
    def t_deel2(b):
        print('Tip:')
        print('S^2 = s_x**2 + s_y**2 + s_z**2')

    @staticmethod
    def t_deel3(b):
        print('Tip:')
        print('<alfa|S^2|alfa> waarbij <alfa| gevormd wordt door',
              'het transponeren van de alfa-kolomvector.')

    @staticmethod
    def a_deel3(b):
        print('Antwoord:')

        alfa, beta = np.array([[1.], [0.]]), np.array([[0.], [1.]])
        ct = 0.5 * hbar
        s_x = np.array([[0, 1], [1, 0]])
        s_y = np.array([[0, -1j], [1j, 0]])
        s_z = np.array([[1, 0], [0, -1]])

        def commutator(A, B):
            return np.allclose(A @ B - B @ A, np.zeros((2, 2)))

        S2 = s_x ** 2 + s_y ** 2 + s_z ** 2
        verw_S2 = np.transpose(alfa) @ S2 @ alfa
        print('De verwachtingswaarde van S2 is 0.5*hbar*{}'.format(float(verw_S2[0][0])))

    @staticmethod
    def t_deel4(b):
        print('Tip:')
        print('Diagonaliseer Sx met np.linalg.eig(...). Normaliseer deze kolomvectoren.')

    @staticmethod
    def a_deel4(b):
        print('Antwoord:')

        s_x = np.array([[0, 1], [1, 0]])
        w, v = np.linalg.eig(s_x)

        # constante maakt in principe weinig uit, nog orthonormalisatie
        # elke kolom is een eigenvector
        v = v / np.abs(v)
        # eerste eigenvector alfa=beta en tweede eigenvector alfa=-beta
        v1 = v[:, 0]
        print('kolomvector 1: ', v1)
        v2 = v[:, 1]
        print('kolomvector 2: ', v2)

    @staticmethod
    def t_deel5(b):
        print('Tip:')
        print('[[<v1|Sz|v1>, <v1|Sz|v2>],\n [<v2|Sz|v1>, <v2|Sz|v2>]]')
        print('Waarbij v1 de eerste eigenvector is en v2 de tweede.')

    @staticmethod
    def a_deel5(b):
        print('Antwoord:')

        s_x = np.array([[0, 1], [1, 0]])
        s_z = np.array([[1, 0], [0, -1]])
        w, v = np.linalg.eig(s_x)

        v = v / np.abs(v)
        v1 = v[:, 0]
        v2 = v[:, 1]

        s_z2 = [[np.transpose(v1) @ v1, np.transpose(v1) @ v2], \
                [np.transpose(v2) @ v1, np.transpose(v2) @ v2]] * s_z

        print(s_z2)

class wc3_oef1:

    @staticmethod
    def t_deel2(b):
        print('Tip:')
        print('Leid W eens af naar c_A en eens naar c_B (gebruik diff(...)) en stel beide afgeleiden ',
              '= 0. Zet beide vergelijkingen om in matrixvorm met sympy.linear_eq_to_matrix(...)',
              'en los het eigenwaardeprobleem op met sympy.solve(...). Vergeet niet dat Haa en ',
              'Hbb hetzelfde zijn, substitueer Hbb dus door Haa.')

    @staticmethod
    def a_deel2(b):
        print('Antwoord:')
        H = Operator('\hat{H}')
        one = Operator('\hat{1}')
        ca, cb = symbols('c_A c_B')
        phi_a, phi_b = symbols('\phi_A \phi_B', constant=True)
        HAA, HBB, HAB, S = symbols('H_{AA} H_{BB} H_{AB} S')

        W = ((ca * Bra(phi_a) + cb * Bra(phi_b)) * H * \
             (ca * Ket(phi_a) + cb * Ket(phi_b))) / \
            ((ca * Bra(phi_a) + cb * Bra(phi_b)) * one * \
             (ca * Ket(phi_a) + cb * Ket(phi_b)))

        W = expand(W)

        W = W.subs(Bra(phi_a) * H * Ket(phi_a), HAA) \
            .subs(Bra(phi_a) * H * Ket(phi_b), HAB) \
            .subs(Bra(phi_b) * H * Ket(phi_a), HAB) \
            .subs(Bra(phi_b) * H * Ket(phi_b), HBB) \
            .subs(Bra(phi_a) * one * Ket(phi_a), 1) \
            .subs(Bra(phi_a) * one * Ket(phi_b), S) \
            .subs(Bra(phi_b) * one * Ket(phi_a), S) \
            .subs(Bra(phi_b) * one * Ket(phi_b), 1)

        W = simplify(W)

        dca = diff(W, ca)
        dcb = diff(W, cb)

        E = Symbol('E')
        dca = simplify(dca.subs(W, E))
        dcb = simplify(dcb.subs(W, E))

        t_dca, _ = fraction(dca)
        t_dcb, _ = fraction(dcb)

        A, b = linear_eq_to_matrix((expand(t_dca), expand(t_dcb)), ca, cb)
        A = A.subs(HBB, HAA)

        E1, E2 = solve(A.det(), E)

        print('E1: ')
        pprint(E1)
        print('E2: ')
        pprint(E2)

class wc3_oef2:

    @staticmethod
    def t_deel1(b):
        print('Tip:')
        print('Maak een (6x3) NumPy-array. Het coordinatenbestand bevat ',
              'alle cijfers als strings. Vergeet deze dus niet om te zetten in floats.')

    @staticmethod
    def a_deel1(b):
        print('Antwoord:')
        coord = []
        with open('benzene.xyz') as file:
            for line in file:
                line = list(map(float, line.rstrip().split()))
                coord.append(line)

        coord = np.array(coord)
        print(coord)

    @staticmethod
    def t_deel2(b):
        print('Tip:')
        print('Gebruik scipy.spatial.distance.cdist(...) om de afstanden tussen ',
              'de verschillende atomen te berekenen. De diagonaalelementen zijn = 0.')

    @staticmethod
    def a_deel2(b):
        print('Antwoord:')
        coord = []
        with open('benzene.xyz') as file:
            for line in file:
                line = list(map(float, line.rstrip().split()))
                coord.append(line)

        coord = np.array(coord)
        bindleng = cdist(coord, coord)
        print(bindleng)

    @staticmethod
    def t_deel3(b):
        print('Tip:')
        print('Maak een (6x6)-matrix met np.zeros(...), vervang de relevante ',
              'posities door alfa en beta. Alfa en beta zijn SymPy-symbolen.')

    @staticmethod
    def a_deel3(b):
        print('Antwoord:')
        coord = []
        with open('benzene.xyz') as file:
            for line in file:
                line = list(map(float, line.rstrip().split()))
                coord.append(line)

        coord = np.array(coord)
        bindleng = cdist(coord, coord)

        huckel = sp.zeros(coord.shape[0])

        a_pos = np.where(bindleng == 0.)
        b_pos = np.where(np.abs(bindleng - 1.445) < 1.445)
        # dit omdat een normale bindingslengte tussen C-C 2.91 bohr bedraagt

        al, be = sp.symbols(r'\alpha \beta')

        for r, k in zip(a_pos[0], a_pos[1]):
            huckel[r, k] = al

        for r, k in zip(b_pos[0], b_pos[1]):
            huckel[r, k] = be

        pprint(huckel)

    @staticmethod
    def t_deel4(b):
        print('Tip:')
        print('Gebruik de methode sp.diagonalize() om de Huckelmatrix ',
              'te diagonaliseren. Zet de eigenvectoren om naar een NumPy array.')

    @staticmethod
    def a_deel4(b):
        print('Antwoord:')
        coord = []
        with open('benzene.xyz') as file:
            for line in file:
                line = list(map(float, line.rstrip().split()))
                coord.append(line)

        coord = np.array(coord)
        bindleng = cdist(coord, coord)

        huckel = sp.zeros(coord.shape[0])

        a_pos = np.where(bindleng == 0.)
        b_pos = np.where(np.abs(bindleng - 1.445) < 1.445)
        # dit omdat een normale bindingslengte tussen C-C 2.91 bohr bedraagt

        al, be = sp.symbols(r'\alpha \beta')

        for r, k in zip(a_pos[0], a_pos[1]):
            huckel[r, k] = al

        for r, k in zip(b_pos[0], b_pos[1]):
            huckel[r, k] = be

        v, w = huckel.diagonalize()
        v = np.array(v)  # .astype(np.float64)
        print(v)

    @staticmethod
    def t_deel5(b):
        print('Tip:')
        print('Gebruik plt.scatter(...) waar je de x- en y-coordinaten meegeeft. ',
              'De eigenvectoren worden in functie van x en y geplot.')

    @staticmethod
    def a_deel5(b):
        print('Antwoord:')
        coord = []
        with open('benzene.xyz') as file:
            for line in file:
                line = list(map(float, line.rstrip().split()))
                coord.append(line)

        coord = np.array(coord)
        bindleng = cdist(coord, coord)

        huckel = sp.zeros(coord.shape[0])

        a_pos = np.where(bindleng == 0.)
        b_pos = np.where(np.abs(bindleng - 1.445) < 1.445)
        # dit omdat een normale bindingslengte tussen C-C 2.91 bohr bedraagt

        al, be = sp.symbols(r'\alpha \beta')

        for r, k in zip(a_pos[0], a_pos[1]):
            huckel[r, k] = al

        for r, k in zip(b_pos[0], b_pos[1]):
            huckel[r, k] = be

        v, w = huckel.diagonalize()
        v = np.array(v)  # .astype(np.float64)

        fig, axes = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(6, 6))

        i = 0

        for lijn in axes:
            for ax in lijn:
                ax.scatter(coord[:, 0], coord[:, 1], c=v[:, i], s=250, cmap=cm.coolwarm)
                # kies zelf welke kleurmap jullie duidelijker vinden
                # ax.scatter(coord[:,0], coord[:,1], c=v[:,i], s=200, cmap='bwr')

                i += 1
                ax.set_xlim([-4, 4])
                ax.set_ylim([-4, 4])
                ax.set(aspect='equal', adjustable='box-forced')

        fig.show()

class hf:

    @staticmethod
    def geef_nuclrep():
        with open('data_HF_SCF/enuc.dat', 'r') as file:
            e_nucl = float(file.readline().rstrip())

        return e_nucl

    @staticmethod
    def geef_coreH():

        def generate_matrix(filename):
            matrix = np.zeros((7, 7))
            with open(filename, 'r') as file:
                for line in file:
                    line = list(map(float, line.rstrip().split()))
                    i, j, s = line[0], line[1], line[2]
                    matrix[int(i - 1)][int(j - 1)] = s
            return np.triu(matrix.T, 1) + matrix

        T = generate_matrix('data_HF_SCF/kinetic.dat')
        V = generate_matrix('data_HF_SCF/nucl_attr.dat')
        core_H = T + V

        return core_H

    @staticmethod
    def geef_overlap():
        def generate_matrix(filename):
            matrix = np.zeros((7, 7))
            with open(filename, 'r') as file:
                for line in file:
                    line = list(map(float, line.rstrip().split()))
                    i, j, s = line[0], line[1], line[2]
                    matrix[int(i - 1)][int(j - 1)] = s
            return np.triu(matrix.T, 1) + matrix

        overlap = generate_matrix('data_HF_SCF/overlap.dat')

        return overlap

    @staticmethod
    def geef_twee_elektron():
        def permutations(mu, nu, lamb, sigma):
            perm_munu = set(itertools.permutations([mu, nu], 2))
            perm_lamsi = set(itertools.permutations([lamb, sigma], 2))
            return set(itertools.chain(set(itertools.product(perm_munu, perm_lamsi)),
                                       set(itertools.product(perm_lamsi, perm_munu))))

        eri = np.zeros((7, 7, 7, 7))
        with open('data_HF_SCF/eri.dat', 'r') as file:
            for line in file:
                line = line.rstrip().split()
                mu, nu, lamb, sigma = int(line[0]) - 1, int(line[1]) - 1, int(line[2]) - 1, int(line[3]) - 1
                value = float(line[4])

                perms = permutations(mu, nu, lamb, sigma)
                for perm in perms:
                    i, j, k, l = perm[0][0], perm[0][1], perm[1][0], perm[1][1]
                    eri[i][j][k][l] = value

        return eri

    @staticmethod
    def t_deel4():
        print('Tip:')
        print('Gebruik np.linalg.eigh(...) om de eigenwaarden en eigenvectoren ',
              'van de overlapmatrix S te vinden. Deze heb je nodig om de orthogonalisatiematrix ',
              'op te stellen.')

    def a_deel4(b):
        print('Antwoord:')
        n = 7

        def generate_matrix(filename):
            matrix = np.zeros((n, n))
            with open(filename, 'r') as file:
                for line in file:
                    line = list(map(float, line.rstrip().split()))
                    i, j, s = line[0], line[1], line[2]
                    matrix[int(i - 1)][int(j - 1)] = s
            return np.triu(matrix.T, 1) + matrix

        overlap = generate_matrix('data_HF_SCF/overlap.dat')

        eigvalues, eigvectors = LA.eigh(overlap)
        diag_matrix = diag(eigvalues)
        S_12 = eigvectors @ np.sqrt(LA.inv(diag_matrix)) @ eigvectors.T
        print(S_12)

    @staticmethod
    def t_deel5(b):
        print('Tip:')
        print('Gebruik hier ook np.linalg.eigh(...). Voor de sommatie over de ',
              'bezette atoomorbitalen gebruik je np.einsum(...).')

    @staticmethod
    def a_deel5(b):
        print('Antwoord:')
        count_lines = wc4_oef1.file_len('data_HF_SCF/overlap.dat')
        n = int((-1 + math.sqrt(1 + 8 * (count_lines))) / 2)

        def generate_matrix(filename):
            matrix = np.zeros((n, n))
            with open(filename, 'r') as file:
                for line in file:
                    line = list(map(float, line.rstrip().split()))
                    i, j, s = line[0], line[1], line[2]
                    matrix[int(i - 1)][int(j - 1)] = s
            return np.triu(matrix.T, 1) + matrix

        overlap = generate_matrix('data_HF_SCF/overlap.dat')
        T = generate_matrix('data_HF_SCF/kinetic.dat')
        V = generate_matrix('data_HF_SCF/nucl_attr.dat')
        core_H = T + V

        eigvalues, eigvectors = LA.eigh(overlap)
        diag_matrix = diag(eigvalues)
        S_12 = eigvectors @ np.sqrt(LA.inv(diag_matrix)) @ eigvectors.T

        f0 = S_12.T @ core_H @ S_12

        def density(fock_orth):
            f_eigvalues, f_eigvectors = LA.eigh(fock_orth)
            coeff = S_12 @ f_eigvectors

            coeff_r = coeff[:, 0:5]
            return np.einsum('ij,kj->ik', coeff_r, coeff_r)

        dens = density(f0)
        print(dens)

    @staticmethod
    def t_deel6(b):
        print('Tip:')
        print('De elektronische energiematrix bereken je door: 2*densiteitsmatrix*core_H. ',
              'Al deze elementen sommeer je om tot de elektronische energie te komen.')

    @staticmethod
    def a_deel6(b):
        print('Antwoord:')
        with open('data_HF_SCF/enuc.dat', 'r') as file:
            e_nucl = float(file.readline().rstrip())

        count_lines = wc4_oef1.file_len('data_HF_SCF/overlap.dat')
        n = int((-1 + math.sqrt(1 + 8 * (count_lines))) / 2)

        def generate_matrix(filename):
            matrix = np.zeros((n, n))
            with open(filename, 'r') as file:
                for line in file:
                    line = list(map(float, line.rstrip().split()))
                    i, j, s = line[0], line[1], line[2]
                    matrix[int(i - 1)][int(j - 1)] = s
            return np.triu(matrix.T, 1) + matrix

        overlap = generate_matrix('data_HF_SCF/overlap.dat')
        T = generate_matrix('data_HF_SCF/kinetic.dat')
        V = generate_matrix('data_HF_SCF/nucl_attr.dat')
        core_H = T + V

        def permutations(mu, nu, lamb, sigma):
            perm_munu = set(itertools.permutations([mu, nu], 2))
            perm_lamsi = set(itertools.permutations([lamb, sigma], 2))
            return set(itertools.chain(set(itertools.product(perm_munu, perm_lamsi)),
                                       set(itertools.product(perm_lamsi, perm_munu))))

        eri = np.zeros((n, n, n, n))
        with open('data_HF_SCF/eri.dat', 'r') as file:
            for line in file:
                line = line.rstrip().split()
                mu, nu, lamb, sigma = int(line[0]) - 1, int(line[1]) - 1, int(line[2]) - 1, int(line[3]) - 1
                value = float(line[4])

                perms = permutations(mu, nu, lamb, sigma)
                for perm in perms:
                    i, j, k, l = perm[0][0], perm[0][1], perm[1][0], perm[1][1]
                    eri[i][j][k][l] = value

        eigvalues, eigvectors = LA.eigh(overlap)
        diag_matrix = diag(eigvalues)
        S_12 = eigvectors @ np.sqrt(LA.inv(diag_matrix)) @ eigvectors.T

        f0 = S_12.T @ core_H @ S_12

        def density(fock_orth):
            f_eigvalues, f_eigvectors = LA.eigh(fock_orth)
            coeff = S_12 @ f_eigvectors

            coeff_r = coeff[:, 0:5]
            return np.einsum('ij,kj->ik', coeff_r, coeff_r)

        dens = density(f0)

        e_el = np.sum(dens * 2 * core_H)
        e_tot = e_el + e_nucl
        print('De totale energie is: ', e_tot)

    @staticmethod
    def t_deel7(b):
        print('Tip:')
        print('Hier moet je de eri-matrix transponeren met np.transpose(...). Ook ',
              'is het gebruikelijk om hier np.einsum(...) te gebruiken.')

    @staticmethod
    def a_deel7(b):
        print('Antwoord:')

    @staticmethod
    def t_deel11(b):
        print('Tip:')
        print('')

    @staticmethod
    def a_deel11(b):
        print('Antwoord:')