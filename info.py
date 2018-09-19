import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from scipy.integrate import quad

class wc1_oef1():
    @staticmethod
    def t_deel1(b):
        print('Tip:')
        print('Maak een functie die enkel het sinusgedeelte',
              'van de golfvergelijking teruggeeft. Hierbij maak je gebruik',
              'van numpy.sin(...) en numpy.pi.')

    @staticmethod
    def a_deel1(b):
        print('Antwoord:')
        print('De functie geeft np.sin( (n*np.pi*x)/l ) terug.')

    @staticmethod
    def t_deel2(b):
        print('Tip:')
        print('Gebruik hiervoor np.arange(...) of np.linspace(...)')

    @staticmethod
    def a_deel2(b):
        print('Antwoord:')
        x = np.linspace(0, 10, 1000)
        print(x)

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

class wc1_oef2():

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

class wc1_oef3():

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