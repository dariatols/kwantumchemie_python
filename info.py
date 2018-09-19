import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps

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
    def t_deel1():
        print('Tip:')
        print('Deze deelvraag is eigenlijk hetzelfde als in oefening 1.',
              ' Gebruik np.sin(...) om deze vraag op te lossen.')

    @staticmethod
    def a_deel1():
        print('Antwoord:')
        print('np.sin( (n * np.pi * x) / a )')

    @staticmethod
    def t_deel2():
        print('Tip:')
        print('Gebruik de functie uit de vorige deelvraag om de 4 eigenfuncties',
              'terug te geven met n=0, n=1, n=2, n=3. Hierna sommeer je deze.')

    @staticmethod
    def t_deel3():
        print('Tip:')
        print('Gebruik vergelijking 1.9 om de probabiliteitsdensiteit te schrijven.',
              'Hierna gebruik je scipy.integrate.simps om te integreren.',
              'Vergeet niet de vierkantswortel te nemen van de bekomen integraal.')

    @staticmethod
    def a_deel3():
        print('Antwoord:')
        x = np.linspace(0, 10, 1000)
        a = 10
        n = 1

        def eigenfunctie(x, a, n):
            return np.sin((n * np.pi * x) / a)

        def golffunctie():
            psi_x = np.zeros(1000)
            for n in range(5):
                psi_x += eigenfunctie(x, l, n)
            return psi_x

        def norm(psi):
            pd = np.conjugate(psi) * psi
            return np.sqrt(simps(pd, x))

        psi_x = golffunctie()
        print(norm(psi_x))

    @staticmethod
    def t_deel4():
        print('Tip:')
        print('Bereken hiervoor de norm van psi en deel de golffunctie',
              'psi door de berekende norm. Sla deze genormaliseerde',
              'golffunctie op in een nieuwe variabele.')

    @staticmethod
    def a_deel4():
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

wc1_oef1.t_deel1()
