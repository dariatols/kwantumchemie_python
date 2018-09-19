import numpy as np
from matplotlib import pyplot as plt

class wc1_oef1():
    @staticmethod
    def t_deel1(b):
        print('Maak een functie die enkel het sinusgedeelte',
              'van de golfvergelijking teruggeeft. Hierbij maak je gebruik',
              'van numpy.sin(...) en numpy.pi.')

    @staticmethod
    def a_deel1(b):
        print('De functie geeft np.sin( (n*np.pi*x)/l ) terug.')

    @staticmethod
    def t_deel2(b):
        print('Gebruik hiervoor np.arange(...) of np.linspace(...)')

    @staticmethod
    def a_deel2(b):
        x = np.linspace(0, 10, 1000)
        print(x)

    @staticmethod
    def t_deel3(b):
        print('Gebruik plt.plot(...) om de drie toestanden te plotten.')

    @staticmethod
    def a_deel3(b):
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

