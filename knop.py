from ipywidgets import widgets
from IPython.display import display

class Knop():
    def __init__(self, soort):
        if soort == 'tip':
            self.knop = widgets.Button(description='Tip')
        else:
            self.knop = widgets.Button(description='Antwoord')

        return self.knop


    def tip_wc1_oef1(self):
        print('Dit is een tip voor deze vraag.')

    def antw_wc1_oef1(self):
        print('Dit is het antwoord.')
