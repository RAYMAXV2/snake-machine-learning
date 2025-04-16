import random
import itertools
import numpy
from NeuralNetwork import *

nbFeatures = 8
nbActions = 4

class Game:
    def __init__(self, hauteur, largeur):
        self.grille = [[0]*hauteur  for _ in range(largeur)]
        self.hauteur, self.largeur = hauteur, largeur
        self.serpent = [[largeur//2-i-1, hauteur//2] for i in range(4)]
        for (x,y) in self.serpent: self.grille[x][y] = 1
        self.direction = 3
        self.accessibles = [[x,y] for (x,y) in list(itertools.product(range(largeur), range(hauteur))) if [x,y] not in self.serpent]
        self.fruit = [0,0]
        self.setFruit()
        self.enCours = True
        self.steps = 0
        self.score = 4
    
    def setFruit(self):
        if (len(self.accessibles)==0): return
        self.fruit = self.accessibles[random.randint(0, len(self.accessibles)-1)][:]
        self.grille[self.fruit[0]][self.fruit[1]] = 2

    def refresh(self):
        nextStep = self.serpent[0][:]
        match self.direction:
            case 0: nextStep[1]-=1
            case 1: nextStep[1]+=1
            case 2: nextStep[0]-=1
            case 3: nextStep[0]+=1

        if nextStep not in self.accessibles:
            self.enCours = False
            return
        self.accessibles.remove(nextStep)
        if self.grille[nextStep[0]][nextStep[1]]==2:
            self.setFruit()
            self.steps = 0
            self.score+=1
        else:
            self.steps+=1
            self.grille[self.serpent[-1][0]][self.serpent[-1][1]] = 0
            self.accessibles.append(self.serpent[-1][:])
            self.serpent = self.serpent[:-1]
            if self.steps>self.hauteur*self.largeur:
                self.enCours = False
                return

        self.grille[nextStep[0]][nextStep[1]] = 1
        self.serpent = [nextStep]+self.serpent

    def getFeatures(self):
        features = numpy.zeros(8)
        x, y = self.serpent[0]

        #1. Est-ce qu’il y a un obstacle directement au dessus de la tête du serpent (0 ou 1) ?
        features[0] = 1 if y == 0 or self.grille[x][y - 1] == 1 else 0

        #2. Est-ce qu’il y a un obstacle directement en dessous de la tête du serpent (0 ou 1)?
        features[1] = 1 if y == self.hauteur-1 or self.grille[x][y + 1] == 1 else 0

        #3. Est-ce qu’il y a un obstacle directement à gauche de la tête du serpent (0 ou 1)?
        features[2] = 1 if x == 0 or self.grille[x - 1][y] == 1 else 0

        #4. Est-ce qu’il y a un obstacle directement à droite de la tête du serpent (0 ou 1)?
        features[3] = 1 if x == self.largeur-1 or self.grille[x + 1][y] == 1 else 0

        #5. Est-ce que le fruit se trouve au dessus (1), en dessous (-1) ou sur la même ligne (0) que la tête du serpent?
        if self.fruit[1] < y:
            features[4] = 1  
        elif self.fruit[1] > y:
            features[4] = -1  
        else:
            features[4] = 0 

        #6. Est-ce que le fruit se trouve à droite (1), à gauche (-1) ou sur la même colonne(0) que la tête du serpent?
        if self.fruit[0] > x:
            features[5] = 1
        elif self.fruit[0] < x:
            features[5] = -1 
        else:
            features[5] = 0 

        #7. Quelle est la direction du serpent (0, 1, 2 ou 3) ?
        features[6] = self.direction

        #8. A quelle distance se trouve le bord, compte tenu de la direction actuelle ?
        if self.direction == 0:
            features[7] = y
        elif self.direction == 1:
            features[7] = self.hauteur - 1 - y
        elif self.direction == 2:
            features[7] = x
        else:
            features[7] = self.largeur - 1 - x

        return features
    
    def print(self):
        print("".join(["="]*(self.largeur+2)))
        for ligne in range(self.hauteur):
            chaine = ["="]
            for colonne in range(self.largeur):
                if self.grille[colonne][ligne]==1: chaine.append("#")
                elif self.grille[colonne][ligne]==2: chaine.append("F")
                else: chaine.append(" ")
            chaine.append("=")
            print("".join(chaine))
        print("".join(["="]*(self.largeur+2))+"\n")

