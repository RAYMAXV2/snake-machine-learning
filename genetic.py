import numpy
from NeuralNetwork import *
from snake import *
from concurrent.futures import ProcessPoolExecutor

def eval(sol, gameParams):
    # On récupère les paramètres de la partie (voir main.py)
    numberGame = gameParams["nbGames"]
    height = gameParams["height"]
    width = gameParams["width"]
    scoreTotal = 0

    # On crée une partie pour chaque individu
    for _ in range(numberGame):
        game = Game(height, width)
        while game.enCours:
            pred = sol.nn.predict(game.getFeatures())
            game.direction = pred
            game.refresh()
        # TODO (maybe ?) Prendre en compte que le score est à 4 de base
        apples = game.score 
        stepsSinceLastApple = game.steps
        scoreTotal += (1000 * apples + stepsSinceLastApple)
    
    sol.score = scoreTotal / (numberGame * height * width * 1000)
    return sol.score

'''
Représente une solution avec
_un réseau de neurones
_un score (à maximiser)

vous pouvez ajouter des attributs ou méthodes si besoin
'''
class Individu:
    def __init__(self, nn):
        self.nn = nn
        self.score = 0


'''
La méthode d'initialisation de la population est donnée :
_on génère N individus contenant chacun un réseau de neurones (de même format)
_on évalue et on trie des individus
'''
def initialization(taillePopulation, arch, gameParams):
    population = []
    for i in range(taillePopulation):
        nn = NeuralNetwork((arch[0],))
        for j in range(1, len(arch)):
            nn.addLayer(arch[j], "elu")
        population.append(Individu(nn))

    for sol in population: eval(sol, gameParams)
    population.sort(reverse=True, key=lambda sol:sol.score)
    
    return population

def optimize(taillePopulation, tailleSelection, pc, arch, gameParams, nbIterations, nbThreads, scoreMax, mr):
    population = initialization(taillePopulation, arch, gameParams)

    for o in range(nbIterations):
        new_population = []

        while len(new_population) < taillePopulation - tailleSelection:
            # Sélection des deux meilleurs parents
            parent1 = population[numpy.random.randint(0, tailleSelection)]
            parent2 = population[numpy.random.randint(0, tailleSelection)]
            croisement(new_population, parent1, parent2, mr, pc, arch)

        # Évaluation et tri de la nouvelle population
        for sol in new_population:
            eval(sol, gameParams)


        # Fusion de la nouvelle population avec les meilleurs de la génération actuelle
        population = new_population + population[:tailleSelection]
        population.sort(reverse=True, key=lambda sol: sol.score)

        print('Itération : ' + str(o))
        print('population : ' + str(len(population)))
        print('Score de la meilleure solution : ' + str(population[0].score))

        # Arrêt si le score maximum est atteint
        if population[0].score >= scoreMax:
            break

    return population[0].nn

    
def croisement(new_population, parent1, parent2, mr, pc, arch):
    # On tire un nombre aléatoire entre 0 et 1
    prob = numpy.random.rand()

    # Création des enfants
    if prob > pc:
        # Clonage des parents
        child1_nn = parent1.nn.clone()
        child2_nn = parent2.nn.clone()
    else:
        # Croisement des parents
        child1_nn = NeuralNetwork((arch[0],))
        child2_nn = NeuralNetwork((arch[0],))
        for i in range(1, len(arch)):
            child1_nn.addLayer(arch[i], "elu")
            child2_nn.addLayer(arch[i], "elu")

            # Mélange des poids et biais
            alpha = numpy.random.rand()
            layer_idx = i - 1  
            for j in range(child1_nn.layers[layer_idx].weights.shape[1]):
                child1_nn.layers[layer_idx].weights[:, j] = (
                alpha * parent1.nn.layers[layer_idx].weights[:, j]
                + (1 - alpha) * parent2.nn.layers[layer_idx].weights[:, j]
                )
                child2_nn.layers[layer_idx].weights[:, j] = (
                (1 - alpha) * parent1.nn.layers[layer_idx].weights[:, j]
                + alpha * parent2.nn.layers[layer_idx].weights[:, j]
                )
                child1_nn.layers[layer_idx].bias[j] = (
                alpha * parent1.nn.layers[layer_idx].bias[j]
                + (1 - alpha) * parent2.nn.layers[layer_idx].bias[j]
                )
                child2_nn.layers[layer_idx].bias[j] = (
                (1 - alpha) * parent1.nn.layers[layer_idx].bias[j]
                + alpha * parent2.nn.layers[layer_idx].bias[j]
                )
            # Mutation pour la couche layer_idx
        mutation(mr, child1_nn, child2_nn, layer_idx)
        # Ajout des enfants à la nouvelle population
        new_population.append(Individu(child1_nn))
        new_population.append(Individu(child2_nn))



def mutation(mr, child1_nn, child2_nn, layer_idx):
    # Probabilité de mutation pour les biais
    pmBias = mr / child1_nn.layers[layer_idx].outputShape[0]
    for j in range(child1_nn.layers[layer_idx].bias.shape[0]):
        if numpy.random.rand() < pmBias:
            child1_nn.layers[layer_idx].bias[j] += numpy.random.randn() * 0.1
        if numpy.random.rand() < pmBias:
            child2_nn.layers[layer_idx].bias[j] += numpy.random.randn() * 0.1
    
    # Probabilité de mutation pour les poids
    pmWeight = mr / child1_nn.layers[layer_idx].inputShape[0]
    for r in range(child1_nn.layers[layer_idx].weights.shape[0]):
        for c in range(child1_nn.layers[layer_idx].weights.shape[1]):
            if numpy.random.rand() < pmWeight:
                child1_nn.layers[layer_idx].weights[r, c] += numpy.random.randn() * 0.1
            if numpy.random.rand() < pmWeight:
                child2_nn.layers[layer_idx].weights[r, c] += numpy.random.randn() * 0.1
