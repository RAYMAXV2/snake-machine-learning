import numpy
from NeuralNetwork import *
from snake import *
from concurrent.futures import ProcessPoolExecutor

def eval_wrapper(args):
    sol, gameParams = args
    return eval(sol, gameParams)

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
    return sol

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
            parent1 = numpy.random.choice(population[:tailleSelection])
            parent2 = numpy.random.choice(population[:tailleSelection])
            croisement(new_population, parent1, parent2, mr, pc, arch)

        # Évaluation et tri de la nouvelle population
        with ProcessPoolExecutor(max_workers=nbThreads) as executor:
            futures = [executor.submit(eval_wrapper, (sol, gameParams)) for sol in new_population]
            evaluated_new_population = [future.result() for future in futures]

        new_population = evaluated_new_population


        # Fusion de la nouvelle population avec les meilleurs de la génération actuelle
        population = new_population + population[:tailleSelection]
        population.sort(reverse=True, key=lambda sol: sol.score)

        print('Itération : ' + str(o))
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
        child1_nn =Individu(parent1.nn.clone())
        child2_nn = Individu(parent2.nn.clone())
    else:
        # Croisement des parents
        child1_nn = NeuralNetwork(parent1.nn.inputShape)
        child2_nn = NeuralNetwork(parent2.nn.inputShape)
        for layer in parent1.nn.layers:
            child1_nn.addLayer(layer.outputShape[0], "elu")
            child2_nn.addLayer(layer.outputShape[0], "elu")

        # Mélange des poids et biais
        for layer_idx in range(len(parent1.nn.layers)):

            parent1_layer = parent1.nn.layers[layer_idx]
            parent2_layer = parent2.nn.layers[layer_idx]
            child1_layer = child1_nn.layers[layer_idx]
            child2_layer = child2_nn.layers[layer_idx]

            alpha_weights = np.random.rand(*parent1_layer.weights.shape)
            alpha_bias = np.random.rand(*parent1_layer.bias.shape)

            child1_layer.weights = alpha_weights * parent1_layer.weights + (1 - alpha_weights) * parent2_layer.weights
            child2_layer.weights = (1 - alpha_weights) * parent1_layer.weights + alpha_weights * parent2_layer.weights

            child1_layer.bias = alpha_bias * parent1_layer.bias + (1 - alpha_bias) * parent2_layer.bias
            child2_layer.bias = (1 - alpha_bias) * parent1_layer.bias + alpha_bias * parent2_layer.bias
        # Mutation pour chacun des enfants
        mutation2(Individu(child1_nn), mr)
        mutation2(Individu(child2_nn), mr)
        # Ajout des enfants à la nouvelle population
        new_population.append(Individu(child1_nn))
        new_population.append(Individu(child2_nn))


def mutation2(child, mr):
    for layer in child.nn.layers:
        layerSize = layer.outputShape[0]
        previousLayerSize = layer.inputShape[0]

        pm_biais = mr / layerSize
        pm_poids = mr / previousLayerSize

        mask_biais = np.random.rand(layerSize) < pm_biais
        mutations_biais = np.random.randn(layerSize) * 0.1
        layer.bias += mask_biais * mutations_biais

        mask_poids = np.random.rand(previousLayerSize, layerSize) < pm_poids
        mutations_poids = np.random.randn(previousLayerSize, layerSize) * 0.1
        layer.weights += mask_poids * mutations_poids

def mutation(mr, child1_nn, child2_nn, layer_idx):
    # Probabilité de mutation pour les biais
    for layer in child1_nn.nn.layers:
        layerSize = layer.outputShape[0]
        previousLayerSize = layer.inputShape[0]

        pm_biais = mr / layerSize
        pm_poids = mr / previousLayerSize

        mask_biais = np.random.rand(layerSize) < pm_biais
        mutations_biais = np.random.randn(layerSize) * 0.1
        layer.bias += mask_biais * mutations_biais

        mask_poids = np.random.rand(previousLayerSize, layerSize) < pm_poids
        mutations_poids = np.random.randn(previousLayerSize, layerSize) * 0.1
        layer.weights += mask_poids * mutations_poids

    for layer in child2_nn.nn.layers:
        layerSize = layer.outputShape[0]
        previousLayerSize = layer.inputShape[0]

        pm_biais = mr / layerSize
        pm_poids = mr / previousLayerSize

        mask_biais = np.random.rand(layerSize) < pm_biais
        mutations_biais = np.random.randn(layerSize) * 0.1
        layer.bias += mask_biais * mutations_biais

        mask_poids = np.random.rand(previousLayerSize, layerSize) < pm_poids
        mutations_poids = np.random.randn(previousLayerSize, layerSize) * 0.1
        layer.weights += mask_poids * mutations_poids

