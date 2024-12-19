# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        minFoodDistance = float('inf')
        for food in currentGameState.getFood().asList():
            minFoodDistance = min(minFoodDistance, manhattanDistance(currentGameState.getPacmanPosition(), food))

        minNewFoodDistance = float('inf')
        for food in newFood.asList():
            minNewFoodDistance = min(minNewFoodDistance, manhattanDistance(newPos, food))

        minNewGhostDistance = float('inf')
        for ghost in newGhostStates:
            minNewGhostDistance = min(minNewGhostDistance, manhattanDistance(newPos, ghost.getPosition()))
        
        if minNewGhostDistance <= 1:
            return 0
        if action == Directions.STOP:
            return 0
        if successorGameState.getScore() > currentGameState.getScore():
            return 40
        if minNewFoodDistance < float('inf') and minNewFoodDistance < minFoodDistance:
            return 20
        return 10


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def __init__(self):
            self.nextAction = None

        def maxPlayer(state, depth):
            depth += 1
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            # initial
            value = float('-inf')
            actions = state.getLegalActions(0)
            for action in actions:
                successor= state.generateSuccessor(0, action)
                minValue = minPlayer(successor, depth, 1)
                if value <= minValue:
                    value = minValue
                    if depth == 0:
                        self.nextAction = action
            return value
        
        def minPlayer(state, depth, agentIndex):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # initial
            value = float('inf')
            actions = state.getLegalActions(agentIndex)
            for action in actions:
                successor= state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    value = min(value, maxPlayer(successor, depth))
                else:
                    # ghosts
                    value = min(value, minPlayer(successor, depth, agentIndex+1))
            return value
        
        maxPlayer(gameState, -1)
        return self.nextAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def __init__(self):
            self.nextAction = None

        def maxPlayer(state, depth, alpha, beta):
            depth += 1
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            # initial
            value = float('-inf')
            actions = state.getLegalActions(0)
            for action in actions:
                successor= state.generateSuccessor(0, action)
                minValue = minPlayer(successor, depth, 1, alpha, beta)
                if value <= minValue:
                    value = minValue
                    if depth == 0:
                        self.nextAction = action
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
            return value
        
        def minPlayer(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # initial
            value = float('inf')
            actions = state.getLegalActions(agentIndex)
            for action in actions:
                successor= state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    value = min(value, maxPlayer(successor, depth, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                else:
                    # ghosts
                    value = min(value, minPlayer(successor, depth, agentIndex+1, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
            return value
        
        maxPlayer(gameState, -1, -float('inf'), float('inf'))
        return self.nextAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def __init__(self):
            self.nextAction = None

        def maxPlayer(state, depth):
            depth += 1
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            # initial
            value = float('-inf')
            actions = state.getLegalActions(0)
            for action in actions:
                successor= state.generateSuccessor(0, action)
                minValue = expPlayer(successor, depth, 1)
                if value <= minValue:
                    value = minValue
                    if depth == 0:
                        self.nextAction = action
            return value
       
        def expPlayer(state, depth, agentIndex):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # initial
            value = 0.0
            actions = state.getLegalActions(agentIndex)
            p=float(1/len(actions))
            for action in actions:
                successor= state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    value += p*maxPlayer(successor, depth)
                else:
                    # ghosts
                    value += p*expPlayer(successor, depth, agentIndex+1)
            return value
        
        maxPlayer(gameState, -1)
        return self.nextAction
    

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()

    ghostStates = currentGameState.getGhostStates()
    minGhostDistance = float('inf')
    scaredTimer = -1
    for ghost in ghostStates:
        scaredTimer = ghost.scaredTimer
        minGhostDistance = min(minGhostDistance, manhattanDistance(pos, ghost.getPosition()))

    minCapsuleDistance = float('inf')
    for capsule in currentGameState.getCapsules():
        minCapsuleDistance = min(minCapsuleDistance, manhattanDistance(pos, capsule))

    minFoodDistance = float('inf')
    for food in currentGameState.getFood().asList():
        minFoodDistance = min(minFoodDistance, manhattanDistance(pos, food))
    
    score = currentGameState.getScore()
    if minGhostDistance <= 1:
        return 0
    if scaredTimer > 0:
        return score + 10/minFoodDistance + 100/minCapsuleDistance + 50/minGhostDistance
    else:
        return score + 10/minFoodDistance + 10/minCapsuleDistance - 5/minGhostDistance
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
