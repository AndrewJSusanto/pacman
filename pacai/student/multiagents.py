import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """
        
        # Useful information you can extract:
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        score = 0
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = currentGameState.getPacmanPosition()
        ghostPositions = currentGameState.getGhostPositions()
        closestGhost = min(distance.manhattan(newPosition, ghostPosition) for ghostPosition in ghostPositions)
        oldFood = currentGameState.getFood()
        closestFood = min(distance.manhattan(newPosition, foodPosition) for foodPosition in oldFood.asList())
        score = closestFood / (closestGhost * 20)
        if action == "Stop":
            score -= 50

        return successorGameState.getScore() + score 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
    
    def getAction(self, gameState): # get_value generates a pair in the form of (value, action), and getAction returns the action (index 1) 
        result = self.get_value(0, 0, gameState)
        return result[1]
        
    def get_value(self, index, depth, gameState): # returns values differing depending on whether or not state is terminal, max-, or min-
        if len(gameState.getLegalActions(index)) == 0: # if index has no legal actions, then terminal case.
            return (gameState.getScore(), "") # Terminal case, game is over
        if index == 0:
            return self.max_value(index, depth, gameState) # Pacman's index is always 0: Max agent
        else:
            return self.min_value(index, depth, gameState) # Ghosts indices are always > 1: Min agent (index is not 0 or )
            
        
    def max_value(self, index, depth, gameState):  # Max utility value for max agent
        legalMoves = gameState.getLegalActions(index) 
        max_value = float("-inf")
        action = ""
        for checkAction in legalMoves:
            successor = gameState.generateSuccessor(index, checkAction)
            successorIndex = index + 1
            successorDepth = depth
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1
            current_value = self.get_value(successor, successorIndex, successorDepth)[0]
            if current_value > max_value:
                max_value = current_value
                action = checkAction
        return max_value, action
    
    def min_value(self, index, depth, gameState): # Min utility value for ghosts / min agent
        legalMoves = gameState.getLegalActions(index)
        min_value = float("inf")
        action = ""
        for checkAction in legalMoves:
            successor = gameState.generateSuccesor(index, checkAction)
            successorIndex = index + 1
            successorDepth = depth
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1
            current_value = self.get_value(successor, successorIndex, successorDepth)[0]
            if current_value < min_value:
                min_value = current_value
                action = checkAction
        return min_value, action
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    return currentGameState.getScore()

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
