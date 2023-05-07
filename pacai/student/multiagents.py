import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.directions import Directions
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
        newPosition = successorGameState.getPacmanPosition()
        ghostPositions = successorGameState.getGhostPositions()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
        closestGhost = \
            min(distance.manhattan(newPosition, ghostPosition) for ghostPosition in ghostPositions)
        farthestGhost = \
            max(distance.manhattan(newPosition, ghostPosition) for ghostPosition in ghostPositions)
        oldFood = currentGameState.getFood()
        closestFood = \
            min(distance.manhattan(newPosition, foodPosition) for foodPosition in oldFood.asList())
        if max(newScaredTimes) >= 0:
            score = score + (1 / (closestFood + 1) * 10) - farthestGhost
        else:
            for ghost in ghostPositions:
                if distance.manhattan(newPosition, ghost) <= 3:
                    closestFood = 9999
            score = score + (closestGhost) / ((closestFood + 1) * 10)
        # high value of closestGhost is good, low value of closestFood is good
        if action == "Stop":
            score -= 100
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

    def getAction(self, gameState):  # get_value returns (value, action) pair
        result = self.get_value(0, 0, gameState)
        return result[1]

    def get_value(self, index, depth, state):  # terminal, max node, min node casing
        if state.isOver() or depth == self.getTreeDepth():
            return self._evaluationFunction(state), ""
        if index == 0:
            return self.max_value(index, depth, state)  # Pacman
        else:
            return self.min_value(index, depth, state)  # Ghost

    def max_value(self, index, depth, state):  # Max utility value for max agent
        legalMoves = state.getLegalActions(index)
        if Directions.STOP in legalMoves:
            legalMoves.remove(Directions.STOP)
        max_value = float("-inf")
        action = ""
        for checkAction in legalMoves:
            successor = state.generateSuccessor(index, checkAction)
            successorIndex = index + 1
            successorDepth = depth
            current_value = self.get_value(successorIndex, successorDepth, successor)[0]
            if current_value > max_value:
                max_value = current_value
                action = checkAction
        return max_value, action

    def min_value(self, index, depth, state):  # Min utility value for ghosts / min agent
        legalMoves = state.getLegalActions(index)
        if Directions.STOP in legalMoves:
            legalMoves.remove(Directions.STOP)
        min_value = float("inf")
        action = ""
        for checkAction in legalMoves:
            successor = state.generateSuccessor(index, checkAction)
            successorIndex = index + 1
            successorDepth = depth
            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1
            current_value = \
                self.get_value(successorIndex, successorDepth, successor)[0]
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

    def getAction(self, gameState):
        result = self.get_value(0, 0, gameState, float("-inf"), float("inf"))
        return result[1]

    def get_value(self, index, depth, state, alpha, beta):
        if state.isOver() or depth == self.getTreeDepth():
            return self._evaluationFunction(state), ""
        if index == 0:
            return self.max_value(index, depth, state, alpha, beta)  # Pacman
        else:
            return self.min_value(index, depth, state, alpha, beta)  # Ghosts

    def max_value(self, index, depth, state, alpha, beta):
        legalMoves = state.getLegalActions(index)
        if Directions.STOP in legalMoves:
            legalMoves.remove(Directions.STOP)
        max_value = float("-inf")
        action = ""
        for checkAction in legalMoves:
            successor = state.generateSuccessor(index, checkAction)
            successorIndex = index + 1
            successorDepth = depth
            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1
            current_value = \
                self.get_value(successorIndex, successorDepth, successor, alpha, beta)[0]
            if current_value > max_value:
                max_value = current_value
                action = checkAction
            alpha = max(alpha, max_value)
            # Pruning
            if max_value > beta:
                return max_value, action
        return max_value, action

    def min_value(self, index, depth, state, alpha, beta):
        legalMoves = state.getLegalActions(index)
        if Directions.STOP in legalMoves:
            legalMoves.remove(Directions.STOP)
        min_value = float("inf")
        action = ""
        for checkAction in legalMoves:
            successor = state.generateSuccessor(index, checkAction)
            successorIndex = index + 1
            successorDepth = depth
            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1
            current_value = \
                self.get_value(successorIndex, successorDepth, successor, alpha, beta)[0]
            if current_value < min_value:
                min_value = current_value
                action = checkAction
            beta = min(beta, min_value)
            # Pruning
            if min_value < alpha:
                return min_value, action
        return min_value, action

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

    def getAction(self, gameState):
        result = self.get_value(0, 0, gameState)
        return result[1]

    def get_value(self, index, depth, state):
        if state.isOver() or depth == self.getTreeDepth():
            return self._evaluationFunction(state), ""
        if index == 0:
            return self.max_value(index, depth, state)  # Pacman
        else:
            return self.exp_value(index, depth, state)  # Ghosts

    def max_value(self, index, depth, state):  # Max utility value for max agent
        legalMoves = state.getLegalActions(index)
        if Directions.STOP in legalMoves:
            legalMoves.remove(Directions.STOP)
        max_value = float("-inf")
        action = ""
        for checkAction in legalMoves:
            successor = state.generateSuccessor(index, checkAction)
            successorIndex = index + 1
            successorDepth = depth
            current_value = self.get_value(successorIndex, successorDepth, successor)[0]
            if current_value > max_value:
                max_value = current_value
                action = checkAction
        return max_value, action

    def exp_value(self, index, depth, state):  # Min utility value for ghost
        legalMoves = state.getLegalActions(index)
        if Directions.STOP in legalMoves:
            legalMoves.remove(Directions.STOP)
        expected_value = 0
        expected_action = ""
        actionProb = 1.0 / len(legalMoves)
        for action in legalMoves:
            successor = state.generateSuccessor(index, action)
            successorIndex = index + 1
            successorDepth = depth
            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1
            current_value = self.get_value(successorIndex, successorDepth, successor)[0]
            expected_value += actionProb * current_value
        return expected_value, expected_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: Create linear combination using the inverse value of closest food, closest ghosts,
    and prioritizing escaping when the ghost comes near. Also takes into account capsules
    which are worth more points.
    """
    currentPosition = currentGameState.getPacmanPosition()
    ghostPositions = currentGameState.getGhostPositions()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.getScaredTimer() for ghostState in ghostStates]
    food = currentGameState.getFood().asList()
    numFood = len(food)
    numCapsules = len(currentGameState.getCapsules())
    closestFood = 1
    closestGhost = \
        min([distance.manhattan(currentPosition, ghost) for ghost in ghostPositions])

    score = currentGameState.getScore()

    # calculate food distances
    food_distances = \
        [distance.manhattan(currentPosition, food) for food in food]

    # closest food
    if numFood > 0:
        closestFood = min(food_distances)

    # ghost distances
    for ghost in ghostPositions:
        ghostDistance = distance.manhattan(currentPosition, ghost)
        # If ghost is too close to pacman, prioritize escaping instead of eating the closest food
        # by resetting the value for closest distance to food
        if ghostDistance < 2:
            closestFood = 99999
    # if the ghosts are scared, then the value of closest
    if min(scaredTimes) > 0:
        score += ((1.0 / closestGhost) * 10)
    # Linear combination of features
    return ((1.0 / closestFood) * 10) \
        + ((score) * 200) + ((numFood) * -100) \
        + ((numCapsules) * -10)

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
