from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}   # A dictionary which holds the q-values for each state.

        # Compute the values here.
        for state in self.mdp.getStates():  # Assign a value of 0 for each state
            if not self.mdp.isTerminal(state):
                self.values[state] = 0.0

        nextVals = self.values.copy()
        states = self.mdp.getStates()
        for i in range(iters):  # for each state in the set of states, compute V(s)
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                if not self.mdp.isTerminal(state):
                    stateVals = {}  # q-values for actions of this state
                    for action in actions:
                        stateVals[action] = self.getQValue(state, action)
                    nextVals[state] = stateVals[max(stateVals, key=stateVals.get)]
            self.values = nextVals.copy()

    def getQValue(self, state, action):
        tProbabilities = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0.0
        for tsp in tProbabilities:  # transition tuple of (state, probability)
            tState, probability = tsp
            reward = self.mdp.getReward(state, action, tState)
            tSample = self.discountRate * self.getValue(tState)  # sample = gamma * value(s')
            qValue += (probability * (reward + tSample))
        return qValue

    def getPolicy(self, state):
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        policy = {}
        for action in actions:
            policy[action] = self.getQValue(state, action)
        return max(policy, key=policy.get)  # returns action with best q-value

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)
