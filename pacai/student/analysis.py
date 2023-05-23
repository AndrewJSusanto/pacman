"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    [Enter a description of what you did here.]
    The noise refers to the probability in which the agent randomizes its action.
    Tuning noise to 0 removes any possibility of an unintended state space being reached,
    and thus the possible transition states when evaluated for their utility would disregard
    any of the 'moat' states with negative impact and follow the bridge to the terminal state
    """
    answerDiscount = 0.9
    answerNoise = 0

    return answerDiscount, answerNoise

def question3a():
    """
    Prefer the close exit (+1), risking the cliff (-10)

    By removing stochasticity from transitions as well as having no living penalty,
    the agent can safely traverse to the close exit while still risking falling off the cliff
    (as no random choices means no chance to accidentally move to a negative state space)
    """
    answerDiscount = 0.3
    answerNoise = 0
    answerLivingReward = 0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Prefer the close exit (+1), but avoiding the cliff (-10)

    In order to prefer the close exit and still choose the long path,
    the discount rate should be stricter (smaller gamma leads to more impatient agents)
    and the noise should be greater than 0 to direct the agent away from risky state spaces.
    """
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Prefer the distant exit (+10), risking the cliff (-10)

    To prefer a distant exit, the discount rate should be weak,
    and stochasticity should be reduced to allow the agent to risk nearing the cliff.
    A living penalty also entices the agent to choose a shorter path.
    """
    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = -1

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    Prefer the distant exit (+10), avoiding the cliff (-10)

    Avoiding the cliff would require high stochasticity
    low living penalty (so that the agent can prioritize greater
    gain when comparing vs the utility of the close exit),
    and a weak discounting factor
    """
    answerDiscount = 0.9
    answerNoise = 0.5
    answerLivingReward = 0.2

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    Avoid both exits (also avoiding the cliff)

    We can avoid both exits by keeping the living reward very high, with no stochasticity.
    The agent will avoid the exits regardless of discounting because living forever would result in
    greater gain in the long run.
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 20.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    Given the circumstances in which the agent has 50 iterations and no visible goal,
    no epsilon 0 < e < 1 or learning rate 0 < e < 1 would result in a >99% likelihood of
    arriving at the optimal policy to cross the bridge. Theoretically, a high learning rate
    is the only possible way for the agent to have a chance in consistently arriving at the
    other end of the bridge within the available iterative span, but epsilon, in conjunction
    with the learning rate, would either push the agent to explore against its learning (if
    learning rate is high) or would lean towards pseudorandom results (learning rate low).
    """

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
