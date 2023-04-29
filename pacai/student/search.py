"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """
    visited = []
    frontier = Stack()
    path = []
    start = (problem.startingState(), [])
    frontier.push(start)
    while frontier:
        state, path = frontier.pop()
        if state not in visited:
            visited.append(state)
            if problem.isGoal(state):
                return path
            children = problem.successorStates(state)
            for child, direction, _ in children:
                path_add = path + [direction]
                frontier.push((child, path_add))
    raise NotImplementedError()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """
    visited = []
    frontier = Queue()
    path = []
    start = (problem.startingState(), [])
    frontier.push(start)
    while frontier:
        state, path = frontier.pop()
        if state not in visited:
            visited.append(state)
            if problem.isGoal(state):
                return path
            children = problem.successorStates(state)
            for child, direction, _ in children:
                path_add = path + [direction]
                frontier.push((child, path_add))
    raise NotImplementedError()

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    visited = []
    frontier = PriorityQueue()
    path = []
    frontier.push((problem.startingState(), [], 0), 0)
    while frontier:
        state, path, curr_cost = frontier.pop()
        if state not in visited:
            visited.append(state)
            if problem.isGoal(state):
                return path
            children = problem.successorStates(state)
            for child, direction, add_cost in children:
                path_add = path + [direction]
                priority = curr_cost + add_cost
                frontier.push((child, path_add, priority), priority)
    raise NotImplementedError()

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    visited = []
    frontier = PriorityQueue()
    path = []
    frontier.push((problem.startingState(), [], 0), 0)
    while frontier:
        state, path, curr_cost = frontier.pop()
        if state not in visited:
            visited.append(state)
            if problem.isGoal(state):
                return path
            children = problem.successorStates(state)
            for child, direction, add_cost in children:
                path_add = path + [direction]
                new_cost = curr_cost + add_cost
                heuristic_cost = new_cost + heuristic(child, problem)
                frontier.push((child, path_add, new_cost), heuristic_cost)
    raise NotImplementedError()
