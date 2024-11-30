# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import*
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):

    """Search the deepest nodes in the search tree first."""
    startState = problem.getStartState()
    if problem.isGoalState(startState):
        return []  # If the start state is already the goal

    # Use a stack to manage frontier, storing tuples of (state, path)
    frontier = util.Stack()
    frontier.push((startState, []))  # Start state with an empty path
    explored = set()  # To keep track of visited nodes

    while not frontier.isEmpty():
        currentState, currentPath = frontier.pop()
        # Add the current state to the explored set
        if currentState in explored:
            continue
        explored.add(currentState)
        # Check if the current state is the goal state
        if problem.isGoalState(currentState):
            return currentPath
        # Expand successors and push them to the stack
        for successor, action, stepCost in problem.getSuccessors(currentState):
            if successor not in explored:
                newPath = currentPath + [action]  # Add the action to the path
                frontier.push((successor, newPath))

    # Return an empty list if no solution is found
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    """ Search the shallowest nodes in the search tree first. """
    currPath = []           # The path that is popped from the frontier in each loop
    currState =  problem.getStartState()    # The state(position) that is popped for the frontier in each loop
    print(f"currState: {currState}")
    if problem.isGoalState(currState):     # Checking if the start state is also a goal state
        return currPath

    frontier = Queue()
    frontier.push( (currState, currPath) )     # Insert just the start state, in order to pop it first
    explored = set()
    while not frontier.isEmpty():
        currState, currPath = frontier.pop()    # Popping a state and the corresponding path
        # To pass autograder.py question2:
        if problem.isGoalState(currState):
            return currPath
        explored.add(currState)
        frontierStates = [ t[0] for t in frontier.list ]
        for s in problem.getSuccessors(currState):
            if s[0] not in explored and s[0] not in frontierStates:
                # Lecture code:
                # if problem.isGoalState(s[0]):
                #     return currPath + [s[1]]
                frontier.push( (s[0], currPath + [s[1]]) )      # Adding the successor and its path to the frontier

    return []       # If this point is reached, a solution could not be found.

def uniformCostSearch(problem):
  
    """Search the node of least total cost first."""
    currPath = []  # The path that is popped from the frontier in each loop
    currState = problem.getStartState()  # The state(position) that is popped from the frontier in each loop
    currCost = 0  # Current cost to reach `currState`

    if problem.isGoalState(currState):  # Checking if the start state is also a goal state
        return currPath

    frontier = PriorityQueue()  # UCS uses a priority queue for the frontier
    frontier.push((currState, currPath, currCost), currCost)  # Push the start state with priority = cost
    explored = set()  # To keep track of explored nodes

    while not frontier.isEmpty():
        currState, currPath, currCost = frontier.pop()  # Popping a state, the corresponding path, and cost
        
        if currState not in explored:
            explored.add(currState)

            if problem.isGoalState(currState):  # Check if the current state is a goal state
                return currPath

            for successor, action, stepCost in problem.getSuccessors(currState):
                if successor not in explored:
                    newCost = currCost + stepCost  # Calculate the new cost to reach `successor`
                    frontier.push((successor, currPath + [action], newCost), newCost)  # Push successor with updated cost

    return []  # If this point is reached, no solution was found


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
