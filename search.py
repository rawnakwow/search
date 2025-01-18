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
import time 
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
    """
    Search the deepest nodes in the search tree first.
    """
    start_time = time.time()  # Start the timer

    startState = problem.getStartState()  # Get the start state of the problem
    if problem.isGoalState(startState):  # Check if the start state is already the goal state
        print("Execution time: {:.6f} seconds".format(time.time() - start_time))
        return []

    frontier = util.Stack()  # Initialize the frontier using a stack
    frontier.push((startState, []))  # Push the start state and an empty path onto the stack
    explored = set()  # Set to keep track of explored states

    while not frontier.isEmpty():
        currentState, currentPath = frontier.pop()  # Pop a state and its path from the stack
        if currentState in explored:  # Skip if the state has already been explored
            continue
        explored.add(currentState)  # Mark the state as explored
        if problem.isGoalState(currentState):  # Check if the current state is the goal state
            print("Execution time: {:.6f} seconds".format(time.time() - start_time))
            return currentPath
        for successor, action, stepCost in problem.getSuccessors(currentState):  # Explore successors
            if successor not in explored:
                newPath = currentPath + [action]  # Create a new path including the current action
                frontier.push((successor, newPath))  # Push the successor and its path onto the stack

    print("Execution time: {:.6f} seconds".format(time.time() - start_time))
    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    start_time = time.time()  # Start the timer

    currState = problem.getStartState()  # Get the start state of the problem
    if problem.isGoalState(currState):  # Check if the start state is already the goal state
        print("Execution time: {:.6f} seconds".format(time.time() - start_time))
        return []

    frontier = util.Queue()  # Initialize the frontier using a queue
    frontier.push((currState, []))  # Push the start state and an empty path onto the queue
    explored = set()  # Set to keep track of explored states

    while not frontier.isEmpty():
        currState, currPath = frontier.pop()  # Pop a state and its path from the queue
        if problem.isGoalState(currState):  # Check if the current state is the goal state
            print("Execution time: {:.6f} seconds".format(time.time() - start_time))
            return currPath
        explored.add(currState)  # Mark the state as explored
        frontierStates = [t[0] for t in frontier.list]  # Get all states currently in the frontier
        for s in problem.getSuccessors(currState):  # Explore successors
            if s[0] not in explored and s[0] not in frontierStates:  # Check if the successor is unexplored
                frontier.push((s[0], currPath + [s[1]]))  # Push the successor and its path onto the queue

    print("Execution time: {:.6f} seconds".format(time.time() - start_time))
    return []

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    start_time = time.time()  # Start the timer

    currState = problem.getStartState()  # Get the start state of the problem
    if problem.isGoalState(currState):  # Check if the start state is already the goal state
        print("Execution time: {:.6f} seconds".format(time.time() - start_time))
        return []

    frontier = util.PriorityQueue()  # Initialize the frontier using a priority queue
    frontier.push((currState, [], 0), 0)  # Push the start state, empty path, and cost 0 onto the priority queue
    explored = set()  # Set to keep track of explored states

    while not frontier.isEmpty():
        currState, currPath, currCost = frontier.pop()  # Pop a state, its path, and cost from the priority queue
        if currState not in explored:  # Check if the state has already been explored
            explored.add(currState)  # Mark the state as explored
            if problem.isGoalState(currState):  # Check if the current state is the goal state
                print("Execution time: {:.6f} seconds".format(time.time() - start_time))
                return currPath
            for successor, action, stepCost in problem.getSuccessors(currState):  # Explore successors
                if successor not in explored:
                    newCost = currCost + stepCost  # Calculate the new cost to reach the successor
                    frontier.push((successor, currPath + [action], newCost), newCost)  # Push the successor onto the priority queue

    print("Execution time: {:.6f} seconds".format(time.time() - start_time))
    return []

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
