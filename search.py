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
import math

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
    n = Directions.NORTH
    s = Directions.SOUTH
    e = Directions.EAST
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def mySearch(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    startState=problem.getStartState()
    childStates=problem.getSuccessors(startState)
    leftChild=childStates[0]

  
    print(startState)
    print(childStates)
    print(leftChild)
    return [s]


def mediumClassicSearch(problem):
    from game import Directions
    n = Directions.NORTH
    s = Directions.SOUTH
    e = Directions.EAST
    w = Directions.WEST
    return  [w,w,w,n,n,n,n,w,w,w,w,w,n,n,n,n,s,s,s,s,e,e,s,s,e,e,e,e,e,e,e,e,e,e,e,e,s,s,e,e,e]

def meduimMazeSearch(problem):
    from game import Directions
    n = Directions.NORTH
    s = Directions.SOUTH
    e = Directions.EAST
    w = Directions.WEST
    return [w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, s, s, s, s, s, s, s, s, s, e, e, e, n, n, n, n, n, n, n, e, e, s, s, s, s, s, s, e, e, n, n, n, n, n, n, e, e, s, s, s, s, e, e, n, n, e, e, e, e, e, e, e, e, s, s, s, e, e, e, e, e, e, e, s, s, s, s, s, s, s, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, s, w, w, w, w, w, w, w, w, w]



def bigMazeSearch(problem):
    from game import Directions
    n = Directions.NORTH
    s = Directions.SOUTH
    e = Directions.EAST
    w = Directions.WEST
    return [n, n, w, w, w, w, n, n, w, w, s, s, w, w, w, w, w, w, w, w, w, w, w, w, w, w, n, n, e, e, n, n, w, w, n, n, n, n, n, n, e, e, e, e, e, e, s, s, e, e, n, n, e, e, e, e, n, n, e, e, s, s, e, e, n, n, n, n, n, n, e, e, e, e, n, n, n, n, n, n, n, n, n, n, w, w, s, s, w, w, w, w, s, s, s, s, s, s, w, w, s, s, s, s, w, w, n, n, w, w, w, w, w, w, w, w, w, w, w, w, n, n, e, e, n, n, n, n, n, n, e, e, e, e, e, e, n, n, n, n, n, n, n, n, w, w, w, w, w, w, s, s, w, w, w, w, s, 
s, s, s, e, e, s, s, w, w, w, w, w, w, w, w, w, w, s, s, s, s, s, s, s, s, s, s, e, e, s, s, s, s, w, w, s, s, s, s, e, e, s, s, w, w, s, s, s, s, w, w, s, s]
    



def test(problem):
    currentState = problem.getStartState()
    children = problem.getSuccessors(currentState)
    print (children)
    return 
def getActionFromTriplet(triple):
    return triple [1]

def depthFirstSearch(problem):
    fringe=util.Stack()
    explored=[]
    startNode=(problem.getStartState(),[])
    fringe.push(startNode)
    while fringe:
        popped=fringe.pop()
        location=popped[0]
        path=popped[1]
        if location not in explored:
            explored.append(location)
            if problem.isGoalState(location):
                print(path)
                return path
            children=problem.getSuccessors(location)
            for child in list(children):
                if child[0] not in explored:
                    fringe.push((child[0],path+[child[1]]))
    return []

    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    fringe = Queue()
    explored=[]
    startNode=(problem.getStartState(),[])
    fringe.push(startNode)
    while fringe:
        popped=fringe.pop()
        location=popped[0]
        path=popped[1]
        if location not in explored:
            explored.append(location)
            if problem.isGoalState(location):
                print(len(path))
                return path
            children=problem.getSuccessors(location)
            for child in list(children):
                if child[0] not in explored:
                    fringe.push((child[0],path+[child[1]]))
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe=util.PriorityQueue()
    explored = set()
    startStateBlock = problem.getStartState()
    fringe.push((startStateBlock, []), 0)
    while (not fringe.isEmpty()):
        state = fringe.pop()
        stateBlock = state[0]
        statePath = state[1]
        explored.add(stateBlock)
        if problem.isGoalState(stateBlock):
            return statePath
        children = problem.getSuccessors(stateBlock)
        for child in children:
            actionToReachChild = child[1]
            costToReachChild = child[2]
            childPath = statePath[:]
            childPath.append(actionToReachChild)
            openList = [ x[0] for x in fringe.heap if x[0]==child[0]]
            inProcress = child[0] in explored or child[0] in openList
            if not inProcress:
                fringe.push( (child[0], childPath), costToReachChild)
            elif child[0] in openList:
                fringe.update((child[0], childPath), costToReachChild)
    return []
def manHattanHueristic (state, problem=None):
    cState = state
    gState = problem.goal
    hCost = abs(gState[0] - cState[0]) + abs(gState[1] - cState[1])
    return hCost

def euclideanDistance (state, problem=None):
    cState = state
    gState = problem.goal
    hCost = math.sqrt( ((gState[0] - cState[0])**2) + ((gState[1] - cState[1])**2) )
    return hCost

def temp(problem):
    fringe=util.PriorityQueue()
    explored = []
    startNode = problem.getStartState()
    cost = 0
    actions = []
    fringe.push((startNode, actions, cost), cost)
    while fringe:
        currentNode = fringe.pop()
        if problem.isGoalState(currentNode[0]):
            return currentNode[1]
        if currentNode[0] not in explored:
            explored.append(currentNode[0])
            for child, action, val in problem.getSuccessors(currentNode[0]):
                if child not in (fringe and explored):
                    fringe.push((child, currentNode[1] + [action], currentNode[2] + val), currentNode[2] + val)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    fringe=util.PriorityQueue()
    explored = set()
    startStateBlock = problem.getStartState()
    gCost = 0
    #hCost = manHattanHueristic(startStateBlock, problem)
    hCost = euclideanDistance(startStateBlock, problem)
    fCost = gCost + hCost
    fringe.push((startStateBlock, []), fCost)
    while (not fringe.isEmpty()):
        state = fringe.pop()
        stateBlock = state[0]
        statePath = state[1]
        explored.add(stateBlock)
        if problem.isGoalState(stateBlock):
            return statePath
        children = problem.getSuccessors(stateBlock)
        for child in children:
            actionToReachChild = child[1]
            gCost = child[2]
            #costToReachChild = child[2]
            #hCost = manHattanHueristic(child[0], problem)
            hCost = euclideanDistance(child[0], problem)
            fCost = gCost + hCost
            childPath = statePath[:]
            childPath.append(actionToReachChild)
            openList = [ x[0] for x in fringe.heap if x[0]==child[0]]
            inProcress = child[0] in explored or child[0] in openList
            if not inProcress:
                fringe.push( (child[0], childPath), fCost)
            elif child[0] in openList:
                fringe.update((child[0], childPath), fCost)
    return []
    
    



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
