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
import copy

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

### This function starts searching from start state and 
# using backtracking finds a path between start and goal
# states, which is 1st soultion in depth. curr is the current 
# state where the algo. is, been is a set of visited vertices, 
# actions is a list of movements(actions) and prolem is the 
# problem(SearchProblem) we'are solving. The function returns
# True if it reaches goal state and othr. - False
# if RET value is True function doesn't make backtracking and
# a real path which we'are searching is saved in actions list.
def my_dfs(curr, been, actions, problem):
    if problem.isGoalState(curr):return True
    if been.__contains__(curr):return False
    been.add(curr)
    for next in problem.getSuccessors(curr):
        actions.append(next[1])
        if my_dfs(next[0], been, actions, problem):return True
        actions.pop()
    been.remove(curr)
    return False
    
### The function(dfs) returns a list of actions which is needed to 
# reach goal state from start state. list is empty if we'are
# already in goal state or there is no solution to reach goal state
def depthFirstSearch(problem):
    #if we'are already in goal state
    if problem.isGoalState(problem.getStartState()): return list()

    been = set()
    actions = list()
    #dfs with rec. is saving path in 'actions' list
    my_dfs(problem.getStartState(), been, actions, problem)
    return actions

### This function(bfs) returns a list of actions which is needed to 
# reach goal state from start state. the algorithm (using Queue) explores 
# all of the neighbor states at the present depth prior to moving 
# on the states at the next lvl.(here in bfs a cost of all edge is 1 or not counted)
# RET list is empty if we'are already in goal state or there is no solution.
from util import Queue
def breadthFirstSearch(problem):
    #if we'are already in goal state
    if problem.isGoalState(problem.getStartState()): return list()
    been = set()
    Q = Queue()
    #put root node in queue, where node is tuple(state, actions's list)
    Q.push((problem.getStartState(), list()))
    while not Q.isEmpty():
        curr = Q.pop()
        # check if goal state is reached
        if problem.isGoalState(curr[0]):return curr[1]
        # check if goal state is visited
        if been.__contains__(curr[0]):continue
        been.add(curr[0])
        # explore the next lvl states in queue which parent is curr
        for next in problem.getSuccessors(curr[0]):
            way = list(curr[1])
            way.append(next[1])
            Q.push((next[0],way))

    return list()

### This function(ucs) returns a list of actions which is needed to 
# reach goal state from start state. the algorithm (using PriorityQueue)
# explores all of the neighbor states at the present depth prior to moving 
# on the states at the next lvl. and each state gives a priority(path cost) which one is  
# better to be reviewed at first, which helps function to find a path with less cost.
# the algorithm is based on bfs, but there a cost of path is counted and provided
# as edges have different costs RET list is empty if we'are 
# already in goal state or there is no solution.
from util import PriorityQueue
def uniformCostSearch(problem):
    #if we'are already in goal state
    if problem.isGoalState(problem.getStartState()): return list()
    been = set()
    Pq = PriorityQueue()
    #put root node in queue, where node is tuple(state, actions's list, cost of path)
    Pq.push((problem.getStartState(),list(),0.0), 0.0)
    while not Pq.isEmpty():
        curr = Pq.pop()
        # check if goal state is reached
        if problem.isGoalState(curr[0]):return curr[1]
        # check if goal state is visited
        if been.__contains__(curr[0]):continue
        been.add(curr[0])
        # explore the next lvl states in queue which parent is curr
        for next in problem.getSuccessors(curr[0]):
            way = list(curr[1])
            way.append(next[1])
            cost = curr[2]+next[2]
            # push new node where priority is cost of path
            Pq.push((next[0], way, cost), cost)
    
    return list()
        

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

### This function(A*) returns a list of actions which is needed to 
# reach goal state from start state. the algorithm (using PriorityQueue)
# explores all of the neighbor states at the present depth prior to moving 
# on the states at the next lvl. and each state gives a priority(path cost + heuristic)
# which one is better to be reviewed at first, which helps function to find a path with less cost.
# the algorithm is based on ucs, but there a priority of state is better and 
# more exact as heuristic function is used. RET list is empty 
# if we'are already in goal state or there is no solution.
def aStarSearch(problem, heuristic=nullHeuristic):
    #if we'are already in goal state
    if problem.isGoalState(problem.getStartState()): return list()
    been = set()
    Pq = PriorityQueue()
    #put root node in queue, where node is tuple(state, actions's list, cost of path + heuristic)
    Pq.push((problem.getStartState(),list(),0.0), 0.0)
    while not Pq.isEmpty():
        curr = Pq.pop()
        # check if goal state is reached
        if problem.isGoalState(curr[0]):return curr[1]
        # check if goal state is visited
        if been.__contains__(curr[0]):continue
        been.add(curr[0])
        # explore the next lvl states in queue which parent is curr
        for next in problem.getSuccessors(curr[0]):
            way = list(curr[1])
            way.append(next[1])
            cost = curr[2]+next[2]
            # push new node where priority is cost of path + heuristic of this node            
            Pq.push((next[0], way, cost), cost+heuristic(next[0],problem))
    
    return list()
    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
