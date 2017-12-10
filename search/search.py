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

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    from game import Directions
    fringe=Stack()   
    close=set()
    route=[]
    if problem.isGoalState(problem.getStartState())==False:
        fringe.push((problem.getStartState(),[]))
        while not fringe.isEmpty() :
            res=[]
            act=[]
            #print "fringe",fringe.list
            t=fringe.pop()
            if not t[0] in close :
                close.add(t[0])
                if problem.isGoalState(t[0]):
                    return t[1]
            #print "t[0]",t
                s=problem.getSuccessors(t[0])
            #print "st",s
                for temp in s:
                    res.append(temp[0])
                    act.append(temp[1])
                #res.reverse()
                #act.reverse()
            #print "temp",temp
            #print "res",res
            #print "act",act
                for i in range(len(res)):
                    if not res[i] in close :
                        newlist=[x for x in t[1]]
                        newlist.append((act[i]))
                        fringe.push((res[i],newlist))
    return 0


    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    from game import Directions
    fringe=Queue()
    close=set()
    route=[]
    if problem.isGoalState(problem.getStartState())==False:
        fringe.push((problem.getStartState(),[]))
        while not fringe.isEmpty() :
            res=[]
            act=[]
            #print "fringe",fringe.list
            t=fringe.pop()
            if not t[0] in close :
                close.add(t[0])
                if problem.isGoalState(t[0]):
                    return t[1]
                    #print "t[0]",t
                s=problem.getSuccessors(t[0])
                #print "st",s
                for temp in s:
                    res.append(temp[0])
                    act.append(temp[1])
                    #res.reverse()
                    #act.reverse()
                    #print "temp",temp
                    #print "res",res
                    #print "act",act
                for i in range(len(res)):
                    if not res[i] in close :
                        newlist=[x for x in t[1]]
                        newlist.append((act[i]))
                        fringe.push((res[i],newlist))
    return 0
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    from game import Directions
    fringe=PriorityQueue()
    close=set()
    if problem.isGoalState(problem.getStartState())==False:
        fringe.update((problem.getStartState(),[],0),0)
        while not fringe.isEmpty() :
            res=[]
            act=[]
            cos=[]
            #print "fringe",fringe.list
            t=fringe.pop()
            #print "t111",t[2]
            if not t[0] in close :
                close.add(t[0])
                if problem.isGoalState(t[0]):
                    return t[1]
                #print "t[1]",t[1]
                #print "t[0][2]",t[0][2]
                s=problem.getSuccessors(t[0])
                #print "st",s
                #print "t1",t[1]
                for temp in s:
                    res.append(temp[0])
                    act.append(temp[1])
                    cos.append(temp[2])
                    #res.reverse()
                    #act.reverse()
                    #print "temp",temp
                    #print "res",res
                #print "cos",cos
                for i in range(len(res)):
                    if not res[i] in close :
                        newlist=[x for x in t[1]]
                        newlist.append((act[i]))
                        #print "s",s[0][2]
                        fringe.update((res[i],newlist,cos[i]+t[2]),cos[i]+t[2])
    return 0
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    from searchAgents import manhattanHeuristic
    from game import Directions
    fringe=PriorityQueue()
    close=set()
    if problem.isGoalState(problem.getStartState())==False:
        fringe.update((problem.getStartState(),[],heuristic(problem.getStartState(),problem)),heuristic(problem.getStartState(),problem))
        while not fringe.isEmpty() :
            res=[]
            act=[]
            cos=[]
            #print "fringe",fringe.list
            t=fringe.pop()
            #print "t111",t[2]
            if not (t[0]) in close :
                close.add(t[0])
                if problem.isGoalState(t[0]):
                    return t[1]
                #print "t[1]",t[1]
                #print "t[0][2]",t[0][2]
                s=problem.getSuccessors(t[0])
                #print "st",s
                #print "t1",t[1]
                for temp in s:
                    res.append(temp[0])
                    act.append(temp[1])
                    cos.append(temp[2])
                    #res.reverse()
                    #act.reverse()
                    #print "temp",temp
                    #print "res",res
                #print "cos",cos
                for i in range(len(res)):
                    if not res[i] in close :
                        newcos=cos[i]+t[2]+heuristic(res[i],problem)
                        newlist=[x for x in t[1]]
                        newlist.append((act[i]))
                        #print "s",s[0][2]
                        fringe.update((res[i],newlist,cos[i]+t[2],newcos),newcos)
    return 0
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
