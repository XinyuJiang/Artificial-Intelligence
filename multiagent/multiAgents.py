# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print "newPos:",newPos
        #print "newFood",newFood.asList()
        #print "newGhoststate:",newGhostStates
        #print "newScaredTimes:",newScaredTimes
        #print "successorGameState:",successorGameState
        #print "successorGameState.getScore:",successorGameState.getScore()
        #print "successorGameState.getPacmanState:",successorGameState.getPacmanState()
        #print "successorGameState.getPacmanPosition:",successorGameState.getPacmanPosition()
        "*** YOUR CODE HERE ***"
        minlist=50
        minghost=5
        consist=10
        foodmain=currentGameState.getNumFood()
        newfoodmain=successorGameState.getNumFood()
        if newfoodmain==0:
          return 1000
        if foodmain>newfoodmain:
          consist=100
        for foodlist in newFood.asList():
        #if newFood.asList(successorGameState.getPacmanPosition):
          minlist=min(manhattanDistance(newPos,foodlist),minlist)
        for ghostlist in newGhostStates:
          minghost=min(manhattanDistance(ghostlist.getPosition(),newPos),minghost)
        for foodlist in newFood.asList():  
          if manhattanDistance(newPos,foodlist)==minlist:
        #    for ghostp in newGhostStates:
              #if action==STOP:
              #  return 0
            #  print "ghostindexxxxxxxxxxxxxxx:",ghostindex
            #  print "ghostpositionxxxxxxxxxxxxxxx:",newGhostStates(ghostindex).getPosition()
            if  minghost<2:
              return -100000
            return (minghost+consist-manhattanDistance(newPos,foodlist))
            #  else :return successorGameState.getScore() 
        #elif successorGameState.getPacmanPosition==newGhostStates:
         # return -1000000*successorGameState.getScore()
        #else :return successorGameState.getScore() 

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #while self.depth!=-1:
          #self=gameState.generateSuccessor(0,action)
          #legalMoves = gameState.getLegalActions()
          #sgamestate = [gameState.generateSuccessor(0,act) for act in legalMoves] #it is not a leaf 
          #MinimaxAgent(MultiAgentSearchAgent).getAction(sgamestate)
        def maxscore(depth,state):#pacman
          score=-999999.9
          #Moves=gameState.getLegalActions
          if depth==0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          else :
            for legalMoves in state.getLegalActions(0):
              score=max(score,minscore(depth,state.generateSuccessor(0,legalMoves),1))
            return score


        def minscore(depth,state,agent_i):#ghost
          score=999999.9
          if depth==0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          elif agent_i!=gameState.getNumAgents()-1 :
            for legalMoves in state.getLegalActions(agent_i):
              score=min(score,minscore(depth,state.generateSuccessor(agent_i,legalMoves),agent_i+1))
          else :
            for legalMoves in state.getLegalActions(agent_i):
              score=min(score,maxscore(depth-1,state.generateSuccessor(agent_i,legalMoves)))
          return score
        score=-999999.9
        act=Directions.STOP
        for legalMoves in gameState.getLegalActions(0):
        #one circle
          #if legalMoves != Directions.STOP:
          #notice the daxiaoxie
            state_next = gameState.generateSuccessor(0,legalMoves) 
            #enter next state
            score_old=score
            score=max(score,minscore(self.depth,state_next,1))
            if score>score_old:
              act=legalMoves
        return act   
         #else prefer stop
        #  for move in legalMoves:
           

        

      
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #global alpha
        #global beta
        global a1
        a1=(-99999)
        global b1
        b1=99999
        def maxscore(depth,state,alpha,beta):#pacman
          score=-999999.9
          #global alpha
          #global beta
          #print "alpha:",alpha
          #print "beta:",beta
          #Moves=gameState.getLegalActions
          #beta=99999
          #alpha=-99999

          if depth==0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          else :
            for legalMoves in state.getLegalActions(0):
              score=max(score,minscore(depth,state.generateSuccessor(0,legalMoves),1,alpha,beta))
              if score>beta:
                b1=score
                return score
              alpha=max(alpha,score)
            return score


        def minscore(depth,state,agent_i,alpha,beta):#ghost
          score=999999.9
          #global alpha
          #global beta
          #beta=99999
          #alpha=-99999
          if depth==0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          if agent_i!=gameState.getNumAgents()-1 :
            for legalMoves in state.getLegalActions(agent_i):
              score=min(score,minscore(depth,state.generateSuccessor(agent_i,legalMoves),agent_i+1,alpha,beta))
              if score<alpha:
                return score
              beta=min(beta,score)
          else :
            for legalMoves in state.getLegalActions(agent_i):
              score=min(score,maxscore(depth-1,state.generateSuccessor(agent_i,legalMoves),alpha,beta))
              if score<alpha:
              #  a1=score
                return score
              beta=min(beta,score)
          return score

        score=-999999.9
        #alpha=-99999
        #beta=99999
        #a1=-999999
        #b1=999999
        act=Directions.STOP
        for legalMoves in gameState.getLegalActions(0):
        #one circle
          #if legalMoves != Directions.STOP:
          #notice the daxiaoxie
            state_next = gameState.generateSuccessor(0,legalMoves) 
            #enter next state
            score_old=score
            score=max(score,minscore(self.depth,state_next,1,a1,b1))
            #a1=score
            #if score>b1:
            #  b1=score
            #  return score
            a1=max(a1,score)
            if score>score_old:
              act=legalMoves
        return act 

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxscore(depth,state):#pacman
          score=-999999.9
          #Moves=gameState.getLegalActions
          if depth==0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          else :
            for legalMoves in state.getLegalActions(0):
              score=max(score,minscore(depth,state.generateSuccessor(0,legalMoves),1))
            return score


        def minscore(depth,state,agent_i):#ghost
          score=0.0
          count=0
          if depth==0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          elif agent_i!=gameState.getNumAgents()-1 :
            for legalMoves in state.getLegalActions(agent_i):
              score+=minscore(depth,state.generateSuccessor(agent_i,legalMoves),agent_i+1)
              count=count+1
          else :
            for legalMoves in state.getLegalActions(agent_i):
              score+=maxscore(depth-1,state.generateSuccessor(agent_i,legalMoves))
              count=count+1
          return score/count
        score=-999999.9
        act=Directions.STOP
        for legalMoves in gameState.getLegalActions(0):
        #one circle
          #if legalMoves != Directions.STOP:
          #notice the daxiaoxie
            state_next = gameState.generateSuccessor(0,legalMoves) 
            #enter next state
            score_old=score
            score=max(score,minscore(self.depth,state_next,1))
            if score>score_old:
              act=legalMoves
        return act   
         #else prefer stop
        #  for move in legalMoves:
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score=currentGameState.getScore()
    minlist=500
    minghost=500
    #consist=10
    foodmain=currentGameState.getNumFood()
    newfoodmain=currentGameState.getNumFood()
    if newfoodmain==0:
      return score+10000
    #if foodmain>newfoodmain:
    #  consist=1000
    for foodlist in newFood.asList():
        #if newFood.asList(successorGameState.getPacmanPosition):
      minlist=min(manhattanDistance(newPos,foodlist),minlist)
    for ghostlist in newGhostStates:
      minghost=min(manhattanDistance(ghostlist.getPosition(),newPos),minghost)
    #for foodlist in newFood.asList():  
    #  if manhattanDistance(newPos,foodlist)==minlist:
        #    for ghostp in newGhostStates:
              #if action==STOP:
              #  return 0
            #  print "ghostindexxxxxxxxxxxxxxx:",ghostindex
            #  print "ghostpositionxxxxxxxxxxxxxxx:",newGhostStates(ghostindex).getPosition()
      if  minghost<3:
        return -100000
    return score-(2*(minghost+1)+(manhattanDistance(newPos,foodlist)+1))
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

