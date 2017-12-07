# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        self.valuestemp = util.Counter()
        for i in range  (self.iterations):#for every iteration
          for state in self.mdp.getStates():#for every state
            if not self.mdp.isTerminal(state):
              self.valuestemp[state] = self.getQValue(state,self.getAction(state))
          for state in self.mdp.getStates():#for every state
              self.values[state] = self.valuestemp[state]#it is not change immediately. After every loop we change the values.

        

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        term=0
        for nexts,prob in self.mdp.getTransitionStatesAndProbs(state,action):
          term += prob * (self.mdp.getReward(state,action,nexts) + self.discount * self.getValue(nexts))
        return term
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestaction = 'exit'
        maxx=-999999
        for action in self.mdp.getPossibleActions(state):
          if maxx < self.getQValue(state,action):
            maxx = self.getQValue(state,action)
            bestaction = action
        return bestaction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        i = 0
        tag = 0
        while tag == 0: 
          for state in self.mdp.getStates():#for every state
            if(i >= self.iterations):
              tag = 1
              break
            i += 1
            if not self.mdp.isTerminal(state):
              self.values[state] = self.getQValue(state,self.getAction(state))
           




class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #Compute predecessors of all states.
        close = set()
        for state in self.mdp.getStates():
          if self.mdp.isTerminal(state):
            continue
          for action in self.mdp.getPossibleActions(state):
            for stateaction in self.mdp.getTransitionStatesAndProbs(state,action):
              close.add((stateaction[0],state))#state is stateaction[0]'s predecessors
        #initialize an empty priority queue.
        pqueue = util.PriorityQueue()
        #for each non-terminal state s
        #Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this max Q-value represents what the value of s should be); call this number diff. Do NOT update self.values[s] in this step.
        for state in self.mdp.getStates():
          if not self.mdp.isTerminal(state):
            diff = abs(self.values[state]-self.getQValue(state,self.getAction(state)))
        #Push s into the priority queue with priority -diff (note that this is negative). We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
            pqueue.update(state,(-1)*diff)
        #for iteration in 0,1,2,...,self.iteration - 1
        for i in range (self.iterations):
          if not pqueue.isEmpty():
            s = pqueue.pop()
        #update s's value
          if not self.mdp.isTerminal(s):
            self.values[s] = self.getQValue(s,self.getAction(s))
        #for each predecessor p of s
          for temp in close:
            if temp[0] == s:
              diff = abs(self.values[temp[1]]-self.getQValue(temp[1],self.getAction(temp[1])))
              if diff > self.theta:
                pqueue.update(temp[1],(-1)*diff)