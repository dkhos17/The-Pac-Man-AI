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


import mdp, util, sys

from learningAgents import ValueEstimationAgent

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

        # Write value iteration code here
        for _ in range(0,self.iterations):
            new_values = util.Counter() #we just need the last values row to calculate news, V[k] new need just V[k-1]
            for state in self.mdp.getStates():
                if mdp.isTerminal(state): # terminal has always 0 
                    new_values[state] = self.values[state]
                    continue
                new_values[state] = -sys.maxint
                # V[k][s] = E(sum for all a with s') T(s,a,s')[R(s,a,s') + gama*V[k-1][s']]
                for action in self.mdp.getPossibleActions(state):
                    V = 0
                    for sp in self.mdp.getTransitionStatesAndProbs(state, action):
                        V += sp[1]*(self.mdp.getReward(state, action, sp[0]) + self.discount*self.values[sp[0]])
                    new_values[state] = max(new_values[state], V)

            self.values = new_values # update values

        
    # return the value of state
    # this value shows what is our 'expected' score from this cell.
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    # This function calculates Q(s,a), Q-value of action from state
    # this value shows how good is this action from this state, as high it is as good it is.
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q_value = 0
        # Q(s,a) = E(sum for all a with s') T(s,a,s')[R(s,a,s') + gama*V(s')]
        for sp in self.mdp.getTransitionStatesAndProbs(state, action):
            Q_value += sp[1]*(self.mdp.getReward(state, action, sp[0]) + self.discount*self.values[sp[0]])

        return Q_value

    # choose the best aciton from values
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        Q_values = util.Counter()
        # action = argmax E(sum for all a with s') T(s,a,s')[R(s,a,s') + gama*V(s')]
        for action in self.mdp.getPossibleActions(state):
            Q_values[action] = self.computeQValueFromValues(state, action)

        return Q_values.argMax() # return action with highest value

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
