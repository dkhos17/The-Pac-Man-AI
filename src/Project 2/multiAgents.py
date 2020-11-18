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
import random, util, sys

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

        "*** YOUR CODE HERE ***"
        x0, y0 = newPos
        # initial value with difference between scores after action, here we also avoid to meet ghost 
        eval = successorGameState.getScore() - currentGameState.getScore()

        dist = sys.maxint
        # counts distance between pacman and nearest ghost
        for gstate in newGhostStates:
          x, y = gstate.getPosition()
          dist = min(dist, abs(x-x0)+abs(y-y0))
        # if ghost is near decrase value to avoid ghost
        if dist <= 2: eval -= dist*100

        dist = sys.maxint
        # counts distance between pacman and nearest food
        for food in currentGameState.getFood().asList():
          x, y = food
          dist = min(dist, abs(x-x0)+abs(y-y0))
        eval -= dist # as near as food is as good it is
        # and while eval_value is good as high it is we need (-dist)

        return eval


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
        """
        "*** YOUR CODE HERE ***"
        # return argmax for pacman all possible actions, set minint and maxint for alpha and beta agents 
        return self.max_agent(self, gameState, -sys.maxint, sys.maxint, self.depth)[1]

    ## This method is for maximaizer agent(alpha-pacman), which always 
    # provides an optimal move (assuming that opponent agents(beta-ghosts) is playing optimally
    # where depth is how many moves left pacman - in other words depth is according to pacman nodes
    @staticmethod
    def max_agent(self, gameState, alpha, beta, depth, idx = 0):
      # check if max-depth is reached or game is finished 
      if gameState.isWin() or gameState.isLose() or depth <= 0:
        return self.evaluationFunction(gameState), None
      
      best_action = Directions.STOP
      Max = -sys.maxint
      # iterate all possible actions
      for action in gameState.getLegalActions(idx):
        # get beta agents score 
        scr,_ = self.min_agent(self, gameState.generateSuccessor(0, action), alpha, beta, depth)
        Max = max(Max, scr) # save max score
        if Max > alpha: # check if score updated and save better action
          best_action = action
          alpha = Max
      # return best score with best action
      return Max, best_action

    ## This method is for minimaizer agents(beta-ghosts), which always 
    # provides an optimal move (assuming that opponent agents(alpha-pacman) is playing optimally
    # where depth is how many moves left pacman - in other words depth is according to pacman nodes
    # and than makes moves for all ghosts
    @staticmethod
    def min_agent(self, gameState, alpha, beta, depth, idx = 1):
      # check if max-depth is reached or game is finished 
      if gameState.isWin() or gameState.isLose() or depth <= 0:
        return self.evaluationFunction(gameState), None
      
      best_action = Directions.STOP 
      Min = sys.maxint
      # iterate all possible actions
      for action in gameState.getLegalActions(idx):
        # check if its last ghost and change depth 
        if idx+1 >= gameState.getNumAgents():
          scr,_ = self.max_agent(self, gameState.generateSuccessor(idx, action), alpha, beta, depth-1)
        else: # if there are ghosts which have to make moves, we dont change depth
          scr,_ = self.min_agent(self, gameState.generateSuccessor(idx, action), alpha, beta, depth, idx+1)
 
        Min = min(Min, scr) # save min value
        if Min < beta: # check if min value updated and save better action
          best_action = action
          beta = Min
      # return min possible score which ghosts can make for pacman, action should be ignored
      return Min, best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # return argmax (by score) for pacman all possible actions, set minint and maxint for alpha and beta agents 
        return self.max_agent(self, gameState, -sys.maxint, sys.maxint, self.depth)[1]

    ## This method is for maximaizer agent(alpha-pacman), which always 
    # provides an optimal move (assuming that opponent agents(beta-ghosts) is playing optimally
    # where depth is how many moves left pacman - in other words depth is according to pacman nodes
    # here different from minimax we avoid extra movements which will not change answer
    @staticmethod
    def max_agent(self, gameState, alpha, beta, depth, idx = 0):
      # check if max-depth is reached or game is finished 
      if gameState.isWin() or gameState.isLose() or depth <= 0:
        return self.evaluationFunction(gameState), None
      
      best_action = Directions.STOP
      Max = -sys.maxint
      # iterate all possible actions
      for action in gameState.getLegalActions(idx):
        scr,_ = self.min_agent(self, gameState.generateSuccessor(0, action), alpha, beta, depth)
        Max = max(Max, scr) # save max score
        if Max > beta: # check if root already has better score and stop expanding extra states 
          return Max, action
        if Max > alpha: # check if value updated and save better action
          best_action = action
          alpha = Max
      # return best score with best action
      return Max, best_action

     ## This method is for minimaizer agents(beta-ghosts), which always 
    # provides an optimal move (assuming that opponent agents(alpha-pacman) is playing optimally
    # where depth is how many moves left pacman - in other words depth is according to pacman nodes
    # and than makes moves for all ghosts. here different from minimax we avoid extra movements which will not change answer.
    # aplha and beta play as they were paying in minimax, but doesn't make extra actions.
    @staticmethod
    def min_agent(self, gameState, alpha, beta, depth, idx = 1):
      # check if max-depth is reached or game is finished 
      if gameState.isWin() or gameState.isLose() or depth <= 0:
        return self.evaluationFunction(gameState), None
      
      best_action = Directions.STOP 
      Min = sys.maxint
      # iterate all possible actions
      for action in gameState.getLegalActions(idx):
        if idx+1 >= gameState.getNumAgents(): # check if its last ghost and change deth
          scr,_ = self.max_agent(self, gameState.generateSuccessor(idx, action), alpha, beta, depth-1)
        else: # there are left ghost to make actions
          scr,_ = self.min_agent(self, gameState.generateSuccessor(idx, action), alpha, beta, depth, idx+1)
          
        Min = min(Min, scr) # save min score
        if Min < alpha: # check if root already has better score and stop expanding extra states 
          return Min, action
        if Min < beta: # check if value updated and save better action
          best_action = action
          beta = Min
      # return best score for gohst and worst for pacman, action should be ignored
      return Min, best_action

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
        return self.pacman_agent(self, gameState, -sys.maxint, 0, self.depth)[1]
    
    ## This method is for maximaizer agent(alpha-pacman), which always 
    # provides an optimal move (assuming that opponent agents(beta-ghosts) is taking
    # average (ecpected mean) score, where depth is how many moves left pacman - in other words depth is according to pacman nodes
    # here alpha agent(pacman) plays as he was palying in minimax and beta(ghosts) plays with Avrg. 
    @staticmethod
    def pacman_agent(self, gameState, alpha, beta, depth, idx = 0):
      # check if max-depth is reached or game is finished 
      if gameState.isWin() or gameState.isLose() or depth <= 0:
        return self.evaluationFunction(gameState), None
      
      best_action = Directions.STOP
      Max = -sys.maxint
      # iterate all possible actions
      for action in gameState.getLegalActions(idx):
        # get ghosts avrg score
        scr,_ = self.ghost_agent(self, gameState.generateSuccessor(0, action), alpha, beta, depth)
        Max = max(Max, scr) # save max score between avrg. scores
        if Max > alpha: # check if max value updated and save better action
          best_action = action
          alpha = Max
      # return max score with best action
      return Max, best_action

    ## This method is for minimaizer agents(beta-ghosts), which always 
    # provides to take avarage score, where depth is how many moves left pacman,
    # and than makes moves for all ghosts. aplha plays as he was paying in minimax, but not as in AlphaBetaPruning.
    @staticmethod
    def ghost_agent(self, gameState, alpha, beta, depth, idx = 1):
      # check if max-depth is reached or game is finished 
      if gameState.isWin() or gameState.isLose() or depth <= 0:
        return self.evaluationFunction(gameState), None
      
      Avrg = 0
      actions = gameState.getLegalActions(idx) # get all possible actions for n
      flt = 1.0/float(len(actions)) # for mean coef. 1/n 
      # iterate all possible actions
      for action in actions:
        if idx+1 >= gameState.getNumAgents(): # check if its last ghost and change depth
          scr,_ = self.pacman_agent(self, gameState.generateSuccessor(idx, action), alpha, beta, depth-1)
        else: # there are gohst left to make actions
          scr,_ = self.ghost_agent(self, gameState.generateSuccessor(idx, action), alpha, beta, depth, idx+1)
        # sum all path scores and multyply it flt (mean coefficient)
        Avrg += flt*scr
      # return Avrg score, action is ignred (no need).
      return Avrg, None

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    def eval_score(GameState): #evals score for gamestate
      # check if game is finished
      if currentGameState.isWin() or currentGameState.isLose(): return currentGameState.getScore()
      pos = currentGameState.getPacmanPosition()
      food = currentGameState.getFood()
      ghosts = currentGameState.getGhostStates()
      
      eval = currentGameState.getScore() # init eval with current Score
      eval -= 2.7*food.count() # decrease with c*(food number) as least it is as good it is. c = eiler_num
      eval -= 3.14*len(ghosts) # decrease with c*(gohst number) as least it is as good it is. c = pi_num

      min_dist = sys.maxint
      # save distance between pacman and nearest food
      for f in food.asList():
        min_dist = min(min_dist, manhattanDistance(pos,f))

      # if there is no food left, we dont need to chane eval
      if min_dist != sys.maxint:eval -= min_dist
      
      return eval
    # return score for crrentGameState. as high it is as good it is.
    return eval_score(currentGameState)

# Abbreviation
better = betterEvaluationFunction

