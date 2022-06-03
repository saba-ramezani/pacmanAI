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
from math import inf
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        action_score = successorGameState.getScore()
        #foods
        foods = newFood.asList()
        dist_to_foods = [inf]
        for food in foods:
            dist_to_food = manhattanDistance(newPos, food)
            dist_to_foods.append(dist_to_food)
        closest_food = min(dist_to_foods)
        action_score += 1/(1+closest_food)
        #ghosts
        ghost_scores = [0]
        i=0
        for ghost in newGhostStates:
            dist_to_ghost = manhattanDistance(newPos, ghost.getPosition())
            ghost_scared_time = newScaredTimes[i]
            if dist_to_ghost < ghost_scared_time:
                ghost_scores.append(inf)
            elif dist_to_ghost <=1:
                ghost_scores.append(-inf)
            i+=1
        best_ghost = min(ghost_scores)
        action_score += best_ghost

        return action_score

        #return successorGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        value, action = self.minimax(self.index, self.depth + 1, gameState)
        return action
        #util.raiseNotDefined()




    def minimax(self,agent, depth, state):
        agent_num = state.getNumAgents()
        best_action = None

        next_depth = depth
        if agent == 0:
            next_depth = depth - 1
        if state.isLose() or state.isWin() or next_depth == 0:
            return self.evaluationFunction(state), best_action
        best_value = inf
        if agent == 0:
            best_value = -inf
        next_agent = agent + 1
        if agent_num == next_agent:
            next_agent = 0
        for a in state.getLegalActions(agent):
            successor_state = state.generateSuccessor(agent, a)
            action_value, _ = self.minimax(next_agent, next_depth, successor_state)
            if agent == 0:
                if action_value >= best_value:
                    best_value = action_value
                    best_action = a
            elif action_value <= best_value:
                best_value = action_value
                best_action = a
        return best_value, best_action




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        _, action = self.alpha_beta(gameState, self.depth + 1, -inf, inf, self.index)
        return action
        #util.raiseNotDefined()



    def alpha_beta(self,state, depth, alpha, beta, agent):

        agent_num = state.getNumAgents()

        next_depth = depth
        if agent == 0:
            next_depth = depth - 1

        best_action = None
        if state.isLose() or state.isWin() or next_depth == 0:
            return self.evaluationFunction(state), best_action

        next_agent = agent + 1
        if agent_num == next_agent:
            next_agent = 0

        best_value = inf
        if agent == 0:
            best_value = -inf

        for a in state.getLegalActions(agent):
            successor_state = state.generateSuccessor(agent, a)
            action_value, _ = self.alpha_beta(successor_state, next_depth, alpha, beta, next_agent)

            if agent == 0:
                if action_value >= best_value:
                    best_value = action_value
                    best_action = a

                if best_value > beta:
                    return best_value, best_action

                alpha = max(alpha, best_value)

            else:
                if action_value <= best_value:
                    best_value = action_value
                    best_action = a

                if best_value < alpha:
                    return best_value, best_action

                beta = min(beta, best_value)

        return best_value, best_action




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
        agent = self.index
        if agent != 0:
            return random.choice(gameState.getLegalActions(agent))

        _, action = self.expectimax(gameState, self.depth + 1, self.index)
        return action
        #util.raiseNotDefined()

    def expectimax(self, state, depth, agent):
        agent_num = state.getNumAgents()
        next_depth = depth
        if agent == 0:
            next_depth = depth - 1

        best_action = None
        best_value = -inf
        if state.isLose() or state.isWin() or next_depth == 0:
            return self.evaluationFunction(state), best_action

        next_agent = agent + 1
        if agent_num == next_agent:
            next_agent = 0

        legal_actions = state.getLegalActions(agent)

        if agent != 0:
            probability = 1.0 / float(len(legal_actions))
            average_value = 0.0

            for a in legal_actions:
                successor_state = state.generateSuccessor(agent, a)
                action_value, _ = self.expectimax(successor_state, next_depth, next_agent)

                average_value += probability * action_value

            return average_value, None

        for a in legal_actions:
            successor_state = state.generateSuccessor(agent, a)
            action_value, _ = self.expectimax(successor_state, next_depth, next_agent)

            if action_value >= best_value:
                best_value = action_value
                best_action = a

        return best_value, best_action




def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    final_score = currentGameState.getScore()
    curr_pos = currentGameState.getPacmanPosition()

    dist_food = []
    for food in currentGameState.getFood().asList():
        dist = 1.0 / manhattanDistance(curr_pos, food)
        dist_food.append(dist)

    if len(dist_food) > 0:
        final_score += max(dist_food)

    dist_ghost = 0
    STEPS = 10
    POW = 2
    for ghost in currentGameState.getGhostStates():
        dist_ghost = manhattanDistance(curr_pos, ghost.getPosition())

        if ghost.scaredTimer > 0:
            dist_ghost += pow(max(STEPS - dist_ghost, 0), POW)

        else:
            dist_ghost -= pow(STEPS, POW)

    final_score += dist_ghost

    dist_capsules = []
    WEIGHT = 50.0
    for capsules in currentGameState.getCapsules():
        dist = WEIGHT / manhattanDistance(curr_pos, capsules)
        dist_capsules.append(dist)

    if len(dist_capsules) > 0:
        final_score += max(dist_capsules)

    return final_score

    #util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
