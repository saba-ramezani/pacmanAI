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

import math
from itertools import cycle

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
        self.values = util.Counter()
        states = self.mdp.getStates()

        for i in range(self.iterations):

            res_val = {}
            for state in states:
                actions = self.mdp.getPossibleActions(state)

                q_state_values = (self.computeQValueFromValues(state, a) for a in actions)

                res_val[state] = max(q_state_values, default=0)

            self.values = util.Counter(res_val)


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
        states_probabilities = self.mdp.getTransitionStatesAndProbs(state, action)

        sum = 0
        for s, prob in states_probabilities:
            r = self.mdp.getReward(state, action, s) + self.discount * self.getValue(s)
            sum += prob * r

        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max_val = -math.inf
        action = None

        for a in self.mdp.getPossibleActions(state):
            val = self.computeQValueFromValues(state, a)
            if val > max_val:
                max_val = val
                action = a

        return action

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
        states = cycle(self.mdp.getStates())

        for i in range(self.iterations):
            state = next(states)
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)

                q_state_values = (self.computeQValueFromValues(state, a) for a in actions)

                self.values[state] = max(q_state_values, default=0)


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
        queue = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):

                actions = self.mdp.getPossibleActions(state)
                q_state_values = (self.computeQValueFromValues(state, a) for a in actions)
                max_val = max(q_state_values, default=0)

                diff = abs(self.values[state] - max_val)
                queue.push(state, -diff)
		
        temp = {}
        for state in self.mdp.getStates():
            temp[state] = []

        predecessors = util.Counter(temp)

        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for s, prob in self.mdp.getTransitionStatesAndProbs(state, action):

                    predecessors[s].append(state)

        for i in range(self.iterations):
            if queue.isEmpty():
                continue

            state = queue.pop()

            actions = self.mdp.getPossibleActions(state)
            q_state_values = (self.computeQValueFromValues(state, a) for a in actions)
            self.values[state] = max(q_state_values, default=0)

            for pre in predecessors[state]:
                actions = self.mdp.getPossibleActions(pre)
                q_state_values = (self.computeQValueFromValues(pre, a) for a in actions)
                max_val = max(q_state_values, default=0)

                diff = abs(self.values[pre] - max_val)

                if diff > self.theta:
                    queue.update(pre, -diff)
