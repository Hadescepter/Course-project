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
        for _ in range(self.iterations):
            updatting_V_value=util.Counter()
            current_states=self.mdp.getStates()
            current_states_num=len(current_states)
            for index in range(current_states_num):
                if current_states[index] == 'TERMINAL_STATE':
                    updatting_V_value['TERMINAL_STATE'] =0
                else:
                    max_Q_value= float('-inf')
                    Possible_actions=self.mdp.getPossibleActions(current_states[index])
                    if len(Possible_actions) is not None:
                        for possible_action in Possible_actions:
                            max_Q_value=max(self.computeQValueFromValues( current_states[index], possible_action),max_Q_value)
                        V_value=max_Q_value
                        updatting_V_value[current_states[index]]=V_value
                    else:
                        updatting_V_value[current_states[index]] = 0
            self.values=updatting_V_value.copy()
            
        


        


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
        total_next_action_list=(self.mdp.getTransitionStatesAndProbs(state, action))
        calculated_Q_value=0
        total_next_action_list_num=len(total_next_action_list)
        
        for  index in range(total_next_action_list_num):
            nextState , prob = total_next_action_list[index]
            possible_action_reward=self.mdp.getReward(state, action, nextState)           
            gamma=self.discount
            possible_single_action_value=possible_action_reward+gamma*self.values[nextState]
            calculated_Q_value+=prob*possible_single_action_value
        
        return(calculated_Q_value)
    

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if state == 'TERMINAL_STATE':
            return None
        Possible_actions=self.mdp.getPossibleActions(state)
        action_policy=None
        util.raiseNotDefined
        max_Q_value=float('-inf')
        if len(Possible_actions) is not None:
                    for possible_action in Possible_actions:
                        possible_action_Q_value=self.computeQValueFromValues( state, possible_action)
                        if max_Q_value < possible_action_Q_value:
                           max_Q_value=possible_action_Q_value
                           action_policy= possible_action
        return action_policy
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
        current_states=self.mdp.getStates()
        current_states_num=len(current_states)
        for epoch in range(self.iterations):
            updatting_V_value=0
            updatting_state=current_states[epoch % current_states_num]
            if updatting_state == 'TERMINAL_STATE':
                    continue
            else:
                max_Q_value= float('-inf')
                Possible_actions=self.mdp.getPossibleActions(updatting_state)
                if len(Possible_actions) is not None:
                    for possible_action in Possible_actions:
                        max_Q_value=max(self.computeQValueFromValues( updatting_state, possible_action),max_Q_value)
                    updatting_V_value=max_Q_value
                else:
                    updatting_V_value= 0
            self.values[updatting_state]=updatting_V_value


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
        predecessors_states = {}
        Prioritized_Sweeping_Value_Queue = util.PriorityQueue()
        current_states=self.mdp.getStates()
        current_states_num=len(current_states)
        for index in range(current_states_num):
            if current_states[index]=='TERMINAL_STATE':
                continue
            Possible_actions=self.mdp.getPossibleActions(current_states[index])
            Possible_actions_num=len(Possible_actions)
            if Possible_actions_num !=0:
                max_Q_value= float('-inf')
                
                for index_2 in range(Possible_actions_num):
                    max_Q_value = max(self.computeQValueFromValues(current_states[index], Possible_actions[index_2]),max_Q_value)
                    for next_state, _ in self.mdp.getTransitionStatesAndProbs(current_states[index], Possible_actions[index_2]):
                        if next_state not in predecessors_states:
                            predecessors_states[next_state] = set()
                        predecessors_states[next_state].add(current_states[index])
                diff = abs(self.values[current_states[index]] - max_Q_value)
                Prioritized_Sweeping_Value_Queue.push(current_states[index], -diff)

        
       
        for _ in range(self.iterations):
            if Prioritized_Sweeping_Value_Queue.isEmpty():
                break
            updatting_state=Prioritized_Sweeping_Value_Queue.pop()
            if updatting_state != 'TERMINAL_STATE':
                Possible_actions=self.mdp.getPossibleActions(updatting_state)
                Possible_actions_num=len(Possible_actions)
                if Possible_actions_num !=0:
                    max_Q_value= float('-inf')
                    for index_2 in range(Possible_actions_num):
                        max_Q_value = max(self.computeQValueFromValues(updatting_state, Possible_actions[index_2]),max_Q_value)
                    self.values[updatting_state]=max_Q_value


            for predecessors_state in predecessors_states.get(updatting_state,[]):
                if predecessors_state == 'TERMINAL_STATE':
                    continue
                Possible_actions=self.mdp.getPossibleActions(predecessors_state)
                Possible_actions_num=len(Possible_actions)
                if Possible_actions_num !=0:
                    max_Q_value= float('-inf')
                    for index_2 in range(Possible_actions_num):
                        max_Q_value = max(self.computeQValueFromValues(predecessors_state, Possible_actions[index_2]),max_Q_value)
                diff = abs(self.values[predecessors_state] - max_Q_value)
                if diff > self.theta:
                    Prioritized_Sweeping_Value_Queue.update(predecessors_state, -diff)



        
        



        

