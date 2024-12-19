# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
       


        "*** YOUR CODE HERE ***"
        self.Qvalues=util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state,action) not in self.Qvalues:
            return 0.0
        else:
            return self.Qvalues[(state,action)]
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        Possible_actions= self.getLegalActions(state)
        Possible_actions_num=len(Possible_actions)
        if Possible_actions_num !=0:
          max_Q_value=float('-inf')
          for index in range(Possible_actions_num):
              max_Q_value=max(self.getQValue(state,Possible_actions[index]),max_Q_value)
          return max_Q_value
        else:
          return 0.0
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        ActionFromQValues=None
        Possible_actions= self.getLegalActions(state)
        Possible_actions_num=len(Possible_actions)
        if Possible_actions_num !=0:
          max_Q_value=self.computeValueFromQValues(state)
          possible_best_actions=[possible_action for possible_action in Possible_actions if self.getQValue(state,possible_action)==max_Q_value]
          ActionFromQValues=random.choice(possible_best_actions)    
        return ActionFromQValues
          
        util.raiseNotDefined()


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        #在epsilond的概率下随机选择
        
        expect_actions = self.getLegalActions(state)
        if not expect_actions:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(expect_actions)
        return self.getPolicy(state)
       
        util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        current_Qvalue = self.getQValue(state, action)
        #先计算出两者的新旧Qvalue的区别
        difference=reward + self.discount * self.getValue(nextState)-current_Qvalue
        #利用学习率算出新的Qvalue
        next_Qvalue =  current_Qvalue + self.alpha *difference
        self.Qvalues[(state, action)] = next_Qvalue
       #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        Grid_features = self.featExtractor.getFeatures(state, action)
        feature_Q_value = 0
        #Q equals sum of weight*feature_value(Q=sum(w*f))
        Grid_features_num=len(Grid_features)
        feature_name_list=list(Grid_features.keys())
        index=0
        while (index < Grid_features_num):
          feature_name=feature_name_list[index]
          feature_Q_value = feature_Q_value + self.weights[feature_name] *Grid_features.get(feature_name)
          index=index+1
        return feature_Q_value
        util.raiseNotDefined
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        diffience = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        Grid_features = self.featExtractor.getFeatures(state, action)
        #根据权重=旧权重+alpha*差分*feature值算出
        Grid_features_num=len(Grid_features)
        feature_name_list=list(Grid_features.keys())
        index=0
        while (index < Grid_features_num):
          feature_name=feature_name_list[index]
          self.weights[feature_name] = self.weights[feature_name] + self.alpha * diffience * Grid_features.get(feature_name)
          index=index+1
        
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
