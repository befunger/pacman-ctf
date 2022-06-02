# myTeam.py
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
import math
import copy
from captureAgents import CaptureAgent
import random, time, util
from util import nearestPoint
from game import Directions
import game
from baselineTeam import ReflexCaptureAgent
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################
def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class MCTSNode():

  def __init__(self, gameState):
    self.v = 0
    self.n = 0
    self.parent = None
    self.children = []
    self.gameState = gameState

  # the teams positions
  # enemy positions
  # food locations
  # current score
  # distance


class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''

  def getEnemyStartPos(self, position):
    start = 0
    return start

  def tracking(self, gameState):
    print "error"

    # starting position
    our_root_node = gameState.getAgentPosition(self, 1)
    print our_root_node
    enemy_start_node = self.getEnemyStartPos()
    ## get legal position where to go
    #legalActions = root_node.getLegalActions(self.index)

  def select(self, Node):
    best_node = Node
    current_max = float('-inf')
    for child in Node.children:
      if child.n == 0:
        return child
      else:
        UCB = child.v + 2*math.sqrt(math.log(Node.n)/child.n)
        if UCB > current_max:
          current_max = UCB
          best_node = child
    return best_node

  def setChildren(self, Node):
    legalActions = Node.gameState.getLegalActions(self.index)
    children = []
    for action in legalActions:
      child = Node.gameState.generateSuccessor(self.index, action)
      node = MCTSNode(child)
      children.append(node)
    Node.children = children

  def expand(self, Node):
    legalActions = Node.gameState.getLegalActions(self.index)
    children = []
    for action in legalActions:
      child = Node.gameState.generateSuccessor(self.index, action)
      node = MCTSNode(child)
      node.parent = Node
      children.append(node)
    Node.children = children
    return children[0]

  def simualate_one_step(self, gameState):
    ca = CaptureAgent.registerInitialState(self, gameState)

    #print "dfsfsdf"

  def simualate(self, node):
    #print "t"
    pass

  def MCTS(self, gameState):
    has_time = True
    root_node = MCTSNode(gameState)
    self.setChildren(root_node)

    while has_time:
      current_node = root_node
      while current_node.children: #travers tree
        last_node = current_node
        current_node = self.select(self, current_node)
      last_node = self.expand(last_node) #add node
      R = self.simulate(last_node) #simulate game
      current_node = last_node
      while current_node.parent != None:
        self.backprop(current_node, R)
        current_node = current_node.parent

    # root_node = gameState
    #
    # future = root_node
    # legalActions = root_node.getLegalActions(self.index)
    # futureStates = []
    # for action in legalActions:
    #   futureState = root_node.generateSuccessor(self.index, action)
    #   futureStates.append(futureState)
    #
    # totalCounter = [float('inf'), float('inf'), float('inf'), float('inf')]
    #
    # for index, state in enumerate(futureStates):
    #   totalCounter[index]
    #   for i in range(1):
    #     future = state
    #     counter = 0
    #     while not future.isOver():
    #       ##print future.getScore()
    #       futureLegalActions = future.getLegalActions(self.index)
    #       action = random.choice(futureLegalActions)
    #       future = future.generateSuccessor(self.index, action)
    #       counter = counter + 1
    #       totalCounter[index] += counter
    #
    # minValue = min(totalCounter)
    # bestMove = totalCounter.index(minValue)
    # print bestMove
    #return legalActions[bestMove]
    #
    # print root_node.getAgentPosition(0)
    # print root_node.getAgentPosition(1)
    # print root_node.getAgentPosition(2)
    # print root_node.getAgentPosition(3)
    # actions = root_node.getLegalActions(self.index)
    # action = random.choice(actions)
    # action = "South"
    # fake_pos = (1,1)

    #
    # print future.getAgentPosition(0)
    # print future.getAgentPosition(1)
    # print future.getAgentPosition(2)
    # print future.getAgentPosition(3)

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    gm = gameState
    ca = self.simualate_one_step(gm)
    #agentState = copy.deepcopy(gameState.data.agentStates[1])
    #agentState.configuration.pos = (int(1), int(1))
    #gameState.data.agentStates[0] = agentState
    #new_state = gameState.generateSuccessor(0, "North")
    #new_state = gameState.generateSuccessor(1, "South")
    #action = self.MCTS(gameState)
    action = self.chooseAction_sim(gameState)
    return action

  def chooseAction_sim(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


