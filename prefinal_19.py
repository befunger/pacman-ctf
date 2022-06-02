# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import numpy as np

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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

  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)

    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    actions = gameState.getLegalActions(self.index)
    whatToDo = self.determineAction(gameState)
    positions = gameState.getAgentDistances()
    foodLeft = len(self.getFood(gameState).asList())
    chaseFood = self.findBestMove(gameState, True)


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
    return chaseFood

  def determineAction(self, gameState):
    map = gameState.getWalls()
    foodToCollect = self.getFood(gameState).asList()
    foodToDefend = self.getFoodYouAreDefending(gameState).asList()
    mylocation = gameState.getAgentState(self.index).getPosition()
    distances = gameState.getAgentDistances()
    # Check for eaten food
    if self.index == 0:

      #print(len(self.getFoodYouAreDefending(gameState).asList()))
      #print(len(self.getFood(gameState).asList()))
      pass
    eatenfood = self.getEatenFood(gameState)
    if(len(eatenfood) > 0):
      #Update belief position
      pass
    friends = self.getTeam(gameState)
    for i in friends:
      if i != self.index:
        friendPos = gameState.getAgentState(i).getPosition()



    return 4

  def getDefensiveAction(self, gameState):
    opponentsIdx = self.getOpponents(gameState)
    distances = gameState.getAgentDistances()
    opponentDistances = [distances[opponentsIdx[0]], distances[opponentsIdx[1]]]
    #print(opponentDistances)

    return 4


  def getEatenFood(self, gameState):
    defendingFoods = self.getFoodYouAreDefending(gameState).asList()
    previousFoods = self.getFoodYouAreDefending(self.getPreviousObservation()).asList() \
      if self.getPreviousObservation() is not None else list()
    eatenFood = []
    if len(defendingFoods) < len(previousFoods):
      eatenFood = list(set(previousFoods) - set(defendingFoods))
    return eatenFood


  def findBestMove(self, gameState, attacking):

    map = gameState.getWalls()
    foodLeft = len(self.getFood(gameState).asList())
    foodToCollect = self.getFood(gameState).asList()
    foodToDefend = self.getFoodYouAreDefending(gameState).asList()
    mylocation = gameState.getAgentState(self.index).getPosition()
    myfriend = self.getTeam(gameState)
    #print("foods", [foodToCollect[0][0], foodToCollect[0][1]])
    if attacking:
      if self.index <= myfriend[0] and self.index <= myfriend[1]:
        best_cost = 100

        for i in foodToCollect:
          move, step = self.planPath(gameState, map, mylocation, [i[0], i[1]])
          average_distance = 0
          for j in foodToCollect:
            average_distance += np.sqrt(abs(i[0] - j[0]) ** 2 + abs(i[1] - j[1]) ** 2)
          average_distance = average_distance / len(foodToCollect)
          step = step + average_distance

          if step < best_cost:
            best_cost = step
            action = move
      else:

        move, step = self.planPath(gameState, map, mylocation, [foodToCollect[0][0], foodToCollect[0][1]])
        action = move

    else:
      action = 'North'
    return action

  def planPath(self, gameState, walls, position, goalPosition):
    done = False
    directions = [[0,1],[1,0], [0,-1], [-1,0]]
    move = ['North', 'East', 'South', 'West']
    queue = []
    visited = []
    visited.append(position)
    a = len(directions)
    firstMove = None
    for i in range(len(directions)):
      newpos = [int(position[0] + directions[i][0]),int(position[1] +directions[i][1]), i]
      posi = [newpos[0], newpos[1]]
      if posi == goalPosition:
        return move[i], 0

      if walls[newpos[0]][newpos[1]] == False:
        queue.append([newpos[0], newpos[1], newpos[2]])
        visited.append([newpos[0], newpos[1]])
    numsteps = 0
    while not done:
      newqueue =[]
      for j in queue:
        for i in range(len(directions)):
          newpos = [int(j[0]+ directions[i][0]), int(j[1] + directions[i][1]), j[2]]
          posi = [newpos[0], newpos[1]]
          if(posi == goalPosition):
            firstMove = newpos[2]
            done = True
            break
          if posi not in visited and walls[posi[0]][posi[1]] == False:
            visited.append(posi)
            newqueue.append(newpos)
      numsteps +=1
      queue = newqueue
    return move[firstMove], numsteps


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


