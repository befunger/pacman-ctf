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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
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

class BasicAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  homebase = None

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
    CaptureAgent.registerInitialState(self, gameState)
    '''
    Your initialization code goes here, if you need any.
    '''

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

  def getActionTowardsPoint(self, gameState, actions, goal):
    '''Calculates the best legal action to move towards a specific point'''
    old_dist = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), goal)
    #print("Distance to my goal is " + str(old_dist))
    
    best_dist = 9999
    best_action = None

    for action in actions:
      successor = self.getSuccessor(gameState, action)
      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()

      #enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      #invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      #if len(invaders) > 0:
      #  dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]

      #self.distancer.getDistance(myPos, goal)
      new_dist = self.getMazeDistance(myPos, goal)
      #print(action + " puts me at distance " + str(new_dist))
      if new_dist < best_dist:
        best_action = action
        best_dist = new_dist

    return best_action

  def getActionAwayFromPoint(self, gameState, actions, goal):
    '''Calculates the best legal action for getting away from a specific point'''
    old_dist = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), goal)
    #print("Distance to my goal is " + str(old_dist))
    
    best_dist = -1
    best_action = None

    for action in actions:
      successor = self.getSuccessor(gameState, action)
      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()

      #enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      #invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      #if len(invaders) > 0:
      #  dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]

      #self.distancer.getDistance(myPos, goal)
      new_dist = self.getMazeDistance(myPos, goal)
      #print(action + " puts me at distance " + str(new_dist))
      if new_dist > best_dist:
        best_action = action
        best_dist = new_dist

    return best_action  

class OffensiveAgent(BasicAgent):
  '''AGENT THAT TRIES TO COLLECT FOOD'''

  def chooseAction(self, gameState):
    '''Picks the offensive agents next move given current state'''

    actions = gameState.getLegalActions(self.index)
    print("Possible legal actions:")
    print(actions)
    actions.remove('Stop')  

    # Register base position
    if self.homebase == None:
      self.homebase = gameState.getAgentState(self.index).getPosition()
      print("Home registered as: ")
      print(self.homebase)

    numInMouth = gameState.getAgentState(self.index).numCarrying
    
    # Checks if any ghost is nearby
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]
    
    
    # If there's a defender close, we run away
    positions = [a.getPosition() for a in defenders]
    dists = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition()) for a in defenders]
    dists.append(1000) #Avoids empty list if no enemies close

    if min(dists) < 4:
      index_of_closest = dists.index(min(dists))
      self.debugDraw([positions[index_of_closest]], [0.5,0.5,0.5], True)
      #return self.minMaxEscape(gameState, actions, positions[index_of_closest], 0)
      return self.getActionAwayFromPoint(gameState, actions, positions[index_of_closest])

    elif min(dists) < 7:
      self.debugDraw(self.homebase, [0.8,0.2,0], True)
      return self.getActionTowardsPoint(gameState, actions, self.homebase)

    else:
      # Get the position of the closest food as goal
      foodList = self.getFood(gameState).asList()
      if len(foodList) > 0:
        myPos = gameState.getAgentState(self.index).getPosition()
        foodDist = [self.getMazeDistance(myPos, food) for food in foodList]
        minDistance = min(foodDist)
        if numInMouth > 40 / minDistance or numInMouth > 5:
          goal = self.homebase
          self.debugDraw([goal], [0,0,1], True)
        else:
          goal = foodList[foodDist.index(minDistance)] # Closest food as goal
          self.debugDraw([goal], [0,1,0], True)
      else:
        goal = self.homebase
        self.debugDraw([goal], [0.8,0.2,0], True)
      
      best_action = self.getActionTowardsPoint(gameState, actions, goal)

    print("Picked " + best_action)
    return best_action

  def minMaxEscape(self, gameState, actions, goal, depth):
    '''To be implemented!'''
    if depth > 5:
      return None #Return evaluated "score"

class DefensiveAgent(BasicAgent):
  '''AGENT THAT TRIES TO STOP ENEMY FROM GRABBING'''
  hasBeenPacman = False

  def chooseAction(self, gameState):
    '''Choses action for the defensive agent given the state'''
    if (not self.hasBeenPacman) and gameState.getAgentState(self.index).isPacman:
      self.hasBeenPacman = True
      print("FINAL HOME POSITION SET TO: ")
      print(self.homebase)
    if self.hasBeenPacman == False:
      self.homebase = gameState.getAgentState(self.index).getPosition()

    actions = gameState.getLegalActions(self.index)
    #print("Possible legal actions:")
    #print(actions)
    # Find all enemies that are on our side of the field and visible
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(invaders) > 0:
      # Pick the action that closes the distance to the nearest invader
      positions = [a.getPosition() for a in invaders]
      dists = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition()) for a in invaders]
      index_of_closest = dists.index(min(dists))
      best_action = self.getActionTowardsPoint(gameState, actions, positions[index_of_closest])
    else:
      # Else, wait at dummy position (temporary fix)
      if not self.hasBeenPacman:
        foodList = self.getFood(gameState).asList()
        myPos = gameState.getAgentState(self.index).getPosition()
        foodDist = [self.getMazeDistance(myPos, food) for food in foodList]
        minDistance = min(foodDist)
        goal = foodList[foodDist.index(minDistance)] # Closest food as goal
      else:
        goal = self.homebase

      best_action = self.getActionTowardsPoint(gameState, actions, goal)

    print("Picked " + best_action)
    return best_action