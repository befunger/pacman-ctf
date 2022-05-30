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
               first = 'OffensiveAgent', second = 'OffensiveAgent'):
               #first = 'OffensiveAgent', second = 'OffensiveAgent'):
               
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
  layout = None
  currentState = None
  # Potentially use this to consider the powerups and improve thinking
  powerupLeft = True
  powerupPosition = None

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
    self.layout = self.stripLayout(gameState.data.layout)
    '''
    Your initialization code goes here, if you need any.
    '''
  def getMovesFromLayout(self, pos):
    '''Returns legal moves from position and stripped map (used for minmax)'''
    #print("Checking position (" + str(pos[0]) + ", " + str(pos[1]) + ")")
    x = int(-pos[1]-1)
    y = int(pos[0]-1+1)
    moves = []
    if self.layout[x][y-1] == ' ':
      moves.append('West')
    #if self.layout[x][y] == ' ':
    #  moves.append('Stop')
    if self.layout[x][y+1] == ' ':
      moves.append('East')
    if self.layout[x-1][y] == ' ':
      moves.append('North')
    if self.layout[x+1][y] == ' ':
      moves.append('South')
    return moves

  def stripLayout(self, layoutRaw):
    '''Creates a layout map with only walls (%) and empty spaces (spacebar)'''
    strippedLayout = []
    layout = layoutRaw.layoutText
    print("Original version:")
    print(layout)
    for row in layout:
      newRow = ''
      for i in range(len(row)):
        if row[i] == '%':
          newRow += '%'
        else:
          newRow += ' '
      strippedLayout.append(newRow)
    print("Stripped version:")
    print(strippedLayout)
    return strippedLayout

  def updatePos(self, pos, action):
    fromDirToCoordinate = {'Stop' : [0, 0], 'North' : [0, 1], 'South' : [0, -1], 'West' : [-1, 0], 'East' : [1, 0]}
    change = fromDirToCoordinate[action]
    change[0] += pos[0]
    change[1] += pos[1]
    return change

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

  def chooseAction(self, oldGameState):
    '''Picks the offensive agents next move given current state'''
    gameState = self.getSuccessor(oldGameState, 'Stop')

    myPos = gameState.getAgentState(self.index).getPosition()
    actions = gameState.getLegalActions(self.index)
    actions.remove('Stop')
    self.currentState = gameState

    self.debugDraw([myPos], [1.0,1.0,1.0], True)

    # Register base position
    if self.homebase == None:
      self.homebase = myPos
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

    if min(dists) < 5:
      index_of_closest = dists.index(min(dists))
      self.debugDraw([positions[index_of_closest]], [0.5,0.5,0.5], False)
      return self.minMaxEscape(myPos, positions[index_of_closest], 5)
      #return self.getActionAwayFromPoint(gameState, actions, positions[index_of_closest])

    #elif min(dists) < 7:
    #  self.debugDraw(self.homebase, [0.8,0.2,0], True)
    #  return self.getActionTowardsPoint(gameState, actions, self.homebase)

    else:
      # Get the position of the closest food as goal
      foodList = self.getFood(gameState).asList()
      if len(foodList) > 0:
        myPos = gameState.getAgentState(self.index).getPosition()
        foodDist = [self.getMazeDistance(myPos, food) for food in foodList]
        minDistance = min(foodDist)
        if numInMouth > 40 / minDistance or numInMouth > 5:
          goal = self.homebase
          self.debugDraw([goal], [0,0,1], False)
        else:
          goal = foodList[foodDist.index(minDistance)] # Closest food as goal
          self.debugDraw([goal], [0,1,0], False)
      else:
        goal = self.homebase
        self.debugDraw([goal], [0.8,0.2,0], False)
      
      best_action = self.getActionTowardsPoint(gameState, actions, goal)

    #print("Picked " + best_action)
    return best_action

  def minMaxEscape(self, myPos, enemyPos, maxDepth):
    '''Uses minmax algorithm to pick best moves to escape'''
    print("ENGAGINGING MINMAX ESCAPE WITH DEPTH " + str(maxDepth))
    #print("Starting pos has player at (" + str(myPos[0]) + ", " + str(myPos[1]) + ") and enemy at (" + str(enemyPos[0]) + ", " + str(enemyPos[1]) + ")")
    ownActions = self.getMovesFromLayout(myPos)
    #print(str(ownActions) + " when friend at (" + str(myPos[0]) + ", " + str(myPos[1]) + ")")
    bestScore = -100
    bestMove = 'Stop'

    for action in ownActions:
      #print("* friend goes " + action)
      newPos = self.updatePos(myPos, action)
      moveScore = self.minMove(newPos, enemyPos, maxDepth-1, alpha=-10000, beta=10000)
      if moveScore > bestScore:
        bestMove = action
        bestScore = moveScore
    print("Moving " + bestMove + " puts me at distance " + str(bestScore))

    return bestMove #Returns move that gives best node given optimal play from both sides

  def minMove(self, myPos, enemyPos, depth, alpha, beta):
    if depth <= 0:
      #print("*"*(3-depth) + " bottomed out")
      return self.evaluationScore(myPos, enemyPos) # Max depth heuristic
    
    if myPos[0] == enemyPos[0] and myPos[1] == enemyPos[1]:
      #print("~Enemy walked into us, score -1")
      return -1 # We are on the enemy and die. This is bad!

    enemyActions = self.getMovesFromLayout(enemyPos)
    #print("*"*(6-depth) + str(enemyActions) + " when enemy at (" + str(enemyPos[0]) + ", " + str(enemyPos[1]) + ")")
    lowestScore = 10000
    for action in enemyActions:
      #print("*"*(6-depth) + " enemy goes " + action)
      newPos = self.updatePos(enemyPos, action)
      moveScore = self.maxMove(myPos, newPos, depth-1, alpha, beta)
      lowestScore = min(moveScore, lowestScore)
      if lowestScore <= alpha: # Alpha pruning
        return lowestScore
      else:
        beta = min(beta, lowestScore)

    return lowestScore #The enemy will pick the move that gives the lowest score (enemy closest to us)

  def maxMove(self, myPos, enemyPos, depth, alpha, beta):
    if depth <= 0:
      #print("*"*(3-depth) + " bottomed out")
      return self.evaluationScore(myPos, enemyPos) # Max depth heuristic

    if myPos[0] == enemyPos[0] and myPos[1] == enemyPos[1]:
      #print("~We walked into enemy, score -1")
      return -1 # We are on the enemy and die. This is bad!

    ownActions = self.getMovesFromLayout(myPos)
    #print("*"*(6-depth) + str(ownActions) + " when friend at (" + str(myPos[0]) + ", " + str(myPos[1]) + ")")

    bestScore = -10000
    for action in ownActions:
      #print("*"*(6-depth) + " friend goes " + action)
      newPos = self.updatePos(myPos, action)
      moveScore = self.minMove(newPos, enemyPos, depth-1, alpha, beta)
      bestScore = max(moveScore, bestScore)
      if bestScore >= beta:
        return bestScore # Beta pruning
      else:
        alpha = max(alpha, bestScore)

    return bestScore #We will pick the move that gives the highest score (greatest distance)
  
  def evaluationScore(self, myPos, enemyPos):
    myPos = (myPos[0], myPos[1])
    enemyPos = (enemyPos[0], enemyPos[1])
    distToEnemy = self.getMazeDistance(myPos, enemyPos)
    distToHome = self.getMazeDistance(myPos, self.homebase)
    #print("score be " + str(distToEnemy + 1.0/distToHome))
    return distToEnemy + 1.0/distToHome

class DefensiveAgent(BasicAgent):
  '''AGENT THAT TRIES TO STOP ENEMY FROM GRABBING'''
  hasBeenPacman = False

  def chooseAction(self, oldGameState):
    '''Choses action for the defensive agent given the state'''
    gameState = self.getSuccessor(oldGameState, 'Stop')

    self.currentState = gameState

    if (not self.hasBeenPacman) and gameState.getAgentState(self.index).isPacman:
      self.hasBeenPacman = True
      print("FINAL HOME POSITION SET TO: ")
      print(self.homebase)
    if self.hasBeenPacman == False:
      prev = self.getPreviousObservation()
      self.homebase = (prev if prev != None else gameState).getAgentState(self.index).getPosition()

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

    #print("Picked " + best_action)
    return best_action