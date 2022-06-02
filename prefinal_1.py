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

from dis import dis
from multiprocessing.pool import RUN
from turtle import update
from behave.decorator import succeeder
from captureAgents import CaptureAgent
import random, time, util
from behave import condition, action, SUCCESS, FAILURE, RUNNING, repeat, failer, forever, not_
from game import Directions
import game
from capture import GameState
import sys
import math
import numpy as np 

#################
# Team creation #
#################

debugPrint = False


def createTeam(firstIndex, secondIndex, isRed,
               first = 'Pacmaniacs', second = 'Pacmaniacs'):
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

class Pacmaniacs(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def __init__(self, isRed):
    self.createBT()
    self.moveDirection = Directions.STOP
    self.gameState = None
    self.legalActions = []
    self.isAttacker = False 
    self.lastRoleChange = 0
    self.myPos = (0,0)
    self.isRed = False
    self.closestEnemy = None
    self.mostFoodEnemyPacman = None 
    self.bestFood = None
    self.lastFoodMatrix = None
    self.eatenFriendlyFood = []
    self.timeSinceFoodEaten = 1000
    self.bestBoost = None
    self.actionsTaken = 0
    self.initialNumBoosts = 0
    CaptureAgent.__init__(self, isRed)
    self.enemyIndices = []
    self.friendlyIndices = []

  @condition 
  def checkIfAttacker(self):
    
    return self.isAttacker

  @condition
  def isCarryingFood(self):
    return self.gameState.getAgentState(self.index).numCarrying >= 1

  @condition 
  def isCarryingTooMuchFood(self):
    foodMatrix = self.getFood(self.gameState)
    foodLeft = foodMatrix.count()
    numFood = self.gameState.getAgentState(self.index).numCarrying
    
    foodPositions = self.getFoodDistances()
    closestFood = 100
    if len(foodPositions) > 0:
      closestFood = foodPositions[0][0]

    distanceHome = self.getMazeDistance(self.myPos, self.getClosestHomePos())

    op1ScaredTimer = self.gameState.getAgentState(self.enemyIndices[0]).scaredTimer
    op2ScaredTimer = self.gameState.getAgentState(self.enemyIndices[1]).scaredTimer

    leastScared = min(op1ScaredTimer, op2ScaredTimer)

    #print "numFood %d, foodLeft %d, closestFood %d, distanceHome %d" % (numFood, foodLeft, closestFood, distanceHome)

    if leastScared > distanceHome:
      return False

    if (numFood >= 1
        and (foodLeft <= 2
        or (numFood >= math.ceil(foodLeft/4) and closestFood >= 2)
        or (closestFood > 7) 
        or (closestFood > 4 and distanceHome < 3)
        )):
      return True
    else: 
      return False 

  
  def getFoodDistances(self):

    foodMatrix = self.getFood(self.gameState)
    chasedFood = SharedData.getChasedFood()
    for x, y in chasedFood:
      foodMatrix[x][y] = False

    # Compute distance to the nearest food
    foodPositions = []
    for foodX in range(foodMatrix.width):
      for foodY in range(foodMatrix.height):
        if not foodMatrix[foodX][foodY]:
          continue
        foodPos = (foodX, foodY)
        dist = self.getMazeDistance(foodPos, self.myPos)
        foodPositions.append((dist, foodPos))
    foodPositions.sort()
    return foodPositions

  @condition 
  def canEatFood(self):
    #check closest food 

    foodPositions = self.getFoodDistances()

    if self.isRed: 
      team_indices = self.gameState.getBlueTeamIndices()
    else: 
      team_indices = self.gameState.getRedTeamIndices()
    
    opponent_1 = self.gameState.getAgentPosition(team_indices[0])
    opponent_2 = self.gameState.getAgentPosition(team_indices[1])
    if opponent_1 is None and opponent_2 is None: 
      self.bestFood = foodPositions[0][1] #foodPos
      return True
    elif opponent_1 is not None and opponent_2 is not None: 
      for dist, foodPos in foodPositions:
        dist_food_opponent_1 = self.getMazeDistance(opponent_1, foodPos)
        dist_food_opponent_2 = self.getMazeDistance(opponent_2, foodPos)
        if dist < dist_food_opponent_1 and dist < dist_food_opponent_2:
          self.bestFood = foodPos 
          return True
    elif opponent_1 is not None:
       for dist, foodPos in foodPositions:
        dist_food_opponent_1 = self.getMazeDistance(opponent_1, foodPos)
        if dist < dist_food_opponent_1: 
          self.bestFood = foodPos
          return True
    elif opponent_2 is not None:
       for dist, foodPos in foodPositions:
        dist_food_opponent_2 = self.getMazeDistance(opponent_2, foodPos)
        if dist < dist_food_opponent_2: 
          self.bestFood = foodPos
          return True
    
    return False
  
  def getClosestHomePos(self):
    width = self.gameState.data.layout.width
    height = self.gameState.data.layout.height

    desiredXPosition = (width + 2)//2

    if self.isRed:
      desiredXPosition = (width + 2)//2 - 1

    minDist = 1000
    bestPos = (desiredXPosition, height//2)
    for y in range(height):
      pos = (desiredXPosition, y)
      if self.gameState.hasWall(pos[0], pos[1]):
        continue
      dist = self.getMazeDistance(self.myPos, pos)
      if dist < minDist:
        minDist = dist
        bestPos = pos
    return bestPos

  @action 
  def returnToFrienlyTerritory(self):

    

    if debugPrint:
      print "state: returnToFrienlyTerritory"

    bestPos = self.getClosestHomePos()

    self.moveDirection = self.moveToward(bestPos)
        
    return RUNNING

  @action 
  def eatClosest(self):

    if debugPrint:
      print "state: eat closest"
    self.moveDirection = self.moveToward(self.bestFood)
    SharedData.setChasedFood(self.index, self.bestFood)
    return RUNNING
    
  @condition
  def enemyGhostClose(self):

    return self.getClosestEnemyDistance(self.gameState) < 3

  def getClosestEnemyDistance(self, gameState):
    if self.isRed: 
      enemyIndices = gameState.getBlueTeamIndices()
    else: 
      enemyIndices = gameState.getRedTeamIndices()
      
    op1ScaredTimer = gameState.getAgentState(enemyIndices[0]).scaredTimer
    op2ScaredTimer = gameState.getAgentState(enemyIndices[1]).scaredTimer
    opponent1 = gameState.getAgentPosition(enemyIndices[0])
    opponent2 = gameState.getAgentPosition(enemyIndices[1])

    dist_op1 = 5
    dist_op2 = 5
    
    if opponent1 is not None:
      dist_op1 = self.getMazeDistance(self.myPos, opponent1)
    else:
      dist_op1 = max(6, gameState.getAgentDistances()[enemyIndices[0]])

    if op1ScaredTimer > 5:
      dist_op1 = max(op1ScaredTimer, dist_op1)

    if opponent2 is not None:
      dist_op2 = self.getMazeDistance(self.myPos, opponent2)
    else:
      dist_op1 = max(6, gameState.getAgentDistances()[enemyIndices[1]])

    if op2ScaredTimer > 5:
      dist_op2 = max(op2ScaredTimer, dist_op2)

    return min(dist_op1, dist_op2) 

  @action 
  def flee(self):
    
    if debugPrint:
      print "state: fleeeee"

    width = self.gameState.data.layout.width
    height = self.gameState.data.layout.height

    desiredXPosition = (width + 2)//2

    if self.isRed:
      desiredXPosition = (width + 2)//2 - 1

    minTargetDist = 1000
    maxDistToEnem = 0
    #bestPos = (desiredXPosition, height//2)

    bestAction = Directions.STOP

    for y in range(height):
      pos = (desiredXPosition, y)
      if self.gameState.hasWall(pos[0], pos[1]):
        continue
      #dist = self.getMazeDistance(self.myPos, pos)

      for action in self.legalActions:
        successor = self.gameState.generateSuccessor(self.index, action)
        newPos = successor.getAgentState(self.index).getPosition()
        targetDist = self.getMazeDistance(newPos, pos)
        newEnemDist = self.getClosestEnemyDistance(successor)

        if newEnemDist - targetDist > maxDistToEnem - minTargetDist:
          maxDistToEnem = newEnemDist
          minTargetDist = targetDist
          bestAction = action 

    if bestAction is Directions.STOP:
      # We can't flee home, just flee away from enemies!
      for action in self.legalActions:
        currEnemDist = self.getClosestEnemyDistance(self.gameState)
        successor = self.gameState.generateSuccessor(self.index, action)
        newEnemDist = self.getClosestEnemyDistance(successor)
        x, y = successor.getAgentPosition(self.index)
        numWalls = 0
        for wx, wy in [(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]:
          if self.gameState.hasWall(wx, wy):
            numWalls +=1 

        if newEnemDist > currEnemDist and numWalls < 3:
          self.moveDirection = action
          
          
          return RUNNING
    else:
      self.moveDirection = bestAction

      return RUNNING
  
  @condition 
  def enemiesVisible(self):
    if self.isRed: 
      enemyIndices = self.gameState.getBlueTeamIndices()
    else: 
      enemyIndices = self.gameState.getRedTeamIndices()
      
    opponent1 = self.gameState.getAgentPosition(enemyIndices[0])
    opponent2 = self.gameState.getAgentPosition(enemyIndices[1])

    return opponent1 is not None or opponent2 is not None
  
  @condition
  def isGhost(self):
    agentState = self.gameState.getAgentState(self.index)
    
    return not agentState.isPacman

  
  @condition
  def visibleEnemyPacman(self):
    if self.isRed: 
      enemyIndices = self.gameState.getBlueTeamIndices()
    else: 
      enemyIndices = self.gameState.getRedTeamIndices()
      
    opponent1 = self.gameState.getAgentPosition(enemyIndices[0])
    opponent2 = self.gameState.getAgentPosition(enemyIndices[1])

    return ((opponent1 is not None and self.gameState.getAgentState(enemyIndices[0]).isPacman)
     or (opponent2 is not None and self.gameState.getAgentState(enemyIndices[1]).isPacman))

  @condition
  def isClosestEnemyPacman(self):
    self.setClosestEnemy()
    return self.closestEnemy is not None and self.gameState.getAgentState(self.closestEnemy).isPacman

  @action 
  def chaseEnemy(self):
    self.setClosestEnemy()
    self.moveDirection = self.moveToward(self.gameState.getAgentPosition(self.closestEnemy))

    if debugPrint: 
      print "chaseEnemy"

    return RUNNING

  @condition 
  def isOurFoodBeingEaten(self):
    return self.timeSinceFoodEaten < 5
  
  @action 
  def goToEatenFood(self):
    self.moveDirection = self.moveToward(self.eatenFriendlyFood[-1])

    if debugPrint: 
      print "goToEatenFood"
      
    return RUNNING

  @action 
  def protect(self):
    ### go to where we have most food 
    foodMatrix = self.getFoodYouAreDefending(self.gameState)
    #Take the power capsules into account
    capsules = []
    capsules = self.getCapsulesYouAreDefending(self.gameState)

    width = self.gameState.data.layout.width
    height = self.gameState.data.layout.height

    minExtents = (3, 3)
    maxExtents = (width - 3, height - 3)

    if self.isRed:
      maxExtents = (width//2 - 3, height - 3)
    else:
      minExtents = (width//2 + 3, 3)

    mostFood = 0.0
    if self.isRed:
      bestPos = (width//2, height//2)
    else:
      bestPos = (width//2 + 1, height//2)

    for x in range(minExtents[0], maxExtents[0]):
      for y in range(minExtents[1], maxExtents[1]):
        pos = (x, y)

        if self.gameState.hasWall(x, y):
          continue

        numWalls = 0
        for wx, wy in [(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]:
          if self.gameState.hasWall(wx, wy):
            numWalls +=1 

        if numWalls >= 3:
          continue
        
        sumFood = 0.0
        for i in range(-4, 5):
          for j in range(-4, 5):
            if x + i < 0 or x + i >= width or y + j < 0 or y + j >= height:
              continue 
            
            tooClose = False
            for defended in SharedData.getDefendedPositions():
              if self.getMazeDistance(defended, pos) <= 4:
                tooClose = True
                break

            if foodMatrix[x + i][y + j] and not tooClose:
              distToFood = self.getMazeDistance(pos, (x + i, y + j)) + 1
              if distToFood <= 4:
                sumFood += 1/distToFood

            position = (x+i, y+j)
            if position in capsules  and not tooClose: 
              distToCapsule = self.getMazeDistance(pos, (x + i, y + j)) + 1
              if distToCapsule <= 4:
                sumFood += 3/distToCapsule
        
        

        if sumFood > mostFood:
          mostFood = sumFood
          bestPos = pos

    SharedData.setDefender(self.index, bestPos)    
    self.moveDirection = self.moveToward(bestPos)

    if debugPrint:
      print "state: protect"

    return RUNNING

  @action 
  def patrol(self):
    #set move direction to going back to center of home territory

    width = self.gameState.data.layout.width
    height = self.gameState.data.layout.height

    desiredXPosition = (width + 2)//2

    if self.isRed:
      desiredXPosition = (width + 2)//2 - 1
  

    upper = range(height//2, height)
    lower = range(height//2)

    allYPoses = []
    for i in range(len(upper) + len(lower)):
      if i%2==0:
        allYPoses.append(upper[i//2])
      else:
        allYPoses.append(lower[i//2])

    for ypos in allYPoses:
      centerPos = (desiredXPosition, ypos)
      if self.gameState.hasWall(centerPos[0], centerPos[1]):
        continue
      else: 
        break 
    
    self.moveDirection = self.moveToward(centerPos)
    
    if debugPrint:
      print "state: patrol"

    return RUNNING
  
  @condition 
  def canEatBoost(self):
    boosts = self.getCapsules(self.gameState) # List of positions (as tuples)
    
    # Must be closer to boost than enemy
    distToClosestBoost = 1000

    opponents = [i for i in self.enemyIndices if self.gameState.getAgentPosition(i) != None]

    for boost in boosts:
      dist = self.getMazeDistance(self.myPos, boost)

      opTooClose = False
      for opIndex in opponents:
        op = self.gameState.getAgentPosition(opIndex)
        opDistToBoost = self.getMazeDistance(op, boost)
        if opDistToBoost < dist:
          opTooClose = True
          break
        
        opDistToSelf = self.getMazeDistance(op, self.myPos)
        action = self.moveToward(boost)
        successor = self.gameState.generateSuccessor(self.index, action)
        newOpDistToSelf = successor.getAgentPosition(opIndex)
        if opDistToSelf >= newOpDistToSelf and newOpDistToSelf <= dist:
          opTooClose = True
          break

      if not opTooClose and dist < distToClosestBoost:
        distToClosestBoost = dist
        self.bestBoost = boost
    
    return self.bestBoost is not None

  @condition 
  def enemiesScared(self):
    op1ScaredTimer = self.gameState.getAgentState(self.enemyIndices[0]).scaredTimer
    op2ScaredTimer = self.gameState.getAgentState(self.enemyIndices[1]).scaredTimer
    
    return op1ScaredTimer >= 5 and op2ScaredTimer >= 5

  @condition 
  def isBoostStrategic(self):
    # [v] Time left
    # [v] Carrying Food
    # [] Enemies Able to intercept <- TODO
    # [v] Enemies Close
    # [v] enemy scared timers 
    # [v] Boosts taken


    heuristic = 0.0
    
    # Time Left and Boosts left 
    boosts = self.getCapsules(self.gameState)

    heuristic += (self.actionsTaken**1.7)/(120**1.7)*20 + len(boosts) * 8 / self.initialNumBoosts
    
    # Enemies scared
    op1ScaredTimer = self.gameState.getAgentState(self.enemyIndices[0]).scaredTimer
    op2ScaredTimer = self.gameState.getAgentState(self.enemyIndices[1]).scaredTimer

    
    heuristic -= (op1ScaredTimer + op2ScaredTimer)/2

    # Food carried
    foodCarried = sum([self.gameState.getAgentState(i).numCarrying for i in self.getTeam(self.gameState)]) 

    heuristic += foodCarried
    
    # Enemy Distance
    heuristic -= max(self.getClosestEnemyDistance(self.gameState), 6)*1.5
    
    # Intercept

    return heuristic > 14
  
  @action  
  def eatBoost(self):
    self.moveDirection = self.moveToward(self.bestBoost)

    return RUNNING

  def my_debugger(self, node, state):
    #print "[%s] -> %s" % (node.name, state)
    pass

  def createBT(self):
    tree = (
      (
        self.checkIfAttacker >> (
        (self.enemiesVisible >> self.isGhost >> self.isClosestEnemyPacman >> self.chaseEnemy)
        | (self.canEatBoost >> not_ * self.enemiesScared >> self.isBoostStrategic >> self.eatBoost)
        | (self.enemyGhostClose >> not_ * self.isGhost >> self.flee)
        | (self.enemyGhostClose >> self.isGhost >> self.patrol)
        | (self.isCarryingTooMuchFood >> self.returnToFrienlyTerritory)
        | (self.canEatFood >> self.eatClosest)
        | self.patrol
        )
      )
      | 
          (not_ * self.isGhost >> self.enemyGhostClose >> self.flee)
        | (not_ * self.isGhost >> self.returnToFrienlyTerritory)
        | (self.enemiesVisible >> self.isGhost >> self.visibleEnemyPacman >> self.chaseEnemy)
        | (self.isOurFoodBeingEaten >> self.goToEatenFood)
        | (self.protect)
    )

    self.BT = tree.debug(self.my_debugger, self)

  def setClosestEnemy(self):
      
    opponent1 = self.gameState.getAgentPosition(self.enemyIndices[0])
    opponent2 = self.gameState.getAgentPosition(self.enemyIndices[1])

    if opponent1 is not None and opponent2 is not None:
      dist_opponent1 = self.getMazeDistance(opponent1, self.myPos)
      dist_opponent2 = self.getMazeDistance(opponent2, self.myPos)
      if dist_opponent1 < dist_opponent2:
        self.closestEnemy = self.enemyIndices[0]
      else: 
        self.closestEnemy = self.enemyIndices[1]  
    elif opponent1 is not None:
      self.closestEnemy = self.enemyIndices[0]
    else: 
      self.closestEnemy = self.enemyIndices[1]

  def setMostFoodEnemyPacman(self):
      
    numFood1 = self.gameState.getAgentState(self.enemyIndices[0]).numCarrying
    numFood2 = self.gameState.getAgentState(self.enemyIndices[1]).numCarrying

    if numFood1 > numFood2:
      self.mostFoodEnemyPacman = self.enemyIndices[0]
    elif numFood2 > numFood1: 
      self.mostFoodEnemyPacman = self.enemyIndices[1]

  def updateEatenFood(self):
    currentFoodMatrix = self.getFoodYouAreDefending(self.gameState)

    if self.lastFoodMatrix is not None:
      for x in range(currentFoodMatrix.width):
        for y in range(currentFoodMatrix.height):
          if currentFoodMatrix[x][y] is not self.lastFoodMatrix[x][y] and not currentFoodMatrix[x][y]:
            self.eatenFriendlyFood.append((x, y))
            self.timeSinceFoodEaten = 0
    
    self.lastFoodMatrix = currentFoodMatrix

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

    if self.index in gameState.getBlueTeamIndices():
      self.isRed = False
    else:
      self.isRed = True 

    if self.isRed: 
      self.enemyIndices = gameState.getBlueTeamIndices()
    else: 
      self.enemyIndices = gameState.getRedTeamIndices()

    self.myPos = gameState.getAgentState(self.index).getPosition()
    self.initialNumBoosts = len(self.getCapsules(gameState))

  def moveToward(self, targetPosition):
    
    currDist = self.getMazeDistance(self.myPos, targetPosition)
    for action in self.legalActions:
      successor = self.gameState.generateSuccessor(self.index, action)
      newPos = successor.getAgentState(self.index).getPosition()
      newDist = self.getMazeDistance(newPos, targetPosition)
      if newDist < currDist:
        return action
    return Directions.STOP


  def calcClosestFood(self, pos):
    foodMatrix = self.getFood(self.gameState)

    chasedFood = SharedData.getChasedFood()
    for x, y in chasedFood:
      foodMatrix[x][y] = False

    # Compute distance to the nearest food
    myPos = pos
    minDistanceToFood = 1000
    minPos = (-1, -1)

    for foodX in range(foodMatrix.width):
      for foodY in range(foodMatrix.height):
        if not foodMatrix[foodX][foodY]:
          continue
        foodPos = (foodX, foodY)
        dist = self.getMazeDistance(foodPos, myPos)

        if minDistanceToFood > dist:
          minDistanceToFood = dist
          minPos = (foodX, foodY)
    
    return (minDistanceToFood, minPos)

  def attackerCheck(self):
    # self.lastRoleChange -= 1
    # if self.lastRoleChange <= 0:
    #   self.lastRoleChange = 3
    # else:
    #   return

    own = self.attackerHeuristic(self.index)
    mates = self.getTeam(self.gameState)
    mates.remove(self.index)
    otherIndex = mates[0] #Assuming 2 agents
    other = self.attackerHeuristic(otherIndex)
    
    score = self.getScore(self.gameState)
    winningScore = max(0, score)
    losingScore = min(0, score)

    foodCarried = sum([self.gameState.getAgentState(i).numCarrying for i in self.getTeam(self.gameState)]) 

    self.isAttacker = own < 10 and winningScore + foodCarried/2 < 5 or own - losingScore/2 < other - winningScore/2
    SharedData.setAttacker(self.index, self.isAttacker)
    # SharedData.setAttacker(otherIndex, not self.isAttacker)

  def attackerHeuristic(self, index):
  
    minDistanceToFood, minPos = self.calcClosestFood(self.gameState.getAgentState(index).getPosition())

    enemies = [i for i in self.getOpponents(self.gameState)]
    defenders = [self.gameState.getAgentState(a) for a in enemies if self.gameState.getAgentState(a).isPacman and self.gameState.getAgentPosition(a) != None]  
    
    #for enemy in defenders:
    #  print enemy.getPosition()

    minDefenderDist = 0

    if len(defenders) > 0:
      dists = [self.getMazeDistance(minPos, a.getPosition()) for a in defenders]
      minDefenderDist = min(dists)
      
    friends = self.getTeam(self.gameState)
    numAttackers = len([i for i in friends if SharedData.isAttacker(i) and self.gameState.getAgentState(i).getPosition() != None and i != self.index])

    
    
    
    # attackerPenalty = 0
    # if numAttackers >= 1:
    #   attackerPenalty = -5
    # if numAttackers >= 2:
    #   closestToHome = self.index
    #   xPos = self.gameState.getAgentState(self.index).getPosition()[0]
    #   for i in friends:
    #     newX = self.gameState.getAgentState(i).getPosition()[0]
    #     if self.isRed and newX < xPos:
    #       xPos = newX
    #     elif not self.isRed and newX > xPos:
    #       xPos = newX
    #   attackerPenalty -= score - 10

    attackerHeuristic = minDistanceToFood - minDefenderDist# + attackerPenalty

    return attackerHeuristic

  def chooseAction(self, gameState):

    self.moveDirection = Directions.STOP

    self.bestFood = None
    self.bestBoost = None
    self.closestEnemyPacman = None
    self.timeSinceFoodEaten += 1
    self.legalActions = gameState.getLegalActions(self.index)
    self.gameState = gameState


    self.updateEatenFood()

    #self.distancer.getMazeDistances()
    self.myPos = self.gameState.getAgentState(self.index).getPosition()
    
    SharedData.setDefender(self.index, None)
    SharedData.setChasedFood(self.index, None)
    self.attackerCheck()
    # else:
      # self.isAttacker = SharedData.isAttacker(self.index)
    #print "attacker: %s position %s" % (self.isAttacker, self.myPos)

    self.createBT()
    BTState = self.BT.tick()
    #print BTState
    #print "index: %d\n" % self.index
    
    self.actionsTaken += 1

    return self.moveDirection

class SharedData:

  #dict of indices of agents that are currently attackers
  # ex. {1 : True, 3: False}
  attackers = {}

  # ex. {1 : (10, 3), 3: (4, 2)}
  defendedPositions = {}

  # ex. {1 : (10, 3), 3: (4, 2)}
  chasedFood = {}

  @classmethod
  def isAttacker(self, index):
    if index not in SharedData.attackers:
      return False
    return SharedData.attackers[index]

  @classmethod
  def setAttacker(self, index, isAttacker):
    SharedData.attackers[index] = isAttacker

  @classmethod
  def setDefender(self, index, position):
    SharedData.defendedPositions[index] = position
  
  @classmethod
  def getDefendedPositions(self):
    return [pos for pos in SharedData.defendedPositions.values() if pos is not None] 
    
  @classmethod
  def setChasedFood(self, index, pos):
    SharedData.chasedFood[index] = pos
  
  @classmethod
  def getChasedFood(self):
    return [pos for pos in SharedData.chasedFood.values() if pos is not None] 