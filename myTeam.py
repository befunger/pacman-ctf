
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

def createTeam(firstIndex, secondIndex, isRed, first = 'OffensiveAgent', second = 'DefensiveAgent'):
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
  isRed = None            # Indicates if we are on the Red team (left side) or the Blue team (right side)
  homebase = None         # Starting position
  layout = None           # A simple map of the currrent layout with only walls (% symbol) and non-walls (blank space)
  currentState = None     # The current gamestate
  defenders = None        # List of visible enemy agents on their own field half
  ownBoundary = None      # List of locations (x, y) that form our boundary against the enemy half of the field
  ownBoundaryX = None     # Used during minmax to check if agent has reached home
  enemyCapsules = []      # Capsules we can eat to power up
  ownCapsules = []        # Capsules the enemy can eat to power up

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

    # Figures out which team you are based on if your starting point is in the left or the right half of the field
    width = len(gameState.data.layout.layoutText[0])
    self.isRed = gameState.getAgentState(self.index).getPosition()[0] < width/2
    
    self.registerCapsules(gameState, width)
    self.layout = self.stripLayout(gameState.data.layout)
    self.ownBoundary = self.getBoundary(gameState)

  def getBoundary(self, gameState):
    '''Compiles all boundary points (where our half of the field meets the enemy field'''
    height = len(self.layout)
    width = len(self.layout[0])

    # If starting on the left side our boundary is on the left line
    bound_x = width/2 -1 if self.isRed else width/2
    self.ownBoundaryX = bound_x
    #print("Boundary at bound_x")

    # Gets all points (bound_x, y) on the boundary that aren't walls
    boundary = [(bound_x, i) for i in range(height) if self.layout[-i-1][bound_x] != '%']
    #print(boundary)  

    #self.debugDraw(boundary, [0.0, 1.0, 0.7], False)
    return boundary
  
  def getClosestBoundary(self, pos):
    '''Gets the closest boundary point to the provided position (useful when trying to get home ASAP)'''
    dists = [self.getMazeDistance(pos, a) for a in self.ownBoundary]
    closestBoundary = self.ownBoundary[dists.index(min(dists))]
    #self.debugDraw([closestBoundary], [0.5,0.5,0.5], False)
    return closestBoundary


  def registerCapsules(self, gameState, width):
    '''Precalculates the position of the capsule powerups'''
    capsules = gameState.data.layout.capsules
    bluesCapsules = [capsulePos for capsulePos in capsules if capsulePos[0] >= width/2]
    redsCapsules = [capsulePos for capsulePos in capsules if capsulePos[0] < width/2]
    if self.isRed:
      self.ownCapsules = redsCapsules
      self.enemyCapsules = bluesCapsules
    else:
      self.ownCapsules = bluesCapsules
      self.enemyCapsules = redsCapsules

  def getMovesFromLayout(self, pos):
    '''Returns legal moves from position and stripped map (used for minmax)'''
    #print("Checking position (" + str(pos[0]) + ", " + str(pos[1]) + ")")
    x = int(-pos[1]-1)
    y = int(pos[0])
    moves = []
    if self.layout[x][y-1] != '%':
      moves.append('West')
    #if self.layout[x][y] == ' ':
    #  moves.append('Stop')
    if self.layout[x][y+1] != '%':
      moves.append('East')
    if self.layout[x-1][y] != '%':
      moves.append('North')
    if self.layout[x+1][y] != '%':
      moves.append('South')
    return moves

  def stripLayout(self, layoutRaw):
    '''Creates a layout map with only walls (%) and empty spaces (spacebar)'''
    strippedLayout = []
    layout = layoutRaw.layoutText
    #print("Original version:")
    #print(layout)
    for i, row in enumerate(layout):
      newRow = ''
      for j in range(len(row)):
        if row[j] == '%':
          newRow += '%'
        else:
          newRow += ' '
      strippedLayout.append(newRow)
    #print("Stripped version:")
    #print(strippedLayout)
    return strippedLayout

  def updatePos(self, pos, action):
    ''''Helper function for updating the position in min-max algorithm'''
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

  ### HMM Code###
  def _init_HMM(self, N, M, variance):
    """
    Initialize A, B, pi
    N - no of states in HMM model
    M - no of observations
    """
    self.A = [[0.0 for j in range(N)] for i in range(N)]
    self.B = [[0.0 for j in range(M)] for i in range(N)]
    self.pi = [[0.0 for j in range(N)]]
        
    for i in range(N):
        row_sch = 0
        for j in range(N-1):
            self.A[i][j]= abs(random.gauss(1/N, variance))
            row_sch += self.A[i][j]
        self.A[i][N-1] = 1 - row_sch
            
        row_sch = 0
        for j in range(M-1):
            self.B[i][j]= abs(random.gauss(1/M, variance))
            row_sch += self.B[i][j]
        self.B[i][M-1] = 1 - row_sch
        
    row_sch = 0
    for j in range(N-1):
        self.pi[0][j]= abs(random.gauss(1/N, variance))
        row_sch += self.pi[0][j]
    self.pi[0][N-1] = 1 - row_sch    
            
    return A, B, pi

  def alphaPass(self, A, B, pi, O):
    """
    computes alpha matrix and list of scaling coefficients C
    """
    #alpha: T*N matrix
    T = len(O)
    N = len(A)
    alpha = [[0.0 for j in range(N)] for t in range(T)]
    C = [0.0 for t in range(T)]

    #initialize alpha0
    for i in range(N):
        alpha[0][i] = pi[0][i] * B[i][O[0]]
        C[0] += alpha[0][i]

    #scale alpha0
    C[0] = 1/C[0]
    for i in range(N):
        alpha[0][i] = C[0] * alpha[0][i]

    #compute alpha
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                alpha[t][i] += A[j][i] * alpha[t-1][j]
            alpha[t][i] = alpha[t][i] * B[i][O[t]]
            C[t] += alpha[t][i]
        #scale
        C[t] = 1/C[t]
        for i in range(N):
            alpha[t][i] = C[t] * alpha[t][i]

    return alpha, C

  def evaluate(self, C):
    """
    computes log probability of given the normalization coefficients
    """
    T = len(C)
    logProb = 0.0

    for t in range(T):
        logProb += math.log(C[t])

    logProb = -1.0 * logProb
    return logProb

  def betaPass(self, A, B, pi, O, C):
    """
    computes beta given the model, observations and normalization coefficients
    C is from alpha pass
    """
    T = len(O)
    N = len(A)
    beta = [[0.0 for j in range(N)] for t in range(T)]

    #initialize
    for i in range(N):
        beta[T-1][i] = C[T-1]

    #compute beta
    for t in range(T-2, -1, -1):
        for i in range(N):
            for j in range(N):
                beta[t][i] += A[i][j] * B[j][O[t+1]] * beta[t+1][j]
            #scale
            beta[t][i] = C[t] * beta[t][i]

    return beta

  def gammaPass(self, A, B, pi, O, alpha, beta):
    """
    computes di-gamma and gamma functions
    """
    T = len(O)
    N = len(A)
    gamma = [[0.0 for j in range(N)] for t in range(T)]
    digamma = [[[0.0 for j in range(N)] for i in range(N)] for t in range(T-1)]

    #compute gamma and digamma
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                digamma[t][i][j] = alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]
                gamma[t][i] += digamma[t][i][j]

    #compute gamma T-1
    for i in range(N):
        gamma[T-1][i] = alpha[T-1][i]

    return gamma, digamma

  def update(self, A, B, pi, O, gamma, digamma):
    """
    updates A, B, pi using gamma and digamma
    """
    T = len(O)
    N = len(A)
    M = len(B[0])

    #update pi
    pi[0] = gamma[0]

    #update A
    for i in range(N):
        denom = 0
        for t in range(T-1):
            denom += gamma[t][i]
        for j in range(N):
            numer = 1e-6
            for t in range(T-1):
                numer += digamma[t][i][j]
            A[i][j] = numer/denom

    #update B
    for i in range(N):
        denom = 0
        for t in range(T-1):
            denom += gamma[t][i]
        for j in range(M):
            numer = 1e-6
            for t in range(T-1):
                if (O[t] == j):
                    numer += gamma[t][i]
            B[i][j] = numer/denom

    return A, B, pi
    
  def learn(self, A, B, pi, O):
    """
    learns A, B, pi from observations
    """
    iters = 0
    maxIters = 100
    oldLogProb = -math.inf

    alpha, C = self.alphaPass(A, B, pi, O)
    logProb = self.evaluate(C)

    while(iters < maxIters and logProb - oldLogProb > 1e-4):
        oldLogProb = logProb

        beta = self.betaPass(A, B, pi, O, C)
        gamma, digamma = self.gammaPass(A, B, pi, O, alpha, beta)
        A, B, pi = self.update(A, B, pi, O, gamma, digamma)

        alpha, C = self.alphaPass(A, B, pi, O)
        logProb = self.evaluate(C)
        iters += 1

    return A, B, pi

  ### Tracking Code###
  def initializeTracking(self, gameState):
    '''Initialze an HMM for each opponent'''
    N = 2
    M = 5
    variance = 0.01
    self.opponents = self.getOpponents(gameState)

    self._tracks = []
    self._moves = []
    self._current = []
    for agent in range(len(self.opponents)):
        A, B, pi = self._init_HMM(N, M, variance)
        self._tracks.append(A)
        self._moves.append(B)
        self._current.append(pi)
    self._observe = [[] for agent in range(len(self.opponents))]

  def inference(self, gameState):
    '''save observationsand track'''
    # store observations
    prob = []
    for agent in range(len(self.opponents)):
        #self._observe[agent].append(??)

        self._tracks[agent], self._moves[agent], self._current[agent] = self.learn(self._tracks[agent], self._moves[agent], self._current[agent], self._observe[agent])
        
        alpha, C = self.alphaPass(self._tracks[agent], self._moves[agent], self._current[agent], self._observe[agent])
        prob.append(self.evaluate(C))
    
        g = prob.index()

  



class OffensiveAgent(BasicAgent):
  '''AGENT THAT TRIES TO COLLECT FOOD'''
  goingHome = False     # Use this for the 'Going home' state, agent should return to the nearest 'home square' (nearest boundary)
  foodToChase = None    # Use this if we pursue a specific food (To avoid an agent near the nearest food, for example)
  verbose = True
  algo = 2
  

  def chooseAction(self, oldGameState):
    '''Picks the offensive agents next move given current state'''

    # Algorithm 1
    if self.algo ==1:
        gameState = self.getSuccessor(oldGameState, 'Stop')

        # If we are on our own field, we have finished going home 
        if not gameState.getAgentState(self.index).isPacman:
          self.goingHome = False

        myPos = gameState.getAgentState(self.index).getPosition()
        self.currentState = gameState

        self.debugDraw([myPos], [1.0,1.0,1.0], True)
        self.debugDraw(self.ownCapsules, [0,0,1])
        #self.debugDraw(self.enemyCapsules, [1,0,0])

        # Register base position at the start of the game
        if self.homebase == None:
          self.homebase = myPos
          if self.verbose: print("Home registered as: ")
          if self.verbose: print(self.homebase)

        numInMouth = gameState.getAgentState(self.index).numCarrying
        # Checks if any ghost is nearby
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]
    
    
        # If there's a defender close, we run away
        positions = [a.getPosition() for a in defenders]
        dists = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition()) for a in defenders]
        dists.append(1000) #Avoids empty list if no enemies close

        if min(dists) < 4:
          self.defenders = defenders # This is saved for the minmax heuristic (Recalculating every time is wasteful)

          index_of_closest = dists.index(min(dists))
          self.debugDraw([positions[index_of_closest]], [0.5,0.5,0.5], False)
          return self.minMaxEscape(myPos, positions[index_of_closest], 9)
          #return self.getActionAwayFromPoint(gameState, actions, positions[index_of_closest])

        #elif min(dists) < 5:
        #  self.debugDraw(self.homebase, [0.8,0.2,0], True)
        #  return self.getActionTowardsPoint(gameState, actions, self.homebase)

        else:
          # Get the position of the closest food as goal
          foodList = self.getFood(gameState).asList()
          if len(foodList) > 0:
            myPos = gameState.getAgentState(self.index).getPosition()
            foodDist = [self.getMazeDistance(myPos, food) for food in foodList]
            minDistance = min(foodDist)
            if numInMouth > 40 / minDistance or numInMouth > 7: # or if only 2 food left
              goal = self.getClosestBoundary(myPos)
              self.debugDraw([goal], [0,0,1], False)
            else:
              goal = foodList[foodDist.index(minDistance)] # Closest food as goal
              self.debugDraw([goal], [0,1,0], False)
          else:
            goal = self.homebase
            self.debugDraw([goal], [0.8,0.2,0], False)
      
          

    elif self.algo == 2:
        gameState = self.getSuccessor(oldGameState, 'Stop')
        myPos = gameState.getAgentState(self.index).getPosition()
        foodList = self.getFood(gameState).asList()
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])

        # Are we in enemy ground?
        if gameState.getAgentState(self.index).isPacman:
            numInMouth = gameState.getAgentState(self.index).numCarrying
            foodCount = len(self.getFood(gameState).asList())

            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]
            if len(defenders) > 0:
              dists = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition()) for a in defenders]
              indexOfClosest = dists.index(min(dists))
              closestEnemy = defenders[indexOfClosest]
              enemyNearby = min(dists) < 5 # To prevent from triggering if the other agent sees a ghost that is nowhere near attacker
            else:
              enemyNearby = False
            

            # Is there an enemy nearby?
            if enemyNearby:

              powerTimeLeft = closestEnemy.scaredTimer # Hopefully this is 0 if enemy is not scared...
              
              # Is enemy scared for much longer?
              if powerTimeLeft > 5:
                if numInMouth > 40 / minDistance or numInMouth > 5 or foodCount <= 2:
                  goal = self.getClosestBoundary(myPos)
                else:
                  goal = self.getClosestFood(gameState)
                  if self.capsuleAvailable(gameState):                  
                    cap = self.getClosestCapsule(gameState)
                    if self.getMazeDistance(myPos, cap) < 3 * self.getMazeDistance(myPos, goal):
                      goal = cap

              # Enemy is close but not scared (or not scared for long)
              else:
                return self.minMaxEscape(myPos, closestEnemy.getPosition(), 9)
            
            # There is no nearby enemy
            else:
              if numInMouth > 40 / minDistance or numInMouth > 7 or foodCount <= 2:
                goal = self.getClosestBoundary(myPos)
              else:
                goal = self.getClosestFood(gameState)
                if self.capsuleAvailable(gameState):                  
                  cap = self.getClosestCapsule(gameState)
                  if self.getMazeDistance(myPos, cap) < 2 * self.getMazeDistance(myPos, goal):
                    goal = cap

        # We are not in enemy ground
        else:
            #go to closest enemy food
            if self.getClosestFood(gameState)!=None:
                goal = self.getClosestFood(gameState)
                if self.verbose: print("Reach for closest food")
            else:
                #go to boundary
                goal = self.getClosestBoundary(gameState.getAgentState(self.index).getPosition())
    
    if self.verbose: print(goal)
    if self.verbose: self.debugDraw([goal], [0,1,0], True)
    actions = gameState.getLegalActions(self.index)
    actions.remove('Stop') # Never allow standing still (does not really contribute to anything)
    best_action = self.getActionTowardsPoint(gameState, actions, goal)

    return best_action


  def minMaxEscape(self, myPos, enemyPos, maxDepth):
    '''Uses minmax algorithm to pick best moves to escape'''
    if self.verbose: print("Calling MINMAX with depth " + str(maxDepth))
    #print("Starting pos has player at (" + str(myPos[0]) + ", " + str(myPos[1]) + ") and enemy at (" + str(enemyPos[0]) + ", " + str(enemyPos[1]) + ")")
    ownActions = self.getMovesFromLayout(myPos)
    #print(str(ownActions) + " when friend at (" + str(myPos[0]) + ", " + str(myPos[1]) + ")")
    bestScore = -100
    bestMove = 'Stop'

    for action in ownActions:
      #print("* friend goes " + action)
      newPos = self.updatePos(myPos, action)
      moveScore = self.minMove(newPos, enemyPos, maxDepth-1, alpha=-10000, beta=10000)
      if self.verbose: print(action + " gives score " + str(moveScore))
      if moveScore > bestScore:
        bestMove = action
        bestScore = moveScore
    if self.verbose: print(bestMove + " picked.")

    return bestMove #Returns move that gives best node given optimal play from both sides

  def minMove(self, myPos, enemyPos, depth, alpha, beta):
    '''Simulates enemy move, tries to minimise the heuristic'''
    if depth <= 0:
      #print("*"*(3-depth) + " bottomed out")
      return self.evaluationScore(myPos, enemyPos) # Max depth heuristic
    
    if myPos[0] == enemyPos[0] and myPos[1] == enemyPos[1]:
      #print("~Enemy walked into us, score -1")
      return -1 # We are on the enemy and die. This is bad!
    
    if myPos[0] == self.ownBoundaryX:
      return 100 * depth # We made it home, success! (Multiply by depth so shorter path (less depth) is better)

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
    '''Simulates our move, tries to maximise the heuristic'''
    if depth <= 0:
      #print("*"*(3-depth) + " bottomed out")
      return self.evaluationScore(myPos, enemyPos) # Max depth heuristic

    if myPos[0] == enemyPos[0] and myPos[1] == enemyPos[1]:
      #print("~We walked into enemy, score -1")
      return -1 # We are on the enemy and die. This is bad!

    if myPos[0] == self.ownBoundaryX:
      return 100 * depth # We made it home, success! (Multiply by depth so shorter path (less depth) is better)

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
    '''The heuristic function used to evaluate board position at max depth'''
    myPos = (myPos[0], myPos[1])
    enemyPos = (enemyPos[0], enemyPos[1])
    distToEnemy = self.getMazeDistance(myPos, enemyPos)
    boundaryPoint = self.getClosestBoundary(myPos)
    distToHomeField = self.getMazeDistance(myPos, boundaryPoint)
    
    #dists = [self.getMazeDistance(myPos, a.getPosition()) for a in self.defenders]
    #dists = [dist for dist in dists if dist < 5]

    return distToEnemy + 1.0/(distToHomeField + 1.0) #+ sum(dists)

  def powerPacman(self, gameState):
    enemies = self.getOpponents(gameState)
    for enemy in enemies:
        if gameState.getAgentState(enemy).scaredTimer > 1:
            self.powerUpTime = gameState.getAgentState(enemy).scaredTimer
            return True
    return False

  def capsuleAvailable(self, gameState):
    '''finds if any capsule available'''
    currFoods = self.getCapsules(gameState)
    if len(currFoods) > 0:
        self.targetCapsule = currFoods[0]
        return True
    else:
        return False

  def getClosestFood(self, gameState):
    foodList = self.getFood(gameState).asList()
    if len(foodList) > 0:
        myPos = gameState.getAgentState(self.index).getPosition()
        foodDist = [self.getMazeDistance(myPos, food) for food in foodList]
        minDistance = min(foodDist)
        return foodList[foodDist.index(minDistance)]
    else:
        return None

  def getClosestCapsule(self, gameState):
    capsules = self.getCapsules(gameState)
    dists = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), capsule) for capsule in capsules]
    indexOfClosest = dists.index(min(dists))
    return capsules[indexOfClosest]

 


class DefensiveAgent(BasicAgent):
  '''AGENT THAT TRIES TO STOP ENEMY FROM GRABBING'''
  hasBeenPacman = False
  lastMissingFood = None
  protectCapsule = None
  verbose = False
  algo = 1
  if verbose: print("Algorithm: ", algo)

  def chooseAction(self, oldGameState):
    '''Choses action for the defensive agent given the state'''  
    # Algorithm 1: Protect boundary 
    if self.algo==1:
        gameState = self.getSuccessor(oldGameState, 'Stop')
        self.currentState = gameState

        # is agent in home ground?
        if gameState.getAgentState(self.index).isPacman == False:
            # is agent scared?
            if gameState.getAgentState(self.index).scaredTimer > 0:
                # move towards boundary
                goal = self.getClosestBoundary(gameState.getAgentState(self.index).getPosition())
                if self.verbose: print("Scared! Go to boundary... ")
            else:
                # are enemies in home?
                enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
                invaders = [a for a in enemies if a.isPacman]
                if len(invaders)> 0:
                    if self.verbose: print("Enemy in home")
                    # are visible enemies there?
                    visible_invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
                    if len(visible_invaders) > 0: 
                        # Pick the action that closes the distance to the nearest invader
                        positions = [a.getPosition() for a in visible_invaders]
                        dists = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition()) for a in visible_invaders]
                        index_of_closest = dists.index(min(dists))
                        goal = positions[index_of_closest]
                        if self.verbose: print("Chasing closest visible enemies ...")
                    else:
                        # is food missing?
                        if self.missingFood(gameState):
                            goal = self.lastMissingFood
                            if self.verbose: print("Chasing invisible enemies ...")
                        elif self.lastMissingFood!=None:
                            goal = self.lastMissingFood
                            if self.verbose: print("Chasing invisible enemies ...")
                        elif self.capsuleAvailable(gameState):
                            goal = self.protectCapsule
                            if self.verbose: print("Protecting capsule ...")
                        else:
                            # move towards boundary
                            goal = self.getClosestBoundary(gameState.getAgentState(self.index).getPosition())
                            if self.verbose: print("Protecting boundary... ")           
                else:
                    # move towards boundary
                    goal = self.getClosestBoundary(gameState.getAgentState(self.index).getPosition())
                    if self.verbose: print("Protecting boundary... ")
        else:
            # move towards boundary
            goal = self.getClosestBoundary(gameState.getAgentState(self.index).getPosition())
            if self.verbose: print("Not in home! Go to boundary... ")
            
        if self.verbose: print(goal)
        if self.verbose: self.debugDraw([goal], [0,1,1], False)
        actions = gameState.getLegalActions(self.index)
        best_action = self.getActionTowardsPoint(gameState, actions, goal)

    # Algorithm 2: Protect power capsules
    elif self.algo==2:
        gameState = self.getSuccessor(oldGameState, 'Stop')
        self.currentState = gameState

        # is agent in home ground?
        if gameState.getAgentState(self.index).isPacman == False:
            # is agent scared?
            if gameState.getAgentState(self.index).scaredTimer > 0:
                # move towards boundary
                goal = self.getClosestBoundary(gameState.getAgentState(self.index).getPosition())
                if self.verbose: print("Scared! Go to boundary... ")
            else:
                # check if enemies in home 
                enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
                invaders = [a for a in enemies if a.isPacman]
                if len(invaders)> 0:
                    # check for visible enemies
                    visible_invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
                    if len(visible_invaders) > 0: 
                        # Pick the action that closes the distance to the nearest invader
                        positions = [a.getPosition() for a in visible_invaders]
                        dists = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition()) for a in visible_invaders]
                        index_of_closest = dists.index(min(dists))
                        goal = positions[index_of_closest]
                        if self.verbose: print("Chasing closest visible enemies ...")
                    else:
                        # check for missing food
                        if self.missingFood(gameState):
                            goal = self.lastMissingFood
                            if self.verbose: print("Chasing invisible enemies ...")
                        elif self.capsuleAvailable(gameState):
                            goal = self.protectCapsule
                            if self.verbose: print("Protecting capsule ...")
                        elif self.lastMissingFood!=None:
                            goal = self.lastMissingFood
                            if self.verbose: print("Chasing invisible enemies ...")
                        else:
                            # move towards boundary
                            goal = self.getClosestBoundary(gameState.getAgentState(self.index).getPosition())
                            if self.verbose: print("Protecting boundary... ")        
                else:
                    # protect capsule
                    if self.capsuleAvailable(gameState):
                        goal = self.protectCapsule
                        if self.verbose: print("Protecting capsule ...")
                    else:
                        # move towards boundary
                        goal = self.getClosestBoundary(gameState.getAgentState(self.index).getPosition())
                        if self.verbose: print("Protecting boundary... ")
        else:
            # move towards boundary
            goal = self.getClosestBoundary(gameState.getAgentState(self.index).getPosition())
            if self.verbose: print("Go to boundary... ")
            
        if self.verbose: print(goal)
        if self.verbose: self.debugDraw([goal], [0,1,1], False)
        actions = gameState.getLegalActions(self.index)
        best_action = self.getActionTowardsPoint(gameState, actions, goal)

    # default algo
    else: 
        gameState = self.getSuccessor(oldGameState, 'Stop')
        self.currentState = gameState

        if (not self.hasBeenPacman) and gameState.getAgentState(self.index).isPacman:
          self.hasBeenPacman = True
          if self.verbose: print("FINAL HOME POSITION SET TO: ")
          if self.verbose: print(self.homebase)

        if self.hasBeenPacman == False:
          prev = self.getPreviousObservation()
          self.homebase = (prev if prev != None else gameState).getAgentState(self.index).getPosition()

        actions = gameState.getLegalActions(self.index)

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
    
    return best_action

  def missingFood(self, gameState):
    '''finds where food goes missing'''
    prevState = self.getPreviousObservation()
    prevFoods = self.getFoodYouAreDefending(prevState).asList()
    currFoods = self.getFoodYouAreDefending(gameState).asList()
    if len(prevFoods) != len(currFoods):
        for food in prevFoods:
            if food not in currFoods:
                self.lastMissingFood = food
        return True
    else:
        return False

  def capsuleAvailable(self, gameState):
    '''finds if any capsule available'''
    currFoods = self.getCapsulesYouAreDefending(gameState)
    if len(currFoods) > 0:
        self.protectCapsule = currFoods[0]
        return True
    else:
        return False