
import math
import random
import sys
import queue

class Problem(object):

    def __init__(self, initial, goal=None):        #neccessary

        self.initial = initial
        self.goal = goal

    def getRandomState(self):

        raise NotImplementedError

    def actions(self, state):                   #neccessary

        raise NotImplementedError

    def result(self, state, action):            #neccessary

        raise NotImplementedError

    def goal_test(self, state):             #neccessary

        raise NotImplementedError

    def path_cost(self, c, state1, action, state2):

        raise NotImplementedError

    def huristic(self, state):

        raise NotImplementedError

    def getGoalstate(self):

        raise NotImplementedError
# ______________________________________________________________________________

class SimpleVaccumeMachine (Problem):

    def __init__(self , agentPosition = 'L', leftRoomStatus = 'D', rightRoomStatus = 'D'):
        self.initial = [agentPosition , leftRoomStatus , rightRoomStatus]

    def actions(self, state):
        if state[0] =='L' :
            actions = ['R' , 'C']
        if state[0] == 'R' :
            actions = ['L' , 'C']
        return actions

    def result(self, state, action):
        nextState = state.copy()
        if action == 'C':
            if state[0] == 'L':
                nextState[1] = 'C'
            if state[0] == 'R':
                nextState[2] = 'C'
            return nextState
        if action == 'L':
            nextState[0] = 'L'
            return nextState
        if action == 'R':
            nextState[0] = 'R'
            return nextState

    def goal_test(self, state):
        if state[1] == 'C' and state[2] == 'C':
            return True
        return False

    def path_cost(self, c, state1, action, state2):
        return c+1

class Puzzle(Problem):
    """state is a dictionary"""
    def __init__(self , initial = None , goal = None):
        #     for initial state
        if initial == None:
            self.initial = self.getRandomState()
        else:
            self.initial = initial
        #     for goal state
        itr_number = 0
        if goal == None:
            valid_values = [1, 2, 3, 4, 5, 6, 7, 8, 0]
            puzzle = {}
            for row in range(0, 3, 1):
                for col in range(0, 3, 1):
                    puzzle[row, col] = itr_number
                    itr_number += 1
            self.goal = puzzle
        else:
            self.goal = goal

    def getCoordinateCell(self , dict , cellValue):
        cellIndex = list(dict.keys())[list(dict.values()).index(cellValue)]
        return cellIndex

    def goal_test(self, state):
        return state == self.goal

    def actions(self, state):
        actions = ['U' ,'D' , 'L' , 'R']
        emptyCell = self.getCoordinateCell(state , 0)
        row = emptyCell[0]
        col = emptyCell[1]

        if row == 0 :
            actions.remove('U')
        if col == 0 :
            actions.remove('L')
        if row == 2 :
            actions.remove('D')
        if col == 2 :
            actions.remove('R')

        return actions

    def result(self, state = {}, action = ''):
        newState = state.copy()
        emptyCell = self.getCoordinateCell(state, 0)
        if action == 'L':
            fullCellValue = state.get((emptyCell[0] , emptyCell[1] - 1))
            newState[emptyCell[0] , emptyCell[1]] = fullCellValue
            newState[emptyCell[0] , emptyCell[1] - 1] = 0
        if action =='R':
            fullCellValue = state.get((emptyCell[0], emptyCell[1] + 1))
            newState[emptyCell[0], emptyCell[1]] = fullCellValue
            newState[emptyCell[0], emptyCell[1] + 1] = 0
        if action =='U':
            fullCellValue = state.get((emptyCell[0] - 1, emptyCell[1]))
            newState[emptyCell[0], emptyCell[1]] = fullCellValue
            newState[emptyCell[0] - 1, emptyCell[1]] = 0
        if action =='D':
            fullCellValue = state.get((emptyCell[0] + 1, emptyCell[1]))
            newState[emptyCell[0], emptyCell[1]] = fullCellValue
            newState[emptyCell[0] + 1, emptyCell[1]] = 0

        return newState

    def getRandomState(self):

        valid_values = [1, 2, 3, 4, 5, 6, 7, 8, 0]
        puzzle = {}
        for row in range(0, 3, 1):
            for col in range(0, 3, 1):
                tmp = random.choice(valid_values)
                valid_values.remove(tmp)
                puzzle[row, col] = tmp
        return puzzle

    def path_cost(self, c, state1, action, state2):
        return c+1

    def huristic(self, state = {}):
        """sum of the belman ford distance of each tile to it's correct place"""
        h = 0
        def makePositive(x):
            if x < 0:
                return -x
            return x

        def difference(x, y):
            return makePositive(x - y)

        def coordinateDifference(c1, c2):
            return makePositive(c1[0] - c2[0]) + makePositive(c1[1] - c2[1])

        for i in range(0,9,1):
            h += coordinateDifference(self.getCoordinateCell(state,i) , self.getCoordinateCell(self.goal,i))
        return h;

class NQueen(Problem):

    def __init__(self , n ):
        self.n = n
        self.initial = []
        i = 0
        for i in range(0,n,1):
            self.initial.append(i)
        print("initialized NQueen state: ", self.initial)

    def getRandomState(self):
        """retrun a random generated state (two queen in the same row is possible!)"""
        randomState = []
        for i in range(0,self.n,1):
            randomState.append(i)
        random.shuffle(randomState)
        return randomState

    def actions(self, state):
        """in every states actions are the same : all changing two columns
         ex: [1,3] - changes the queen in second column with the fourth column"""
        actions = []
        j = 0
        i = 1
        while j < len(state) - 1:
            while i < len(state):
                actions.append([j, i])
                i += 1
            j += 1
            i = j + 1
        return actions

    def result(self, state, action):
        nextState = state.copy()
        i = action[0]
        j = action[1]
        temp = nextState[i]
        nextState[i] = nextState[j]
        nextState[j] = temp
        return nextState

    def goal_test(self, state):
        if self.huristic(state) == 0:
            return True
        return False

    def path_cost(self, c , state1, action, state2 ):
        return c+1

    def isConflict(self, x1, x2, y1, y2):
        if x1 == x2:
            return True
        if y1 == y2:
            return True
        if (x1 - x2) / (y1 - y2) == 1:
            return True
        if (x1 - x2) / (y1 - y2) == -1:
            return True
        return False

    def huristic(self ,state):
        """The heuristic cost function h is the number of pairs of numbers that are attacking each other"""
        counter = j = 0
        i = 1
        while j < len(state) - 1:
            while i < len(state):
                # print(state[j],state[i])
                if self.isConflict(j, i, state[j], state[i]):
                    counter += 1
                i += 1
            j += 1
            i = j + 1
        return counter

class Maze(Problem):

    def __init__(self, m = int , n = int, blocks = []):
        self.blocks = blocks
        self.initial = [1,1]
        self.goal = [m,n]
        print("MAZE initialized m:" , m ,"n:", n ,"blocks:", blocks)

    def actions(self, state):

        actions , y , x = [] , state[0], state[1]
        m , n = self.goal[0] , self.goal[1]

        if x + 1 <= n :
            if [y, x+1] not in self.blocks :
                actions.append("right")
        if y + 1 <= m :
            if [y+1 , x] not in self.blocks :
                actions.append("down")
        if x - 1 > 0 :
            if [y , x-1] not in self.blocks :
                actions.append("left")
        if y - 1 > 0 :
            if [y-1 , x] not in self.blocks :
                actions.append("up")
        return actions

    def result(self, state, action):

        y, x =  state[0], state[1]

        if action == "right" :
            x += 1
        if action == "down":
            y += 1
        if action == "left":
            x -= 1
        if action == "up":
            y -= 1
        return [y,x]

    def goal_test(self, state):
        """The Goal is Fixed in [m,n] where m is row and n is collumn """
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def huristic(self, state):

        raise NotImplementedError
    def getGoalstate(self):
        return self.goal

class Node:

    def __init__(self, state, parent=None, action=None, path_cost=0):

        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def getPathCost(self):
        return self.path_cost

    def getChilds (self , problem):
        """return childs list (just state attribute) of current node in the specified problem"""
        childs=[]
        for action in problem.actions(self.state):
            childs.append(problem.result(self.state , action))
        return childs

    def expand2(self , problem):
        """child_node and expand toghether"""
        actions = problem.actions(self.state)
        childs = []
        for action in actions:
            childs.append(Node(problem.result(self.state,action),self,action,problem.path_cost(self.path_cost,self.state , action , problem.result(self.state , action))))
        return childs

    def expand(self, problem):
        """return the created childs of this node"""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """create and return child node of this node"""
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solve(self):
        """return the action sequence to this state"""
        actionList = []
        node = self
        while node.parent:
            # print(node.state , node.action)
            actionList.append(node.action)
            node = node.parent
        actionList.reverse()         #moving this step into print(next line) makes some problem !!!
        print("depth:",self.depth,"\npath cost:",self.path_cost,"\nfounded path : " ,actionList)
        return

    def solve_backtrack(self , path):
        """return the action sequence to this state (recursively)"""
        path.append(self.action)
        if not self.parent:
            print(path)
            return
        return solve_backtrack(self.parent , path)

    def solution(self):
        """return the path to this state"""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """return the path to this state"""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def getG(self):
        """return the total path cost to this node"""
        node = self
        g = 0
        while node:
            g = g + node.path_cost
            node = node.parent
        return g

class MyQueue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def enqueueSortAsc(self, item):
        self.items.insert(0,item)
        self.items.sort(reverse=True)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def showItems(self):
        for item in self.items:
            print(item)

class MyPriorityQueue(MyQueue):

    def sortAsc(self):
        self.items.sort()

    def sortDesc(self):
        self.items.sort(reverse=True)

class MyStack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.size() == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def showItems(self):
        print(self.items)

def hill_climbing(problem):
    """all hill climbing huristics are descending. means: state with lower h is better """
    currentNode = Node(problem.initial)
    pq = MyQueue()
    expandedNodes = createdNodes = 0
    while True:
        neighbours = currentNode.expand(problem)
        expandedNodes += 1
        for neighbour in neighbours:
            createdNodes += 1
            pq.enqueueSortAsc((problem.huristic(neighbour.state),neighbour))
        neighbour = pq.dequeue()
        neighbour = neighbour[1]
        if problem.huristic(neighbour.state) >= problem.huristic(currentNode.state):
            print("Created Nodes:" , createdNodes)
            print("Expanded Nodes:" , expandedNodes)
            return currentNode.state
        else:
            currentNode = neighbour
            print("state:",currentNode.state , "with huristic:" , problem.huristic(currentNode.state))

def hill_climbing_stochastic(problem , n = 100):
    """selects a random neighbour which is better than the current state (lower h).
     n is the number of try for finding a better state randomly in each loop.
     by default n = state space ex: array lengh"""

    currentNode = Node(problem.initial)
    q = MyQueue()
    expandedNodes = createdNodes = 0
    while True:
        counter = len(problem.initial)
        counter = n
        while counter > 0:
            counter -= 1
            neighbours = currentNode.expand(problem)
            expandedNodes += 1
            createdNodes += 1
            neighbour = random.choice(neighbours)
            if problem.huristic(neighbour.state) <= problem.huristic(currentNode.state):
                currentNode = neighbour
                print("state:", currentNode.state, "with huristic:", problem.huristic(currentNode.state))
                break
        if counter < 1:
            break
    print("Created Nodes:", createdNodes)
    print("Expanded Nodes:", expandedNodes)
    return currentNode.state

def hill_climbing_random_restart(problem , run_counter = 8):

    currentNode = Node(problem.initial)
    pq = MyQueue()
    while True:
        if run_counter < 0:
            return currentNode.state
        print("state:" , currentNode.state , " huristic:" , problem.huristic(currentNode.state))
        neighbours = currentNode.expand(problem)
        for neighbour in neighbours:
            pq.enqueueSortAsc((problem.huristic(neighbour.state), neighbour))
        neighbour = pq.dequeue()
        neighbour = neighbour[1]
        if problem.huristic(neighbour.state) > problem.huristic(currentNode.state):
            currentNode = Node(problem.getRandomState())
            run_counter -= 1
        else:
            currentNode = neighbour
            print(currentNode.state)

# TODO modify it (analytical report)

def dfs_rec (problem):

    root = Node(problem.initial)
    e = []
    sys.setrecursionlimit(50000)
    depthFirstSearch(problem , root , e)
    return

def depthFirstSearch(problem = Problem, node = Node, visited=[]):

    if problem.goal_test(node.state):
        print("reached the goal :", node.state)
        # print("# of expanded nodes : ", expanded_counter)
        node.solve()
        return
    visited.append(node.state)
    neighbours = node.expand(problem)
    for neighbour in neighbours:
        if neighbour.state not in visited:
            depthFirstSearch(problem , neighbour , visited)

def bfs_tree(problem):

    root = Node(problem.initial)
    f = MyQueue()
    f.enqueue(root)
    visitedNodesCounter = 1
    expanded_counter = 1
    memory_counter = 2
    while f :
        print(memory_counter, " nodes are in memory(f) right know")
        currentNode = f.dequeue()
        memory_counter -= 1
        if problem.goal_test(currentNode.state):
            print("reached the goal :" , currentNode.state)
            print("# of expanded nodes : " , expanded_counter)
            return currentNode.solution()
        else:
            childs = currentNode.expand(problem)
            print(currentNode.state , "   expanded")
            expanded_counter += 1
            for child in childs:
                f.enqueue(child)
                memory_counter += 1
                visitedNodesCounter += 1
                print(child.state , "   added")
        print( "# of visited nodes : ", visitedNodesCounter , "\n")
    return "end of bfs and nothing"

def bfs_graph(problem):
    path=[] #for solve_backtrack(path)
    root = Node(problem.initial)
    e = []
    f = MyQueue()
    f.enqueue(root)
    visitedNodesCounter = 1
    expanded_counter = 1
    memory_counter = 2
    while f :
        # print("e : " , e[:] , "\nf : " , f.items)
        print(memory_counter, " nodes are in memory(f) right know")
        currentNode = f.dequeue()
        e.append(currentNode.state)                         #cheto mishe akhe currentNode age be e ezaf konim kar nemikone !!
        memory_counter -= 1
        if problem.goal_test(currentNode.state):
            print("reached the goal :" , currentNode.state)
            print("# of expanded nodes : " , expanded_counter)
            currentNode.solve()
            return currentNode.solution()
        else:
            childs = currentNode.expand(problem)
            print(currentNode.state , "   expanded")
            expanded_counter += 1
            for child in childs:
                if child not in f.items and child.state not in e:    #cheto mishe akhe currentNode age be e ezaf konim kar nemikone !!
                    f.enqueue(child)
                    memory_counter += 1
                    visitedNodesCounter += 1
                    print(child.state , "   added")
        print( "# of visited nodes : ", visitedNodesCounter , "\n")
    return "end of bfs and nothing"

def ucs_graph(problem):
    root = Node(problem.initial)
    f = MyPriorityQueue()
    f.enqueue([root , root.path_cost])
    visitedNodesCounter = 1
    expanded_counter = 1
    memory_counter = 1
    e = [root.state]
    while f :
        f.sortDesc()
        print(memory_counter, " nodes are in memory(f) right know")
        tmp = f.dequeue()
        currentNode = tmp[0]
        e.append(currentNode.state)
        memory_counter -= 1
        if problem.goal_test(currentNode.state):
            print("reached the goal :" , currentNode.state)
            print("# of expanded nodes : " , expanded_counter)
            currentNode.solve()
            return currentNode.solution()
        else:
            childs = currentNode.expand(problem)
            print(currentNode.state , "   expanded")
            expanded_counter += 1
            for child in childs:
                if child.state not in e:
                    f.enqueue([child , child.path_cost])
                    memory_counter += 1
                    visitedNodesCounter += 1
                    print(child.state , "   added with path cost:" , child.path_cost)
        print( "# of visited nodes : ", visitedNodesCounter , "\n")
    return "end of bfs and nothing"

def astar_graph(problem):
    root = Node(problem.initial)
    f = MyPriorityQueue()
    f.enqueue([root , problem.huristic(root.state)])
    visitedNodesCounter = 1
    expanded_counter = 1
    memory_counter = 1
    e = [root.state]
    while f :
        f.sortDesc()
        print(memory_counter, " nodes are in memory(f) right know")
        tmp = f.dequeue()
        currentNode = tmp[0]
        e.append(currentNode.state)
        memory_counter -= 1
        if problem.goal_test(currentNode.state):
            print("reached the goal :" , currentNode.state)
            print("# of expanded nodes : " , expanded_counter)
            currentNode.solve()
            return currentNode.solution()
        else:
            childs = currentNode.expand(problem)
            print(currentNode.state , "   expanded")
            expanded_counter += 1
            for child in childs:
                if child.state not in e:
                    f.enqueue([child , problem.huristic(child.state)])
                    memory_counter += 1
                    visitedNodesCounter += 1
                    print(child.state , "   added with huristic:" , problem.huristic(child.state))
        print( "# of visited nodes : ", visitedNodesCounter , "\n")
    return "end of bfs and nothing"

def dfs_tree(problem):

    root = Node(problem.initial)
    f = MyStack()
    f.push(root)
    visitedNodesCounter = 1
    expanded_counter = 1
    memory_counter = 2
    while f :
        print(memory_counter, " nodes are in memory(f) right know")
        currentNode = f.pop()
        memory_counter -= 1
        if problem.goal_test(currentNode.state):
            print("reached the goal :" , currentNode.state)
            print("# of expanded nodes : " , expanded_counter)
            currentNode.solve()
            return currentNode.solution()
        else:
            childs = currentNode.expand(problem)
            print(currentNode.state , "   expanded")
            expanded_counter += 1
            for child in childs:
                f.push(child)
                memory_counter += 1
                visitedNodesCounter += 1
                print(child.state , "   added")
        print( "# of visited nodes : ", visitedNodesCounter , "\n")
    return "end of bfs and nothing"

def dfs_graph(problem):
    path=[] #for solve_backtrack(path)
    root = Node(problem.initial)
    e = []
    f = MyStack()
    f.push(root)
    visitedNodesCounter = 1
    expanded_counter = 1
    memory_counter = 2
    while f :
        print("e : " , e , "\nf : " , f.items)
        print(memory_counter, " nodes are in memory(f) right know")
        currentNode = f.pop()
        e.append(currentNode.state)                         #cheto mishe akhe currentNode age be e ezaf konim kar nemikone !!
        memory_counter -= 1
        if problem.goal_test(currentNode.state):
            print("reached the goal :" , currentNode.state)
            print("# of expanded nodes : " , expanded_counter)
            currentNode.solve()
            return currentNode.solution()
        else:
            childs = currentNode.expand(problem)
            print(currentNode.state , "   expanded")
            expanded_counter += 1
            for child in childs:
                if child not in f.items and child.state not in e:    #cheto mishe akhe currentNode age be e ezaf konim kar nemikone !!
                    f.push(child)
                    memory_counter += 1
                    visitedNodesCounter += 1
                    print(child.state , "   added")
        print( "# of visited nodes : ", visitedNodesCounter , "\n")
    return "end of bfs and nothing"



#-------------------------------------------- node function tests
# maze = Maze(4,4,[2,2])
# print(maze.path_cost(1,[3,3] ,"down" , [4,4]))
# node1 = Node([3,1])
# node2 = Node(2,node1)
# node3 = Node(3,node2)
# node4 = Node([1,1],action="wow action",parent=node3)
# print(node4.child_node(maze,"down"))
# print(node1.expand(maze))

#-------------------------------------------- search algorithm tests
# maze = Maze(6,6,[[1,2],[1,3],[3,1]])
# maze = Maze(3,3)
# nqueen = NQueen(4)
# smv = SimpleVaccumeMachine()
# bfs_graph(nqueen)
# dfs_rec(nqueen)
# bfs_graph(nqueen)
# bfs_tree(nqueen)
# ucs_graph(maze)
# astar_graph(nqueen)
# print(dfs_graph(maze))
# print(dfs_tree(maze))
# dfs_recurisve(maze)
# hill_climbing(nqueen)
# hill_climbing_stochastic(nqueen)
# hill_climbing_random_restart(nqueen,10)
#-------------------------------------------- Queue tests
# q = MyQueue()
# q.enqueue(2)
# q.enqueue(2)
# q.enqueue(2)
# q.enqueue(3)
# q.enqueue(3)
# q.enqueue(3)
# print(q.items)

#-------------------------------------------- Stack tests
# simpleStack = MyStack()
# simpleStack.push(3)
# simpleStack.push(3)
# simpleStack.push(2)
# simpleStack.push(2)
# simpleStack.showItems()
# simpleStack.pop()
# simpleStack.showItems()
# simpleStack.pop()
# simpleStack.pop()
# simpleStack.showItems()
#-------------------------------------------- NQueen tests
# array = []
# for action in nqueen.actions(nqueen.initial):
#     next = nqueen.result(nqueen.initial,action)
#     array.append(next)
#     print(next , action , nqueen.huristic(next))
# print(array)
# nqueen = NQueen(5)
# print(nqueen.getRandomState())
#-------------------------------------------- Routing (8-Puzzle Game) tests

initialStatePuzzle = {(0,0):1 , (0,1):6 , (0,2):2 , (1,0):5 , (1,1):3 , (1,2):0 , (2,0):4 , (2,1):7 , (2,2):8}
goalStatePuzzle = {(0,0):1 , (0,1):2 , (0,2):3 , (1,0):4 , (1,1):5 , (1,2):6 , (2,0):7 , (2,1):8 , (2,2):0}

puzzle = Puzzle(initialStatePuzzle , goalStatePuzzle) #initializing puzzle

bfs_graph(puzzle)
# ucs_graph(puzzle)
# astar_graph(puzzle)
