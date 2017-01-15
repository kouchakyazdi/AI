
from utils import *
from grid import distance
from queue import *
import math
import random
import sys
import bisect
import queue

class Problem(object):

    def __init__(self, initial, goal=None):

        self.initial = initial
        self.goal = goal

    def getRandomState(self):

        raise NotImplementedError

    def actions(self, state):

        raise NotImplementedError

    def result(self, state, action):

        raise NotImplementedError

    def goal_test(self, state):

        raise NotImplementedError

        # if isinstance(self.goal, list):
        #     return is_in(state, self.goal)
        # else:
        #     return state == self.goal

    def path_cost(self, c, state1, action, state2):

        raise NotImplementedError

    def huristic(self, state):

        raise NotImplementedError
# ______________________________________________________________________________

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
        """The heuristic cost function h is the number of pairs of queens that are attacking each other"""
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
# ______________________________________________________________________________

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

class Node:

    def __init__(self, state, parent=None, action=None, path_cost=0):

        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __lt__(self, node):
        return self.state < node.state

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

class PSA:

    def __init__(self, initial_state=None):
        self.state = initial_state
        self.seq = []

    def __call__(self, percept):
        self.state = self.update_state(self.state, percept)
        if not self.seq:
            goal = self.formulate_goal(self.state)
            problem = self.formulate_problem(self.state, goal)
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq.pop(0)

    def update_state(self, percept):
        raise NotImplementedError

    def formulate_goal(self, state):
        raise NotImplementedError

    def formulate_problem(self, state, goal):
        raise NotImplementedError

    def search(self, problem):
        raise NotImplementedError

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

# TODO fix it. dosent work :(
def dfs_recurisve(problem):
    root = Node(problem.initial,action=problem.actions(problem.initial))
    # f = []
    e = []
    def inner_dfs_rec(node , e):
        current_node = node
        if problem.goal_test(current_node.state):
            # f.append(node.action)
            return current_node
        else:
            childs = current_node.expand(problem)
            for child in childs not in e:
                e.append(child)
                return inner_def_rec(child , e)
    return inner_dfs_rec(root , e)

def bfs_tree(problem):

    root = Node(problem.initial)
    f = MyQueue()
    f.enqueue(root)
    round_counter = 1
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
                print(child.state , "   added")
        round_counter += 1
        print( "# of visited nodes : ", round_counter , "\n")
    return "end of bfs and nothing"


def bfs_graph(problem):
    path=[] #for solve_backtrack(path)
    root = Node(problem.initial)
    e = []
    f = MyQueue()
    f.enqueue(root)
    round_counter = 1
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
                    print(child.state , "   added")
        round_counter += 1
        print( "# of visited nodes : ", round_counter , "\n")
    return "end of bfs and nothing"

def ucs_graph(problem):
    root = Node(problem.initial)
    f = MyPriorityQueue()
    f.enqueue([root , root.path_cost])
    round_counter = 1
    expanded_counter = 1
    memory_counter = 2
    e = [root.state]
    while f :
        f.sortDesc()
        # print(memory_counter, " nodes are in memory(f) right know")
        tmp = f.dequeue()
        currentNode = tmp[0]
        e.append(currentNode.state)
        print(currentNode.state)
        # memory_counter -= 1
        if problem.goal_test(currentNode.state):
            print("reached the goal :" , currentNode.state)
            # print("# of expanded nodes : " , expanded_counter)
            return currentNode.solution()
        else:
            childs = currentNode.expand(problem)
            # print(currentNode.state , "   expanded")
            expanded_counter += 1
            for child in childs:
                if child.state not in e:
                    f.enqueue([child , child.path_cost])

                # memory_counter += 1
                # print(child.state , "   added")
                # print( child.path_cost)
        # round_counter += 1
        # print( "# of visited nodes : ", round_counter , "\n")
    return "end of bfs and nothing"

def dfs_tree(problem):

    root = Node(problem.initial)
    f = MyStack()
    f.push(root)
    round_counter = 1
    expanded_counter = 1
    memory_counter = 2
    while f :
        print(memory_counter, " nodes are in memory(f) right know")
        currentNode = f.pop()
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
                f.push(child)
                memory_counter += 1
                print(child.state , "   added")
        round_counter += 1
        print( "# of visited nodes : ", round_counter , "\n")
    return "end of bfs and nothing"

def dfs_graph(problem):
    path=[] #for solve_backtrack(path)
    root = Node(problem.initial)
    e = []
    f = MyStack()
    f.push(root)
    round_counter = 1
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
            # currentNode.solve()
            return currentNode.solution()
        else:
            childs = currentNode.expand(problem)
            print(currentNode.state , "   expanded")
            expanded_counter += 1
            for child in childs:
                if child not in f.items and child.state not in e:    #cheto mishe akhe currentNode age be e ezaf konim kar nemikone !!
                    f.push(child)
                    memory_counter += 1
                    print(child.state , "   added")
        round_counter += 1
        print( "# of visited nodes : ", round_counter , "\n")
    return "end of bfs and nothing"


def tree_search(problem, frontier):

    frontier.append(Node(problem.initial))
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None

def breadth_first_tree_search(problem):

    return tree_search(problem, FIFOQueue())


def depth_first_tree_search(problem):

    return tree_search(problem, Stack())



def breadth_first_search(problem):

    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = FIFOQueue()
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None


def best_first_graph_search(problem, f):

    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None


def uniform_cost_search(problem):

    return best_first_graph_search(problem, lambda node: node.path_cost)


def depth_limited_search(problem, limit=50):

    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None

    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):

    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result


greedy_best_first_graph_search = best_first_graph_search


def astar_search(problem, h=None):

    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


def recursive_best_first_search(problem, h=None):

    h = memoize(h or problem.h, 'h')

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node, 0   # (The second value is immaterial)
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, infinity
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            # Order by lowest f value
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = infinity
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result, best.f

    node = Node(problem.initial)
    node.f = h(node)
    result, bestf = RBFS(problem, node, infinity)
    return result




class NQueensProblem(Problem):


    def __init__(self, N):
        self.N = N
        self.initial = [None] * N

    def actions(self, state):

        if state[-1] is not None:
            return []  # All columns filled; no successors
        else:
            col = state.index(None)
            return [row for row in range(self.N)
                    if not self.conflicted(state, row, col)]

    def result(self, state, row):

        col = state.index(None)
        new = state[:]
        new[col] = row
        return new

    def conflicted(self, state, row, col):

        return any(self.conflict(row, col, state[c], c)
                   for c in range(col))

    def conflict(self, row1, col1, row2, col2):

        return (row1 == row2 or  # same row
                col1 == col2 or  # same column
                row1 - col1 == row2 - col2 or  # same \ diagonal
                row1 + col1 == row2 + col2)   # same / diagonal

    def goal_test(self, state):

        if state[-1] is None:
            return False
        return not any(self.conflicted(state, state[col], col)
                       for col in range(len(state)))













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
# maze = Maze(3,3,[[1,2],[1,3],[3,1]])
nqueen = NQueen(4)
# maze = Maze(3,3)
# bfs_graph(nqueen)
bfs_tree(nqueen)
# breadth_first_search(nqueen)
# print(dfs_graph(maze))
# print(dfs_tree(maze))
# dfs_recurisve(maze)
ucs_test(nqueen)
# astar_search(maze)
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