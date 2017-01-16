# import queue as Q
#
# pq = Q.PriorityQueue()
#
# pq = Q.PriorityQueue()
# pq.put((10,'ten'))
# pq.put((1,'one'))
# pq.put((5,'five'))
# while not pq.empty():
#     print (pq.get())
#
# mylist = [{"a" , 5},{"b" , 6},{"c" , 7},{"d" , 8}]
#
# def f1 ():
#     for item in mylist:
#     # if item not in
#         print(item.values())
# print(f1())


# def isConflict(x1,x2,y1,y2):
#     if x1 == x2:
#         return True
#     if y1 == y2:
#         return True
#     if (x1-x2)/(y1-y2) == 1:
#         return True
#     if (x1-x2)/(y1-y2) == -1:
#         return True
#     return False
#
# def conflictCount(state):
#     actions = []
#     counter  = j = 0
#     i = 1
#     while j < len(state) - 1:
#         while i < len(state):
#             actions.append([j,i])
#             i += 1
#         j += 1
#         i = j + 1
#     return actions
#
# state = [1,3,0,1]
# print(conflictCount(state))

# pairs = [ [3,4],[2,5]]
# pairs.sort()
# print(pairs)

# class MyQueue:
#     def __init__(self):
#         self.items = []
#
#     def isEmpty(self):
#         return self.items == []
#
#     def enqueue(self, item):
#         self.items.insert(0,item)
#         self.items.sort()
#
#     def dequeue(self):
#         return self.items.pop()
#
#     def size(self):
#         return len(self.items)
#
#     def showItems(self):
#         print(self.items)
#
# q = MyQueue()
# q.enqueue([3,"a"])
# q.enqueue([4,"b"])
# q.enqueue([2,"c"])
# print(q.dequeue())

# import random
#
#
# def myrRandom():
#     randomState = []
#     i = 0
#     for i in range(0, 4, 1):
#         rand  = (int)((random.random()*10 ) % self.n)
#         randomState.append(rand)
#     print(randomState)
#    # print/(rand)
# myrRandom()
#
# neighbours = [1,2,3,4,5]

# randIndex = (int)(random.random() * 10) % len(neighbours)
# print(randIndex)

# i = 10
# while True:
#     while i > 0:
#         i -= 1
#     break
# #
# import sys , random
#
# list = [1,2,3,4,5,6]
# arr = []
# arr = random.shuffle(list)
# # print(random.choice(list))
# # arr = []
# # for item in list:
# #     arr.append(random.choice(list))
# print(arr , listr)
# # for each in range(sys.maxsize):
# #     print(each)
#
# class MyQueue:
#     def __init__(self):
#         self.items = []
#
#     def isEmpty(self):
#         return self.items == []
#
#     def enqueue(self, item):
#         self.items.insert(0,item)
#
#     def enqueueSortAsc(self, item):
#         self.items.insert(0,item)
#         self.items.sort(reverse=True)
#
#     def dequeue(self):
#         return self.items.pop()
#
#     def size(self):
#         return len(self.items)
#
#     def sortAsc(self):
#         return self.items.sort(reverse=True)
#     def showItems(self):
#         for item in self.items:
#             print(item)
#
#     # add  to items in pairs : [node , g]
# q = MyQueue()
# q.enqueue(["node1",1])
# q.enqueue(["node4",4])
# q.enqueue(["node2",2])
# q.enqueue(["node3",3])
# q.enqueue(["node5",5])
# q.sortAsc()
# print(q.items)

# import  random
# valid_values = [1,2,3,4,5,6,7,8,0]
#
# matrix = {}
# for row in range(0,3,1):
#     for col in range(0,3,1):
#         tmp = random.choice(valid_values)
#         valid_values.remove(tmp)
#         matrix[row, col] = tmp
#
# for row in range(0,3,1):
#     tmplist = []
#     for col in range(0,3,1):
#         # print("row:", row , "col:",col,"value:" , matrix[row,col])
#         tmplist.append(matrix[row ,col])
#     print(tmplist)
# print(matrix)

def makePositive(x):
    if x < 0:
        return -x
    return x

def getCoordinate(matrix , value):

    coordinate = list(matrix.keys())[list(matrix.values()).index(value)]
    return coordinate

def difference (x,y):
    return  makePositive(x - y)

def coordinateDifference (c1 , c2):
    return makePositive(c1[0] - c2[0]) + makePositive(c1[1] - c2[1])

initialStatePuzzle = {(0,0):1 , (0,1):6 , (0,2):2 , (1,0):5 , (1,1):3 , (1,2):0 , (2,0):4 , (2,1):7 , (2,2):8}
goalStatePuzzle = {(0,0):1 , (0,1):2 , (0,2):3 , (1,0):4 , (1,1):5 , (1,2):6 , (2,0):7 , (2,1):8 , (2,2):0}


correct = getCoordinate(goalStatePuzzle,7)
real = getCoordinate(initialStatePuzzle,7)

print(coordinateDifference(correct , real))

# print("asdads",goalStatePuzzle.get((0,2)))
# goalStatePuzzle[0,0] = 0
# print(goalStatePuzzle)

# def getCoordinateCell( dict, cellValue):
#     cellIndex = list(puzzle.keys())[list(puzzle.values()).index(cellValue)]
#     return cellIndex
# print(matrix.items())
# matrix[1,1] = 1
# matrix[0,0] = None

# print(matrix[2,2])
# valid_values.remove(1)
# print(valid_values)
# lsit = ['a' , 'b' , 'c']
# list.remove('a')
# print(list)







