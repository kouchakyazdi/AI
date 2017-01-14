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

















