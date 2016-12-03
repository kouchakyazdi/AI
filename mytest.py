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

