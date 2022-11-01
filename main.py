import queue
import copy
from telnetlib import SE
import time

class node:
    #Class to keep the state of a node in the search queue

    def __init__(self,state,depth) -> None:
        self.state = copy.deepcopy(state)
        self.depth = depth

class problem:
    #Abstract class

    def __init__(self, init_state, goal_state) -> None:
        self.init_state = copy.deepcopy(init_state)
        self.goal_state = copy.deepcopy(goal_state)
    
class eight_tile(problem):
    #Class to keep initial and goal state for the problem space

    def __init__(self, init_state, goal_state) -> None:
        super().__init__(init_state, goal_state)

    #Function to check if the current state is the goal state
    def goal_test(self,state):
        dim = len(state)
        for i in range(dim):
            for j in range(dim):
                if state[i][j] != self.goal_state[i][j]:
                    return False
        return True
    
def h(state, h_n: bool):
    #If h_n = False we are using missing tiles
    dist = 0
    dimension = len(state)
    for i in range(dimension):
        for j in range(dimension):
            # f(x,y) = dim * y + x, in the given example dimension = 3
            if (state[i][j] != dimension*i + j + 1):
                # Adding g(n)
                dist = dist + 1
                
                # if we're using manhattan, h_n is True
                # we don't care about the distance of the empty tile
                if (state[i][j] != pow(dimension,2) and h_n) : 
                    goal_j = (state[i][j] - 1) % dimension # f(x,y) % dim = x 
                    goal_i = (state[i][j] - 1) // dimension # floor[f(x,y) / dim] = y

                    # Adding h(n) and not counting blank square, which im leaving as 9 for simplicity
                    dist = dist + abs(i - goal_i) + abs(j - goal_j)
    return dist

#Initializes a que with a single node
def make_que(node: node) -> queue.Queue:
    Que = queue.Queue()
    Que.put(node)
    return Que

#Creates a que node for the given state
def make_node(state,depth) -> node:
    return node(state,depth)

#I wasn't sure how to take an operator as an argument so I just made expand fit the square problem
def expand(node: node):
    state = copy.deepcopy(node.state)
    dim = len(state)
    children = []
    for i in range(dim):
        for j in range(dim):
            #Empty square is dim^2
            if state[i][j] == pow(dim,2):
                #We can move the top square down, swap 9 and i,j
                if i > 0:
                    state1 = copy.deepcopy(state)
                    state1[i][j],state1[i-1][j] = state1[i-1][j],state1[i][j]
                    children.append(state1)
                
                #We can move the bottom square up, swap 9 and i,j
                if i < dim - 1:
                    state2 = copy.deepcopy(state)
                    state2[i][j],state2[i+1][j] = state2[i+1][j],state2[i][j]
                    children.append(state2)

                #We can move the left square to the right, swap 9 and i,j
                if j > 0:
                    state3 = copy.deepcopy(state)
                    state3[i][j],state3[i][j-1] = state3[i][j-1],state3[i][j]
                    children.append(state3)

                #We can move the right square to the left, swap 9 and i,j
                if j < dim - 1:
                    state4 = copy.deepcopy(state)
                    state4[i][j],state4[i][j+1] = state4[i][j+1],state4[i][j]
                    children.append(state4)
                return children

#Que's the children passed in to the que
def queing_function(nodes: queue.Queue,children,depth) -> queue.Queue:
    for child in children:
        new_node = node(child,depth+1)
        nodes.put(new_node)
    return nodes

#Our main search; que's based on the passed in queing function
def search(problem,queing_function):
    nodes = make_que(make_node(problem.init_state,0))
    i = 0
    max_que = 0
    visited_nodes = {}
    while not nodes.empty():
        i = i + 1
        max_que = max(max_que,nodes.qsize())
        node = nodes.get()
        if problem.goal_test(node.state):
            print(f" Solution found \n Depth: {node.depth} \n Max queue size: {max_que} \n Nodes expanded: {i}")
            return node
        nodes = queing_function(nodes,expand(node),node.depth)
    return "Failure"


#Main Code
goal_state = [[1,2,3],[4,5,6],[7,8,9]]
test1 = [[2,3,4],[5,6,7],[8,9,1]]
test2 = [[16,2,3,4],[5,6,7,8],[9,10,11,12],[13,1,15,14]]
test3 = [[1,2,4],[3,9,6],[7,8,5]]
test4 = [[8,2,3],[4,5,6],[7,1,9]]
test5 = [[1,3,6],[5,9,2],[4,7,8]]


Problem = eight_tile(test5,goal_state)
time1 = time.time()
#search(Problem,queing_function)
time2 = time.time()
total_time = ((time2 - time1)*10//1)/10
print(f" Search took {total_time} seconds")