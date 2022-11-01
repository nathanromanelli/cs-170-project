import queue

class node:
    def __init__(self,state) -> None:
        self.state = state

class problem:
    def __init__(self, init_state, goal_state) -> None:
        self.init_state = init_state
        self.goal_state = goal_state
    
    def goal_test(state):
        pass

class eight_tile(problem):
    def __init__(self, init_state, goal_state) -> None:
        super().__init__(init_state, goal_state)

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
            if (state[i][j] != dimension*i + j + 1):
                # f(x,y) = dim * y + x, in the given example dimension = 3
                # Adding g(n)
                # dist = dist + 1
                
                # if we're using manhattan, h_n is True
                # we don't care about the distance of the empty tile
                if (state[i][j] != pow(dimension,2) and h_n) : 
                    goal_j = (state[i][j] - 1) % dimension # f(x,y) % dim = x 
                    goal_i = (state[i][j] - 1) // dimension # floor[f(x,y) / dim] = y

                    # Adding h(n) and not counting blank square, which im leaving as 9 for simplicity
                    dist = dist + abs(i - goal_i) + abs(j - goal_j)
    return dist

def make_que(node: node) -> queue.Queue:
    Que = queue.Queue()
    Que.put(node)
    return Que

def make_node(state) -> node:
    return node(state)

def search(problem,queing_function):
    nodes = make_que(make_node(problem.init_state))
    while not nodes.empty():
        node = nodes.get()
        if problem.goal_test(node.state()):
            return node
        nodes = queing_function(nodes,expand(node,problem.operator()))
    return "Failure"



#Main Code
goal_state = [[1,2,3],[4,5,6],[7,8,9]]
test1 = [[2,3,4],[5,6,7],[8,9,1]]
test2 = [[16,2,3,4],[5,6,7,8],[9,10,11,12],[13,1,15,14]]
test3 = [[1,2,4],[3,9,6],[7,8,5]]
test4 = [[8,2,3],[4,5,6],[7,1,9]]

print(h(test1,h_n=True))
print(h(test2,h_n=True))
print(h(test3,h_n=True))
print(h(test4,h_n=True))
print(h(goal_state,h_n=True))

#n = input("Puzzle dimension: ")

#Manhattan distance testing
temp = []
for i in range(4):
    for j in range(4):
        some = i*4 + j + 1
        temp.append((some-1)%4)
    print(temp)
    temp = []

que = queue.Queue()
print(que.empty())