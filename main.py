class node:
    pass

def missing_heuristic(state):
    dist = 0
    for i in range(len(state)):
        for j in range(len(state[i])):
            if (state[i][j] != len(state[i])*i + j + 1):
                
                dist = dist + 1
    return dist

def manhat_heuristic(state, h_n):
    dist = 0
    dimension = len(state[0])
    for i in range(dimension):
        for j in range(dimension):
            if (state[i][j] != dimension*i + j + 1):
                # f(x,y) = dim * y + x, in the given example dimension = 3
                # Adding g(n)
                # dist = dist + 1
                
                # if we're using manhattan, h_n is True, else we just use missing squares
                # we don't care about the distance of the empty tile
                if (state[i][j] != pow(dimension,2) and h_n) : 
                    goal_j = (state[i][j] - 1) % dimension # f(x,y) % dim = x 
                    goal_i = (state[i][j] - 1) // dimension # floor[f(x,y) / dim] = y

                    # Adding h(n) and not counting blank square, which im leaving as 9 for simplicity
                    dist = dist + abs(i - goal_i) + abs(j - goal_j)
    return dist


def make_que(node):
    pass

def make_node(problem,initial_state):
    pass

def astar_search(problem,queing_function):
    nodes = make_que(make_node(problem,initial_state))
    while 1:
        if nodes.empty():
            return "failure"
        node = nodes.remove_front()
        if problem.goal_test(node.state()):
            return node
        nodes = queing_function(nodes,expand(node,problem.operator()))
        #end while


goal_state = [[1,2,3],[4,5,6],[7,8,9]]
test1 = [[2,3,4],[5,6,7],[8,9,1]]
test2 = [[16,2,3,4],[5,6,7,8],[9,10,11,12],[13,1,15,14]]
test3 = [[1,2,4],[3,9,6],[7,8,5]]
test4 = [[8,2,3],[4,5,6],[7,1,9]]

print(manhat_heuristic(test1,h_n=True))
print(manhat_heuristic(test2,h_n=True))
print(manhat_heuristic(test3,h_n=True))
print(manhat_heuristic(test4,h_n=True))
print(manhat_heuristic(goal_state,h_n=True))

#n = input("Puzzle dimension: ")

#Manhattan distance testing
temp = []
for i in range(4):
    for j in range(4):
        some = i*4 + j + 1
        temp.append((some-1)%4)
    print(temp)
    temp = []