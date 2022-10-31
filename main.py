class node:
    pass

def missing_heuristic(state):
    dist = 0
    for i in range(len(state)):
        for j in range(len(state[i])):
            if (state[i][j] != len(state[i])*i + j + 1):
                
                dist = dist + 1
    return dist

def manhat_heuristic(state):
    dist = 0
    dimension = len(state[0])
    for i in range(len(state)):
        for j in range(len(state[i])):
            if (state[i][j] != dimension*i + j + 1):
                #f(x,y) = dim * y + x, in the given example dimension = 3
                dist = dist + 1
                #Adding g(n)
                goal_i = state[i][j] % dimension # f(x,y) % dim = x 
                goal_j = state[i][j] - goal_i * dimension
                dist = dist
                #Adding h(n)
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
#goal_state = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
test1 = [[2,3,4],[5,6,7],[8,9,1]]
test2 = [[16,2,3,4],[5,6,7,8],[9,10,11,12],[13,1,15,14]]
test3 = [[1,2,4],[3,0,6],[7,8,5]]

print(missing_heuristic(test1))
print(missing_heuristic(test2))
print(missing_heuristic(test3))
print(missing_heuristic(goal_state))

print(manhat_heuristic(test1))
print(manhat_heuristic(test2))
print(manhat_heuristic(test3))
print(manhat_heuristic(goal_state))

#n = input("Puzzle dimension: ")

#Manhattan distance testing
temp = []
for i in range(4):
    for j in range(4):
        some = i*4 + j + 1
        temp.append(some % 3)
    print(temp)
    temp = []