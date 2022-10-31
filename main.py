class node:
    pass

def missing_heuristic(state):
    dist = 0
    for i in range(len(state)):
        for j in range(len(state[i])):
            if (state[i][j] != len(state[i])*i + j):
                dist = dist + 1
    return dist

def manhat_heuristic(state):
    dist = 0
    for i in range(len(state)):
        for j in range(len(state[i])):
            if (state[i][j] != len(state[i])*i + j):
                dist = dist + 1
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

print(missing_heuristic(test1))
print(missing_heuristic(test2))
print(missing_heuristic(goal_state))

#n = input("Puzzle dimension: ")
