import numpy as np
import cvxpy as cp
import json
import os
import random

DAMAGE_ARROW = 1    #scaled down
DAMAGE_BLADE = 2    #scaled down
FACTOR_HEALTH_MM = 25

W={}
C={}
E={}
N={}
S={}
MAP_STATE_TO_ACTIONS = {}
MAP_ACTIONS_TO_STATE = {}
MAP_PROBAB_TO_STATE = {}
MAP_ACTIONS_TO_NAME = {}
PROBAB_EAST = {}
PROBAB_WEST = {}
PROBAB_NORTH = {}
PROBAB_SOUTH = {}
PROBAB_CENTER = {}

MAP_ACTIONS_TO_NAME["U"] = "UP"
MAP_ACTIONS_TO_NAME["D"] = "DOWN"
MAP_ACTIONS_TO_NAME["L"] = "LEFT"
MAP_ACTIONS_TO_NAME["R"] = "RIGHT"
MAP_ACTIONS_TO_NAME["S"] = "STAY"
MAP_ACTIONS_TO_NAME["F"] = "SHOOT"
MAP_ACTIONS_TO_NAME["H"] = "HIT"
MAP_ACTIONS_TO_NAME["G"] = "GATHER"
MAP_ACTIONS_TO_NAME["C"] = "CRAFT"
MAP_ACTIONS_TO_NAME["X"] = "NONE"

state_MM = []
state_MM.append("D")
state_MM.append("R")
pos_IJ = []
pos_IJ.append("W")
pos_IJ.append("N")
pos_IJ.append("E")
pos_IJ.append("S")
pos_IJ.append("C")
POS_TO_INDEX = {}
POS_TO_INDEX["W"] = 0
POS_TO_INDEX["N"] = 1
POS_TO_INDEX["E"] = 2
POS_TO_INDEX["S"] = 3
POS_TO_INDEX["C"] = 4
MAP_MM_INDEX = {}
MAP_MM_INDEX["D"] = 0
MAP_MM_INDEX["R"] = 1


MAX_ARROWS = 3
MAX_MM_HEALTH = 4
MAX_MATERIALS = 2


PROB_MOVE_SUCCESSFUL = 0.85 #rest goes to EAST with prob = 0.15


PROB_ARROW_HIT_CENTER = 0.5
PROB_BLADE_HIT_CENTER = 0.1


CRAFT_ONE = 0.5
CRAFT_TWO = 0.35
CRAFT_THREE = 0.15


GATHER_MATERIAL = 0.75


PROB_ARROW_HIT_EAST = 0.9
PROB_BLADE_HIT_EAST = 0.2


#actions from west and east have 100% chance of occuring
PROB_ARROW_HIT_WEST = 0.25
PROB_BLADE_HIT_WEST = 0.0


MM_READY_STATE = 0.2
MM_DORMANT_STATE = 0.8
MM_ATTACK = 0.5     #affect only if IJ on center and east square => IJ drops arrows + MM health regain + CANCEL INDIANA action + REWARD -40
MM_REGAIN_HEALTH = 1 #scaled down
NEGATIVE_REWARD_MM = 40


STAY_STEP_COST = -10 #TEAM NUMBER = 61 => 61%3 => 1 => Y = arr[1] => Y = 1 ; 0 for task2.2, -10 for task 1
GENERAL_STEP_COST = -10
GAMMA = 0.999 #0.999 for task 1 and 0.25 for task 2.3
DELTA = 0.001


START_STATE_ONE = ['C',2,3,1,4]
START_STATE_TWO = ['C',2,3,1,4]

WEST_ACTIONS = ["S","R","F"]
W["R"] = "C"
W["S"] = "W"
W["F"] = "W"
PROBAB_WEST["R"] = 1.0
PROBAB_WEST["S"] = 1.0
PROBAB_WEST["F"] = 0.25


EAST_ACTIONS = ["L","S","F","H"]
E["L"] = "C"   #task 2.1 for W + C for task 1
E["S"] = "E"
E["F"] = "E"
E["H"] = "E"
PROBAB_EAST["L"] = 1.0 
PROBAB_EAST["S"] = 1.0
PROBAB_EAST["F"] = 0.9
PROBAB_EAST["H"] = 0.2


SOUTH_ACTIONS = ["U","S","G"]
S["U"] = "C"
S["S"] = "S"
S["G"] = "S"
PROBAB_SOUTH["U"] = 0.85
PROBAB_SOUTH["S"] = 0.85
PROBAB_SOUTH["G"] = 0.75


NORTH_ACTIONS = ["D","S","C"]
N["D"] = "C"
N["S"] = "N"
N["C"] = "N"
PROBAB_NORTH["D"] = 0.85 
PROBAB_NORTH["S"] = 0.85
PROBAB_NORTH["C"] = 1.0 

CENTER_ACTIONS = ["U","S","L","R","D","F","H"]
C["U"] = "N"
C["D"] = "S"
C["L"] = "W"
C["R"] = "E"
C["S"] = "C"
C["F"] = "C"
C["H"] = "C"
PROBAB_CENTER["U"] = 0.85
PROBAB_CENTER["D"] = 0.85
PROBAB_CENTER["L"] = 0.85
PROBAB_CENTER["R"] = 0.85
PROBAB_CENTER["S"] = 0.85
PROBAB_CENTER["F"] = 0.5
PROBAB_CENTER["H"] = 0.1

MAP_STATE_TO_ACTIONS['W'] = WEST_ACTIONS
MAP_STATE_TO_ACTIONS['N'] = NORTH_ACTIONS
MAP_STATE_TO_ACTIONS['E'] = EAST_ACTIONS
MAP_STATE_TO_ACTIONS['S'] = SOUTH_ACTIONS
MAP_STATE_TO_ACTIONS['C'] = CENTER_ACTIONS

MAP_ACTIONS_TO_STATE['W'] = W
MAP_ACTIONS_TO_STATE['N'] = N
MAP_ACTIONS_TO_STATE['E'] = E
MAP_ACTIONS_TO_STATE['S'] = S
MAP_ACTIONS_TO_STATE['C'] = C 

MAP_PROBAB_TO_STATE["W"] = PROBAB_WEST
MAP_PROBAB_TO_STATE["N"] = PROBAB_NORTH
MAP_PROBAB_TO_STATE["E"] = PROBAB_EAST
MAP_PROBAB_TO_STATE["S"] = PROBAB_SOUTH
MAP_PROBAB_TO_STATE["C"] = PROBAB_CENTER

MAP_ACTION_TO_VARIABLE = {}
MAP_ACTION_TO_VARIABLE["UP"] = "U"
MAP_ACTION_TO_VARIABLE["DOWN"] = "D"
MAP_ACTION_TO_VARIABLE["LEFT"] = "L"
MAP_ACTION_TO_VARIABLE["RIGHT"] = "R"
MAP_ACTION_TO_VARIABLE["SHOOT"] = "F"
MAP_ACTION_TO_VARIABLE["STAY"] = "S"
MAP_ACTION_TO_VARIABLE["HIT"] = "H"
MAP_ACTION_TO_VARIABLE["GATHER"] = "G"
MAP_ACTION_TO_VARIABLE["CRAFT"] = "C"

NUM_OF_STATES = 5*3*4*2*5 #positionIJ * material * arrow * stateMM * healthMM
NUM_OF_COLUMNS = 0


def generate_hash(state):
    if state[0] != "C" and state[0]!="W" and state[0]!="E" and state[0]!="S" and state[0]!="N":
        return state[0]*120 + state[1]*40 + state[2]*10 + state[3]*5 + state[4]
    else:
        pos = POS_TO_INDEX[state[0]]
        return pos*120 + state[1]*40 + state[2]*10 + state[3]*5 + state[4]


def move(localPos,localMat,localArr,localState,localHealth,action):
    #char,int,int,int,int,char
    # print(localPos)
    # print(action)
    if action == "X":
        return [],0
    step = MAP_ACTIONS_TO_STATE[localPos]
    probab = MAP_PROBAB_TO_STATE[localPos]
    finPos = step[action]
    finPos = POS_TO_INDEX[finPos]
    finMat = localMat
    finArr = localArr
    finHea = localHealth
    finSta = localState
    scale = 1.0
    sucPro = probab[action]
    states = []
    avgR = GENERAL_STEP_COST

    if finSta==1 and (localPos == "C" or localPos=="E"):
        #C,E + Ready
        scale = MM_ATTACK
        curState = [POS_TO_INDEX[localPos],localMat,0,1-finSta,min(MAX_MM_HEALTH,finHea+1)]
        curPro = scale
        avgR += (scale*(-40))
        states.append([curState,curPro])
        if action == "C":
            if finMat> 0:
                finMat-=1
                finArr = min(MAX_ARROWS,finArr+1)
                curState = [finPos,finMat,finArr,finSta,finHea]
                curPro = CRAFT_ONE * (1.0 - scale)
                states.append([curState,curPro])

                finArr = min(MAX_ARROWS,finArr+1)
                curState = [finPos,finMat,finArr,finSta,finHea]
                curPro = CRAFT_TWO * (1.0 - scale)
                states.append([curState,curPro])

                finArr = min(MAX_ARROWS,finArr+1)
                curState = [finPos,finMat,finArr,finSta,finHea]
                curPro = CRAFT_THREE * (1.0 - scale)
                states.append([curState,curPro])
        elif action == "G":
            curState = [finPos,finMat,finArr,finSta,finHea]
            curPro = (1.0-sucPro) * (1.0 - scale)
            states.append([curState,curPro])

            finMat = min(MAX_MATERIALS,finMat+1)
            curState = [finPos,finMat,finArr,finSta,finHea]
            curPro = (sucPro) * (1.0 - scale)
            states.append([curState,curPro])
        elif action == "H":
            curState = [finPos,finMat,finArr,finSta,finHea]
            curPro = (1.0-sucPro) * (1.0 - scale)
            states.append([curState,curPro])

            finHea = max(0,finHea-DAMAGE_BLADE)
            curState = [finPos,finMat,finArr,finSta,finHea]
            curPro = (sucPro) * (1.0 - scale)
            states.append([curState,curPro])
        elif action == "F":
            if finArr > 0:
                finArr -= 1
                curState = [finPos,finMat,finArr,finSta,finHea]
                curPro = (1.0-sucPro) * (1.0 - scale)
                states.append([curState,curPro])

                finHea = max(0,finHea-DAMAGE_ARROW)
                curState = [finPos,finMat,finArr,finSta,finHea]
                curPro = (sucPro) * (1.0 - scale)
                states.append([curState,curPro])
        else:
            curState = [finPos,finMat,finArr,finSta,finHea]
            curPro = (sucPro) * (1.0 - scale)
            states.append([curState,curPro])
            
            curState = [POS_TO_INDEX["E"],finMat,finArr,finSta,finHea]
            curPro = (1.0-sucPro) * (1.0 - scale)
            states.append([curState,curPro])    

    else:
        if finSta == 0:
            scale = MM_READY_STATE
        else:
            scale = MM_ATTACK
        if action == "C":
            if finMat> 0:
                finMat-=1
                finArr = min(MAX_ARROWS,finArr+1)
                curState = [finPos,finMat,finArr,finSta,finHea]
                curPro = CRAFT_ONE * (1.0 - scale)
                states.append([curState,curPro])
                curState = [finPos,finMat,finArr,1-finSta,finHea]
                curPro = CRAFT_ONE * scale
                states.append([curState,curPro])

                finArr = min(MAX_ARROWS,finArr+1)
                curState = [finPos,finMat,finArr,finSta,finHea]
                curPro = CRAFT_TWO * (1.0 - scale)
                states.append([curState,curPro])
                curState = [finPos,finMat,finArr,1-finSta,finHea]
                curPro = CRAFT_TWO * scale
                states.append([curState,curPro])

                finArr = min(MAX_ARROWS,finArr+1)
                curState = [finPos,finMat,finArr,finSta,finHea]
                curPro = CRAFT_THREE * (1.0 - scale)
                states.append([curState,curPro])
                curState = [finPos,finMat,finArr,1-finSta,finHea]
                curPro = CRAFT_THREE * scale
                states.append([curState,curPro])
            # else:
            #     finPos = POS_TO_INDEX[localPos]
            #     curState = [finPos,finMat,finArr,finSta,finHea]
            #     states.append([curState,1.0])

        elif action == "G":
            curState = [finPos,finMat,finArr,finSta,finHea]
            curPro = (1.0-sucPro) * (1.0 - scale)
            states.append([curState,curPro])
            curState = [finPos,finMat,finArr,1-finSta,finHea]
            curPro = (1.0-sucPro) * (scale)
            states.append([curState,curPro])

            finMat = min(MAX_MATERIALS,finMat+1)
            curState = [finPos,finMat,finArr,finSta,finHea]
            curPro = (sucPro) * (1.0 - scale)
            states.append([curState,curPro])
            curState = [finPos,finMat,finArr,1-finSta,finHea]
            curPro = (sucPro) * (scale)
            states.append([curState,curPro])

        elif action == "H":
            curState = [finPos,finMat,finArr,finSta,finHea]
            curPro = (1.0-sucPro) * (1.0 - scale)
            states.append([curState,curPro])
            curState = [finPos,finMat,finArr,1-finSta,finHea]
            curPro = (1.0-sucPro) * (scale)
            states.append([curState,curPro])

            finHea = max(0,finHea-DAMAGE_BLADE)
            curState = [finPos,finMat,finArr,finSta,finHea]
            curPro = (sucPro) * (1.0 - scale)
            states.append([curState,curPro])
            curState = [finPos,finMat,finArr,1-finSta,finHea]
            curPro = (sucPro) * (scale)
            states.append([curState,curPro])

        elif action == "F":
            if finArr > 0:
                finArr -= 1
                curState = [finPos,finMat,finArr,finSta,finHea]
                curPro = (1.0-sucPro) * (1.0 - scale)
                states.append([curState,curPro])
                curState = [finPos,finMat,finArr,1-finSta,finHea]
                curPro = (1.0-sucPro) * (scale)
                states.append([curState,curPro])

                finHea = max(0,finHea-DAMAGE_ARROW)
                curState = [finPos,finMat,finArr,finSta,finHea]
                curPro = (sucPro) * (1.0 - scale)
                states.append([curState,curPro])
                curState = [finPos,finMat,finArr,1-finSta,finHea]
                curPro = (sucPro) * (scale)
                states.append([curState,curPro])
        else:
            curState = [finPos,finMat,finArr,finSta,finHea]
            curPro = (sucPro) * (1.0 - scale)
            states.append([curState,curPro])
            curState = [finPos,finMat,finArr,1-finSta,finHea]
            curPro = (sucPro) * (scale)
            states.append([curState,curPro])
            
            curState = [POS_TO_INDEX["E"],finMat,finArr,finSta,finHea]
            curPro = (1.0-sucPro) * (1.0 - scale)
            states.append([curState,curPro])
            curState = [POS_TO_INDEX["E"],finMat,finArr,1-finSta,finHea]
            curPro = (1.0-sucPro) * (scale)
            states.append([curState,curPro])
    
        
    return states,avgR

def get_state(i):
    localPos = pos_IJ[ i//120 ]
    i = i%120
    localMat = i//40
    i=i%40
    localArr = i//10
    i=i%10
    localSta = i//5
    i=i%5
    localHea = i
    #char,int,int,int,int
    return localPos,localMat,localArr,localSta,localHea


def checkActions(localPos,localMat,localArr,localSta,localHea,actions):
    temp_actions = []
    #C,F,HIT,GATHER,L,R,U,S,D
    for action in actions:
        if action == "C":
            if localMat > 0:
                temp_actions.append(action)
        elif action == "F":
            if localArr > 0:
                temp_actions.append(action)
        else:
            temp_actions.append(action)
    return temp_actions

def get_dimension():
    dim = 0
    for i in range(NUM_OF_STATES):
        #char,int,int,int,int
        localPos,localMat,localArr,localSta,localHea = get_state(i)
        actions = MAP_STATE_TO_ACTIONS[localPos]
        if localHea == 0:
            dim += 1
        else:
            dim += len(checkActions(localPos,localMat,localArr,localSta,localHea,actions))
    return dim


# 5 * 3 * 4 * 2 * 5
# 3 * 4 * 2 * 5 => 120
# 4 * 2 * 5 => 40
# 2 * 5 => 10
# 5 => 5
def generate_AR():
    # print(NUM_OF_COLUMNS)
    A = np.zeros((NUM_OF_STATES, NUM_OF_COLUMNS), dtype=np.float64)
    R = np.full((1, NUM_OF_COLUMNS), GENERAL_STEP_COST)
    idx = 0
    for i in range(NUM_OF_STATES):
        localPos,localMat,localArr,localSta,localHea = get_state(i)
        #char,int,int,int,int
        actions2 = []
        if localHea != 0:
            actions = MAP_STATE_TO_ACTIONS[localPos]
            actions2 = checkActions(localPos,localMat,localArr,localSta,localHea,actions)
        else:
            actions2 = ["X"]
        for action in actions2:
            A[i][idx] = 1
            localPos,localMat,localArr,localSta,localHea = get_state(i)
            next_states,avgR = move(localPos,localMat,localArr,localSta,localHea,action)
            R[0][idx] = avgR #avgR
            for next_state in next_states:
                A[generate_hash(next_state[0])][idx] -= next_state[1]
            idx += 1
    return A,R

def generate_alpha():
    alpha = np.zeros((NUM_OF_STATES, 1))
    s = generate_hash(START_STATE_ONE)
    alpha[s][0] = 1
    return alpha

def slve():
    x = cp.Variable((NUM_OF_COLUMNS, 1), 'x')
    A,R = generate_AR()
    alpha = generate_alpha()
    constraints = [
        cp.matmul(A, x) == alpha,
        x >= 0
    ]
    objective = cp.Maximize(cp.matmul(R, x))
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()
    arr = list(x.value)
    l = [ float(val) for val in arr]
    return l,solution

NUM_OF_COLUMNS = get_dimension()
# print(NUM_OF_COLUMNS)
a,r = generate_AR()
alpha = generate_alpha()
alp = [float(val) for val in alpha]
x,objective = slve()
# print(x)
x = np.array(x)
# for i in range(600):
#     print(a[i][1568])
# policy_sel = []
# for i in range(5): # pos{{W = 0} {N = 1} {E = 2} {S = 3} {c = 4}}
#     temp_mat1 = []
#     for j in range(3): # materials
#         temp_arrow1 = []
#         for k in range(4): # arrows
#             temp_state1 = []
#             for l in range(2): # state of MM {{D = 0} {R = 1}}
#                 temp_Health1 = []
#                 for m in range(5): # heath {{%{health}/25}}
#                     if(m>=1):
#                         temp_Health1.append("")
#                     else:
#                         temp_Health1.append("NONE")
#                 temp_state1.append(temp_Health1)
#             temp_arrow1.append(temp_state1)
#         temp_mat1.append(temp_arrow1)
#     policy_sel.append(temp_mat1)

policy = []
idx = 0
for i in range(NUM_OF_STATES):
    localPos,localMat,localArr,localSta,localHea = get_state(i)
    actions2 = []
    if localHea != 0:
        actions = MAP_STATE_TO_ACTIONS[localPos]
        actions2 = checkActions(localPos,localMat,localArr,localSta,localHea,actions)
    else:
        actions2 = ["X"]
    # if(localPos == "C") and localHea!=0:
        # print(actions2)
    act_idx = 0
    maxi = x[idx+act_idx]
    for j in range(1,len(actions2)):
        if x[idx+j] > maxi:
            maxi = x[idx+j]
            act_idx = j
    idx = idx+len(actions2)
    best_action = actions2[act_idx]
    local = [[localPos,localMat,localArr,state_MM[localSta],localHea*25],MAP_ACTIONS_TO_NAME[best_action]]
    policy.append(local)

solDict = {}
solDict["a"] = a.tolist()
r = [float(val) for val in np.transpose(r)]
solDict["r"] = r
solDict["alpha"] = alp
solDict["x"] = x.tolist()
solDict["policy"] = policy
solDict["objective"] = float(objective)
os.makedirs('outputs', exist_ok=True)
path = "outputs/part_3_output.json"
# path2 = "./output.json"
json_object = json.dumps(solDict, indent=4)
with open(path, 'w+') as f:
    f.write(json_object)
# print(objective)


# print(MAP_ACTIONS_TO_STATE)
# print(MAP_STATE_TO_ACTIONS)
# print(MAP_PROBAB_TO_STATE)
# exit()
# START_STATE_ONE = ['C',2,3,'R',100]
# START_STATE_ONE[4] = START_STATE_ONE[4] // FACTOR_HEALTH_MM

# strt_ctr = 0
# while(START_STATE_ONE[4] != 0) and strt_ctr != 100:
#     strt_ctr +=1
#     localPos = START_STATE_ONE[0] #character
#     matWithIJ = START_STATE_ONE[1] #int
#     arrowWithIJ = START_STATE_ONE[2] #int
#     stateMM = START_STATE_ONE[3] #character
#     healthMM = START_STATE_ONE[4] #with scale
#     # print(POS_TO_INDEX[localPos])
#     # print(matWithIJ)
#     # print(arrowWithIJ)
#     # print(MAP_MM_INDEX[stateMM])
#     # print(healthMM//FACTOR_HEALTH_MM)
#     action = policy_sel[POS_TO_INDEX[localPos]][matWithIJ][arrowWithIJ][MAP_MM_INDEX[stateMM]][healthMM]
#     print(START_STATE_ONE)
#     print("Take Action: "+action)
#     if action == "NONE":
#         break
#     action = MAP_ACTION_TO_VARIABLE[action]
#     actions = MAP_STATE_TO_ACTIONS[localPos]
#     move = MAP_ACTIONS_TO_STATE[localPos]    
#     probabArr = MAP_PROBAB_TO_STATE[localPos]
#     if stateMM == "D":
#         probab = random.uniform(0.0,1.0)
#         if probab<=0.2:
#             START_STATE_ONE[3] = "R"
#         else:
#             START_STATE_ONE[3] = "D"
#         if action == "U" or action == "D" or action =="L" or action == "R" or action == "S":
#             probab = random.uniform(0.0,1.0)
#             if probab <= probabArr[action]:
#                 START_STATE_ONE[0] = move[action]
#             else:
#                 START_STATE_ONE[0] = "E"
#         elif action == "C":
#             if matWithIJ>0:
#                 START_STATE_ONE[1] -= 1
#                 probab = random.uniform(0.0,1.0)
#                 newArr = arrowWithIJ
#                 if probab <= CRAFT_ONE:
#                     newArr = min(MAX_ARROWS,1+newArr)
#                 elif probab <= CRAFT_ONE+CRAFT_TWO:
#                     newArr = min(MAX_ARROWS,2+newArr)
#                 else:
#                     newArr = min(MAX_ARROWS,3+newArr)
#                 START_STATE_ONE[2] = newArr
#         elif action == "G":
#             probab = random.uniform(0.0,1.0)
#             if probab <= probabArr[action]:
#                 START_STATE_ONE[1] = min(MAX_MATERIALS,1+START_STATE_ONE[1])
#         elif action == "F":
#             probab = random.uniform(0.0,1.0)
#             if probab <= probabArr[action]:
#                 if arrowWithIJ > 0:
#                     START_STATE_ONE[2] -= 1
#                     START_STATE_ONE[4] -= 1
#             else:
#                 if arrowWithIJ > 0:
#                     START_STATE_ONE[2] -= 1
#         elif action == "H":
#             probab = random.uniform(0.0,1.0)
#             if probab <= probabArr[action]:
#                     START_STATE_ONE[4] = max(0,START_STATE_ONE[4] - 2)
#     else:
#         probab = random.uniform(0.0,1.0)
#         print(probab)
#         if probab <= MM_ATTACK and (START_STATE_ONE[0] == "C" or START_STATE_ONE[0] == "E"):
#             START_STATE_ONE[2] = 0
#             START_STATE_ONE[3] = "D"
#             START_STATE_ONE[4] = min(4,START_STATE_ONE[4]+1)
#             # exit()
#         else: 
#             factor = 1.0
#             if probab <= MM_ATTACK:
#                 START_STATE_ONE[3] = "D"
#                 START_STATE_ONE[4] = min(4,START_STATE_ONE[4]+1)
#                 factor = 1.0
#             else:
#                 factor = 1.0
            
#             if action == "U" or action == "D" or action =="L" or action == "R" or action == "S":
#                 probab = random.uniform(0.0,1.0)
#                 if probab <= probabArr[action]*factor:
#                     START_STATE_ONE[0] = move[action]
#                 else:
#                     START_STATE_ONE[0] = "E"
#             elif action == "C":
#                 if matWithIJ>0:
#                     START_STATE_ONE[1] -= 1
#                     probab = random.uniform(0.0,1.0)
#                     newArr = arrowWithIJ
#                     if probab <= CRAFT_ONE*factor:
#                         newArr = min(MAX_ARROWS,1+newArr)
#                     elif probab <= (CRAFT_ONE+CRAFT_TWO)*factor:
#                         newArr = min(MAX_ARROWS,2+newArr)
#                     elif probab <= (CRAFT_TWO+CRAFT_THREE+CRAFT_ONE)*factor:
#                         newArr = min(MAX_ARROWS,3+newArr)
#                     START_STATE_ONE[2] = newArr
#             elif action == "G":
#                 probab = random.uniform(0.0,1.0)
#                 if probab <= probabArr[action]*factor:
#                     START_STATE_ONE[1] = min(MAX_MATERIALS,1+START_STATE_ONE[1])
#             elif action == "F":
#                 probab = random.uniform(0.0,1.0)
#                 if probab <= probabArr[action]*factor:
#                     if arrowWithIJ > 0:
#                         START_STATE_ONE[2] -= 1
#                         START_STATE_ONE[4] -= 1
#                 else:
#                     if arrowWithIJ > 0:
#                         START_STATE_ONE[2] -= 1
#             elif action == "H":
#                 probab = random.uniform(0.0,1.0)
#                 if probab <= probabArr[action]*factor:
#                         START_STATE_ONE[4] = max(0,START_STATE_ONE[4] - 2)

# print(START_STATE_ONE)
# if(strt_ctr == 100):
#     print("ended due to limitation")