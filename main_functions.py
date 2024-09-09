import argparse
import numpy as np
from environment import MountainCar, GridWorld

def raw_encode(state):
    vec = np.zeros(2)
    for k in state.keys():
        vec[k] = state[k]
    return vec

def tile_encode(state, states):
    vec= np.zeros(states)
    for k in state.keys():
        vec[k] = 1.00
    return vec
#state, reward, done
def randomaction(env,vec,weights,bias,space):
    a=np.random.randint(0, space)
    q = np.dot(vec, weights[:,a]) + bias
    return q, a

def nonrand(env,vec,weights,bias,space):
    qlist=[]
    for i in range(space):
        qlist.append(np.dot(vec, weights[:,i]) + bias)
    q=max(qlist)
    a=qlist.index(q)
    return q, a

def update(a,lr,qoldvec,qnewvec,qold,qnew,reward,gamma,bias,weights,rewardsum):
    weights[:,a]-=lr*(qold - (reward + (gamma*qnew)))*qoldvec
    bias-=lr*(qold-(reward+(gamma*qnew)))
    rewardsum+=reward
    return weights,bias,rewardsum

def main(args):
    # Command line inputs
    mode = args.mode
    weight_out = args.weight_out
    returns_out = args.returns_out
    episodes = args.episodes
    max_iterations = args.max_iterations
    epsilon = args.epsilon
    gamma = args.gamma
    learning_rate = args.learning_rate
    debug = args.debug
    # We will initialize the environment for you:
    if args.environment == 'mc':
        env = MountainCar(mode=mode, debug=debug)
        space=3
    else:
        env = GridWorld(mode=mode, debug=debug)
        space=4
    if mode=='raw':
        flag='r'
        states=2

    if mode!='raw' and args.environment=='mc':
        flag='g'
        states=2048
    if mode!='raw' and args.environment=='gw':
        states=12
        flag='g'
    res=q(space,states,env,weight_out,returns_out,episodes,max_iterations,epsilon,gamma,learning_rate,flag)
    return



def q(space,states,env,weight_out,returns_out,episodes,max_iterations,epsilon,gamma,lr,flag):
    weights = np.zeros((states,space))  # Our shape is |A| x |S|, if this helps.
    bias =0
    returns = []
    for episode in range(episodes):
        rewardsum=0
        state = env.reset()
        if flag=='g':
            vec=tile_encode(state, states)

        else:
            vec=raw_encode(state)
        for it in range(max_iterations):
            if np.random.rand()<=epsilon:
                q, a=randomaction(env,vec,weights,bias,space)
                newstate, reward, done=env.step(a)
            else:
                 q,a=nonrand(env,vec,weights,bias,space)
                 newstate, reward, done=env.step(a)
            if flag=='g':
                newvec=tile_encode(newstate,states)
            else:
                newvec=raw_encode(newstate)
            newq,newa=nonrand(env,newvec,weights,bias,space)
            weights,bias,rewardsum=update(a,lr,vec,newvec,q,newq,reward,gamma,bias,weights,rewardsum)
            state=newstate
            vec=newvec
            if done:
                break
        returns.append(rewardsum)


    weightz=open(weight_out,'w')
    weightz.write(str(bias))
    weightz.write('\n')
    for row in range(states):
       for col in range(space):
           w=weights[row,col]
           weightz.write(str(w))
           weightz.write('\n')


   # np.savetxt(weight_out,weights, delimiter='\n')
    np.savetxt(returns_out,returns,delimiter='\n')
