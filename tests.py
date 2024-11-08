import buffer as buffer 
import copy
import numpy as np
import jax
from jax import numpy as jnp
from model import DQN, DQNAgent
import flax


def test_buffer():

    new_buffer = buffer.init_buffer(15, (2,1))
    #print(empty_buffer)

    si = np.array([[1], [1]])

    key = jax.random.key(43)

    for i in range(10):

        #Update s0
        s0 = copy.deepcopy(si)
        si[0][0] = si[0][0]  + 1

        #Add transtion 
        transition = (s0, 1, -1, 1, si)

        #Update buffer
        new_buffer = buffer.add_transition(new_buffer, transition)

        sample_transition = buffer.sample_transition(key, new_buffer)
        print(f"i ={i}:{sample_transition}")



    return


def test_DQN_init():

    key1, key2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.uniform(key1, (4,4))

    model = DQN(2, 4)
    dummy_input = jnp.ones((1, 28, 28, 1))  # (N, H, W, C) format
    # initialize
    key = jax.random.key(0)
    key, subkey = jax.random.split(key, 2)
    parameters = model.init(subkey, dummy_input)

    print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, flax.core.unfreeze(parameters)))

    
    return model, parameters

def test_action():

    
    model, parameters = test_DQN_init()
    #model.apply(parameters, dummy_input)

    agent = DQNAgent.initialize_agent_state
    key = jax.random.key(1)

    action = agent.select_action(model, key, parameters, jnp.array([1,1,1,1]), 0.1 )




if __name__ == '__main__':

    #1
    #test_buffer()

    #test_DQN_init()

    test_action()
    

        