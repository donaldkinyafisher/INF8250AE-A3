import buffer as buffer 
import copy
import numpy as np
import jax
from jax import numpy as jnp
from model import DQN, DQNAgent
import flax

from model import (
    DQNTrainingArgs, DQNTrainState, DQN, DQNParameters, DQNAgent,
    select_action, compute_loss, update_target, initialize_agent_state,
    SimpleDQNAgent
)

def create_buffer(): 
     
    new_buffer = buffer.init_buffer(15, (4,))
    #print(empty_buffer)

    si = np.array([1, 1, 0, 0])

    key = jax.random.key(43)

    for i in range(10):

        #Update s0
        s0 = copy.deepcopy(si)
        si[0] = si[0]  + 1

        #Add transtion 
        transition = (s0, 1, -1, 1, si)

        #Update buffer
        new_buffer = buffer.add_transition(new_buffer, transition)

    return new_buffer

def test_DQN_init():

    # Example parameters
    state_shape = (4,) #[4, 84, 84, 3]  # e.g., (batch_size, height, width, channels)
    n_actions = 2

    rng = jax.random.key(0)


    dqn = DQN(n_actions=n_actions, state_shape=state_shape)
    state = jnp.ones((4, *state_shape[1:]))  # Example input with batch size of 32
    params = dqn.init(rng, state)

    agent = SimpleDQNAgent = DQNAgent(
    dqn=dqn,
    initialize_agent_state=initialize_agent_state,
    select_action=select_action,
    compute_loss=compute_loss,
    update_target=update_target,
)
    action = SimpleDQNAgent.select_action(agent.dqn, rng, params = params,state=state,epsilon=0.1 )
    
    return action

def test_loss():

    # Example parameters
    state_shape = (4,) #[4, 84, 84, 3]  # e.g., (batch_size, height, width, channels)
    n_actions = 2

    rng = jax.random.key(0)

    dqn = DQN(n_actions=n_actions, state_shape=state_shape)
    state = jnp.ones((4, *state_shape[1:]))  # Example input with batch size of 32
    params = dqn.init(rng, state)
    target_params = dqn.init(rng, state)

    agent = DQNAgent(
    dqn=dqn,
    initialize_agent_state=initialize_agent_state,
    select_action=select_action,
    compute_loss=compute_loss,
    update_target=update_target,
)

    new_buffer = create_buffer()

    transition = buffer.sample_transition(rng, new_buffer)

    agent_state = agent.initialize_agent_state(dqn, rng , DQNTrainingArgs)

    new_state = agent.update_target(agent_state)

    loss = agent.compute_loss(agent.dqn, params, target_params, transition, 0.95)

    return loss
    

if __name__ == '__main__':

    #1
    #test_buffer()
    action = test_DQN_init()
    loss = test_loss()
    

        