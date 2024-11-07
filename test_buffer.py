import buffer as buffer 
import copy
import numpy as np
import jax

if __name__ == '__main__':

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

        if i == 3 or i == 7 or i==9:
            sample_transition = buffer.sample_transition(key, new_buffer)
            print(f"i ={i}:{sample_transition}")

        