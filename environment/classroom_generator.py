import random
import numpy as np
import pandas as pd

class Engine:
    def __init__(self):
        # Define 'universe' of positions and actions
        self.legal_actions = ['left', 'right', 'up', 'down']
        self.x_range = [-5,5]
        self.y_range = [-5,5]
        self.Classrooms = {}

        # Initialise complete classroom setup
        self.classroom_init = pd.DataFrame()
        for n1,x in enumerate(range(self.x_range[0], self.x_range[1])):
            for n2,y in enumerate(range(self.y_range[0], self.y_range[1])):
                state = str(x) + "_" + str(y)
                state_x = x
                state_y = y
                state_id = "NONE"
                prob = 1
                reward = 0
                terminal = False
                df_row = pd.DataFrame({'state':state, 'state_x':state_x, 'state_y':state_y, 'state_ids':state_id, 'prob':prob, 'reward':reward, 'terminal':terminal}, 
                                                                              index = [(n1*(self.y_range[1]-self.y_range[0])) + n2])
                self.classroom_init = pd.concat([self.classroom_init,df_row])

    def Classroom_A(self):
        # Define Class A
        classroom_id = 'A'
        #------------------------------------------------------------------------------
        # Adding example classroom to initialise environment
        ## 0. Create a copy of the initialized environment
        ## 1. Add state id name to x,y position
        ## 2. Add probability of command being followed to x,y position
        #----------------------------
        ## Define the x,y position of each state (manually and fixed for now)
        self.start_state_list = [[4,1],[3,1],[2,1],[1,1],[1,2],[1,3],[2,3],[3,2]]
        #self.start_state_list = [[1,1]]
        x_list = [4,3,2,1,1,1,2,3,3,4,4]
        y_list = [1,1,1,1,2,3,3,3,2,3,2]
        terminal_states = ['4_2', '4_3']
        rewards = [0,0,0,0,0,0,0,0,0,1,-1]
        ## Define the probability of each student following commands (manually and fixed for now)
        ### NOTE: 'Trap' states are defined by a 0 probability and are those for which the paper cannot move from (e.g. bins)
        state_probs = [0.4, 0.6, 0.5, 0.8, 0.6, 0.9, 0.9, 1, 0.2, 0, 0]
        ## Define state ids for our reference, not to be used by agent directly
        state_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'M', 'recycling', 'trash']
        #----------------------------
        Classroom_A = self.classroom_init.copy()
        for item in range(0,len(state_ids)):
            state_id = state_ids[item]
            state_prob = state_probs[item]
            state_reward = rewards[item]
            x = x_list[item]
            y = y_list[item]
            if str(x)+'_'+str(y) in terminal_states:
                terminal = True
            else:
                terminal = False

            Classroom_A['state_ids'] = np.where((Classroom_A['state_x']==x)&(Classroom_A['state_y']==y),
                                                        state_id,
                                                        Classroom_A['state_ids'])
            Classroom_A['prob'] = np.where((Classroom_A['state_x']==x)&(Classroom_A['state_y']==y),
                                                        state_prob,
                                                        Classroom_A['prob'])
            Classroom_A['reward'] = np.where((Classroom_A['state_x']==x)&(Classroom_A['state_y']==y),
                                                        state_reward,
                                                        Classroom_A['reward'])
            Classroom_A['terminal'] = np.where((Classroom_A['state_x']==x)&(Classroom_A['state_y']==y),
                                                        terminal,
                                                        Classroom_A['terminal'])
        # Add this classroom to Classroom dictionary
        self.Classrooms['Classroom_'+str(classroom_id)] = Classroom_A     
        start_pos = random.choice(self.start_state_list)   
        return start_pos

    def reset(self):
        # Start episode position
        env_reset_obs = random.choice(self.start_state_list)
        return env_reset_obs 

    
    @staticmethod
    def action_outcome(state_x,state_y,action,states_df):
        # Produces x and y directional vectors for the action given the current x,y position
        # If this produces an outcome where there is no state (i.e. an empty slot) then the position won't change
        # Define basic action outcomes
        if action == 'left':
            u = -1
            v = 0
        elif action == 'right':
            u = 1
            v = 0
        elif action == 'up':
            u = 0
            v = 1
        elif action == 'down':
            u = 0
            v = -1
        else:
            print("Error: Invalid action given")

        # Define overrides now based on max class ranges
        new_x = state_x + u
        new_y = state_y + v
        states_df_state = states_df[(states_df['state_x']==state_x)&(states_df['state_y']==state_y)]
        states_df_new_state = states_df[(states_df['state_x']==new_x)&(states_df['state_y']==new_y)]
        # If current state has probability 0, then this is trap state and we do not move
        if states_df_state['prob'].iloc[0] == 0:
            u = 0
            v = 0
        # If next state doesn't exist, don't move
        elif len(states_df_new_state) == 0:
            u = 0
            v = 0
        # If this returns a valid result, outcome acceptable
        elif states_df_new_state['state_ids'].iloc[0] != "NONE":
            u = u
            v = v
        # Otherwise, a wall is hit and paper doesn't move from current state
        else:
            u = 0
            v = 0
        return (u, v)
    
    def step(self, classroom_id, state_x, state_y, action):
        classroom = self.Classrooms['Classroom_'+str(classroom_id)]
        # Find current state and given probability
        state_data = classroom[(classroom['state_x']==state_x)&(classroom['state_y']==state_y)]
        prob = state_data['prob'].iloc[0]

        # Take action as successful or pick another random action if not
        action_rng = np.random.rand()
        if action_rng <= prob:
            action = action
        else:
            action_sub_list = self.legal_actions.copy()
            action_sub_list.remove(action)
            action = random.choice(action_sub_list)

        # Find movement direction given current state and action that ended up being taken
        current_action_outcome = Engine.action_outcome(state_x, state_y, action, classroom)
        u = current_action_outcome[0]
        v = current_action_outcome[1]
        next_state_x = int(state_x + u)
        next_state_y = int(state_y + v)
        next_state_data = classroom[(classroom['state_x']==next_state_x)&(classroom['state_y']==next_state_y)]
        reward = next_state_data['reward'].iloc[0]
        terminated = next_state_data['terminal'].iloc[0]

        return(next_state_x, next_state_y, reward, terminated)




