import random
from tqdm import tqdm
import time
# ------ Imports -----------------------------------------
from environment.classroom_generator import Engine
# Adapter
from adapters.classroom_A import ClassroomALanguage
# Agent Setup
from helios_rl.environment_setup.imports import ImportHelper
# Evaluation standards
from helios_rl.environment_setup.results_table import ResultsTable
from helios_rl.environment_setup.helios_info import HeliosInfo


STATE_ADAPTER_TYPES = {
    "Language": ClassroomALanguage
}

class Environment:

    def __init__(self, local_setup_info: dict):
        # Init classroom A env
        self.env = Engine()
        self.start_obs = self.env.Classroom_A()
        self.legal_actions = self.env.legal_actions
        
        # Language adapter
        #self.adapter = ClassroomALanguage()

        # Agent
        Imports = ImportHelper(local_setup_info)
        self.agent, self.agent_type, self.agent_name, self.agent_state_adapter = Imports.agent_info(STATE_ADAPTER_TYPES)
        self.num_train_episodes, self.num_test_episodes, self.training_action_cap, self.testing_action_cap, self.reward_signal = Imports.parameter_info()  
        # Training or testing phase flag
        self.train = Imports.training_flag()
        # --- HELIOS
        self.live_env, self.observed_states, self.experience_sampling = Imports.live_env_flag()
        # Results formatting
        self.results = ResultsTable(local_setup_info)
        # HELIOS input function
        # - We only want to init trackers on first batch otherwise it resets knowledge
        self.helios = HeliosInfo(self.observed_states, self.experience_sampling)
        # Env start position for instr input
        # Enable sub-goals
        if (local_setup_info['sub_goal'] is not None) & (local_setup_info['sub_goal']!=["None"]) & (local_setup_info['sub_goal']!="None"):
            self.sub_goal:list = local_setup_info['sub_goal']
        else:
            self.sub_goal:list = None


    def episode_loop(self):
        # Mode selection (already initialized)
        if self.train:
            number_episodes = self.num_train_episodes
        else:
            number_episodes = self.num_test_episodes

        for episode in tqdm(range(0, number_episodes)):
            start_obs = self.start_obs
            state_x = start_obs[0]
            state_y = start_obs[1]
            state = self.agent_state_adapter.adapter(state_x=state_x, state_y=state_y, legal_moves=self.legal_actions, episode_action_history=None, encode=True)
            # ---
            start_time = time.time()
            episode_reward:int = 0
            action_history = []
            for action in range(0,self.training_action_cap):
                if self.live_env:
                    # Agent takes action
                    agent_action = self.agent.policy(state, self.legal_actions)
                    action_history.append(agent_action)
                    
                    next_state_x, next_state_y, reward, terminated = self.env.step(classroom_id='A', state_x=state_x, state_y=state_y, action=agent_action)
                    # Override reward per action with small negative punishment
                    if reward==0:
                        reward = -0.05
                    
                    next_state = self.agent_state_adapter.adapter(state_x=next_state_x, state_y=next_state_y, legal_moves=self.legal_actions, episode_action_history=None, encode=True)
                    # HELIOS trackers    
                    self.helios.observed_state_tracker(engine_observation=tuple([next_state_x, next_state_y]),
                                                        language_state=self.agent_state_adapter.adapter(state_x=next_state_x, state_y=next_state_y, legal_moves=self.legal_actions, episode_action_history=None, encode=False))
                    
                    # TODO MUST COME BEFORE SUB-GOAL CHECK OR 'TERMINAL STATES' WILL BE FALSE
                    self.helios.experience_sampling_add(state, agent_action, next_state, reward, terminated)
                    # Trigger end on sub-goal if defined
                    if self.sub_goal:
                        next_state_tuple = tuple([next_state_x,next_state_y])
                        if next_state_tuple in self.sub_goal:
                            reward = self.reward_signal[0]
                            terminated = True                        
                else:
                    #print(self.helios.experience_sampling[tuple(state.cpu().numpy().flatten())])
                    # override state with previous actions outcome, 
                    # - if first action in game this wont exist so use init first state
                    try:
                        state = next_state
                    except:
                        state = state
                    legal_moves = self.helios.experience_sampling_legal_actions(state)
                    # Unknown state, have no experience to sample from so force break episode
                    if legal_moves == None:
                        break
                    
                    agent_action = self.agent.policy(state, legal_moves)
                    next_state, reward, terminated = self.helios.experience_sampling_step(state, agent_action)

                if self.train:
                    self.agent.learn(state, next_state, reward, agent_action)
                    

                episode_reward+=reward
                if terminated:
                    break
                else:    
                    state = next_state
                    if self.live_env:
                        state_x = next_state_x
                        state_y = next_state_y
            end_time = time.time()
            agent_results = self.agent.q_result()
            if self.live_env:
                self.results.results_per_episode(self.agent_name, None, episode, action, episode_reward, (end_time-start_time), action_history, agent_results[0], agent_results[1]) 

        return self.results.results_table_format()
                    