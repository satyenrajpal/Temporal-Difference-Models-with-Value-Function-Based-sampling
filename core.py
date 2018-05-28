# -*- coding: utf-8 -*-
import warnings
from copy import deepcopy
import numpy as np
from keras.callbacks import History
import pickle
from callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
import os
import datetime
# def compute_reward(achieved_goal, goal, reward_type='sparse'):        # Compute distance between goal and the achieved goal.
    
#         d = np.linalg.norm(achieved_goal - goal, axis=-1)

#         if reward_type == 'sparse':
#             return -(d > 0.05).astype(np.float32)
#         else:
#             return -d



class Agent(object):
    """Abstract base class for all implemented agents.

    Each agent interacts with the environment (as defined by the `Env` class) by first observing the
    state of the environment. Based on this observation the agent changes the environment by performing
    an action.

    Do not use this abstract base class directly but instead use one of the concrete agents implemented.
    Each agent realizes a reinforcement learning algorithm. Since all agents conform to the same
    interface, you can use them interchangeably.

    To implement your own agent, you have to implement the following methods:

    - `forward`
    - `backward`
    - `compile`
    - `load_weights`
    - `save_weights`
    - `layers`

    # Arguments
        processor (`Processor` instance): See [Processor](#processor) for details.
    """
    def __init__(self, val_goal,processor=None):
        self.processor = processor
        self.training = False
        self.step = 0
        self.val_goal=val_goal

    def get_config(self):
        """Configuration of the agent for serialization.
        """
        return {}

    def fit(self, env, nb_steps,folder_to_save,action_repetition=1, callbacks=None, verbose=1,max_tau=10,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=5000,
            nb_max_episode_steps=None,reward_type='sparse'):
        """Trains the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()
        history_success=[]
        test_history=[]
        episode = 0
        self.step = 0
        states = None
        episode_reward = None
        episode_step = None
        did_abort = False

        reward_flag=1 if reward_type.lower()=='sparse' else 0 

        try:
            episode_exp=[]
            while self.step < nb_steps:
                if states is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = 0
                    episode_reward = 0.

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    states= deepcopy(env.reset())
                    if self.processor is not None:
                        states = self.processor.process_observation(states)
                    assert states is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    # nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                    # for _ in range(nb_random_start_steps):
                    #     if start_step_policy is None:
                    #         action = env.action_space.sample()
                    #     else:
                    #         action = start_step_policy(observation)
                    #     if self.processor is not None:
                    #         action = self.processor.process_action(action)
                    #     callbacks.on_action_begin(action)
                    #     observation, reward, done, info = env.step(action)
                    #     observation = deepcopy(observation)
                    #     if self.processor is not None:
                    #         observation, reward, done, info = self.processor.process_step(observation, reward, done, info)
                    #     callbacks.on_action_end(action)
                    #     if done:
                    #         warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                    #         observation = deepcopy(env.reset())
                    #         if self.processor is not None:
                    #             observation = self.processor.process_observation(observation)
                    #         break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert states is not None

                # Run a single step.
                is_succ_hist=callbacks.on_step_begin(episode_step)
                if is_succ_hist!=-1:
                    # print("In Core!!", is_succ_hist)
                    history_success.append(is_succ_hist)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                observation=states['observation']
                goal_state=states['desired_goal']
                tau_step=nb_max_episode_steps-episode_step
                action = self.forward(observation,goal_state,tau_step)
                
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    new_states, r, done, info = env.step(action)
                    achieved_goal=new_states['achieved_goal']
                    next_observation=new_states['observation']
                                    #1-Sparse           0- Dense
                    r=env.compute_reward(achieved_goal,goal_state,reward_flag)
                    # observation = deepcopy(observation)
                    episode_exp.append([observation, action,r, next_observation, done, goal_state, achieved_goal])

                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(observation, r, done, info)

                    for key, value in info.items():
                        if np.all(value==1):
                            value=1
                        else: 
                            value=0
                            # if not np.isreal(value):
                                # continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward += r
                    states=deepcopy(new_states)
                    if done:
                        break
                
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics = self.backward(reward, max_tau,terminal=done,reward_flag=reward_flag) #TODO:Check this
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'goal_state':goal_state,
                    'info': accumulated_info,
                }

                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1
                
                if done:
                    # ############################HER STARTS HERE###################################
                    
                    # # num_time_steps_left=np.random.randint(0,max_tau,size=self.batch_size)
                    # for counter, step_exp in enumerate(episode_exp):
                    #     # counter += 1
                    #     observation_l, action_l, r_l, next_observation_l, done_l, goal_state_l, achieved_goal_l = step_exp
                    #     self.memory.append(observation_l, action_l,r_l, next_observation_l, done_l, goal_state_l, achieved_goal_l)
                       
                    #     future = np.random.randint(counter,len(episode_exp), size=4)                        #GET THE INDICES OF STATES TO BE RELABELED AS GOALS
                    #     for i in future:                                                                    #list(set(future))
                    #         _, _, _, _, _, _, resampled_goal_l = episode_exp[i]
                    #         updated_reward = env.compute_reward(achieved_goal_l, resampled_goal_l, reward_flag) #CHANGE THIS ACCORDING TO THE RESAMPLED GOAL
                    #         self.memory.append(observation_l, action_l,updated_reward, next_observation_l, done_l, resampled_goal_l, achieved_goal_l)#STORE THE NEW CONTENT
                    # ############################HER ENDS HERE#####################################
                    # episode_exp = []                                                                #B

                    ############################tdm replay STARTS HERE###################################
                    tau = max_tau
                    possible_goals = []
                    Gt_episode=[]
                    if self.val_goal:
                        G_t=0
                        for epi in reversed(episode_exp):
                            G_t+=1*epi[2]
                            epi[2]=G_t
                            # print("Reward:", G_t)

                    for counter, step_exp in enumerate(episode_exp):
                        #observation, action,   r,    next_observation,     done, goal_state, achieved_goal
                        observation_l, action_l, r_l, next_observation_l, done_l, goal_state_l, achieved_goal_l = step_exp
                        possible_goals.append(achieved_goal_l)
                        Gt_episode.append(r_l)
                        self.memory.append(observation_l, action_l,r_l, next_observation_l,
                         done_l,tau, episode, counter) 
                        tau -= 1
                        if tau<0:
                            tau = max_tau
                    possible_goals.append(goal_state_l)
                    self.episode_mem.append(episode,possible_goals,np.array(Gt_episode))

                    ############################tdm replay ENDS HERE#####################################
                    # possible_goals = []
                    # for i in range(0,len(episode_exp)):                                             #list(set(future))
                    #     _, _, _, _, _, _, possible_goal = episode_exp[i]
                    #     possible_goals.append(possible_goal)
                    # possible_goals.append(goal_state_l)
                    # self.episode_mem.append(possible_goals)
                    episode_exp = []                                                                #B

                    # if(self.step%5000==0 and self.step>=10000):
                    #     test_history.append(self.test(env, nb_episodes=50,visualize=False))
                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    states = None
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()
        log_history=np.array(history_success)
        
        with open(os.path.join('results',folder_to_save,'TDM_'+reward_type+'.pkl'),'wb') as f:
            pickle.dump(log_history,f)

        return history

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins."
        """
        if not self.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        # self.step = 0

        # callbacks = [] if not callbacks else callbacks[:]

        # if verbose >= 1:
        #     callbacks += [TestLogger()]
        # if visualize:
        #     callbacks += [Visualizer()]
        # history = History()
        # callbacks += [history]
        # callbacks = CallbackList(callbacks)
        # if hasattr(callbacks, 'set_model'):
        #     callbacks.set_model(self)
        # else:
        #     callbacks._set_model(self)
        # callbacks._set_env(env)
        # params = {
        #     'nb_episodes': nb_episodes,
        # }
        # if hasattr(callbacks, 'set_params'):
        #     callbacks.set_params(params)
        # else:
        #     callbacks._set_params(params)

        # self._on_test_begin()
        # callbacks.on_train_begin()
        for episode in range(nb_episodes):
            # callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            states = deepcopy(env.reset())
            if self.processor is not None:
                states = self.processor.process_observation(states)
            assert states is not None
            history_success=[]
            observation=states['observation']
            goal_state=states['desired_goal']
            
            done = False
            while not done:
                # history_success.append(callbacks.on_step_begin(episode_step))

                # callbacks.on_step_begin(episode_step)

                action = self.forward(observation,goal_state) #<- Change this to include goal 
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    # callbacks.on_action_begin(action)
                    new_states, r, done, info = env.step(action)
                    achieved_goal=new_states['achieved_goal']
                    next_observation=new_states['observation']
                    
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(observation, r, d, info)
                    # callbacks.on_action_end(action)
                    reward += r
                    # for key, value in info.items():
                    #     if not np.isreal(value):
                    #         continue
                    #     if key not in accumulated_info:
                    #         accumulated_info[key] = np.zeros_like(value)
                    #     accumulated_info[key] += value
                    if done:
                        history_success.append(info['is_success'])
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                # self.backward(reward, terminal=done)
                episode_reward += reward

                # step_logs = {
                #     'action': action,
                #     'observation': observation,
                #     'reward': reward,
                #     'episode': episode,
                #     'info': accumulated_info,
                # }
                # callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                # self.step += 1

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            # callbacks.on_episode_end(episode, episode_logs)
        # callbacks.on_train_end()
        # self._on_test_end()
        history=np.mean(np.array(history_success).flatten())
        print("Test success: {}, for {} episodes".format(history,nb_episodes))
        return history

    def reset_states(self):
        """Resets all internally kept states after an episode is completed.
        """
        pass

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        raise NotImplementedError()

    def backward(self, reward,max_tau, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.

        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.
        """
        raise NotImplementedError()

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.

        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        raise NotImplementedError()

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.

        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.

        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).
        
        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.
        """
        raise NotImplementedError()

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        pass

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        pass

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        pass

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        pass


class Processor(object):
    """Abstract base class for implementing processors.

    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.

    Do not use this abstract base class directly but instead use one of the concrete implementations
    or write your own.
    """

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.

        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.

        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        """
        return observation

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.
        """
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.
        """
        return action

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.
        """
        return batch

    @property
    def metrics(self):
        """The metrics of the processor, which will be reported during training.

        # Returns
            List of `lambda y_true, y_pred: metric` functions.
        """
        return []

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []


# Note: the API of the `Env` and `Space` classes are taken from the OpenAI Gym implementation.
# https://github.com/openai/gym/blob/master/gym/core.py


# class Env(object):
#     """The abstract environment class that is used by all agents. This class has the exact
#     same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
#     OpenAI Gym implementation, this class only defines the abstract methods without any actual
#     implementation.
#     """
#     reward_range = (-np.inf, np.inf)
#     action_space = None
#     observation_space = None

#     def step(self, action):
#         """Run one timestep of the environment's dynamics.
#         Accepts an action and returns a tuple (observation, reward, done, info).

#         # Arguments
#             action (object): An action provided by the environment.

#         # Returns
#             observation (object): Agent's observation of the current environment.
#             reward (float) : Amount of reward returned after previous action.
#             done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
#             info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
#         """
#         raise NotImplementedError()

#     def reset(self):
#         """
#         Resets the state of the environment and returns an initial observation.
        
#         # Returns
#             observation (object): The initial observation of the space. Initial reward is assumed to be 0.
#         """
#         raise NotImplementedError()

#     def render(self, mode='human', close=False):
#         """Renders the environment.
#         The set of supported modes varies per environment. (And some
#         environments do not support rendering at all.) 
        
#         # Arguments
#             mode (str): The mode to render with.
#             close (bool): Close all open renderings.
#         """
#         raise NotImplementedError()

#     def close(self):
#         """Override in your subclass to perform any necessary cleanup.
#         Environments will automatically close() themselves when
#         garbage collected or when the program exits.
#         """
#         raise NotImplementedError()

#     def seed(self, seed=None):
#         """Sets the seed for this env's random number generator(s).
        
#         # Returns
#             Returns the list of seeds used in this env's random number generators
#         """
#         raise NotImplementedError()

#     def configure(self, *args, **kwargs):
#         """Provides runtime configuration to the environment.
#         This configuration should consist of data that tells your
#         environment how to run (such as an address of a remote server,
#         or path to your ImageNet data). It should not affect the
#         semantics of the environment.
#         """
#         raise NotImplementedError()

#     def __del__(self):
#         self.close()

#     def __str__(self):
#         return '<{} instance>'.format(type(self).__name__)


# class Space(object):
#     """Abstract model for a space that is used for the state and action spaces. This class has the
#     exact same API that OpenAI Gym uses so that integrating with it is trivial.
#     """

#     def sample(self, seed=None):
#         """Uniformly randomly sample a random element of this space.
#         """
#         raise NotImplementedError()

#     def contains(self, x):
#         """Return boolean specifying if x is a valid member of this space
#         """
#         raise NotImplementedError()
