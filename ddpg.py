from __future__ import division
from collections import deque
import os
import warnings

import numpy as np
import keras.backend as K
import keras.optimizers as optimizers

from core import Agent
from random_rl import OrnsteinUhlenbeckProcess
from util import *
import sys

#TERMINAL - 0 -> Ended
#         - 1 -> Not ended
#Compute Reward- 0 - dense
#              - 1 - Sparse
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def compute_reward_scalar(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return -np.linalg.norm(goal_a - goal_b, axis=-1)
    # return -np.abs(goal_a - goal_b)

def compute_reward_vector(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return -np.abs(goal_a - goal_b)


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

def get_relabelled_batch(env,episode_num_batch,step_num_batch,episode_mem,terminal1_batch,max_tau,reward_flag,vectorized):
    batch_size=len(terminal1_batch)
    tau_batch = np.random.randint(0, max_tau, size = batch_size)
    resampled_goal_batch = []
    reward_batch=[]
    
    for i in range(batch_size):
        resampled_goal = episode_mem.sample_goal(step_num_batch[i], episode_num_batch[i], tau_batch[i])
        resampled_goal_batch.append(resampled_goal)
        achieved_goal=episode_mem.get_achieved_goal(step_num_batch[i],episode_num_batch[i])
        terminal1_batch[i] = terminal1_batch[i]*(tau_batch[i]!=0)
        if goal_distance(achieved_goal,resampled_goal)<0.05:
            terminal1_batch[i] = 0

        # desired_goal=episode_mem.get_desired_goal(episode_num_batch[i]
        # reward=compute_reward(achieved_goal, resampled_goal,reward_flag)
        if not vectorized:
        	reward=compute_reward_scalar(achieved_goal, resampled_goal)
        else:
        	reward=compute_reward_vector(achieved_goal, resampled_goal)
        
        assert np.all(reward<=0)
        if not reward_flag:
            reward*=10
        reward_batch.append(reward*terminal1_batch[i])
    
    return np.array(reward_batch), np.array(terminal1_batch), np.array(resampled_goal_batch) ,np.expand_dims(tau_batch,axis=-1)

class DDPGAgent(Agent):
    """Write me
    """
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory,env,critic_goal_input,tau_input,
                 gamma=.99, batch_size=128, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={}, target_model_update=.001,val_goal=0,vectorized=0, **kwargs):
        # if hasattr(actor.output, '__len__') and len(actor.output) > 1:
        #     raise ValueError('Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.format(actor))
        print("Actor has {} outputs".format(len(actor.output)))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError('Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))
        if critic_action_input not in critic.input:
            raise ValueError('Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError('Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.format(critic))

        super(DDPGAgent, self).__init__(val_goal,**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
            print("Hard Update")
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)
            print("Soft update")

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_goal_input=critic_goal_input
        self.critic_action_input_idx = self.critic.input.index(critic_action_input)
        self.memory,self.episode_mem = memory
        self.val_goal=val_goal
        if vectorized:
        	self.goal_dim=env.reset()['desired_goal'].shape[0]
        else:
        	self.goal_dim=1
        self.tau_input=tau_input

        # State.
        self.compiled = False
        self.reset_states()
        self.env=env
        self.vectorized=vectorized

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip))

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        combined_inputs = []
        actor_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                actor_inputs.append(i)
        
        actions_tanh,_=self.actor(actor_inputs)

        combined_inputs[self.critic_action_input_idx] = actions_tanh

        predictions = self.critic(combined_inputs)
        
        #Expectation of Q values as the loss, ACTOR UPDATE!!!!!
        # print("Goal_diff shape:",K.sum(K.abs(predictions-K.squeeze(self.critic_goal_input,axis=1)),axis=-1))
        # loss_=K.sum(K.abs(predictions-K.squeeze(self.critic_goal_input,axis=1)),axis=-1)
                            #^Shape has to be (?,3) 
        # loss_ = K.sum(K.abs(predictions),axis=-1) 
        if self.vectorized:
        	loss_ = K.mean(K.abs(predictions))
        else:
        	loss_ = -(K.mean(predictions))
        # loss_0=K.sum(K.abs(self.next_state_input-self.critic_goal_input),axis=-1)
        # loss_=loss_+K.mean(loss_0*K.cast(K.equal(self.tau_input,0),'float32'))
        
        updates = actor_optimizer.get_updates(
            params=self.actor.trainable_weights, loss=loss_)
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN
        
        # Finally, combine it all into a callable function.
        if K.backend() == 'tensorflow':
            self.actor_train_fn = K.function(actor_inputs + [K.learning_phase()],
                                             self.actor(actor_inputs), updates=updates)
        else:
            if self.uses_learning_phase:
                actor_inputs += [K.learning_phase()]
            self.actor_train_fn = K.function(actor_inputs, self.actor(actor_inputs), updates=updates)
        
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        # self.recent_action = None
        # self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state,goal_state, tau):
    
        state_batch = self.process_state_batch([state])
        goal_batch = self.process_state_batch([goal_state])
        tau_batch = self.process_state_batch([tau])
        #ACTUAL
        # action, _ = self.actor.predict_on_batch([state_batch,np.abs(goal_batch - state_batch[:,:,:3]), tau_batch])
        action, _ = self.actor.predict_on_batch([state_batch,goal_batch, tau_batch])        
        action=action.flatten()
        assert action.shape == (self.nb_actions,)
        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action

    def forward(self, observation,goal_state, tau):
        # Select an action. 
        state = [observation]
        goal_state=[goal_state]
        tau=[tau]
        action = self.select_action(state,goal_state, tau) 

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names
    
    def backward(self, reward,max_tau, terminal=False, reward_flag=1):

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)

            assert len(experiences) == self.batch_size

            obs_batch = []
            n_obs_batch = []
            action_batch = []
            terminal1_batch = []
            tau_batch=[]
            episode_num_batch=[]
            step_num_batch=[]
            reward_batch=[]

            for e in experiences:
                obs_batch.append(e.state0)
                action_batch.append(e.action)
                n_obs_batch.append(e.state1)
                terminal1_batch.append(0. if e.terminal1 else 1.)
                tau_batch.append(e.num_of_steps_left)
                episode_num_batch.append(e.episode_number)
                step_num_batch.append(e.step_number)
                reward_batch.append(e.reward)

            #env,episode_num_batch,step_num_batch,episode_mem,terminal1_batch,max_tau
            reward_batch, terminal1_batch, resampled_goal_batch, tau_batch = \
             get_relabelled_batch(self.env,episode_num_batch,step_num_batch,self.episode_mem,
                terminal1_batch,max_tau,reward_flag, self.vectorized)
            
            # Prepare and validate parameters.
            obs_batch = self.process_state_batch(obs_batch) #returns np.array
            n_obs_batch = self.process_state_batch(n_obs_batch)
            action_batch = np.array(action_batch)
 
            if not self.vectorized:
            	reward_batch=np.expand_dims(reward_batch,axis=-1)

            assert reward_batch.shape == (self.batch_size,self.goal_dim)
            assert terminal1_batch.shape == (self.batch_size,)
            assert action_batch.shape == (self.batch_size, self.nb_actions)
            assert tau_batch.shape == (self.batch_size,1)
            
            goal_batch_reshape=resampled_goal_batch

            resampled_goal_batch=np.expand_dims(resampled_goal_batch,axis=1) #To make it as (Batch_size,1,goal_dim)
            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                next_tau_batch=tau_batch-1
                next_tau_batch[next_tau_batch<0] = 0
                #ACTUAL                                                             #Give g-goalify(s)
                # target_actions, _ = self.target_actor.predict_on_batch([n_obs_batch,np.abs(resampled_goal_batch-n_obs_batch[:,:,:3]),next_tau_batch]) #Change this!
                target_actions, _ = self.target_actor.predict_on_batch([n_obs_batch,resampled_goal_batch,next_tau_batch]) 

                assert target_actions.shape == (self.batch_size, self.nb_actions)
                n_obs_batch_w_action = [target_actions,n_obs_batch, resampled_goal_batch, next_tau_batch]
                target_qf = self.target_critic.predict_on_batch(n_obs_batch_w_action) #<O/P is f(inputs)- goal
                assert target_qf.shape == (self.batch_size,self.goal_dim)

                # target_q_values=np.linalg.norm(goal_pred_batch-goal_batch_reshape,ord=1,axis=-1)
                # target_q_values+=np.linalg.norm(np.squeeze(n_obs_batch[:,:,:3])-goal_batch_reshape,ord=1,axis=-1)
                # target_q_values=np.multiply(target_q_values,(tau_batch==0).flatten())
                

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                           #f(inputs)      -    g
                if self.vectorized:
                	target_qf=np.abs(target_qf) #<-(128,3)
                # target_q_values=target_q_values*(tau_batch>0) ###CHECK THIS! TRY WITHOUT THIS AS WELL

                discounted_reward_batch = self.gamma * target_qf
                discounted_reward_batch *= np.expand_dims(terminal1_batch,axis=-1)
                # discounted_reward_batch *= terminal1_batch
                # reward_batch=np.expand_dims(reward_batch,axis=-1)
                #This basically says that reward is only acheived if state is terminal
                assert discounted_reward_batch.shape == reward_batch.shape ==(self.batch_size,self.goal_dim)
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size,self.goal_dim)
                # targets=np.linalg.norm(targets,ord=1,axis=-1)
                assert targets.shape==(self.batch_size,self.goal_dim)
                # Perform a single batch update on the critic network.
                ##########################################################
                obs_batch_w_action = [obs_batch,resampled_goal_batch, tau_batch]

                obs_batch_w_action.insert(self.critic_action_input_idx, action_batch)
                metrics = self.critic.train_on_batch(obs_batch_w_action, targets)
                ##########################################################
                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                ##################################################
                # inputs = [obs_batch,np.abs(resampled_goal_batch-obs_batch[:,:,:3]), tau_batch]
                inputs = [obs_batch,resampled_goal_batch, tau_batch]

                if self.uses_learning_phase:
                    inputs += [self.training]
                ######################################################
                action_values, _ = self.actor_train_fn(inputs)
                action_values=np.array(action_values)
                assert action_values.shape == (self.batch_size, self.nb_actions)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics
