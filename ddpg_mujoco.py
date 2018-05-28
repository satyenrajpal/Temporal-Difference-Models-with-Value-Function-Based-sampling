import numpy as np

import gym,sys
from gym import wrappers

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate,Subtract
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
# from keras.utils import plot_model
import os, datetime
from processors import WhiteningNormalizerProcessor
from ddpg import DDPGAgent
from memory import MemoryWithHer, EpisodeMemoryForHer
from random_rl import OrnsteinUhlenbeckProcess
import argparse

class MujocoProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, -5., 5.)

def parse_args():
    parser = argparse.ArgumentParser(description='TDM with Value function bro!')
    parser.add_argument('--reward_type', dest='reward_type',
                        help='type of reward',
                        default='dense', type=str)
    parser.add_argument('--ENV_NAME', dest='ENV_NAME',
    					help='ENV_NAME', type=str, default='FetchReach-v1')
    parser.add_argument('--output_dir_add', dest='output_dir_add',
    					help='Folder to save', type=str, default='')
    parser.add_argument('--val_goal', dest='val_goal', default=0,
    	help='To sample with value function or not', type=int)
    parser.add_argument('--vectorized', dest='vectorized',default=0,
    	help='To vectorize or not!?',type=int)
    args = parser.parse_args()
    return args

def make_models(env, vectorized):
	action_dim = env.action_space.shape[0]
	obs=env.reset()
	observation_dim=obs['observation'].shape
	goal_dim=obs['desired_goal'].shape

	action_input = Input(shape=(action_dim,), name='action_input')
	observation_input = Input(shape=(1,)+observation_dim, name='observation_input')
	goal_input = Input(shape=(1,)+goal_dim, name='goal_input')
	tau_input= Input(batch_shape=(None,1), name='tau_input')

	flattened_observation = Flatten()(observation_input)
	flattened_goal = Flatten()(goal_input)
	y = Concatenate()([flattened_observation,flattened_goal,tau_input])
	# y = Concatenate()([flattened_observation,flattened_goal])
	y = Dense(300)(y)
	y = Activation('relu')(y)
	# y = Concatenate()([y,tau_input])
	y = Dense(300)(y)
	y = Activation('relu')(y)
	# y = Dense(64)(y)
	# y = Activation('relu')(y)
	y = Dense(action_dim)(y)
	f = Activation('tanh')(y)
	actor= Model(inputs=[observation_input,goal_input,tau_input], 
		outputs=[f,y])
	# print(actor.summary())

	#Critic network
	x = Concatenate()([action_input,flattened_observation,flattened_goal,tau_input])
	# x = Concatenate()([action_input,flattened_observation,flattened_goal])
	x = Dense(300)(x)
	x = Activation('relu')(x)
	# x = Concatenate()([x,tau_input])
	x = Dense(300)(x)
	x = Activation('relu')(x)
	# x = Dense(64)(x)
	# x = Activation('relu')(x)
	if not vectorized:
		x = Dense(1)(x)
	else:
		x=Dense(goal_dim[0])(x)
		x = Subtract()([x,flattened_goal]) ##CHECK THIS
	# x = Dense(1)(x)
	x = Activation('linear')(x)
	critic = Model(inputs=[action_input, observation_input,goal_input,tau_input], 
		outputs=x)

	# print(critic.summary())
	return actor, critic,action_input,goal_input,tau_input
	

if __name__=="__main__":
	args=parse_args()
	if(args.output_dir_add==''):
		print("Enter Something to add to the folder directory name")
		print("Add it as python ddpg_mujoco.py --output_dir_add=blah")
		sys.exit()
	
	reward_type=args.reward_type
	print("Reward Type: ", reward_type)
	ENV_NAME = args.ENV_NAME

	# Get the environment and extract the number of actions.
	env = gym.make(ENV_NAME)
	# env = wrappers.Monitor(env, '/tmp/{}'.format(ENV_NAME), force=True)
	np.random.seed(123)
	env.seed(123)

	actor, critic,action_input,goal_input,tau_input=make_models(env,args.vectorized)
	nb_actions=env.action_space.shape[0]
	assert len(env.action_space.shape) == 1
	
	output_dir_add=args.output_dir_add

	folder_to_save = os.path.join(str(datetime.datetime.now().time()).split('.')[0].replace(':','_')+'_TDM_DDPG_'+output_dir_add+reward_type)
	if not os.path.exists(folder_to_save):
	    os.makedirs(folder_to_save)

	memory = MemoryWithHer(limit=1000000, window_length=1)
	episode_mem=EpisodeMemoryForHer(args.val_goal)
	random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.1, mu=0., sigma=.1)
	
	agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,tau_input=tau_input,
		env=env,memory=(memory,episode_mem), critic_goal_input=goal_input,delta_clip=1.0, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
		random_process=random_process, gamma=0.98, target_model_update=0.001, val_goal=args.val_goal,vectorized=args.vectorized)

	agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

	agent.fit(env, nb_steps=200000, folder_to_save=folder_to_save, visualize=False, 
		verbose=1,reward_type=reward_type,nb_max_episode_steps=50,max_tau=12)

	agent.save_weights(os.path.join(folder_to_save,'ddpg_{}_weights_{}.h5f'.format(ENV_NAME,reward_type)), 
		overwrite=True)
	sys.exit()
	# Finally, evaluate our algorithm for 5 episodes.
	agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=50)
