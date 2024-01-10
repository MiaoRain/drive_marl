# import torch as th
# from torch import nn
import configparser
import argparse
from llmagent import llmagent
from DriveLikeAHuman.HELLM_MARL import DriveLikeAHumanClass
config_dir = 'configs/configs_ppo.ini'
config = configparser.ConfigParser()
config.read(config_dir)
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
# th.manual_seed(torch_seed)
# th.backends.cudnn.benchmark = False
# th.backends.cudnn.deterministic = True
# from torch.optim import Adam, RMSprop
import sys
sys.path.append("../highway-env")
import highway_env
import numpy as np
import os, logging
from copy import deepcopy
# from single_agent.Memory_common import OnPolicyReplayMemory
# from single_agent.Model_common import ActorNetwork, CriticNetwork
from common.utils import index_to_one_hot, to_tensor_var, VideoRecorder
import gym

# evaluation the learned agent(this block is directly copy from chendong's evaluation because i think LLM don't need to train)
rewards = []
infos = []
avg_speeds = []
steps = []
eval_episodes=100 ## set evaluation episode as 100, i arbitarly set this
vehicle_speed = []
vehicle_position = []
video_recorder = None
seeds = [20] #1
traffic_density=2
env = gym.make('merge-multi-agent-v0')
output_dir='result/'
# llmagent=llmagent()

#step-Environment Initialization and state sample
for i in range(eval_episodes):
    avg_speed = 0
    step = 0
    rewards_i = []
    infos_i = []
    done = False

    if traffic_density == 1: # easy mode: 1-3 CAVs + 1-3 HDVs
        state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=1)
    elif traffic_density == 2:# hard mode: 2-4 CAVs + 2-4 HDVs
        state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=2)
    elif traffic_density == 3:# hard mode: 4-6 CAVs + 3-5 HDVs
        state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=4)
    else:
        state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i])

    n_agents = len(env.controlled_vehicles)
    rendered_frame = env.render(mode="rgb_array")

    # #recording video
    # video_filename = os.path.join(output_dir,
    #                                 "testing_episode{}".format(eval_episodes + 1) + '_{}'.format(i) +
    #                                 '.mp4')
    # # Init video recording
    # if video_filename is not None:
    #     print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
    #                                                             5))
    #     video_recorder = VideoRecorder(video_filename,
    #                                     frame_size=rendered_frame.shape, fps=5)
    #     video_recorder.add_frame(rendered_frame)
    # else:
    #     video_recorder = None

    DLHuman = DriveLikeAHumanClass(5, env)
    while not done:
        step += 1
        # action = action_gen(state, n_agents)
        ## assume we only control llm agent(RL cars):
        control_agents=env.controlled_vehicles
        i = 0
        actions = []
        for agent_id in control_agents:

            print("agent_id:",agent_id)
            state_ = state[i] ## state: num_control_agent* 25= num_agent* (5car* 5[id,x,y,vx,vy])
            # action = llmagent.generate_decision(state)
            state_ = state_.reshape(5, 5) #5car* 5[id,x,y,vx,vy], first cars are "control vech"
            action = DLHuman.run(state_)
            actions.append(action)
            print("action:", action)
            i += 1#num_agent* 25= num_agent* (5car* 5[id,x,y,vx,vy])

        state, reward, done, info = env.step(actions)
        avg_speed += info["average_speed"]
        rendered_frame = env.render(mode="rgb_array")
        if video_recorder is not None:
            video_recorder.add_frame(rendered_frame)

        rewards_i.append(reward)
        infos_i.append(info)

    vehicle_speed.append(info["vehicle_speed"])
    vehicle_position.append(info["vehicle_position"])
    rewards.append(rewards_i)
    infos.append(infos_i)
    steps.append(step)
    avg_speeds.append(avg_speed / step)

    if video_recorder is not None:
        video_recorder.release()
    env.close()
