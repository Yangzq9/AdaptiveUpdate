import argparse
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
import time
import pickle
import shutil
import os
import sys
import copy
import random
import gym
import pybullet_envs

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg_attention_MF import MADDPGAgentTrainerAll, make_update
from multiagent.core import Revise
# import RflySim APIs
from OU_noise.OU_noise import OrnsteinUhlenbeckActionNoise
from threading import Thread
from gym import spaces
tf.disable_v2_behavior()

time.sleep(np.random.random() * 1)
with open('seed.txt', 'r+') as f:
    seed = int(f.read())
    f.seek(0)
    f.truncate()
    f.write(str(seed + 1))

env_config = [['InvertedPendulumBulletEnv-v0', 5000, 500], ['InvertedDoublePendulumBulletEnv-v0', 25000, 500],
              ['InvertedPendulumSwingupBulletEnv-v0', 3500, 500], ['HopperBulletEnv-v0', 25000, 500],
              ['Walker2DBulletEnv-v0', 50000, 500], ['AntBulletEnv-v0', 13000, 500]]
al_config = [[0.01, 1, 'soft'], [0.1, 1, 'soft'], [0.5, 1, 'soft'], [0.5, 1, 'soft-t'], [1.0, 1, 'soft-s1']]
al_num = 0

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_defender", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=0, help="maximum episode length,better not to exceed 1k")
    parser.add_argument("--num-episodes", type=int, default=2000, help="number of episodes")
    parser.add_argument("--good-policy", type=str, default="ddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="ddpg", help="policy of adversaries")
    parser.add_argument("--ctrl-attacker", default=False, help="RL control either attacker or defender")
    # Core training parameters.
    parser.add_argument("--τ", type=float, default=0.0, help="soft parameter")
    parser.add_argument("--d", type=int, default=0, help="update interval")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='env', help="name of the experiment")
    parser.add_argument("--soft-update", type=str, default=' ', help="type of soft update")
    parser.add_argument("--save-dir", type=str, default="./Saveofsimple/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--display-death", default=False)
    parser.add_argument("--estimator", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--Decritic", default=False)
    parser.add_argument("--para-share", default=False)
    parser.add_argument("--sample-eval", default=False)
    parser.add_argument("--new-display", default=False)
    parser.add_argument("--loss-weight-eval", default=False)
    parser.add_argument("--dynamic-attacker-policy", default=True)
    parser.add_argument("--save-position", default=False, help="if you want enable estimator,set this parameter to True in training")
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./Saveofsimple/learning_curves-test/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = tf.compat.v1.layers.dense(out, 1 * num_units, tf.nn.relu)
        out = tf.compat.v1.layers.dense(out, 1 * num_units, tf.nn.relu)
        out = tf.compat.v1.layers.dense(out, 1 * num_units, tf.nn.relu)
        out = tf.compat.v1.layers.dense(out, 1 * num_units, tf.nn.relu)
        out = tf.compat.v1.layers.dense(out, 1 * num_units, tf.nn.relu)
        out = tf.compat.v1.layers.dense(out, num_outputs, None)
        return out

def sample_eval_model(input, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = tf.compat.v1.layers.dense(out, 4 * num_units, tf.nn.relu)
        out = tf.compat.v1.layers.dense(out, 2 * num_units, tf.nn.relu)
        out = tf.compat.v1.layers.dense(out, 2, None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    if arglist.ctrl_attacker:
        world.agents = world.attackers
        world.num_agents = world.num_attackers
        world.Non_agents = world.defenders
        world.num_Non_agents = world.num_defenders
    else:
        world.agents = world.defenders
        world.num_agents = world.num_defenders
        world.Non_agents = world.attackers
        world.num_Non_agents = world.num_attackers
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    sample_model = sample_eval_model
    trainer = MADDPGAgentTrainerAll
    trainers.append(trainer('defender', model, model, sample_model, obs_shape_n, [spaces.Discrete(env.action_space.shape[0] * 3)],
                       arglist, policy_name=arglist.good_policy))
    return trainers

def act(env, obs_n, trainers, adj):
    # set action to unit.action.u
    action_agent = trainers[0].action([obs_n[i][None] for i in range(len(obs_n))])
    action_agent = [action_agent[i][0] for i in range(len(action_agent))]
    action_ = copy.deepcopy(action_agent)
    # set action of agents
    trainers[0].acts.append(action_[0])
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.mean(trainers[0].acts, 0), x0=trainers[0].acts[-1], theta=0.15, sigma=0.2)
    noise = ou_noise()
    action_[0] = action_[0] + noise * trainers[0].u_noise
    action_temp = np.zeros(int(len(action_agent[0])/3))
    for j in range(len(action_temp)):
        action_temp[j] = np.clip(action_[0][j*3] - action_[0][j*3+1], -1.0, 1.0)
    trainers[0].acts[-1] = action_[0] + 0.0
    action_agent[0] = action_[0] + 0.0
    # set action of Non agents
    return action_temp * adj, action_agent

def train(arglist):
    with U.single_threaded_session():
        # np.random.seed(0)
        # Create environment
        env = gym.make(arglist.env_name)
        # Create agent trainers
        obs_shape_n = [env.observation_space.shape]
        adj = env.action_space.high

        # random seed set according to exp_name
        np.random.seed(seed+1000)
        tf.set_random_seed(seed+1000)
        os.environ['PYTHONHASHSEED'] = str(seed+1000)
        random.seed(seed+1000)
        env.seed(seed+1000)

        trainers = get_trainers(env, obs_shape_n, arglist)
        print('Using policy {}'.format(arglist.good_policy))

        # Initialize
        U.initialize()

        op, polyak, _ = make_update(trainers[0].name, RL_al=arglist.good_policy)
        sess = tf.get_default_session()
        sess.run([op], feed_dict={polyak: [1.0]})

        # Load previous results, if necessary
        arglist.load_dir = arglist.save_dir + arglist.exp_name + '/'
        if arglist.restore:
            arglist.exp_name = arglist.exp_name + '_test'
        arglist.save_dir = arglist.save_dir + arglist.exp_name + '/'
        if arglist.restore:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)


        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(1)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = [env.reset()]
        episode_step = 0
        train_step = 0
        train_step_late = 0
        t_start = time.time()

        t0 = 0.
        t1 = 0.
        t2 = 0.

        τ = [[] for _ in range(len(trainers))]
        delta = [[] for _ in range(len(trainers))]

        # the data need by estimator
        losses = [[] for i in range(len(trainers))]

        print('Starting iterations...')
        while True:
            # set action to unit.action.u
            t_rec = time.time()
            action_temp, action_agent = act(env, obs_n, trainers, adj)
            t0 += time.time() - t_rec
            # environment step
            t_rec = time.time()
            new_obs_n, rew_n, done_n, info_n = env.step(action_temp)
            new_obs_n = [new_obs_n] + []
            rew_n = [rew_n] + []

            episode_step += 1
            done = done_n
            # statistics the status of all agents and targets
            # collect experience
            trainers[0].replay_buffer[0].add(obs_n[0], action_agent[0], rew_n[0], new_obs_n[0], float(done_n), 0.0)

            # compute reward of every agent
            obs_n = new_obs_n
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
            t1 += time.time() - t_rec

            if done:
                t_rec = time.time()

                obs_n = [env.reset()]
                trainers[0].acts = []
                # save pos_rec and vel_rec when arglist.display is True

                episode_rewards[-1] = episode_rewards[-1]
                episode_rewards.append(0)
                for a in agent_rewards:
                    a[-1] = a[-1]/episode_step
                    a.append(0)
                agent_info.append([[]])
                episode_step = 0

                for i in range(len(trainers)):
                    τ[i].append(trainers[i].τ_equal)
                    delta[i].append(trainers[i].delta_equal)
                t2 += time.time() - t_rec

            # increment global step counter
            train_step += 1
            if not arglist.display and len(episode_rewards) >= 2 * arglist.save_rate and train_step % 100 == 0:
                # update all trainers, if not in display or benchmark mode
                loss = None
                for i, trainer in enumerate(trainers):
                    trainer.preupdate()
                    loss = trainers[0].update(trainers, train_step)
                    if loss is not None:
                        losses[i].append(loss)

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.004)
                env.render()
                continue

            # save model, display training output
            if (done) and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                mean_rew = [round(np.mean(rew[-arglist.save_rate:]), 3) for rew in agent_rewards]
                mean_τ = [round(np.mean(τ_d[-arglist.save_rate:]), 3) for τ_d in τ]
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                agent_rewards_temp = np.array(agent_rewards)
                final_ep_ag_rewards.append(np.mean(agent_rewards_temp[:, -arglist.save_rate:], axis=1))

                print("steps: {:.3f}, episodes: {}, mean episode reward: {:.3f}, agent episode reward: {}, polyak_mean: {},time: {:.3f}"
                    .format((train_step - train_step_late) / arglist.save_rate, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                            np.array(mean_rew), np.array(mean_τ), time.time() - t_start))
                # print('action time:', t0, 'step time:', t1, 'done time(train):', t2)
                t0 = 0.
                t1 = 0.
                t2 = 0.

                train_step_late = train_step
                t_start = time.time()

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) >= arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(np.array(episode_rewards), fp)
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(np.array(agent_rewards), fp)
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_tau.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(np.array(τ), fp)
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_delta.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(np.array(delta), fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    for i in range(len(env_config)):
        for j in range(len(al_config)):
            arglist = parse_args()
            arglist.τ = al_config[j][0]
            arglist.d = al_config[j][1]
            arglist.exp_name = arglist.exp_name + '_' + str(i) + '_' + str(j + al_num) + '_' + str(seed)
            arglist.env_name = env_config[i][0]
            arglist.num_episodes = env_config[i][1]
            arglist.soft_update = al_config[j][2]
            arglist.save_rate = env_config[i][2]

            train(arglist)

            tf.reset_default_graph()