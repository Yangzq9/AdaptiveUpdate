import argparse
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
import time
import pickle
import os
import random
import gym
import pybullet_envs

import ddpg.common.tf_util as U
from ddpg.trainer.core import DDPGAgentTrainer, make_update
from OU_noise.OU_noise import OrnsteinUhlenbeckActionNoise
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
al_config = [['', 0.01, 0.01, 1, ''], ['', 0.1, 0.1, 1, ''], ['', 0.5, 0.5, 1, ''],
             ['', 0.5, 0.5, 1, 't-soft'], ['', 1.0, 1.0, 3, ''], ['', 1.0, 1.0, 1, 'adaptive1']]

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for adaptive update method of target network")
    # Environment
    parser.add_argument("--num-episodes", type=int, default=1000, help="number of episodes")
    # Core training parameters.
    parser.add_argument("--tau", type=float, default=0.01, help="target network update rate")
    parser.add_argument("--xi-1", type=float, default=0.999, help="xi_1")
    parser.add_argument("--xi-2", type=float, default=0.9, help="xi_2")
    parser.add_argument("--d", type=int, default=1, help="update interval")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='train', help="name of the experiment")
    parser.add_argument("--soft-update", type=str, default='', help="type of soft update")
    parser.add_argument("--save-dir", type=str, default="./Saveofsimple/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="./Saveofsimple/learning_curves/", help="directory where plot data is saved")
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

def get_trainer(env, obs_shape, arglist):
    model = mlp_model
    trainer = DDPGAgentTrainer
    trainer = trainer('defender', model, model, obs_shape, spaces.Discrete(env.action_space.shape[0] * 3), arglist)
    return trainer

def act(env, obs, trainer, adj):
    # set action to unit.action.u
    action_agent = trainer.action([obs[None]])
    action_agent = action_agent[0]
    action_ = [action_agent + 0.0]
    # set action of agents
    trainer.acts.append(action_[0])
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.mean(trainer.acts, 0), x0=trainer.acts[-1], theta=0.15, sigma=0.2)
    noise = ou_noise()
    action_[0] = action_[0] + noise * trainer.u_noise
    action_temp = np.zeros(int(len(action_agent)/3))
    for i in range(len(action_temp)):
        action_temp[i] = np.clip(action_[0][i*3] - action_[0][i*3+1], -1.0, 1.0)
    trainer.acts[-1] = action_[0] + 0.0
    action_agent = action_[0] + 0.0
    # set action of Non agents
    return action_temp * adj, action_agent

def train(arglist):
    with U.single_threaded_session():
        # np.random.seed(0)
        # Create environment
        env = gym.make(arglist.env_name)
        # Create agent trainer
        obs_shape = env.observation_space.shape
        adj = env.action_space.high

        # random seed set according to exp_name
        np.random.seed(seed)
        tf.set_random_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        env.seed(seed)

        trainer = get_trainer(env, obs_shape, arglist)

        # Initialize
        U.initialize()

        op, polyak, _ = make_update(trainer.name)
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


        episode_rewards = [0.0]  # sum of rewards
        episode_time_step = []
        agent_rewards = [0.0]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs = env.reset()
        episode_time = time.time()
        episode_step = 0
        train_step = 0
        train_step_late = 0
        t_start = time.time()

        t0 = 0.
        t1 = 0.
        t2 = 0.

        tau = []
        delta = []

        _total = arglist.num_episodes - 2 * arglist.save_rate
        decay_rounds = _total / 2
        base = pow(arglist.tau_min / arglist.tau_max, 1 / decay_rounds)

        # the data need by estimator
        losses = []

        print('Starting iterations...')
        while True:
            # set action to unit.action.u
            t_rec = time.time()
            action_temp, action_agent = act(env, obs, trainer, adj)
            t0 += time.time() - t_rec
            # environment step
            t_rec = time.time()
            new_obs, rew, done, info = env.step(action_temp)

            episode_step += 1
            # statistics the status of all agents and targets
            # collect experience
            trainer.replay_buffer.add(obs, action_agent, rew, new_obs, float(done))

            # compute reward of every agent
            obs = new_obs
            episode_rewards[-1] += rew
            agent_rewards[-1] += rew
            t1 += time.time() - t_rec

            if done:
                t_rec = time.time()

                obs = env.reset()
                trainer.acts = []

                if arglist.decay_name == 'cosi':
                    # Cosine decay
                    _round = len(episode_rewards) - 2 * arglist.save_rate
                    if _round <= decay_rounds and _round >= 0:
                        tau_decay = (np.cos(2 * np.pi * (_round / _total)) + 1) / 2 * (
                                    arglist.tau_max - arglist.tau_min) + arglist.tau_min
                    else:
                        tau_decay = arglist.tau_min
                    trainer.tau = tau_decay
                elif arglist.decay_name == 'poly':
                    # Polynomial decay
                    _round = len(episode_rewards) - 2 * arglist.save_rate
                    if _round <= decay_rounds and _round >= 0:
                        tau_decay = ((1 - _round / decay_rounds) ** arglist.power) * (
                                    arglist.tau_max - arglist.tau_min) + arglist.tau_min
                    else:
                        tau_decay = arglist.tau_min
                    trainer.tau = tau_decay
                elif arglist.decay_name == 'expo':
                    # Exponential decay
                    _round = len(episode_rewards) - 2 * arglist.save_rate
                    tau_decay = arglist.tau_max
                    if _round >= 0:
                        if _round <= decay_rounds:
                            tau_decay = trainer.tau * base
                        else:
                            tau_decay = arglist.tau_min
                    trainer.tau = tau_decay

                episode_rewards[-1] = episode_rewards[-1]
                episode_rewards.append(0)
                episode_time_step.append([time.time() - episode_time, episode_step])
                episode_time = time.time()
                agent_rewards[-1] = agent_rewards[-1] / episode_step
                agent_rewards.append(0)
                agent_info.append([[]])
                episode_step = 0

                tau.append(trainer.tau_equal)
                delta.append(trainer.delta_equal)
                t2 += time.time() - t_rec

            # increment global step counter
            train_step += 1
            if not arglist.display and len(episode_rewards) >= 2 * arglist.save_rate and train_step % 100 == 0:
                # update the trainer, if not in display or benchmark mode
                trainer.preupdate()
                loss = trainer.update(train_step)
                if loss is not None:
                    losses.append(loss)

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.004)
                env.render()
                continue

            # save model, display training output
            if (done) and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                mean_rew = round(np.mean(agent_rewards[-arglist.save_rate:]), 3)
                mean_tau = round(np.mean(tau[-arglist.save_rate:]), 3)
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                agent_rewards_temp = np.array(agent_rewards)
                final_ep_ag_rewards.append(np.mean(agent_rewards_temp[-arglist.save_rate:]))

                print("steps: {:.3f}, episodes: {}, mean episode reward: {:.3f}, agent episode reward: {}, polyak_mean: {},time: {:.3f}"
                    .format((train_step - train_step_late) / arglist.save_rate, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                            np.array(mean_rew), np.array(mean_tau), time.time() - t_start))
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
                    pickle.dump(np.array(tau), fp)
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_delta.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(np.array(delta), fp)
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_ts.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(np.array(episode_time_step), fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    for i in range(len(env_config)):
        for j in range(len(al_config)):
            arglist = parse_args()
            arglist.decay_name = al_config[j][0]
            arglist.tau = al_config[j][1]
            arglist.tau_max = al_config[j][1]
            arglist.tau_min = al_config[j][2]
            arglist.d = al_config[j][3]
            arglist.exp_name = arglist.exp_name + '_' + str(i) + '_' + str(j) + '_' + str(seed)
            arglist.env_name = env_config[i][0]
            arglist.num_episodes = env_config[i][1]
            arglist.soft_update = al_config[j][4]
            arglist.power = 0.5
            arglist.save_rate = env_config[i][2]

            train(arglist)

            tf.reset_default_graph()