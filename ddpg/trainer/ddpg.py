import numpy as np
import random
import tensorflow._api.v2.compat.v1 as tf
import ddpg.common.tf_util as U
import time
import copy

from ddpg.common.distributions import make_pdtype
from ddpg import AgentTrainer
from ddpg.trainer.replay_buffer import ReplayBuffer
tf.disable_v2_behavior()

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update(scope, reuse = None):
    with tf.variable_scope(scope, reuse=reuse):
        transfer = []
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        vars = p_func_vars + q_func_vars
        target_vars = target_p_func_vars + target_q_func_vars

        delta_equal = []
        polyak = tf.placeholder(tf.float32, 1)
        para_num = 0
        for i in range(len(vars)):
            para = tf.cast(tf.size(vars[i]), tf.float32)
            delta = vars[i] - target_vars[i]

            transfer.append(target_vars[i].assign(target_vars[i] + polyak * delta))
            delta_equal.append(tf.reduce_sum(tf.abs(delta)))
            para_num += para

        delta_equal = tf.reduce_sum(delta_equal) / para_num
        return transfer, polyak, delta_equal

def make_update_t(scope, reuse = None, tao = 0.5):
    with tf.variable_scope(scope, reuse=reuse):
        transfer = []
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        vars = p_func_vars + q_func_vars
        target_vars = target_p_func_vars + target_q_func_vars

        W = tf.placeholder(tf.float32, len(vars))
        sigma2 = tf.placeholder(tf.float32, len(vars))
        v = tf.placeholder(tf.float32, 1)
        sigma2_next = []
        W_next = []
        tau_equal = []
        delta_equal = []
        para_num = 0
        for i in range(len(vars)):
            para = tf.cast(tf.size(vars[i]), tf.float32)
            delta = vars[i] - target_vars[i]
            delta2 = tf.reduce_mean(tf.square(delta))
            w = (v + 1) / (v + delta2 / sigma2[i])
            tau = w/(W[i] + w)

            tau_equal.append(tau * para)
            delta_equal.append(tf.reduce_sum(tf.abs(delta)))
            para_num += para
            transfer.append(target_vars[i].assign(target_vars[i] + tau * delta))
            tau_sigma = tao * w * v / (v + 1)
            sigma2_next.append((1 - tau_sigma) * sigma2[i] + tau_sigma * delta2)
            W_next.append((1 - tao) * (W[i] + w))

        sigma2_next = tf.concat(sigma2_next, axis=0)
        W_next = tf.concat(W_next, axis=0)
        tau_equal = tf.reduce_sum(tau_equal) / para_num
        delta_equal = tf.reduce_sum(delta_equal) / para_num

        return transfer, W_next, sigma2_next, tau_equal, W, sigma2, v, delta_equal

def make_update_s1(scope, reuse = None, beta_1=0.999, beta_2=0.9):
    with tf.variable_scope(scope, reuse=reuse):
        transfer = []
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        vars = p_func_vars + q_func_vars
        target_vars = target_p_func_vars + target_q_func_vars
        polyak = tf.placeholder(tf.float32, 1)
        t = tf.placeholder(tf.float32, 1)

        m = []
        v = []
        m_op = []
        v_op = []
        tau_equal = []
        delta_equal = []
        para_num = 0

        for var in vars:
            m.append(tf.Variable(tf.zeros(var.shape), name='moment1_estimate'))
            v.append(tf.Variable(tf.zeros(var.shape), name='moment2_estimate'))

        for i in range(len(vars)):
            para = tf.cast(tf.size(vars[i]), tf.float32)
            delta =  vars[i] - target_vars[i]
            m_op.append(m[i].assign(beta_1 * m[i] + (1 - beta_1) * delta))
            v_op.append(v[i].assign(beta_2 * v[i] + (1 - beta_2) * delta ** 2))

            m_temp = m[i] / (1 - beta_1 ** t)
            v_temp = v[i] / (1 - beta_2 ** t)
            tau = tf.sqrt(tf.abs(m_temp) / (tf.sqrt(v_temp + 1e-10)))
            tau = tf.clip_by_value(tau, 0.0, 1.0)
            tau = polyak * tau
            s_temp = delta * tau

            transfer.append(target_vars[i].assign(target_vars[i] + s_temp))
            tau_equal.append(tf.reduce_sum(tau))
            delta_equal.append(tf.reduce_sum(tf.abs(delta)))
            para_num += para

        tau_equal = tf.reduce_sum(tau_equal) / para_num
        delta_equal = tf.reduce_sum(delta_equal) / para_num

        return m_op, v_op, transfer, tau_equal, polyak, t, delta_equal

def make_update_s2(scope, reuse = None, beta_1=0.999, beta_2=0.9):
    with tf.variable_scope(scope, reuse=reuse):
        transfer = []
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        vars = p_func_vars + q_func_vars
        target_vars = target_p_func_vars + target_q_func_vars
        polyak = tf.placeholder(tf.float32, 1)
        t = tf.placeholder(tf.float32, 1)

        m = []
        v = []
        m_op = []
        v_op = []
        tau_equal = []
        delta_equal = []
        para_num = 0

        for var in vars:
            m.append(tf.Variable(tf.zeros(var.shape), name='moment1_estimate'))
            v.append(tf.Variable(tf.zeros(var.shape), name='moment2_estimate'))

        for i in range(len(vars)):
            para = tf.cast(tf.size(vars[i]), tf.float32)
            delta =  vars[i] - target_vars[i]
            m_op.append(m[i].assign(beta_1 * m[i] + (1 - beta_1) * delta))
            v_op.append(v[i].assign(beta_2 * v[i] + (1 - beta_2) * delta ** 2))

            m_temp = m[i] / (1 - beta_1 ** t)
            v_temp = v[i] / (1 - beta_2 ** t)
            tau = tf.sqrt(tf.abs(m_temp) / (tf.sqrt(v_temp + 1e-10)))
            tau = tf.reduce_mean(tau)
            tau = tf.clip_by_value(tau, 0.0, 1.0)
            tau = polyak * tau
            s_temp = delta * tau

            transfer.append(target_vars[i].assign(target_vars[i] + s_temp))
            tau_equal.append(tau * para)
            delta_equal.append(tf.reduce_sum(tf.abs(delta)))
            para_num += para

        tau_equal = tf.reduce_sum(tau_equal) / para_num
        delta_equal = tf.reduce_sum(delta_equal) / para_num

        return m_op, v_op, transfer, tau_equal, polyak, t, delta_equal

def make_update_s3(scope, reuse = None, beta_1=0.999, beta_2=0.9):
    with tf.variable_scope(scope, reuse=reuse):
        transfer = []
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        vars = p_func_vars + q_func_vars
        target_vars = target_p_func_vars + target_q_func_vars
        polyak = tf.placeholder(tf.float32, 1)
        t = tf.placeholder(tf.float32, 1)

        m = []
        v = []
        m_op = []
        v_op = []
        tau_equal = []
        delta_equal = []
        taus = []
        deltas = []
        para_num = 0

        for var in vars:
            m.append(tf.Variable(tf.zeros(var.shape), name='moment1_estimate'))
            v.append(tf.Variable(tf.zeros(var.shape), name='moment2_estimate'))

        for i in range(len(vars)):
            para = tf.cast(tf.size(vars[i]), tf.float32)
            delta = vars[i] - target_vars[i]
            m_op.append(m[i].assign(beta_1 * m[i] + (1 - beta_1) * delta))
            v_op.append(v[i].assign(beta_2 * v[i] + (1 - beta_2) * delta ** 2))

            m_temp = m[i] / (1 - beta_1 ** t)
            v_temp = v[i] / (1 - beta_2 ** t)
            tau = tf.sqrt(tf.abs(m_temp) / (tf.sqrt(v_temp + 1e-10)))
            taus.append(tf.reduce_sum(tau))
            deltas.append(delta)
            para_num += para

        tau = tf.reduce_sum(taus) / para_num
        tau = tf.clip_by_value(tau, 0.0, 1.0)
        tau = polyak * tau
        for i in range(len(vars)):
            s_temp = deltas[i] * tau
            transfer.append(target_vars[i].assign(target_vars[i] + s_temp))
            delta_equal.append(tf.reduce_sum(tf.abs(deltas[i])))

        tau_equal = tau
        delta_equal = tf.reduce_sum(delta_equal) / para_num

        return m_op, v_op, transfer, tau_equal, polyak, t, delta_equal

def critics_train(make_obs_ph, act_space, q_func, optimizer, num_defender_close=0,
                  grad_norm_clipping=None, policy_name=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype = make_pdtype(act_space)

        # set up placeholders
        obs_ph = make_obs_ph
        act_ph = act_pdtype.sample_placeholder([None], name="action")
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        q_input = tf.concat([obs_ph, act_ph], 1)

        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))  # loss function is MSE
        # viscosity solution to Bellman differential equation in place of an initial condition
        # q_reg = tf.reduce_mean(tf.square(q))
        loss = tf.reduce_sum(q_loss)

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)
        # Create callable functions
        train = U.function(inputs=[obs_ph] + [act_ph] + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function([obs_ph] + [act_ph], q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:, 0]
        target_q_values = U.function([obs_ph] + [act_ph], target_q)

        return train, {'q_values': q_values, 'target_q_values': target_q_values}


def actors_train(make_obs_ph, act_space, p_func, q_func, optimizer, num_defender_close=0, grad_norm_clipping=None, policy_name=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype = make_pdtype(act_space)

        # set up placeholders
        obs_ph = make_obs_ph
        act_ph = act_pdtype.sample_placeholder([None], name="action")

        p_input = obs_ph

        p = p_func(tf.convert_to_tensor(p_input), int(act_pdtype.param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype.pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input = act_ph
        act_input = act_pd.sample()
        q_input = tf.concat([obs_ph, act_input], 1)

        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]
        pg_loss = -tf.reduce_mean(q)
        loss = tf.reduce_sum(pg_loss) + tf.reduce_sum(p_reg) * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=[obs_ph] + [act_ph], outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph], outputs=act_sample)
        p_values = U.function([obs_ph], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype.param_shape()[0]), scope="target_p_func", num_units=num_units)

        target_act_sample = act_pdtype.pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph], outputs=target_act_sample)

        return act, train, {'p_values': p_values, 'target_act': target_act}

class DDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, p_model, q_model, obs_shape, act_space, args, policy_name=False):
        self.name = name
        self.args = args
        self.n = 1
        self.sample_eval_train_record = []
        self.eval = args.sample_eval
        self.loss_weight_eval = args.loss_weight_eval
        self.update_time = 0
        num_defender_close = 2
        self.tau = args.tau
        self.d = args.d
        self.acts = []
        self.u_noise = 1.0
        obs_ph = U.BatchInput(obs_shape, name="observation").get()
        # Create all the functions necessary to train the model
        self.q_train, self.q_debug = critics_train(
            scope=self.name,
            make_obs_ph=obs_ph,
            act_space=act_space,
            q_func=q_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            policy_name=policy_name,
            num_units=args.num_units,
            num_defender_close=num_defender_close
        )
        self.act, self.p_train, self.p_debug = actors_train(
            scope=self.name,
            make_obs_ph=obs_ph,
            act_space=act_space,
            p_func=p_model,
            q_func=q_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            policy_name=policy_name,
            num_units=args.num_units,
            num_defender_close=num_defender_close
        )
        # Create experience buffer
        self.replay_buffer = [ReplayBuffer(1e6)]
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

        self.tau_equal = 0
        self.delta_equal = 0
        self.epsilon = 0.1
        self.t = 0.0
        self.soft_update = args.soft_update
        if self.soft_update == 'soft':
            self.op, self.polyak, self.delta = make_update(self.name)
        if self.soft_update == 'soft-t':
            self.op, self.W_next, self.sigma2_next, self.tau_next, self.W, self.sigma2, self.v, self.delta = make_update_t(self.name, tao=self.tau)
            self.sigma2_value = np.array([self.epsilon ** 2 for i in range(self.sigma2.shape[0])])
            self.W_value = np.array([(1 - self.tau) / (self.tau) for i in range(self.W.shape[0])])
        if self.soft_update == 'soft-s1':
            self.m_op, self.v_op, self.op, self.tau_next, self.polyak, self.t_input, self.delta = make_update_s1(self.name)
        if self.soft_update == 'soft-s2':
            self.m_op, self.v_op, self.op, self.tau_next, self.polyak, self.t_input, self.delta = make_update_s2(self.name)
        if self.soft_update == 'soft-s3':
            self.m_op, self.v_op, self.op, self.tau_next, self.polyak, self.t_input, self.delta = make_update_s3(self.name)

    def update_target(self):
        sess = U.get_session()
        if self.soft_update == 'soft':
            [_, self.delta_equal] = sess.run([self.op, self.delta], feed_dict={self.polyak: [self.tau]})
            self.tau_equal = self.tau
        if self.soft_update == 'soft-t':
            [_, self.W_value, self.sigma2_value, self.tau_equal, self.delta_equal] = \
                sess.run([self.op, self.W_next, self.sigma2_next, self.tau_next, self.delta],
                         feed_dict={self.W: self.W_value, self.sigma2: self.sigma2_value, self.v: [1.0]})
        if self.soft_update == 'soft-s1':
            self.t = self.t + 1
            sess.run([self.m_op, self.v_op])
            [_, self.tau_equal, self.delta_equal] = sess.run([self.op, self.tau_next, self.delta],
                                                    feed_dict={self.polyak: [self.tau], self.t_input:[self.t]})
        if self.soft_update == 'soft-s2':
            self.t = self.t + 1
            sess.run([self.m_op, self.v_op])
            [_, self.tau_equal, self.delta_equal] = sess.run([self.op, self.tau_next, self.delta],
                                                    feed_dict={self.polyak: [self.tau], self.t_input:[self.t]})
        if self.soft_update == 'soft-s3':
            self.t = self.t + 1
            sess.run([self.m_op, self.v_op])
            [_, self.tau_equal, self.delta_equal] = sess.run([self.op, self.tau_next, self.delta],
                                                    feed_dict={self.polyak: [self.tau], self.t_input:[self.t]})

    def action(self, obs):
        return self.act(*obs)

    def sample_eval(self, obs, act, new_obs, rew, done, trainers):
        done = np.array([done])[:, np.newaxis]
        rew = np.array([rew])[:, np.newaxis]
        obs = obs[np.newaxis, :]
        act = act[np.newaxis, :]
        new_obs = new_obs[np.newaxis, :]
        score = self.sample_eval_(*([obs] + [act] + [new_obs] + [rew] + [done]))[0][0]
        return score

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, t):
        # self.replay_sample_index = [self.replay_buffer[i].make_index(self.args.batch_size, mean=not self.eval) for i in range(self.n)]
        self.replay_sample_index = self.replay_buffer[0].make_index(self.args.batch_size, mean=not self.eval)
        # collect replay sample from all agents
        index = self.replay_sample_index

        obs, act, rew, obs_next, done = self.replay_buffer[0].sample_index(index)

        # train q network
        target_act_next = self.p_debug['target_act'](*[obs_next])
        target_q_next = self.q_debug['target_q_values'](*([obs_next] + [target_act_next]))
        target_q = rew + self.args.gamma * (1.0 - done) * target_q_next

        q_loss = self.q_train(*([obs] + [act] + [target_q]))
        self.update_time += 1
        # train p network
        p_loss = self.p_train(*([obs] + [act]))
        self.update_target()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]