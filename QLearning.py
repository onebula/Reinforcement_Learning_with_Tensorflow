# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 16:38:09 2018

@author: ck
"""

import tensorflow as tf
import numpy as np

class Model():
    def __init__(self, in_size, net_size, out_size, activation):
        def add_layer(inputs,in_size,out_size,n_layer,activation_function=None): #activation_function=None线性函数  
            layer_name="layer%s" % n_layer
            with tf.name_scope(layer_name):
                with tf.name_scope('weights'):
                    self.Weights = tf.Variable(tf.random_normal([out_size,in_size])) #Weight中都是随机变量  
                with tf.name_scope('biases'):
                    self.biases = tf.Variable(tf.zeros([out_size,1])+0.1) #biases推荐初始值不为0  
                with tf.name_scope('Wx_plus_b'):
                    self.Wx_plus_b = tf.matmul(self.Weights,inputs)+self.biases #inputs*Weight+biases  
                if activation_function is None:
                    self.outputs = self.Wx_plus_b
                else:
                    self.outputs = activation_function(self.Wx_plus_b)
                return self.outputs
        
        tf.reset_default_graph()

        with tf.name_scope('inputs'): #结构化  
            self.xs = tf.placeholder(tf.float32,[None,1],name='x_input')
            self.ys = tf.placeholder(tf.float32,[None,1],name='y_input')
        
        self.l1 = add_layer(self.xs,in_size,net_size,n_layer=1,activation_function=activation) #隐藏层  
        self.l2 = add_layer(self.l1,net_size,out_size,n_layer=2,activation_function=None) #输出层

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys-self.l2),reduction_indices=[1])) #square()平方,sum()求和,mean()平均值  
        with tf.name_scope('train'):
            self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss) #0.1学习效率,minimize(loss)减小loss误差  

        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()
    
    def init_train(self):
        self.sess.run(self.init)
    
    def learn(self, currentstate, qupdate):
        self.sess.run(self.train_step,feed_dict={self.xs:currentstate,self.ys:qupdate})
    
    def get_loss(self, currentstate, qupdate):
        mse = self.sess.run(self.loss,feed_dict={self.xs:currentstate,self.ys:qupdate})
        return mse
    
    def get_Q(self, currentstate):
        qcurrentstate = self.sess.run(self.l2,feed_dict={self.xs:currentstate})
        return qcurrentstate

class RL_Agent():
    def __init__(self, env, net_size, max_timestep, learn_rate=0.01, gamma=1.0):
        self.env = env
        self.max_timestep = max_timestep
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.model = Model(env.observation_space.shape[0], net_size, env.action_space.n, tf.nn.relu)

    def act(self, currentstate):
        action = np.argmax(self.model.get_Q(currentstate))
        return action
    
    def act_epsilon(self, currentstate):
        if np.random.rand(1) < 0.01:
            action = self.env.action_space.sample()
        else:
            action = self.act(currentstate)
        return action

class RL_QLearning(RL_Agent):
    def __init__(self, env, net_size, max_timestep, learn_rate=0.01, gamma=1.0):
        super().__init__(env, net_size, max_timestep, learn_rate, gamma)
        
    def learn(self):
        self.model.init_train()
        i = 0
        while True:
            obs, done = self.env.reset(), False
            episode_reward = 0
            while not done:
                self.env.render()
                currentstate = self.env.state
                q_currentstate = self.model.get_Q(currentstate)
                
                action = self.act_epsilon(currentstate)
                
                obs, reward, done, _ = self.env.step(action)
                
                q_newstate = self.model.get_Q(obs)
                
                q_currentstate_action = q_currentstate[action] + self.learn_rate*(reward+self.gamma*np.max(q_newstate)-q_currentstate[action])
                q_update = q_currentstate
                q_update[action] = q_currentstate_action                
                self.model.learn(currentstate,q_update)
                
                i = i + 1
                episode_reward += reward
            print(['Train', episode_reward, env.counts])
            if i >= self.max_timestep:
                break

if __name__ == '__main__':
    from Car2D import Car2DEnv
    env = Car2DEnv()
    
    RL = RL_QLearning(env,10,10000)
    #RL = RL_Sarsa(env,10,10000)
    RL.learn()
    
    print('======================Done!=====================')
    
    while True:
        obs, done = env.reset(), False
        episode_reward = 0
        steps = 0
        while not done:
            env.render()
            obs, reward, done, _ = env.step(RL.act(obs))
            episode_reward += reward
            steps = steps + 1
            if steps > 100:
                done = 1
        print(['Test', episode_reward, env.counts])