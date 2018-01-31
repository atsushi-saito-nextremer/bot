'''
Created on Jun 18, 2016

@author: xiul
'''

from .utils import *

import tensorflow as tf
import numpy as np
import math

class OptimizerWithHyperParams(object):
    def __init__(self): 
        self._learning_rate = 0.001
        self._decay_rate = 0.999
        self._momentum = 0.1
        self._grad_clip =  1e-3
        self._smooth_eps = 1e-8
        self._sdg_type = 'rmsprop'
        self._reg_cost = 1e-3
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
    @property
    def regc(self):
        return self._reg_cost
    @property
    def learning_rate(self):
        return self._learning_rate
    @property
    def decay_rate(self):
        return self._decay_rate
    @property
    def momentum(self):
        return self._momentum
    @property
    def grad_clip(self):
        return self._grad_clip
    @property
    def smooth_eps(self):
        return self._smooth_eps
    @property
    def sdg_type(self):
        return self._sdg_type

class QNet(object):
    def __init__(self, tf_var_scope_head, input_size, hidden_size, output_size):
        self.shape_dict = {"input_size":input_size, "output_size":output_size, "hidden_size":hidden_size}
        self._component_names = ["h1", "b1", "h2", "b2"]
        self._qvars, self._tvars, self._best_vars = {}, {}, {}

        with tf.variable_scope(tf_var_scope_head + self._component_names[0],  reuse=tf.AUTO_REUSE, initializer=tf.initializers.random_uniform):    
            h1_shape = [self.shape_dict["input_size"], self.shape_dict["hidden_size"]]
            self._qvars["h1"] = tf.get_variable(shape=h1_shape, dtype=tf.float32, name="q_net",  trainable=True)
            self._tvars["h1"] = tf.get_variable(shape=h1_shape, dtype=tf.float32, name="target_net", trainable=False)
            self._best_vars["h1"] = tf.get_variable(shape=h1_shape, dtype=tf.float32, name="best_net", trainable=False)

        with tf.variable_scope(tf_var_scope_head + self._component_names[1],  reuse=tf.AUTO_REUSE, initializer=tf.initializers.zeros):    
            b1_shape = [self.shape_dict["hidden_size"]]
            self._qvars["b1"] = tf.get_variable(shape=b1_shape, dtype=tf.float32, name="q_net",  trainable=True)
            self._tvars["b1"] = tf.get_variable(shape=b1_shape, dtype=tf.float32, name="target_net", trainable=False)
            self._best_vars["b1"] = tf.get_variable(shape=b1_shape, dtype=tf.float32, name="best_net", trainable=False)

        with tf.variable_scope(tf_var_scope_head + self._component_names[2],  reuse=tf.AUTO_REUSE, initializer=tf.initializers.random_uniform):
            h2_shape = [self.shape_dict["hidden_size"], self.shape_dict["output_size"]]
            self._qvars["h2"] = tf.get_variable(shape=h2_shape, dtype=tf.float32, name="q_net",  trainable=True)
            self._tvars["h2"] = tf.get_variable(shape=h2_shape, dtype=tf.float32, name="target_net",  trainable=False)
            self._best_vars["h2"] = tf.get_variable(shape=h2_shape, dtype=tf.float32, name="best_net", trainable=False)

        with tf.variable_scope(tf_var_scope_head + self._component_names[3],  reuse=tf.AUTO_REUSE, initializer=tf.initializers.zeros):    
            b2_shape = [self.shape_dict["output_size"]]
            self._qvars["b2"] = tf.get_variable(shape=b2_shape, dtype=tf.float32, name="q_net",  trainable=True)
            self._tvars["b2"] = tf.get_variable(shape=b2_shape, dtype=tf.float32, name="target_net", trainable=False)
            self._best_vars["b2"] = tf.get_variable(shape=b2_shape, dtype=tf.float32, name="best_net", trainable=False)

    @property
    def placeholders(self):
        return [self._ph_xs]

    @property
    def variable_dicts(self):
        return self._qvars, self._tvars

    @property
    def component_names(self):
        return self._component_names

    def build_graph_q_and_t(self):
        # Set False to trainable of the target network                
        input_size = self.shape_dict["input_size"]
        output_size = self.shape_dict["output_size"]
        self._ph_xs_q = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name="input_q_xs")
        self._ph_xs_t = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name="input_t_xs")
        self._ph_target_qs = tf.placeholder(dtype=tf.float32, shape=[None, output_size], name="target_qs")

        q_h1 = tf.matmul(self._ph_xs_q, self._qvars["h1"]) + self._qvars["b1"]
        t_h1 = tf.matmul(self._ph_xs_t, self._tvars["h1"]) + self._tvars["b1"]

        q_h1 = tf.nn.relu(q_h1)
        t_h1 = tf.nn.relu(t_h1)

        q_h2 = tf.matmul(q_h1, self._qvars["h2"]) + self._qvars["b2"]
        t_h2 = tf.matmul(t_h1, self._tvars["h2"]) + self._tvars["b2"]

        self._q_out = q_h2
        self._t_out = t_h2
        print self._q_out, self._t_out

        self._copy_q_net_to_target_ops = [self._tvars[name].assign(self._qvars[name]) for name in self.component_names]
        self._copy_q_net_to_best_net_ops = [self._best_vars[name].assign(self._qvars[name]) for name in self.component_names]
        self._copy_best_net_to_q_net_ops = [self._qvars[name].assign(self._best_vars[name])for name in self.component_names]

        keys=["h1", "h2"]
        sqrt_scale = lambda x:math.sqrt(6.0/x)
        q_scales = dict(zip(keys, map(sqrt_scale, [sum(self._qvars[name].get_shape().as_list()) for name in ["h1", "h2"]])))
        b_scales = dict(zip(keys, map(sqrt_scale, [sum(self._best_vars[name].get_shape().as_list()) for name in ["h1", "h2"]])))
        t_scales = dict(zip(keys, map(sqrt_scale, [sum(self._tvars[name].get_shape().as_list()) for name in ["h1", "h2"]])))

        print "###\n"
        print "q,b,t: nums", q_scales, b_scales, t_scales
        print "###\n"

        print "###\n"
        print "q,b,t: scales", q_scales, b_scales, t_scales
        print "###\n"

        self._normalize_init_weights_ops = []
        for name in ["h1","h2"]:
            for s, v in zip([q_scales[name], b_scales[name], t_scales[name]],
                            [self._qvars[name], self._best_vars[name], self._tvars[name]]):
                if name == "h2":
                    s = s*0.1
                self._normalize_init_weights_ops.append(v.assign(s*(2.0*v - 1.0)))

    @property
    def q_out_op(self):
        return self._q_out
    
    @property
    def t_out_op(self):
        return self._t_out
    
    @property
    def ph_target_qs(self):
        return self._ph_target_qs
    
    @property
    def ph_inputs_q_net(self):
        return self._ph_xs_q

    @property
    def ph_inputs_t_net(self):
        return self._ph_xs_t

    @property
    def copy_q_net_to_target_ops(self):
        return self._copy_q_net_to_target_ops

    @property
    def copy_q_net_to_best_net_ops(self):
        return self._copy_q_net_to_best_net_ops

    @property
    def copy_best_net_to_q_net_ops(self):
        return self._copy_best_net_to_q_net_ops

    @property
    def normalize_init_weights_ops(self):
        return self._normalize_init_weights_ops
            
class AgentQNet(QNet):
    def __init__(self, tf_var_scope_head, opt_with_hyper_params, input_size, hidden_size, output_size, gamma=0.9):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        self._sess = tf.Session(config=config)
        self._opt = opt_with_hyper_params
        self._gamma = gamma
        
        self._num_actions = output_size

        super(AgentQNet, self).__init__(tf_var_scope_head, input_size, hidden_size, output_size)
        self.build_graph_q_and_t()
        self._set_loss_op()
        self._set_update_op()
        self.saver = tf.train.Saver()
        self._sess.run(tf.global_variables_initializer())
        self._sess.graph.finalize()
        self.copy_best_net_to_q_net()
        self.copy_q_net_to_target()
        self.normalize_init_weights()

    @property
    def tf_sess(self):
        return self._sess

    def normalize_init_weights(self):
        self._sess.run(self.normalize_init_weights_ops)

    def copy_q_net_to_target(self):
        self._sess.run(self.copy_q_net_to_target_ops)

    def copy_q_net_to_best_net(self):
        self._sess.run(self.copy_q_net_to_best_net_ops)

    def copy_best_net_to_q_net(self):
        self._sess.run(self.copy_best_net_to_q_net_ops)

    def q_calc_forward(self, batched_inputs):
        feed_dict = {self.ph_inputs_q_net:batched_inputs}
        return self._sess.run(self.q_out_op, feed_dict=feed_dict)

    def t_calc_forward(self, batched_inputs):
        feed_dict = {self.ph_inputs_t_net:batched_inputs}
        return self._sess.run(self.t_out_op, feed_dict=feed_dict)

    def _set_loss_op(self):
        self._loss_op = tf.reduce_mean(tf.reduce_sum(tf.square((self.q_out_op - self.ph_target_qs)), axis=1), axis=0)

    def _set_update_op(self, regularizer=""):
        print self.loss_op
        if regularizer == "L2":
            loss_op = self.loss_op
            for layer in ["h1", "h2"]:
                loss_op += self._opt.regc*tf.nn.l2_loss(self._qvars[layer])
            self._reg_loss_op = loss_op
        else:
            loss_op = self.loss_op

        grads_and_vars =self._opt.optimizer.compute_gradients(loss_op)
        capped_gvs = [(tf.clip_by_value(grad, -self._opt.grad_clip, self._opt.grad_clip), var) for grad, var in grads_and_vars]
        self._updateq_q_op = self._opt.optimizer.apply_gradients(capped_gvs)

    @property
    def update_q_net_op(self):
        return self._updateq_q_op

    @property
    def reg_loss_op(self):
        if hasattr(self, "_reg_loss_op"):
            return self._reg_loss_op
        else:
            return self._loss_op

    @property
    def loss_op(self):
        return self._loss_op

    def calc_loss(self, batched_inputs, target_q_vals):
        return self._sess.run(self.loss_op, feed_dict={self.ph_inputs_q_net: batched_inputs, self.ph_target_qs: target_q_vals})

    def calc_target_q_values(self,  batched_transitions):
        actions = np.array([t[1] for t in batched_transitions], dtype=np.int32)
        states = np.array([t[0].reshape(-1) for t in batched_transitions], dtype=np.float32)
        rewards = np.array([t[2] for t in batched_transitions], dtype=np.float32)
        next_states = np.array([t[3].reshape(-1) for t in batched_transitions], dtype=np.float32)
        dones = np.array([t[4] for t in batched_transitions], dtype=bool)
        targets = self.q_calc_forward(states)
        next_ys = self.t_calc_forward(next_states)

        for i, z in enumerate(zip(next_ys, rewards, dones)):
            next_y, reward, done = z
            n_action = np.nanargmax(next_y)
            max_next_y = next_y[n_action]

            target_y = reward
            if not done:
                target_y += self._gamma*max_next_y

            targets[i, actions[i]] = target_y

        return targets

    @property
    def batch_size(self):
        return self._batch_size

    def update_q_net(self, batched_transitions):
        batched_states = [ t[0].reshape(-1) for t in batched_transitions ]
        target_q_values = self.calc_target_q_values(batched_transitions)
        feed_dict={self.ph_inputs_q_net: batched_states, self.ph_target_qs: target_q_values}
        loss, _ = self._sess.run([self.loss_op, self.update_q_net_op], feed_dict=feed_dict)
        return loss, None

    def predict(self, batched_inputs, params, **kwargs):
        ys = self.q_calc_forward(batched_inputs)
        return np.argmax(ys)


class BuckupedDQN:

    def __init__(self, input_size, hidden_size, output_size):
        self.model = {}
        # input-hidden
        self.model['Wxh'] = initWeight(input_size, hidden_size)
        self.model['bxh'] = np.zeros((1, hidden_size))

        # hidden-output
        self.model['Wd'] = initWeight(hidden_size, output_size)*0.1
        self.model['bd'] = np.zeros((1, output_size))

        self.update = ['Wxh', 'bxh', 'Wd', 'bd']
        self.regularize = ['Wxh', 'Wd']

        self.step_cache = {}


    def getStruct(self):
        return {'model': self.model, 'update': self.update, 'regularize': self.regularize}


    """Activation Function: Sigmoid, or tanh, or ReLu"""
    def fwdPass(self, Xs, params, **kwargs):
        predict_mode = kwargs.get('predict_mode', False)
        active_func = params.get('activation_func', 'relu')

        # input layer to hidden layer
        Wxh = self.model['Wxh']
        bxh = self.model['bxh']
        Xsh = Xs.dot(Wxh) + bxh

        hidden_size = self.model['Wd'].shape[0] # size of hidden layer
        H = np.zeros((1, hidden_size)) # hidden layer representation

        if active_func == 'sigmoid':
            H = 1/(1+np.exp(-Xsh))
        elif active_func == 'tanh':
            H = np.tanh(Xsh)
        elif active_func == 'relu': # ReLU
            H = np.maximum(Xsh, 0)
        else: # no activation function
            H = Xsh

        # decoder at the end; hidden layer to output layer
        Wd = self.model['Wd']
        bd = self.model['bd']
        Y = H.dot(Wd) + bd

        # cache the values in forward pass, we expect to do a backward pass
        cache = {}
        if not predict_mode:
            cache['Wxh'] = Wxh
            cache['Wd'] = Wd
            cache['Xs'] = Xs
            cache['Xsh'] = Xsh
            cache['H'] = H

            cache['bxh'] = bxh
            cache['bd'] = bd
            cache['activation_func'] = active_func

            cache['Y'] = Y

        return Y, cache

    def bwdPass(self, dY, cache):
        Wd = cache['Wd']
        H = cache['H']
        Xs = cache['Xs']
        Xsh = cache['Xsh']
        Wxh = cache['Wxh']

        active_func = cache['activation_func']
        n,d = H.shape

        dH = dY.dot(Wd.transpose())
        # backprop the decoder
        dWd = H.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims=True)

        dXsh = np.zeros(Xsh.shape)
        dXs = np.zeros(Xs.shape)

        if active_func == 'sigmoid':
            dH = (H-H**2)*dH
        elif active_func == 'tanh':
            dH = (1-H**2)*dH
        elif active_func == 'relu':
            dH = (H>0)*dH # backprop ReLU
        else:
            dH = dH

        # backprop to the input-hidden connection
        dWxh = Xs.transpose().dot(dH)
        dbxh = np.sum(dH, axis=0, keepdims = True)

        # backprop to the input
        dXsh = dH
        dXs = dXsh.dot(Wxh.transpose())

        return {'Wd': dWd, 'bd': dbd, 'Wxh':dWxh, 'bxh':dbxh}


    """batch Forward & Backward Pass"""
    def batchForward(self, batch, params, predict_mode = False):
        caches = []
        Ys = []
        for i,x in enumerate(batch):
            Xs = np.array([x['cur_states']], dtype=float)

            Y, out_cache = self.fwdPass(Xs, params, predict_mode = predict_mode)
            caches.append(out_cache)
            Ys.append(Y)

        # back up information for efficient backprop
        cache = {}
        if not predict_mode:
            cache['caches'] = caches

        return Ys, cache

    def batchDoubleForward(self, batch, params, clone_dqn, predict_mode = False):
        caches = []
        Ys = []
        tYs = []

        for i,x in enumerate(batch):
            Xs = x[0]
            Y, out_cache = self.fwdPass(Xs, params, predict_mode = predict_mode)
            caches.append(out_cache)
            Ys.append(Y)

            tXs = x[3]
            tY, t_cache = clone_dqn.fwdPass(tXs, params, predict_mode = False)

            tYs.append(tY)

        # back up information for efficient backprop
        cache = {}
        if not predict_mode:
            cache['caches'] = caches

        return Ys, cache, tYs

    def batchBackward(self, dY, cache):
        caches = cache['caches']

        grads = {}
        for i in xrange(len(caches)):
            single_cache = caches[i]
            local_grads = self.bwdPass(dY[i], single_cache)
            mergeDicts(grads, local_grads) # add up the gradients wrt model parameters

        return grads


    """ cost function, returns cost and gradients for model """
    def costFunc(self, batch, params, clone_dqn):
        regc = params.get('reg_cost', 1e-3)
        gamma = params.get('gamma', 0.9)

        # batch forward
        Ys, caches, tYs = self.batchDoubleForward(batch, params, clone_dqn, predict_mode = False)

        loss_cost = 0.0
        dYs = []
        for i,x in enumerate(batch):
            Y = Ys[i]
            nY = tYs[i]

            action = np.array(x[1], dtype=int)
            reward = np.array(x[2], dtype=float)

            n_action = np.nanargmax(nY[0])
            max_next_y = nY[0][n_action]

            eposide_terminate = x[4]

            target_y = reward
            if eposide_terminate != True: target_y += gamma*max_next_y

            pred_y = Y[0][action]

            nY = np.zeros(nY.shape)
            nY[0][action] = target_y
            Y = np.zeros(Y.shape)
            Y[0][action] = pred_y

            # Cost Function
            loss_cost += (target_y - pred_y)**2

            dY = -(nY - Y)
            #dY = np.minimum(dY, 1)
            #dY = np.maximum(dY, -1)
            dYs.append(dY)

        # backprop the RNN
        grads = self.batchBackward(dYs, caches)

        # add L2 regularization cost and gradients
        reg_cost = 0.0
        if regc > 0:
            for p in self.regularize:
                mat = self.model[p]
                reg_cost += 0.5*regc*np.sum(mat*mat)
                grads[p] += regc*mat

        # normalize the cost and gradient by the batch size
        batch_size = len(batch)
        reg_cost /= batch_size
        loss_cost /= batch_size
        for k in grads: grads[k] /= batch_size

        out = {}
        out['cost'] = {'reg_cost' : reg_cost, 'loss_cost' : loss_cost, 'total_cost' : loss_cost + reg_cost}
        out['grads'] = grads
        return out


    """ A single batch """
    def singleBatch(self, batch, params, clone_dqn):
        learning_rate = params.get('learning_rate', 0.001)
        decay_rate = params.get('decay_rate', 0.999)
        momentum = params.get('momentum', 0.1)
        grad_clip = params.get('grad_clip', -1e-3)
        smooth_eps = params.get('smooth_eps', 1e-8)
        sdg_type = params.get('sdgtype', 'rmsprop')
        activation_func = params.get('activation_func', 'relu')

        for u in self.update:
            if not u in self.step_cache:
                self.step_cache[u] = np.zeros(self.model[u].shape)

        cg = self.costFunc(batch, params, clone_dqn)

        cost = cg['cost']
        grads = cg['grads']

        # clip gradients if needed
        if activation_func.lower() == 'relu':
            if grad_clip > 0:
                for p in self.update:
                    if p in grads:
                        grads[p] = np.minimum(grads[p], grad_clip)
                        grads[p] = np.maximum(grads[p], -grad_clip)

        # perform parameter update
        for p in self.update:
            if p in grads:
                if sdg_type == 'vanilla':
                    if momentum > 0:
                        dx = momentum*self.step_cache[p] - learning_rate*grads[p]
                    else:
                        dx = -learning_rate*grads[p]
                    self.step_cache[p] = dx
                elif sdg_type == 'rmsprop':
                    self.step_cache[p] = self.step_cache[p]*decay_rate + (1.0-decay_rate)*grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                elif sdg_type == 'adgrad':
                    self.step_cache[p] += grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)

                self.model[p] += dx

        out = {}
        out['cost'] = cost
        return out

    """ prediction """
    def predict(self, Xs, params, **kwargs):
        Ys, caches = self.fwdPass(Xs, params, predict_model=True)
        pred_action = np.argmax(Ys)

        return pred_action
