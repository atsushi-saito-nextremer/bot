#!/usr/bin/env python
# coding: utf-8

'''
Created on Jun 18, 2016

An DQN Agent

- An DQN
- Keep an experience_replay pool: training_data <State_t, Action, Reward, State_t+1>
- Keep a copy DQN

Command: python .\run.py --agt 9 --usr 1 --max_turn 40 --movie_kb_path .\deep_dialog\data\movie_kb.1k.json --dqn_hidden_size 80 --experience_replay_pool_size 1000 --replacement_steps 50 --per_train_epochs 100 --episodes 200 --err_method 2


@author: xiul
'''

import random, copy, json
import cPickle as pickle
import numpy as np

from deep_dialog import dialog_config

from agent import Agent
# from deep_dialog.qlearning import DQN
from deep_dialog.qlearning import AgentQNet
from deep_dialog.qlearning import OptimizerWithHyperParams

import tensorflow as tf


class AgentDQN(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None, agent_name="AgentDQN"):
        assert isinstance(agent_name, str)
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        self.experience_replay_pool = []  # experience replay pool <s_t, a_t, r_t, s_t+1>

        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)

        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn

#        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions)
#        self.clone_dqn = copy.deepcopy(self.dqn)
        opt_with_hyper_params = OptimizerWithHyperParams()
        opt_with_hyper_params.optimizer = tf.train.RMSPropOptimizer(learning_rate=opt_with_hyper_params.learning_rate) 
        self.dqn = AgentQNet(agent_name, opt_with_hyper_params, self.state_dimension, self.hidden_size, self.num_actions)
        # self.tf_saver = tf.train.Saver()

        self.cur_bellman_err = 0
        self.bot_type = "movie"
        self.limit_experience_replay_pool_size = True
        

        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            # self.dqn.model = copy.deepcopy(self.load_trained_DQN(params['trained_model_path']))
            # self.clone_dqn = copy.deepcopy(self.dqn)
            self.load_trained_DQN(params['trained_model_path'])
            self.predict_mode = True
            self.warm_start = 2

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0
        self.phase = 0
        if self.bot_type == "movie":
            self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
        elif self.bot_type == "hotel":
            self.request_set = ['reservation', 'name', 'date', 'roomtype']
        else:
            assert False

    # state_to_actionをオーバーライドする
    def state_to_action(self, state):
        """ DQN: Input state, output action """
        # 受け取った対話ステートをエージェントが学習可能な表現に変換
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

    #
    def prepare_state_representation(self, state):
        """ Create the representation for each state """

        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        # 1 * 発話タイプ数次元の配列を準備
        user_act_rep = np.zeros((1, self.act_cardinality))
        # 前回にユーザが発した発話タイプ部分を1で埋める
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        # 1 * スロット数次元の配列を準備
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        # 前回ユーザが発したinform_slotsで該当するkey部分を1で埋める
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot.replace("\r","")]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        # 1 * スロット数次元の配列を準備
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        # 前回ユーザが発したrequest_slotsで該当するkey部分を1で埋める
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        # 1 * スロット数次元の配列を準備
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        # current_slotsのinform_slots(ユーザ・エージェントを総合した現状のスロット)で該当するkey部分を1で埋める
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot.replace("\r","")]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        # 初回でなければ，エージェントの最後の発話行為に該当する部分を1で埋める
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        # 初回でなければ，エージェントの最後のinforms_slotsに該当する部分を1で埋める
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot.replace("\r","")]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        # 初回でなければ，エージェントの最後のrequests_slotsに該当する部分を1で埋める
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        # ターンを10で割ったものもベクトルにする
        turn_rep = np.zeros((1, 1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        # 1 * 最大ターン数分の配列を作って，現ターンに該当する部分を1で埋める
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        # 1 * スロット数文の配列を作り，全ての制約を満たす映画の数を100で割って値を入れる
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        # 1 * スロット数文の配列を作り，全ての制約を満たす映画が1以上あれば1で埋める
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
            kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)

        # 作成した配列を全て結合する
        self.final_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation

    def run_policy(self, representation):
        """ epsilon-greedy policy """

        # ランダム動作 (< ε)
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        # ネットに従ったgreedy動作(ε <)
        else:
            # warm startフラグが立っていた場合
            if self.warm_start == 1:
                # warm start の終了条件？
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy()
            else:
                return self.dqn.predict(representation, {}, predict_model=True)

    def rule_policy(self):
        """ Rule Policy """
        # print("self.pahse:{}".format(self.phase))
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1
            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                                 'request_slots': {}}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)

    def action_index(self, act_slot_response):
        """ Return the index of action """
        #print "self.feasible_actions", self.feasible_actions,"\n"
        #print "act_slot_response", act_slot_response
        #assert False
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """ Register feedback from the environment, to be stored as future training data """

        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)

        # トレーニングモードなら，warm startの時にER(experience replay)に登録
        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
                # ##print training_example
                # ##assert False
        # 予測モードならERに登録
        else:  # Prediction Mode
            if self.limit_experience_replay_pool_size and len(self.experience_replay_pool) > self.experience_replay_pool_size:
                self.experience_replay_pool.reverse()
                self.experience_replay_pool.pop()
                self.experience_replay_pool.reverse()
                self.experience_replay_pool.append(training_example)
            else:
                self.experience_replay_pool.append(training_example)

    def train(self, batch_size=1, num_batches=100):
        """ Train DQN with experience replay """

        for iter_batch in range(num_batches):
            self.cur_bellman_err = 0.
            for iter in range(len(self.experience_replay_pool) / (batch_size)):
                # バッチはERからランダムに選ぶ
                batch = [random.choice(self.experience_replay_pool) for i in xrange(batch_size)]
                # batch = reduce(lambda x,y: np.concatenate((x,y)),[random.choice(self.experience_replay_pool) for i in xrange(batch_size)])
                # batch_struct = self.dqn.singleBatch(batch, {'gamma': self.gamma}, self.clone_dqn)
                #batch = batch[0][0].shape
                #assert False
                bell_err, _ = self.dqn.update_q_net(batch)
                assert bell_err is not None
                self.cur_bellman_err += bell_err
                #self.cur_bellman_err += batch_struct['cost']['total_cost']

            print ("cur bellman err %.4f, experience replay pool %s" % (
                float(self.cur_bellman_err) / len(self.experience_replay_pool), len(self.experience_replay_pool)))

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print 'saved model in %s' % (path,)
        except Exception, e:
            print 'Error: Writing model fails: %s' % (path,)
            print e

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """

        # trained_file = pickle.load(open(path, 'rb'))
        # model = trained_file['model']
        self.dqn.saver.restore(self.dqn.tf_sess, path)
        # print "trained DQN Parameters:", json.dumps(trained_file['params'], indent=2)
        # return model
