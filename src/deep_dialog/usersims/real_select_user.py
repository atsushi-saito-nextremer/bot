#!/usr/bin/env python
# coding: utf-8

"""
Created on May 14, 2016

a rule-based user simulator

-- user_goals_first_turn_template.revised.v1.p: all goals
-- user_goals_first_turn_template.part.movie.v1.p: moviename in goal.inform_slots
-- user_goals_first_turn_template.part.nomovie.v1.p: no moviename in goal.inform_slots

@author: xiul, t-zalipt
"""

from .usersim import UserSimulator
import argparse, json, random, copy

from deep_dialog import dialog_config


class RealSelectUser(UserSimulator):
    """ A rule-based user simulator for testing dialog policy """

    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """

        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.max_turn = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']

        self.simulator_run_mode = params['simulator_run_mode']
        self.simulator_act_level = params['simulator_act_level']

        self.learning_phase = params['learning_phase']

    def initialize_episode(self):
        """ Initialize a new episode (dialog)
        state['history_slots']: keeps all the informed_slots
        state['rest_slots']: keep all the slots (which is still in the stack yet)
        """

        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0

        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        # self.goal =  random.choice(self.start_set)
        self.goal = self._sample_goal(self.start_set)
        self.goal['request_slots']['ticket'] = 'UNK'
        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

        """ Debug: build a fake goal mannually """
        # self.debug_falk_goal()

        # sample first action
        user_action = self._sample_action()
        assert (self.episode_over != 1), ' but we just started'
        return user_action

    def _sample_action(self):
        """ randomly sample a start action based on user goal """
        # 最初はrequest
        self.state['diaact'] = random.choice(dialog_config.start_dia_acts.keys())

        # "sample" informed slots
        if len(self.goal['inform_slots']) > 0:
            # 1つだけ通知する情報を選択
            known_slot = random.choice(self.goal['inform_slots'].keys())
            self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]

            if 'name' in self.goal['inform_slots'].keys():  # 'moviename' must appear in the first user turn
                self.state['inform_slots']['name'] = self.goal['inform_slots']['name']

            for slot in self.goal['inform_slots'].keys():
                if known_slot == slot or slot == 'name': continue
                self.state['rest_slots'].append(slot)

        # まだinformされていないスロットはrest_slotsに保存
        self.state['rest_slots'].extend(self.goal['request_slots'].keys())

        # "sample" a requested slot
        request_slot_set = list(self.goal['request_slots'].keys())
        request_slot_set.remove('ticket')
        # 他にrequestを選べないときのみ，ticketをリクエストする
        if len(request_slot_set) > 0:
            request_slot = random.choice(request_slot_set)
        else:
            request_slot = 'ticket'
        self.state['request_slots'][request_slot] = 'UNK'

        # request_slotsがなければinformタイプの発話
        if len(self.state['request_slots']) == 0:
            self.state['diaact'] = 'inform'

        # thanksかclosingタイプの発話なら，対話を終了する
        if (self.state['diaact'] in ['thanks', 'closing']):
            self.episode_over = True  # episode_over = True
        else:
            self.episode_over = False  # episode_over = False

        sample_action = {}
        sample_action['diaact'] = self.state['diaact']
        sample_action['inform_slots'] = self.state['inform_slots']
        sample_action['request_slots'] = self.state['request_slots']
        sample_action['turn'] = self.state['turn']

        # 最後に発話行為を元に自然言語を追加
        self.add_nl_to_action(sample_action)
        return sample_action

    def _sample_goal(self, goal_set):
        """ sample a user goal  """

        sample_goal = random.choice(self.start_set[self.learning_phase])
        return sample_goal

    def next(self, system_action):
        """ Generate next User Action based on last System Action """

        self.state['turn'] += 2
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        sys_act = system_action['diaact']

        response_action = {}

        # ターン数が制限を超えていれば対話を終了
        if (self.max_turn > 0 and self.state['turn'] > self.max_turn):
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.episode_over = True
            self.state['diaact'] = "closing"
        # ターン数が制限内なら対話を続行
        else:
            self.state['history_slots'].update(self.state['inform_slots'])
            self.state['inform_slots'].clear()

            # 応答する
            # 応答文を取得
            user_sentence = raw_input('>')

            # 発話行為タイプごとのnl_pairsを取得
            nl_pairs = self.nlg_model.diaact_nl_pairs["dia_acts"]

            # 返すべき発話行為を取得
            response_action = self.get_response_act(user_sentence, nl_pairs)

        response_action['turn'] = self.state['turn']

        return response_action, self.episode_over, self.dialog_status

    def get_response_act(self, user_sentence, nl_pairs):
        for dia_act in nl_pairs:
            for dialog in nl_pairs[dia_act]:
                # テンプレートとマッチしたら発話行為を返す
                res_act = self.check_sentence_match(user_sentence, dialog)
                if res_act != None:
                    res_act["diaact"] = dia_act
                    res_act["nl"] = user_sentence
                    return res_act
        # テンプレートとマッチしなかった
        print(user_sentence)
        raise Exception("sentence not found")

    def check_sentence_match(self, user_sentence, template_dialog):
        # ユーザ発話を単語にバラす
        user_words = user_sentence.split(" ")
        # プレースホルダの値を保存しておく
        template_words = template_dialog["nl"]["usr"].split(" ")
        tmp_slots = {}

        # 発話の長さが一致しなければマッチしていない
        if len(user_words) != len(template_words):
            return None

        # ユーザ発話がテンプレートと一致するチェック
        for i, t_word in enumerate(template_words):
            # プレースホルダなら値を保存
            if '$' in t_word:
                placeholder = t_word.replace("$", "")
                tmp_slots[placeholder] = user_words[i]
                continue
            # マッチしなければNoneを返す
            elif user_words[i] != t_word:
                return None
        # 一致してれば，発話行為を返す
        # request_slotとinform_slotを埋める
        return self.fill_slots(template_dialog, tmp_slots)

    def fill_slots(self, dialog, slots):
        # inform_slotsとrequest_slotsを埋める
        inform_slots = {}
        request_slots = {}
        for slot in dialog["inform_slots"]:
            inform_slots[slot] = slots[slot]
        for slot in dialog["request_slots"]:
            request_slots[slot] = slots[slot]
        res_act = {}
        res_act["inform_slots"] = inform_slots
        res_act["request_slots"] = request_slots
        return res_act


def main(params):
    user_sim = RealSelectUser()
    user_sim.initialize_episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)

    print ("User Simulator Parameters:")
    print (json.dumps(params, indent=2))

    main(params)
