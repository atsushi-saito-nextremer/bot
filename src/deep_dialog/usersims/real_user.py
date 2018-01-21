"""
Created on May 17, 2016
@author: xiul, t-zalipt
"""

"Suitable only for message given"

import random

from deep_dialog import dialog_config
from .usersim import UserSimulator


class RealUser(UserSimulator):
    def __init__(self, movie_dict, act_set, slot_set, start_set, params):
        """ Constructor shared by all user simulators """

        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.max_turn = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']

        # self.simulator_run_mode = params['simulator_run_mode']
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
        """
        user_action = {'request_slots': {'ticket': 'UNK'}, 'turn': 0, 'nl': 'Can I get 2 tickets for hail caesar?',
                       'diaact': 'request', 'inform_slots': {'numberofpeople': '2', 'moviename': 'hail caesar'}}
        """
        assert (self.episode_over != 1), ' but we just started'
        return user_action

    def _sample_goal(self, goal_set):
        """ sample a user goal  """

        sample_goal = random.choice(self.start_set[self.learning_phase])
        return sample_goal

    def _sample_action(self):
        """ randomly sample a start action based on user goal """

        self.state['diaact'] = random.choice(list(dialog_config.start_dia_acts.keys()))

        # "sample" informed slots
        if len(self.goal['inform_slots']) > 0:
            known_slot = random.choice(list(self.goal['inform_slots'].keys()))
            self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]

            if 'moviename' in list(self.goal['inform_slots'].keys()):  # 'moviename' must appear in the first user turn
                self.state['inform_slots']['moviename'] = self.goal['inform_slots']['moviename']

            for slot in list(self.goal['inform_slots'].keys()):
                if known_slot == slot or slot == 'moviename': continue
                self.state['rest_slots'].append(slot)

        self.state['rest_slots'].extend(list(self.goal['request_slots'].keys()))

        # "sample" a requested slot
        request_slot_set = list(self.goal['request_slots'].keys())
        request_slot_set.remove('ticket')
        if len(request_slot_set) > 0:
            request_slot = random.choice(request_slot_set)
        else:
            request_slot = 'ticket'
        self.state['request_slots'][request_slot] = 'UNK'

        if len(self.state['request_slots']) == 0:
            self.state['diaact'] = 'inform'

        if (self.state['diaact'] in ['thanks', 'closing']):
            self.episode_over = True  # episode_over = True
        else:
            self.episode_over = False  # episode_over = False

        sample_action = {}
        sample_action['diaact'] = self.state['diaact']
        sample_action['inform_slots'] = self.state['inform_slots']
        sample_action['request_slots'] = self.state['request_slots']
        sample_action['turn'] = self.state['turn']

        self.add_nl_to_action(sample_action)
        return sample_action

    def state_to_action(self, message):
        """ Generate an action by getting input interactively from the command line """
        act_slot_value_response = self.generate_diaact_from_nl(message)

        return {"act_slot_response": act_slot_value_response, "act_slot_value_response": act_slot_value_response}

    def corrupt(self, user_action):
        """ Randomly corrupt an action with error probs (slot_err_probability and slot_err_mode) on Slot and Intent (intent_err_probability). """

        for slot in list(user_action['inform_slots'].keys()):
            slot_err_prob_sample = random.random()
            if slot_err_prob_sample < self.slot_err_probability:  # add noise for slot level
                if self.slot_err_mode == 0:  # replace the slot_value only
                    if slot in list(self.movie_dict.keys()): user_action['inform_slots'][slot] = random.choice(
                        self.movie_dict[slot])
                elif self.slot_err_mode == 1:  # combined
                    slot_err_random = random.random()
                    if slot_err_random <= 0.33:
                        if slot in list(self.movie_dict.keys()): user_action['inform_slots'][slot] = random.choice(
                            self.movie_dict[slot])
                    elif slot_err_random > 0.33 and slot_err_random <= 0.66:
                        del user_action['inform_slots'][slot]
                        random_slot = random.choice(list(self.movie_dict.keys()))
                        user_action[random_slot] = random.choice(self.movie_dict[random_slot])
                    else:
                        del user_action['inform_slots'][slot]
                elif self.slot_err_mode == 2:  # replace slot and its values
                    del user_action['inform_slots'][slot]
                    random_slot = random.choice(list(self.movie_dict.keys()))
                    user_action[random_slot] = random.choice(self.movie_dict[random_slot])
                elif self.slot_err_mode == 3:  # delete the slot
                    del user_action['inform_slots'][slot]

        intent_err_sample = random.random()
        if intent_err_sample < self.intent_err_probability:  # add noise for intent level
            user_action['diaact'] = random.choice(list(self.act_set.keys()))

    def next(self, system_action):
        """ Generate next User Action based on last System Action """

        self.state['turn'] += 2
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        sys_act = system_action['diaact']

        if (self.max_turn > 0 and self.state['turn'] > self.max_turn):
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.episode_over = True
            self.state['diaact'] = "closing"
        else:
            self.state['history_slots'].update(self.state['inform_slots'])
            self.state['inform_slots'].clear()

            self.msg = raw_input()

            if sys_act == "closing":
                self.episode_over = True
                self.state['diaact'] = "thanks"

        self.corrupt(self.state)

        response_action = {}
        response_action = self.generate_diaact_from_nl(self.msg)

        # response_action['diaact'] = self.state['diaact']
        # response_action['inform_slots'] = self.state['inform_slots']
        # response_action['request_slots'] = self.state['request_slots']
        response_action['turn'] = self.state['turn']
        # response_action['nl'] = ""

        # print(response_action)

        # add NL to dia_act
        # self.add_nl_to_action(response_action)
        return response_action, self.episode_over, self.dialog_status

    def generate_diaact_from_nl(self, string):
        """ Generate Dia_Act Form with NLU """

        user_action = {}
        user_action['diaact'] = 'UNK'
        user_action['inform_slots'] = {}
        user_action['request_slots'] = {}

        if len(string) > 0:
            user_action = self.nlu_model.generate_dia_act(string)

        user_action['nl'] = string

        return user_action
