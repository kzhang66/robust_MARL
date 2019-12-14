#
# Copyright (c) 2019 Kaiqing Zhang @ Amazon AI Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import copy
from typing import Union
from collections import OrderedDict
import random


import numpy as np

from rl_coach.agents.actor_critic_agent import ActorCriticAgent
from rl_coach.agents.agent import Agent
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import DDPGActorHeadParameters, DDPGVHeadParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.base_parameters import NetworkParameters, AlgorithmParameters, \
    AgentParameters, EmbedderScheme
from rl_coach.core_types import ActionInfo, EnvironmentSteps, Batch
from rl_coach.exploration_policies.ou_process import OUProcessParameters
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.spaces import BoxActionSpace, GoalsSpace
from rl_coach.core_types import TimeTypes


class MADDPGCriticNetworkParameters(NetworkParameters):
    def __init__(self, use_batchnorm=False):
        super().__init__()
        self.input_embedders_parameters = {'observation_n': InputEmbedderParameters(batchnorm=use_batchnorm),
                                            'action_n': InputEmbedderParameters(scheme=EmbedderScheme.Shallow)}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [DDPGVHeadParameters()]
        self.optimizer_type = 'Adam'
        self.batch_size = 64
        self.async_training = False
        self.learning_rate = 0.001
        self.adam_optimizer_beta2 = 0.999
        self.optimizer_epsilon = 1e-8
        self.create_target_network = True
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = False
        # self.l2_regularization = 1e-2


class MADDPGActorNetworkParameters(NetworkParameters):
    def __init__(self, use_batchnorm=False):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(batchnorm=use_batchnorm)}
        self.middleware_parameters = FCMiddlewareParameters(batchnorm=use_batchnorm)
        self.heads_parameters = [DDPGActorHeadParameters(batchnorm=use_batchnorm)]
        self.optimizer_type = 'Adam'
        self.batch_size = 64
        self.adam_optimizer_beta2 = 0.999
        self.optimizer_epsilon = 1e-8
        self.async_training = False
        self.learning_rate = 0.0001
        self.create_target_network = True
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = False


class MADDPGAlgorithmParameters(AlgorithmParameters):
    """
    :param num_steps_between_copying_online_weights_to_target: (StepMethod)
        The number of steps between copying the online network weights to the target network weights.

    :param rate_for_copying_weights_to_target: (float)
        When copying the online network weights to the target network weights, a soft update will be used, which
        weight the new online network weights by rate_for_copying_weights_to_target

    :param num_consecutive_playing_steps: (StepMethod)
        The number of consecutive steps to act between every two training iterations

    :param use_target_network_for_evaluation: (bool)
        If set to True, the target network will be used for predicting the actions when choosing actions to act.
        Since the target network weights change more slowly, the predicted actions will be more consistent.

    :param action_penalty: (float)
        The amount by which to penalize the network on high action feature (pre-activation) values.
        This can prevent the actions features from saturating the TanH activation function, and therefore prevent the
        gradients from becoming very low.

    :param clip_critic_targets: (Tuple[float, float] or None)
        The range to clip the critic target to in order to prevent overestimation of the action values.

    :param use_non_zero_discount_for_terminal_states: (bool)
        If set to True, the discount factor will be used for terminal states to bootstrap the next predicted state
        values. If set to False, the terminal states reward will be taken as the target return for the network.
    """
    def __init__(self):
        super().__init__()
        self.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1)
        self.rate_for_copying_weights_to_target = 0.001
        self.num_consecutive_playing_steps = EnvironmentSteps(1)
        self.use_target_network_for_evaluation = False
        self.action_penalty = 0
        self.clip_critic_targets = None  # expected to be a tuple of the form (min_clip_value, max_clip_value) or None
        self.use_non_zero_discount_for_terminal_states = False


class MADDPGAgentParameters(AgentParameters):
    def __init__(self, agent_index, use_batchnorm=False):
    # def __init__(self, use_batchnorm=False, name, model, obs_shape_n, act_space_n, agent_index, local_q_func=False):
    #     self.agent_index = agent_index
        super().__init__(algorithm=MADDPGAlgorithmParameters(),
                         exploration=OUProcessParameters(),
                         memory=EpisodicExperienceReplayParameters(),
                         networks=OrderedDict([("actor"+str(agent_index), MADDPGActorNetworkParameters(use_batchnorm=use_batchnorm)),
                                               ("critic"+str(agent_index), MADDPGCriticNetworkParameters(use_batchnorm=use_batchnorm))]))


    @property
    def path(self):
        return 'rl_coach.agents.maddpg_agent:MADDPGAgent'


# MA Deep Deterministic Policy Gradients Network - https://arxiv.org/pdf/1706.02275.pdf
class MADDPGAgent(ActorCriticAgent):
    def __init__(self, agent_parameters, name, num_agents, agent_index, action_dim, parent: Union['LevelManager', 'CompositeAgent']=None, local_q_func=False):
        super().__init__(agent_parameters, parent)
        # new arguments for MARL
        self.name = name
        self.n = num_agents
        self.agent_index = agent_index
        self.action_dim = action_dim


        self.q_values = self.register_signal("Q")
        self.TD_targets_signal = self.register_signal("TD targets")
        self.action_signal = self.register_signal("actions")


    def learn_from_batch(self, batch): # we change this one, too
        actor = self.networks['actor'+str(self.agent_index)]
        critic = self.networks['critic'+str(self.agent_index)]
        act_i_dim = self.action_dim


        actor_keys = self.ap.network_wrappers['actor'+str(self.agent_index)].input_embedders_parameters.keys()
        critic_keys = self.ap.network_wrappers['critic'+str(self.agent_index)].input_embedders_parameters.keys()

        # TD error = r + discount*max(q_st_plus_1) - q_st
        next_actions, actions_mean = actor.parallel_prediction([
            (actor.target_network, batch.next_states(actor_keys)),
            (actor.online_network, batch.states(actor_keys))
        ])



        critic_inputs = copy.copy(batch.next_states(critic_keys))
        # critic_inputs['action'+str(self.agent_index)] = next_actions
        q_st_plus_1 = critic.target_network.predict(critic_inputs)[0] # this is where next Q is calculated

        # calculate the bootstrapped TD targets while discounting terminal states according to
        # use_non_zero_discount_for_terminal_states
        if self.ap.algorithm.use_non_zero_discount_for_terminal_states:
            TD_targets = batch.rewards(expand_dims=True) + self.ap.algorithm.discount * q_st_plus_1
        else:
            TD_targets = batch.rewards(expand_dims=True) + \
                         (1.0 - batch.game_overs(expand_dims=True)) * self.ap.algorithm.discount * q_st_plus_1

        # clip the TD targets to prevent overestimation errors
        if self.ap.algorithm.clip_critic_targets:
            TD_targets = np.clip(TD_targets, *self.ap.algorithm.clip_critic_targets)

        self.TD_targets_signal.add_sample(TD_targets)

        # get the gradients of the critic output with respect to the "mean action" that is calculated from current on-policy
        critic_inputs = copy.copy(batch.states(critic_keys))
        critic_inputs['action_n'] = copy.copy(batch.states(['mean_action_n']))
        action_gradients_full = critic.online_network.predict(critic_inputs,
                                                         outputs=critic.online_network.gradients_wrt_inputs[1]['action_n'])

        # extract the gradient with respect to the local a^i only
        action_gradients = action_gradients_full[:, self.agent_index*act_i_dim:(self.agent_index+1)*act_i_dim]

        # train the critic, we do not need to change anything about the action_n, since it is already in the batch data
        critic_inputs = copy.copy(batch.states(critic_keys))
        # critic_inputs['action'] = batch.actions(len(batch.actions().shape) == 1)

        # also need the inputs for when applying gradients so batchnorm's update of running mean and stddev will work
        result = critic.train_and_sync_networks(critic_inputs, TD_targets, use_inputs_for_apply_gradients=True)
        total_loss, losses, unclipped_grads = result[:3]

        # apply the gradients from the critic to the actor
        initial_feed_dict = {actor.online_network.gradients_weights_ph[0]: -action_gradients}
        gradients = actor.online_network.predict(batch.states(actor_keys),
                                                 outputs=actor.online_network.weighted_gradients[0],
                                                 initial_feed_dict=initial_feed_dict)

        # also need the inputs for when applying gradients so batchnorm's update of running mean and stddev will work
        if actor.has_global:
            actor.apply_gradients_to_global_network(gradients, additional_inputs=copy.copy(batch.states(critic_keys)))
            actor.update_online_network()
        else:
            actor.apply_gradients_to_online_network(gradients, additional_inputs=copy.copy(batch.states(critic_keys)))

        return total_loss, losses, unclipped_grads

    def train_multiagent(self, agents): # we overwrite train() to handle the multi-agent case
        # return Agent.train(self)
        loss = 0
        if self._should_train():
            if self.ap.is_batch_rl_training:
                # when training an agent for generating a dataset in batch-rl, we don't want it to be counted as part of
                # the training epochs. we only care for training epochs in batch-rl anyway.
                self.training_epoch += 1
            for network in self.networks.values():
                network.set_is_training(True)

            # At the moment we only support a single batch size for all the networks
            networks_parameters = list(self.ap.network_wrappers.values())
            assert all(net.batch_size == networks_parameters[0].batch_size for net in networks_parameters)


            batch_size = networks_parameters[0].batch_size

            # get prepared for sample_with_index
            transitions_idx = np.random.randint(self.num_transitions_in_complete_episodes(), size=batch_size)

            # get prepared for get_shuffled_training_data_generator_with_index
            # we suppose that all agents having the same get_last_training_set_transition_id
            shuffled_transition_indices = list(range(self.memory.get_last_training_set_transition_id()))
            random.shuffle(shuffled_transition_indices)


            # we either go sequentially through the entire replay buffer in the batch RL mode,
            # or sample randomly for the basic RL case.
            training_schedules = []
            for i in range(self.n):
                if self.ap.is_batch_rl_training:
                    training_schedules.append(
                        agents[i].call_memory('get_shuffled_training_data_generator_with_index', batch_size,
                                              shuffled_transition_indices))
                else:
                    training_schedules.append([agents[i].call_memory('sample_with_index', transitions_idx) for _ in range(self.ap.algorithm.num_consecutive_training_steps)])


            training_schedule = training_schedules[self.agent_index] # get its own training_schedule
            # tmp_obs = np.array([])
            # tmp_act = np.array([])
            # tmp_next_obs = np.array([])
            # tmp_next_act = np.array([])
            # for i in range(self.n):
            #     actor_i = agents[i].networks['actor'+str(i)]
            #     tmp_next_act_all = actor_i.parallel_prediction(
            #         [(actor_i.online_network, training_schedules[i].states('observation'))])
            #     for tmp_batch in training_schedules[i]:
            #         tmp_obs = np.concatenate((tmp_obs, tmp_batch.state['observation']), axis=0) if tmp_obs.size else tmp_batch.state['observation']
            #         tmp_act = np.concatenate((tmp_act, tmp_batch.action), axis=0) if tmp_act.size else tmp_batch.action
            #         tmp_next_obs = np.concatenate((tmp_next_obs, tmp_batch.state['observation']), axis=0) if tmp_next_obs.size else tmp_batch.state['observation']
            #         tmp_next_act = np.concatenate((tmp_next_act, tmp_batch.state['observation']), axis=0) if tmp_next_obs.size else tmp_batch.state['observation']

            tmp_curr_mean_act_all = []
            tmp_next_act_all = []
            for i in range(self.n):
                actor_i = agents[i].networks['actor' + str(i)]
                actor_keys = agents[i].ap.network_wrappers['actor'+str(i)].input_embedders_parameters.keys()
                tmp_curr_mean_act_all_i, tmp_next_act_all_i = actor_i.parallel_prediction(
                    [(actor_i.online_network, training_schedules[i].states(actor_keys)), (actor_i.target_network, training_schedules[i].next_states(actor_keys))])
                tmp_curr_mean_act_all.append(tmp_curr_mean_act_all_i)
                tmp_next_act_all.append(tmp_next_act_all_i)

            # update the training_schedule of the current agent
            for t in len(training_schedule):
                tmp_obs = np.array([])
                tmp_act = np.array([])
                tmp_curr_mean_act = np.array([])
                tmp_next_obs = np.array([])
                tmp_next_act = np.array([])
                for i in range(self.n):
                    # for tmp_batch in training_schedules[i]:
                    tmp_batch = training_schedules[i]
                    tmp_obs = np.concatenate((tmp_obs, tmp_batch.state['observation']), axis=0) if tmp_obs.size else \
                    tmp_batch.state['observation']
                    tmp_act = np.concatenate((tmp_act, tmp_batch.action),
                                             axis=0) if tmp_act.size else tmp_batch.action
                    tmp_curr_mean_act = np.concatenate((tmp_curr_mean_act, tmp_curr_mean_act_all[i][t]),
                                                  axis=0) if tmp_next_obs.size else tmp_curr_mean_act_all[i][t]
                    tmp_next_obs = np.concatenate((tmp_next_obs, tmp_batch.state['observation']),
                                                  axis=0) if tmp_next_obs.size else tmp_batch.state['observation']
                    tmp_next_act = np.concatenate((tmp_next_act, tmp_next_act_all[i][t]),
                                                  axis=0) if tmp_next_obs.size else tmp_next_act_all[i][t]

                # note that the difference between action_n and mean_action_n is that the former is from the batch data (off-policy); while the latter comes from the current network
                training_schedule[t].state['observation_n'] = tmp_obs
                training_schedule[t].state['action_n'] = tmp_act
                training_schedule[t].state['mean_action_n'] = tmp_curr_mean_act
                # training_schedule[t].action = tmp_act
                # we include both the joint observation and joint action in the "next_state"
                training_schedule[t].next_state['observation_n'] = tmp_next_obs
                training_schedule[t].next_state['action_n'] = tmp_next_act
                # new_info = {'action': tmp_act}
                # training_schedule[t].update_info(new_info)


            for batch in training_schedule:
                # update counters
                self.training_iteration += 1
                if self.pre_network_filter is not None:
                    batch = self.pre_network_filter.filter(batch, update_internal_state=False, deep_copy=False)

                # if the batch returned empty then there are not enough samples in the replay buffer -> skip
                # training step
                if len(batch) > 0:
                    # train
                    batch = Batch(batch)
                    total_loss, losses, unclipped_grads = self.learn_from_batch(batch)
                    loss += total_loss

                    self.unclipped_grads.add_sample(unclipped_grads)

                    # TODO: this only deals with the main network (if exists), need to do the same for other networks
                    #  for instance, for DDPG, the LR signal is currently not shown. Probably should be done through the
                    #  network directly instead of here
                    # decay learning rate
                    if 'main' in self.ap.network_wrappers and \
                            self.ap.network_wrappers['main'].learning_rate_decay_rate != 0:
                        self.curr_learning_rate.add_sample(self.networks['main'].sess.run(
                            self.networks['main'].online_network.current_learning_rate))
                    else:
                        self.curr_learning_rate.add_sample(networks_parameters[0].learning_rate)

                    if any([network.has_target for network in self.networks.values()]) \
                            and self._should_update_online_weights_to_target():
                        for network in self.networks.values():
                            network.update_target_network(self.ap.algorithm.rate_for_copying_weights_to_target)

                        self.agent_logger.create_signal_value('Update Target Network', 1)
                    else:
                        self.agent_logger.create_signal_value('Update Target Network', 0, overwrite=False)

                    self.loss.add_sample(loss)

                    if self.imitation:
                        self.log_to_screen()

            if self.ap.visualization.dump_csv and \
                    self.parent_level_manager.parent_graph_manager.time_metric == TimeTypes.Epoch:
                # in BatchRL, or imitation learning, the agent never acts, so we have to get the stats out here.
                # we dump the data out every epoch
                self.update_log()

            for network in self.networks.values():
                network.set_is_training(False)

            # run additional commands after the training is done
            self.post_training_commands()

        return loss


    def choose_action(self, curr_state):
        if not (isinstance(self.spaces.action, BoxActionSpace) or isinstance(self.spaces.action, GoalsSpace)):
            raise ValueError("DDPG works only for continuous control problems") # box means continuous action space
        # convert to batch so we can run it through the network
        tf_input_state = self.prepare_batch_for_inference(curr_state, 'actor' + str(self.agent_index))
        if self.ap.algorithm.use_target_network_for_evaluation:
            actor_network = self.networks['actor' + str(self.agent_index)].target_network
        else:
            actor_network = self.networks['actor' + str(self.agent_index)].online_network

        action_values = actor_network.predict(tf_input_state).squeeze()

        action = self.exploration_policy.get_action(action_values)

        self.action_signal.add_sample(action)

        # get q value
        tf_input_state = self.prepare_batch_for_inference(curr_state, 'critic' + str(self.agent_index))
        action_batch = np.expand_dims(action, 0)
        if type(action) != np.ndarray:
            action_batch = np.array([[action]])
        tf_input_state['action'] = action_batch
        q_value = self.networks['critic'+ str(self.agent_index)].online_network.predict(tf_input_state)[0]
        self.q_values.add_sample(q_value)

        action_info = ActionInfo(action=action,
                                 action_value=q_value)

        return action_info