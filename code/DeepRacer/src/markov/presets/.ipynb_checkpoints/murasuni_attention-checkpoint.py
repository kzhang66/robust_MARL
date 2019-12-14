from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, DistributedCoachSynchronizationType
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.observation.observation_rgb_to_y_filter import ObservationRGBToYFilter
from rl_coach.filters.observation.observation_stacking_filter import ObservationStackingFilter
from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule
from rl_coach.architectures.tensorflow_components.layers import Conv2d
import tensorflow as tf

import json

class Attention(object):
    def __init__(self, units: int):
        self.units = units

    def __call__(self, input_layer, name: str=None, kernel_initializer=None, activation=None, is_training=None):

        score_weights = tf.layers.dense(input_layer,
                                        self.units, name="Score_{}".format(name),
                                        kernel_initializer=kernel_initializer,
                                        activation=None)
        score_activation = tf.nn.tanh(score_weights)

        attention_weights = tf.layers.dense(score_activation,
                                            1,
                                            name="Attention_{}".format(name), kernel_initializer=kernel_initializer,
                                            activation=None)

        attention_weights_activation = tf.nn.softmax(attention_weights)
        context_vector = tf.multiply(attention_weights_activation, input_layer)
        context_conv = tf.reduce_sum(context_vector, axis=1, keepdims=True)
        return context_conv

    def __str__(self):
        return "Conv2dWithAttention (num outputs = {})".format(self.units)


####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(100000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(40)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = ClippedPPOAgentParameters()

agent_params.network_wrappers['main'].learning_rate = 0.0003
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = [Conv2d(32, 8, 4),
                                                                                          Conv2d(64, 4, 2),
                                                                                          Conv2d(64,3,1),
                                                                                          Attention(128)]
agent_params.network_wrappers['main'].middleware_parameters.activation_function = 'relu'
agent_params.network_wrappers['main'].batch_size = 64
agent_params.network_wrappers['main'].optimizer_epsilon = 1e-5
agent_params.network_wrappers['main'].adam_optimizer_beta2 = 0.999

agent_params.algorithm.clip_likelihood_ratio_using_epsilon = 0.2
agent_params.algorithm.clipping_decay_schedule = LinearSchedule(1.0, 0, 1000000)
agent_params.algorithm.beta_entropy = 0.01
agent_params.algorithm.gae_lambda = 0.95
agent_params.algorithm.discount = 0.999
agent_params.algorithm.optimization_epochs = 10
agent_params.algorithm.estimate_state_value_using_gae = True
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentEpisodes(20)
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentEpisodes(20)

agent_params.algorithm.distributed_coach_synchronization_type = DistributedCoachSynchronizationType.SYNC
agent_params.exploration = CategoricalParameters()

###############
# Environment #
###############
DeepRacerInputFilter = InputFilter(is_a_reference_filter=True)
DeepRacerInputFilter.add_observation_filter('observation', 'to_grayscale', ObservationRGBToYFilter())
DeepRacerInputFilter.add_observation_filter('observation', 'to_uint8', ObservationToUInt8Filter(0, 255))
DeepRacerInputFilter.add_observation_filter('observation', 'stacking',
                                              ObservationStackingFilter(1))

env_params = GymVectorEnvironment()
env_params.default_input_filter = DeepRacerInputFilter
env_params.level = 'DeepRacerRacetrackCustomActionSpaceEnv-v0'

vis_params = VisualizationParameters()
vis_params.dump_mp4 = False

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 400
preset_validation_params.max_episodes_to_achieve_reward = 10000

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
