""" Required submission file

In this file, you should implement your `AgentSpec` instance, and **must** name it as `agent_spec`. 
As an example, this file offers a standard implementation.
"""

import pickle
import gym

from pathlib import Path

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent import AgentPolicy, AgentSpec

from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from config import make_config

tf = try_import_tf()


class PolicyWrapper(AgentPolicy):
    def __init__(self, loader, params):
        self._policy = loader(*params)
        self._prep = None

    def set_preprocessor(self, prep):
        self._prep = prep

    def set_weights(self, weights):
        self._policy.set_weights(weights)

    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        if isinstance(obs, list):
            # batch infer
            obs = [self._prep.transform(o) for o in obs]
            action = self._policy.compute_actions(obs, explore=False)[0]
        else:
            # single infer
            obs = self._prep.transform(obs)
            action = self._policy.compute_actions([obs], explore=False)[0][0]

        return action


class RLlibTFCheckpointPolicy:
    def __init__(
        self, load_path, algorithm, policy_names, observation_space, action_space
    ):
        self._checkpoint_path = load_path
        self._algorithm = algorithm
        self._policy_mapping = dict.fromkeys(policy_names, None)
        self._observation_space = observation_space
        self._action_space = action_space
        self._sess = None

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupported action space")

        if self._sess:
            return

        if self._algorithm == "PPO":
            from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy as LoadPolicy
        elif self._algorithm in ["A2C", "A3C"]:
            from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy as LoadPolicy
        elif self._algorithm == "PG":
            from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy as LoadPolicy
        elif self._algorithm == "DQN":
            from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy as LoadPolicy
        else:
            raise TypeError("Unsupport algorithm")

        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        self._sess = tf.Session(graph=tf.Graph())
        self._sess.__enter__()

        objs = pickle.load(open(self._checkpoint_path, "rb"))
        objs = pickle.loads(objs["worker"])
        state = objs["state"]

        for name in self._policy_mapping:
            with tf.variable_scope(name):
                # obs_space need to be flattened before passed to PPOTFPolicy
                flat_obs_space = self._prep.observation_space
                self._policy_mapping[name] = PolicyWrapper(
                    LoadPolicy, params=(flat_obs_space, self._action_space, {})
                )
                self._policy_mapping[name].set_preprocessor(self._prep)
                weights = state[name]
                self._policy_mapping[name].set_weights(weights)

    def policies(self):
        return self._policy_mapping


config = make_config(use_stacked_observation=True, use_rgb=False, action_type=1,)
load_path = "checkpoint_1/checkpoint-1"


# load saved model
# NOTE: the saved model includes two agent policy model with name AGENT-0 and AGENT-1 respectively
policy_handler = RLlibTFCheckpointPolicy(
    Path(__file__).parent / load_path,
    "DQN",
    [f"AGENT-{i}" for i in range(2)],
    config.spec["obs"],
    config.spec["act"],
)

# Agent specs in your submission must be correlated to each scenario type, in other words, one agent spec for one scenario type.
# DO NOT MODIFY THIS OBJECT !!!
scenario_dirs = [
    "crossroads",
    "double_merge",
    "ramp",
    "roundabout",
    "t_junction",
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = AgentSpec(
        **config.agent, interface=AgentInterface(**config.interface)
    )
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`
    agent_specs[k].policy_builder = lambda: policy_handler.policies()[f"AGENT-{i % 2}"]


__all__ = ["agent_specs"]
