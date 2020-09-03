import math
import sys
import gym
import cv2
import numpy as np
import copy

from typing import Dict
from collections import namedtuple

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation.observation_function import ObservationFunction
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy

from smarts.core.sensors import Observation
from smarts.core.controllers import ActionSpaceType


Config = namedtuple(
    "Config", "name, agent, interface, policy, learning, other, trainer, spec"
)


SPACE_LIB = dict(
    distance_to_center=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
    heading_errors=lambda shape: gym.spaces.Box(low=-1.0, high=1.0, shape=shape),
    speed=lambda shape: gym.spaces.Box(low=-330.0, high=330.0, shape=shape),
    steering=lambda shape: gym.spaces.Box(low=-1.0, high=1.0, shape=shape),
    goal_relative_pos=lambda shape: gym.spaces.Box(low=-1e2, high=1e2, shape=shape),
    neighbor=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
    img_gray=lambda shape: gym.spaces.Box(low=0.0, high=1.0, shape=shape),
)


def _cal_angle(vec):
    if vec[1] < 0:
        base_angle = math.pi
        base_vec = np.array([-1.0, 0.0])
    else:
        base_angle = 0.0
        base_vec = np.array([1.0, 0.0])

    cos = vec.dot(base_vec) / np.sqrt(vec.dot(vec) + base_vec.dot(base_vec))
    angle = math.acos(cos)
    return angle + base_angle


def _get_closest_vehicles(ego, neighbor_vehicles, n):
    ego_pos = ego.position[:2]
    groups = {i: (None, 1e10) for i in range(n)}
    partition_size = math.pi * 2.0 / n
    # get partition
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        # calculate its partitions
        angle = _cal_angle(rel_pos_vec)
        i = int(angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        if dist < groups[i][1]:
            groups[i] = (v, dist)

    return groups


class ActionSpace:
    @staticmethod
    def from_type(action_type: int):
        space_type = ActionSpaceType(action_type)
        if space_type == ActionSpaceType.Continuous:
            return gym.spaces.Box(
                low=np.array([0.0, 0.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32,
            )
        elif space_type == ActionSpaceType.Lane:
            return gym.spaces.Discrete(4)
        else:
            raise NotImplementedError


class EasyOBSFn(ObservationFunction):
    @staticmethod
    def filter_obs_dict(agent_obs: dict, agent_id):
        res = copy.copy(agent_obs)
        res.pop(agent_id)
        return res

    @staticmethod
    def filter_act_dict(policies):
        return {_id: policy.action_space for _id, policy in policies}

    def __call__(self, agent_obs, worker, base_env, policies, episode, **kw):
        return {
            agent_id: {
                "own_obs": obs,
                **EasyOBSFn.filter_obs_dict(agent_obs, agent_id),
                **{f"{_id}_action": 0.0 for _id in agent_obs},
            }
            for agent_id, obs in agent_obs.items()
        }


class CalObs:
    """ Feature engineering for Observation, feature by feature.
    """

    @staticmethod
    def cal_goal_relative_pos(env_obs: Observation, **kwargs):
        ego_pos = env_obs.ego_vehicle_state.position[:2]
        goal_pos = env_obs.ego_vehicle_state.mission.goal.positions[0]

        vector = np.asarray([goal_pos[0] - ego_pos[0], goal_pos[1] - ego_pos[1]])
        space = SPACE_LIB["goal_relative_pos"](vector.shape)
        return vector / (space.high - space.low)

    @staticmethod
    def cal_distance_to_center(env_obs: Observation, **kwargs):
        """ Calculate the signed distance to the center of the current lane.
        """

        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        signed_dist_to_center = closest_wp.signed_lateral_error(ego.position)
        lane_hwidth = closest_wp.lane_width * 0.5
        norm_dist_from_center = signed_dist_to_center / lane_hwidth

        dist = np.asarray([norm_dist_from_center])
        return dist

    @staticmethod
    def cal_heading_errors(env_obs: Observation, **kwargs):
        look_ahead = kwargs["look_ahead"]
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        closest_path = waypoint_paths[closest_wp.lane_index][:look_ahead]

        heading_errors = [
            math.sin(math.radians(wp.relative_heading(ego.heading)))
            for wp in closest_path
        ]

        if len(heading_errors) < look_ahead:
            last_error = heading_errors[-1]
            heading_errors = heading_errors + [last_error] * (
                look_ahead - len(heading_errors)
            )

        return np.asarray(heading_errors)

    @staticmethod
    def cal_speed(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        res = np.asarray([ego.speed])
        return res / 120.0

    @staticmethod
    def cal_steering(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        return np.asarray([ego.steering / 45.0])

    @staticmethod
    def cal_neighbor(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
        closest_neighbor_num = kwargs.get("closest_neighbor_num", 8)
        features = np.zeros((closest_neighbor_num, 5))
        surrounding_vehicles = _get_closest_vehicles(
            ego, neighbor_vehicle_states, n=closest_neighbor_num
        )

        heading_angle = math.radians(ego.heading + 90.0)
        ego_heading_vec = np.asarray([math.cos(heading_angle), math.sin(heading_angle)])
        for i, v in surrounding_vehicles.items():
            if v[0] is None:
                continue
            v = v[0]
            rel_pos = np.asarray(
                list(map(lambda x: x[0] - x[1], zip(v.position[:2], ego.position[:2])))
            )

            rel_dist = np.sqrt(rel_pos.dot(rel_pos))
            v_heading_angle = math.radians(v.heading)
            v_heading_vec = np.asarray(
                [math.cos(v_heading_angle), math.sin(v_heading_angle)]
            )

            ego_heading_norm_2 = ego_heading_vec.dot(ego_heading_vec)
            rel_pos_norm_2 = rel_pos.dot(rel_pos)
            v_heading_norm_2 = v_heading_vec.dot(v_heading_vec)
            ego_cosin = ego_heading_vec.dot(rel_pos) / np.sqrt(
                ego_heading_norm_2 + rel_pos_norm_2
            )

            v_cosin = v_heading_vec.dot(rel_pos) / np.sqrt(
                v_heading_norm_2 + rel_pos_norm_2
            )

            if ego_cosin <= 0 < v_cosin:
                rel_speed = 0
            else:
                rel_speed = ego.speed * ego_cosin - v.speed * v_cosin

            ttc = min(rel_dist / max(1e-5, rel_speed), 1e3)

            features[i, :] = np.asarray(
                [rel_dist, rel_speed, ttc, rel_pos[0], rel_pos[1]]
            )

        return features.reshape((-1,))

    @staticmethod
    def cal_img_gray(env_obs: Observation, **kwargs):
        resize = kwargs["resize"]

        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

        rgb_ndarray = env_obs.top_down_rgb
        gray_scale = (
            cv2.resize(
                rgb2gray(rgb_ndarray), dsize=resize, interpolation=cv2.INTER_CUBIC
            )
            / 255.0
        )
        return gray_scale


class EasyCallbacks(DefaultCallbacks):
    """ See example from: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
    """

    def on_episode_start(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        print("episode {} started".format(episode.episode_id))
        episode.user_data["ego_speed"] = dict()
        episode.user_data["step_heading_error"] = dict()

    def on_episode_step(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        ego_speed = episode.user_data["ego_speed"]
        for _id, obs in episode._agent_to_last_raw_obs.items():
            if ego_speed.get(_id, None) is None:
                ego_speed[_id] = []
            if obs.get("speed", None) is not None:
                ego_speed[_id].append(obs["speed"])

    def on_episode_end(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        ego_speed = episode.user_data["ego_speed"]
        mean_ego_speed = {
            _id: np.mean(speed_hist) for _id, speed_hist in ego_speed.items()
        }

        distance_travelled = {
            _id: np.mean(info["score"])
            for _id, info in episode._agent_to_last_info.items()
        }

        speed_list = list(map(lambda x: round(x, 3), mean_ego_speed.values()))
        dist_list = list(map(lambda x: round(x, 3), distance_travelled.values()))
        reward_list = list(map(lambda x: round(x, 3), episode.agent_rewards.values()))

        for _id, speed in mean_ego_speed.items():
            episode.custom_metrics[f"mean_ego_speed_{_id}"] = speed
        for _id, distance in distance_travelled.items():
            episode.custom_metrics[f"distance_travelled_{_id}"] = distance

        print(
            f"episode {episode.episode_id} ended with {episode.length} steps: [mean_speed]: {speed_list} [distance_travelled]: {dist_list} [reward]: {reward_list}"
        )


class ActionAdapter:
    @staticmethod
    def from_type(action_type):
        space_type = ActionSpaceType(action_type)
        if space_type == ActionSpaceType.Continuous:
            return ActionAdapter.continuous_action_adapter
        elif space_type == ActionSpaceType.Lane:
            return ActionAdapter.discrete_action_adapter
        else:
            raise NotImplementedError

    @staticmethod
    def continuous_action_adapter(model_action):
        assert len(model_action) == 3
        return np.asarray(model_action)

    @staticmethod
    def discrete_action_adapter(model_action):
        assert model_action in [0, 1, 2, 3]
        return model_action


def _update_obs_by_item(
    ith, obs_placeholder: dict, tuned_obs: dict, space_dict: gym.spaces.Dict
):
    for key, value in tuned_obs.items():
        if obs_placeholder.get(key, None) is None:
            obs_placeholder[key] = np.zeros(space_dict[key].shape)
        obs_placeholder[key][ith] = value


def _cal_obs(env_obs: Observation, space, **kwargs):
    obs = dict()
    for name in space.spaces:
        if hasattr(CalObs, f"cal_{name}"):
            obs[name] = getattr(CalObs, f"cal_{name}")(env_obs, **kwargs)
    return obs


def subscribe_features(**kwargs):
    res = dict()

    for k, config in kwargs.items():
        if bool(config):
            res[k] = SPACE_LIB[k](config)

    return res


def get_observation_adapter(observation_space, **kwargs):
    def observation_adapter(env_obs):
        obs = dict()
        if isinstance(env_obs, list) or isinstance(env_obs, tuple):
            for i, e in enumerate(env_obs):
                temp = _cal_obs(e, observation_space, **kwargs)
                _update_obs_by_item(i, obs, temp, observation_space)
        else:
            temp = _cal_obs(env_obs, observation_space, **kwargs)
            _update_obs_by_item(0, obs, temp, observation_space)
        return obs

    return observation_adapter


def default_info_adapter(shaped_reward: float, raw_info: dict):
    return raw_info


def get_submission_num(scenario_root):
    previous_path = sys.path.copy()
    sys.path.append(str(scenario_root))

    import scenario

    sys.modules.pop("scenario")
    sys.path = previous_path

    if hasattr(scenario, "agent_missions"):
        return len(scenario.agent_missions)
    else:
        return -1
