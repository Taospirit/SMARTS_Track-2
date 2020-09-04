import logging

import ray
import gym
import argparse

from pathlib import Path

from smarts.core.agent_interface import AgentInterface, ActionSpaceType
from smarts.core.agent import AgentSpec, AgentPolicy
from smarts.core.utils.episodes import episodes
from smarts.core.controllers import DiscreteAction

from utils.common import get_submission_num


logging.basicConfig(level=logging.INFO)


class Policy(AgentPolicy):
    def act(self, obs):
        return DiscreteAction.keep_lane


@ray.remote
def evaluate(episode, eval_scenario, agent_specs):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[eval_scenario],
        agent_specs=agent_specs,
        headless=False,
        visdom=False,
        timestep_sec=0.1,
        # If you want deterministic tests, set the seed to a constant.
        # We vary the seed because depending on the scenario, a fixed seed can
        # place the vehicle in an impossible situation.
        seed=episode,
    )

    agents = {_id: agent_spec.build_agent() for _id, agent_spec in agent_specs.items()}

    observations = env.reset()
    dones = {"__all__": False}
    while not dones["__all__"]:
        agent_actions = {_id: agents[_id].act(obs) for _id, obs in observations.items()}
        observations, rewards, dones, infos = env.step(agent_actions)

    total_evaluation_score = sum([info["score"] for info in infos.values()])
    env.close()

    # log your evaluation score / emit tensorboard metrics
    print(f"Evaluation after episode {episode}: {total_evaluation_score:.2f}")


@ray.remote
def train(args, agent_specs, eval_interval: int = None):
    scenario = Path(args.scenario).absolute()
    eval_scenario = Path(args.eval_scenario).absolute()
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[scenario],
        agent_specs=agent_specs,
        headless=False,
        visdom=False,
        timestep_sec=0.1,
    )

    agents = {_id: agent_spec.build_agent() for _id, agent_spec in agent_specs.items()}

    for episode in episodes(n=50):
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_actions = {
                _id: agents[_id].act(obs) for _id, obs in observations.items()
            }
            observations, rewards, dones, infos = env.step(agent_actions)
            episode.record_step(observations, rewards, dones, infos)

        if eval_interval and episode.index % eval_interval == 0:
            # Block for evaluation
            ray.wait([evaluate.remote(episode.index, eval_scenario, agent_specs)])
            # Optionally, instead, you can run your evaluation concurrently by omitting the `ray.wait([..])`.
            #
            #   evaluate.remote(episode.index, args.eval_scenario, agent)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hiway-multi-instance-example")
    parser.add_argument(
        "scenario",
        help="Either the scenario to run (see scenarios/ for some samples you can use) OR a directory of scenarios to sample from",
        type=str,
    )
    parser.add_argument(
        "eval_scenario",
        help="Same as `scenario`, but this scenario will be used for evaluation",
        type=str,
    )
    args = parser.parse_args()

    mission_num = get_submission_num(Path(args.scenario).absolute())
    eval_mission_num = get_submission_num(Path(args.eval_scenario).absolute())

    assert (
        mission_num == eval_mission_num
    ), f"Inconsistency between learning and evaluation on the number of missions!"
    if mission_num == -1:
        mission_num = 1
        eval_mission_num = 1

    AGENT_IDS = [f"AGENT-{i}" for i in range(mission_num)]

    agent_specs = {
        _id: AgentSpec(
            interface=AgentInterface(
                max_episode_steps=1000,
                waypoints=True,
                action=ActionSpaceType.Lane,
                debug=True,
            ),
            policy_builder=Policy,
        )
        for _id in AGENT_IDS
    }

    N_TRAINERS = 1
    EVAL_INTERVAL = 3  # run evaluation every 3 episodes

    ray.init(ignore_reinit_error=True)
    ray.wait(
        [train.remote(args, agent_specs, EVAL_INTERVAL) for _ in range(N_TRAINERS)]
    )
