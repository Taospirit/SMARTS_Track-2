import os
import argparse

import gym

from pathlib import Path

from smarts.core.agent import AgentPolicy
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import DiscreteAction

from utils.common import get_submission_num


WORK_SPACE = os.path.dirname(os.path.realpath(__file__))


class KeeplanePolicy(AgentPolicy):
    def act(self, obs):
        return DiscreteAction.keep_lane


def parse_args():
    parser = argparse.ArgumentParser(
        "Simple multi-agent case with lane following control."
    )
    parser.add_argument(
        "--scenario", type=str, help="Path to scenario",
    )
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )

    return parser.parse_args()


def main(_args):
    scenario_path = Path(args.scenario).absolute()
    mission_num = get_submission_num(scenario_path)

    if mission_num == -1:
        mission_num = 1

    AGENT_IDS = [f"AGENT-{i}" for i in range(mission_num)]

    agent_interface = AgentInterface.from_type(AgentType.Laner)

    agent_specs = [
        AgentSpec(interface=agent_interface, policy_builder=lambda: KeeplanePolicy())
        for _ in range(mission_num)
    ]

    agents = dict(zip(AGENT_IDS, agent_specs))

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[scenario_path],
        agent_specs=agents,
        headless=_args.headless,
        visdom=False,
        seed=42,
    )

    agents = {_id: agent_spec.build_agent() for _id, agent_spec in agents.items()}

    import webbrowser
    webbrowser.open('http://localhost:8081/')
    
    for ie in range(30):
        step = 0
        print(f"\n---- Starting episode: {ie}...")
        observations = env.reset()
        total_reward = 0.0
        dones = {"__all__": False}

        while not dones["__all__"]:
            step += 1
            agent_actions = {
                _id: agents[_id].act(obs) for _id, obs in observations.items()
            }
            observations, rewards, dones, _ = env.step(agent_actions)
            total_reward += sum(rewards.values())

            if (step + 1) % 10 == 0:
                print(f"* Episode: {ie} * step: {step} * acc-Reward: {total_reward}")
        print("Accumulated reward:", total_reward)

    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
