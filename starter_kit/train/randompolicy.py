import argparse
import numpy as np
import random
import gym

from pathlib import Path

from smarts.core.agent import AgentPolicy
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType

from utils.common import get_submission_num


class RandomPolicy(AgentPolicy):
    """Random policy with continuous control
    """

    def act(self, obs):
        return np.asarray(
            [random.uniform(0, 1), random.uniform(0, 1), random.uniform(-1, 1)],
            dtype=np.float32,
        )


def parse_args():
    parser = argparse.ArgumentParser("Run simple keep lane agent")
    parser.add_argument("--scenario", type=str, help="Path to scenario")
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )

    return parser.parse_args()


def main(args):
    scenario_path = Path(args.scenario).absolute()
    agent_num = get_submission_num(scenario_path)

    AGENT_IDS = [f"AGENT-{i}" for i in range(agent_num)]
    agent_interface = AgentInterface.from_type(AgentType.StandardWithAbsoluteSteering)

    agent_specs = [
        AgentSpec(interface=agent_interface, policy_builder=lambda: RandomPolicy())
        for _ in range(agent_num)
    ]

    agents = dict(zip(AGENT_IDS, agent_specs))
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[scenario_path],
        agent_specs=agents,
        headless=args.headless,
        visdom=False,
        seed=42,
    )

    agents = {_id: agent_spec.build_agent() for _id, agent_spec in agents.items()}

    for ie in range(10):
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
