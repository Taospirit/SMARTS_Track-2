import argparse
import gym

from pathlib import Path
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType

from randompolicy import RandomPolicy
from utils.common import get_submission_num


def parse_args():
    parser = argparse.ArgumentParser("Train on multi scenarios")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        type=str,
        help="Scenario dir includes multiple scenarios.",
    )
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main(args):
    scenarios = [Path(e).absolute() for e in args.scenarios]
    agent_interface = AgentInterface.from_type(AgentType.StandardWithAbsoluteSteering)
    agent_num_list = [get_submission_num(s) for s in scenarios]

    min_v, max_v = max(agent_num_list), min(agent_num_list)
    assert min_v == max_v, "Mission number mismatch."
    AGENT_IDS = [f"AGENT-{i}" for i in range(max_v)]
    agents = {
        _id: AgentSpec(interface=agent_interface, policy_builder=lambda: RandomPolicy())
        for _id in AGENT_IDS
    }

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
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

        print(f"load scenario={env.scenario_log['scenario_map']}")

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
