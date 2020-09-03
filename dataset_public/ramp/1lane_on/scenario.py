import os

from smarts.sstudio import gen_traffic, gen_missions
from smarts.sstudio.types import (
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    Mission,
)

scenario = os.path.dirname(os.path.realpath(__file__))

agent_missions = [
    Mission(Route(begin=("left_in", 0, 80), end=("merged", (0,), 40))),
    Mission(Route(begin=("left_in", 0, 50), end=("merged", (0,), 40))),
    Mission(Route(begin=("ramp_in", 0, 80), end=("merged", (0,), 60))),
    Mission(Route(begin=("ramp_in", 0, 50), end=("merged", (0,), 60))),
]

gen_missions(scenario, agent_missions, overwrite=True)

gen_traffic(
    scenario,
    Traffic(
        flows=[
            Flow(
                route=RandomRoute(), rate=3600, actors={TrafficActor(name="car"): 1.0},
            )
        ]
    ),
    name="random",
    overwrite=True,
)
