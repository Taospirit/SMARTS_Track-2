import os

from smarts.sstudio import gen_traffic, gen_missions
from smarts.sstudio.types import (
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    Mission,
    Distribution,
    LaneChangingModel,
    JunctionModel
)

scenario = os.path.dirname(os.path.realpath(__file__))

agent_missions = [
    Mission(Route(begin=("edge-west-WE", 0, 100), end=("edge-east-WE", (1,), 40))),
    Mission(Route(begin=("edge-south-SN", 0, 50), end=("edge-west-EW", (0,), 40))),
    Mission(Route(begin=("edge-east-EW", 0, 50), end=("edge-west-EW", (1,), 40))),
    Mission(Route(begin=("edge-north-NS", 0, 50), end=("edge-south-NS", (1,), 30)))
]

gen_missions(scenario, agent_missions, overwrite=True)

impatient_car = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=0.5),
    lane_changing_model=LaneChangingModel(impatience=1, cooperative=0.25),
    junction_model=JunctionModel(
        drive_after_red_time=1.5, drive_after_yellow_time=1.0, impatience=1.0
    ),
)

patient_car = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=0.5),
    lane_changing_model=LaneChangingModel(impatience=0, cooperative=0.5),
    junction_model=JunctionModel(drive_after_yellow_time=1.0, impatience=0.5),
)

vertical_routes = [
    ("north-NS", "south-NS"),
    ("south-SN", "north-SN"),
]

horizontal_routes = [
    ("west-WE", "east-WE"),
    ("east-EW", "west-EW"),
]

turn_left_routes = [
    ("south-SN", "west-EW"),
    ("west-WE", "north-SN"),
    ("north-NS", "east-WE"),
    ("east-EW", "south-NS"),
]

turn_right_routes = [
    ("south-SN", "east-WE"),
    ("west-WE", "south-NS"),
    ("north-NS", "west-EW"),
    ("east-EW", "north-SN"),
]

for name, routes in {
    "vertical": vertical_routes,
    "horizontal": horizontal_routes,
    "unprotected_left": turn_left_routes,
    "turns": turn_left_routes + turn_right_routes,
    "all": vertical_routes + horizontal_routes + turn_left_routes + turn_right_routes,
}.items():
    traffic = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"edge-{r[0]}", 0, "random"),
                    end=(f"edge-{r[1]}", 0, "random"),
                ),
                rate=1,
                actors={patient_car: 1.0},
            )
            for r in routes
        ]
    )

    gen_traffic(scenario, traffic, name=name, overwrite=True)
