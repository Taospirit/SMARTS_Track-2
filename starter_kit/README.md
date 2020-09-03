# starter-kit 

The starter-kit includes multiple training examples, a submission example.

## Structure

**Training Examples**: We offer some training examples listed in the `starter_kit/train`.
- `keeplane.py`: muti-agent learning with LaneFollowing policy
- `multi_instance.py`: run multiple training and evaluation instance concurrently with Ray.
- `randompolicy.py`: random policy multi-agent learning with random policy
- `rllib_marl.py`: RLlib-based multi-agent learning, with PPO algorithm

**Submission Example**: We offer a submission example locates in `stater_kit/submission`. To submit your solution, you must include the following
files in your submission:

- `agent.py`: you **must** implement this file in your submission, since our evaluation tool will import your agent model from this file. For the implementation, you must offer an `AgentSpec` instance named `agent_spec`. More details please read the example `starter_kit/submission/agent.py`.
- a checkpoint directory: the checkpoint directory includes your saved model.

**NOTE**: to submit your solution to our platform, you must compress it as a zip file.


## Prerequisites

**Environment**: `SMARTS` is developed for `Ubuntu>=16.04` and `macOS>=10.15`, but not available for WSL1 and WSL2. If you wanna run `SMARTS` on the Windows System, some prerequisites need to be met: (1) system version >= 10; (2) install it via Docker(>=19.03.7). We provide the Dockerfile, you can build an image with it or pull from `drive-ml:5000/competition/smarts-codalab` (more details can be found on the page of Participat::Get Data). For the python environment, the `Conda` or `Virtualenv` is recommended. If not, you need to resolve an issue of Docs with the following guidance.

**Setup**: Install with the guidance of `setup.md`. If you complete the installation in a bare Python environment (neither `Conda` or `Virtualenv`), you need to resolve the Docs issue:

```bash
Error: No docs found, try running:
    make docs
```

That means `scl` cannot find the `smarts_docs` in `/usr`, instead in `/usr/local`. You can fix this error with soft link: `ln -s /usr/local/smarts_docs /usr/smarts_docs`, then `scl docs` will work succesfully.

## Quick Start

```bash
# we provide `public_dataset`, you can download it from the **Participate/Data** page, then compile it locally.
cp ${directory_of_your_dowloaded_dataset_public}/dataset_public.zip ./
unzip dataset_public.zip

# if this command throw error or choked, that means you haven't enough resources to compile it concurrently.
# an alternative approach is to compile the scenarios one by one: `scl scenario build ${scenario_dir}`
scl scenario build-all dataset_public

# open one tab to run envision, or using scl envision start -s dataset_public
scl envision start -s dataset_public -p 8081

# run simulation
python keeplane.py --scenario dataset_public/crossroads/2lane

# open localhost:8081 to see render
# open localhost:8082 to see SMARTS docs to get to know more details
scl docs
```

## !!!Important!!!: Submission

In Track 2, there are 5 scenario types as the `dataset_public` shows. Specifically, they are: _crossroads_, _double_merge_, _ramp_, _roundabout_ and _t_junction_ respectively. For your submission, you **must** implement an `dict` instance named `agent_specs` which specifies 5 `AgentSpec` instances for different scenario types. e.g.

```python
agent_specs = {
    "crossroads": AgentSpec(...),
    "double_merge": AgentSpec(...),
    "ramp": AgentSpec(...),
    "roundabout": AgentSpec(...),
    "t_junction": AgentSPec(...),
}
```

Our evaluation program will extract `AgentSpec` with given scenario key, like: `agent_specs["crossroads"]`, so, **DO NOT** modify the original key space.

**Enjoy it !**
