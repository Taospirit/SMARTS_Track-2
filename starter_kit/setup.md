# Setup SMARTS Environment


## Setup from Dockerfile

We recommend use docker to setup SMARTS envirionment. A Dockerfile which contains dependencies of SMARTS is supplied in 
your starter-kit. Follow the example command below and create your instance.

```bash
# get docker
docker build -t smarts/dai .
# recommend also use -v to mount your working directory to docker, like -v ~/dai:~/dai
docker run -itd -p 6006:6006 -p 8081:8081 -p 8082:8082 --name smarts smarts/dai bash
docker exec -it smarts bash

# install the dependencies in the docker
pip install smarts-xxx.whl
pip install smarts-xxx.whl[train]
pip install smarts-xxx.whl[dev]

```

## Setup from scratch

To setup the environment for SMARTS, run the following commands. Currently SMARTS can be run in MAC OS system and
Ubuntu System.


```bash

# unzip the starter_kit and place somewhere convenient on your machine. (e.x. ~/src/starter_kit)

cd ~/src/starter_kit
./install_deps.sh
# ...and, follow any on-screen instructions

# test that the sumo installation worked
sumo-gui

# setup virtual environment (Python 3.7 is required)
# or you can use conda environment if you like.
python3.7 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# install the dependencies
pip install smarts-xxx.whl
pip install smarts-xxx.whl[train]
pip install smarts-xxx.whl[dev]

# download the public datasets from Codalab to ./dataset_public

# test that the sim works
python train_example/keeplane_example.py --scenario xxx

```

##  Common Questions


1. **Exception: Could not open window.**

    If you are running on a computer with GUI interface and occur this problem, **do not** try the solution below and try to use docker solution or contact us.

    Otherwise if you are running on a server without GUI, you can try the following instructions to solve it.


    ```bash
    # set DISPLAY 
    vim ~/.bashrc
    # write following command into bashrc
    export DISPLAY=":1"
    # refresh
    source ~/.bashrc

    # set xorg server
    sudo wget -O /etc/X11/xorg.conf http://xpra.org/xorg.conf
    sudo /usr/bin/Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile ./xdummy.log -config /etc/X11/xorg.conf $DISPLAY &
    ```

2. **Problems like cannot use sumo, you can export sumo path to bashrc manually:**

    ```bash
    # set SUMO HOME
    vim ~/.bashrc
    # write following command into bashrc for ubuntu
    export SUMO_HOME="/usr/share/sumo"

    # for macos
    export SUMO_HOME="/usr/local/opt/sumo/share/sumo"
    # refresh
    source ~/.bashrc
    ```

3. **Fatal server error: (EE) Server is already active for display 1**
    
    Just ignore this error. This is to make sure the Xorg is running.

4. **Can smarts simulator run on Windows?**

    Our simulator was developed for ubuntu (>=16.04) and macOS(>=10.15), but not suitable for WSL1 and WSL2. To install it on the Windows system, some prerequisites need to be met: (1) system version >= 10; (2) install it via docker (>=19.03.7).

5. **When I run scl docs, it returns error.** 
    
    The reason is that you install the smarts package without any virtual environment likes virtualenv or conda (in other words, virtualenv and conda are recommended). It will return error:

    ```bash
    Error: No docs found, try running:
        make docs
    ```

    That means scl cannot find the `smarts_docs` in `/usr/`, instead in `/usr/local/`. You can fix this error with soft link: `ln -s /usr/local/smarts_docs /usr/smarts_docs`, then `scl docs` will work successfully !

6. **Core dumped when build scenaios.**

    Since scenario building is parallel, this error means you do not have enough resources to do building cocurrently. Try to build scenario one by one.

7. **ERROR:RemoteAgentBuffer:Failed to initialize remote agent**

    Ignore it.


## Docs

To look at the documentation call:

```bash
# Browser will attempt to open on localhost:8082
scl docs
```

## NOTE

Please read the docs and [FAQ](https://competitions.codalab.org/competitions/26013#learn_the_details-faq) carefully before raising an issue in WeChat group or forums. Thanks!


