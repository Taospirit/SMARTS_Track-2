# DAI2020_SMARTS_Competition_Track-2

|time|note|todo |version|
|--|--|--|--|
|2020.8.18|INIT|upload official files & datas||
|2020.9.3|update README || 
|2020.9.4|updata Q&A ||

## 文件说明：
`__MACOSX`: 好像为window准备的, 不用管它.

`orgin_zip_files/`：官方提供的原始zip文件, 解压出来的就是下面两个文件, 同不用管.

`starter_kit/`: 安装脚本、说明文档、smarts的whl文件等

`dataset_pulic/`：各种场景下的测试集数据

## 安装说明：
    本文档只考虑本地ubuntu安装配置，window暂未测试

0. 环境说明：
    - ubuntu 16.04
    - anconda python>=3.7

1. 新建环境
   
**由于环境要求是python3.7, 推荐使用conda先新建一个python3.7环境**：
```bash
conda create -n sumo python=3.7 # 环境取名sumo
conda activate sumo
```
2. 安装sumo和相关依赖
```bash
cd starter_kit
chmod +x install_deps.sh # 不给权限好像无法直接执行
./install_deps.sh # 官方的安装脚本
```
如果以上安装出现问题，可用下面的命令自己手动安装
```bash
sudo add-apt-repository ppa:sumo/stable && sudo apt-get update
sudo apt-get install -y \
        libspatialindex-dev \
        sumo sumo-tools sumo-doc
```

3. 安装smarts包和相关
```bash
# 注意在conda环境下
pip install --upgrade pip
# install the dependencies
cd starter_kit
pip install smarts-0.3.7-py3-none-any.whl
# install gym
pip install gym
```

4. 测试安装成功
```bash
# test sumo
sumo-gui
# test that the sim works
python train_example/keeplane_example.py --scenario xxx
```

5. 编译测试集场景(已经生成了编译文件,可跳过)
```bash
# 注：scl是安装好smarts后就会有的命令
# an alternative approach is to compile the scenarios one by one: `scl scenario build ${scenario_dir}`
scl scenario build-all dataset_public
```

## 运行环境

2. 打开envision可视化
```bash
# open one tab to run envision, or using scl envision start -s dataset_public
scl envision start -s dataset_public -p 8081
```
在本地浏览器中 `localhost:8081` 查看 `smarts` 的可视化界面. 初始化是一片空白

3. 运行demo
```bash
# run simulation
cd starter_kit
python train/keeplane.py --scenario ../dataset_public/crossroads/2lane
```

4. 查看smarts的环境api的说明文档
```bash
scl docs
```

## Q&A
### 1. ImportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so: undefined symbol: PyCObject_Type
- 这是经典的ros下的cv2不支持python3的错误.
- 需要针对python3重新编译才行,相关资料在googel直接搜索即可.
- 目前我在官方源代码中已经根据我自己环境改好了

### 2. ModuleNotFoundError: No module named 'scenario'
- 文件路径问题
- 场景没编译

### 3. sumo路径问题
```bash
echo "export SUMO_HOME="/usr/share/sumo"" >> ~/.bashrc
source ~/.bashrc
```