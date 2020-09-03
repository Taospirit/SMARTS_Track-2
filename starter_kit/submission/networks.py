import gym

from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.models.preprocessors import get_preprocessor


tf = try_import_tf()


def observation_adapter(obs: dict):
    # convert obs to an array
    features = []
    for key, value in obs.items():
        if len(value.shape) > 3:
            continue
        features.append(value)
    features_array = tf.concat(features, axis=-1)

    # move axis: (height, width, channel)
    features_array = tf.transpose(features_array, perm=[0, 2, 1])
    res = {"feature_input": features_array}
    if obs.get("img_gray", None) is not None:
        img_gray = tf.transpose(obs["img_gray"], perm=[0, 2, 3, 1])
        res["vision_input"] = img_gray
    return res


def get_shapes(space_dict: gym.spaces.Dict):
    feature_dim, vision_shape = 0, None
    stack_size = 1
    for key, space in space_dict.spaces.items():
        if key == "img_gray":
            vision_shape = space.shape
        else:
            feature_dim += space.shape[-1]
        stack_size = space.shape[0]

    if vision_shape is not None:
        return (feature_dim, stack_size), vision_shape[1:] + (vision_shape[0],)
    else:
        return (feature_dim, stack_size), None


class DictCNN(TFModelV2):
    NAME = "DictCNN"

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        super(DictCNN, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        custom_model_config = model_config["custom_model_config"]
        obs_space_dict = kwargs.get("obs_space_dict", None) or custom_model_config.get(
            "obs_space_dict", None
        )
        assert obs_space_dict is not None

        # convert mix observation to ...
        feature_shape, vision_shape = get_shapes(obs_space_dict)
        feature_input = tf.keras.Input(feature_shape)
        vision_input = (
            tf.keras.Input(vision_shape) if vision_shape is not None else None
        )

        feature_conv_1 = tf.keras.layers.Conv1D(
            filters=16, kernel_size=3, activation=tf.nn.tanh
        )(feature_input)
        feature_flat = tf.keras.layers.Flatten()(feature_conv_1)

        vision_flat = None
        if vision_shape is not None:
            vision_conv_1 = tf.keras.layers.Conv2D(
                filters=32, kernel_size=4, strides=2, activation=tf.nn.tanh
            )(vision_input)
            vision_conv_2 = tf.keras.layers.Conv2D(
                filters=64, kernel_size=11, strides=1, activation=tf.nn.tanh
            )(vision_conv_1)
            vision_flat = tf.keras.layers.Flatten()(vision_conv_2)

        if vision_flat is not None:
            self._use_vision = True
            concat_state = tf.keras.layers.Concatenate(axis=-1)(
                [feature_flat, vision_flat]
            )
        else:
            self._use_vision = False
            concat_state = feature_flat
        value_layer = tf.keras.layers.Dense(units=1)(concat_state)
        value_layer = tf.reshape(value_layer, (-1,))
        # output_layer = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)(concat_state)
        output_layer = tf.keras.layers.Dense(units=num_outputs)(concat_state)

        self.base_model = (
            tf.keras.Model([feature_input, vision_input], [output_layer, value_layer])
            if vision_shape is not None
            else tf.keras.Model([feature_input], [output_layer, value_layer])
        )
        self.register_variables(self.base_model.variables)
        self._value_out = None

    @override(TFModelV2)
    def value_function(self):
        return self._value_out

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        obs = observation_adapter(obs)
        inputs = (
            [obs["feature_input"], obs["vision_input"]]
            if self._use_vision
            else [obs["feature_input"]]
        )
        model_out, self._value_out = self.base_model(inputs)
        return model_out, state


class FCModel(FullyConnectedNetwork):
    NAME = "FCModel"


class CCModel(TFModelV2):
    NAME = "CCModel"
    CRITIC_OBS = "CRITIC_OBS"

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        super(CCModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        # ordered dict
        agent_number = 4
        critic_obs = gym.spaces.Dict(
            {
                **{f"AGENT-{i}": obs_space for i in range(agent_number)},
                **{f"AGENT-{i}-action": action_space for i in range(agent_number)},
            }
        )

        self.critic_preprocessor = get_preprocessor(critic_obs)(critic_obs)
        self.obs_preprocessor = get_preprocessor(obs_space)(obs_space)
        self.act_preprocessor = get_preprocessor(action_space)(action_space)
        model_config["custom_model_config"] = dict()
        # inner network
        self.action_model = DictCNN(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_action",
            **kwargs,
        )
        self.value_model = FullyConnectedNetwork(
            gym.spaces.Box(low=-1e10, high=1e10, shape=self.critic_preprocessor.shape),
            action_space,
            1,
            model_config,
            name + "_vf",
        )
        self.register_variables(self.action_model.variables())
        self.register_variables(self.value_model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.action_model.forward(input_dict, state, seq_lens)

    def central_value_function(self, obs):
        # TODO(ming): make inputs as dicts is better
        value, _ = self.value_model({"obs": obs})
        return value

    @override(TFModelV2)
    def value_function(self):
        return self.model.value_function()


ModelCatalog.register_custom_model(CCModel.NAME, CCModel)
ModelCatalog.register_custom_model(DictCNN.NAME, DictCNN)
ModelCatalog.register_custom_model(FCModel.NAME, FCModel)
