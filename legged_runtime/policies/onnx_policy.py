import numpy as np
import onnxruntime as ort


class ONNXPolicy:

    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape

    @property
    def obs_dim(self):
        return self.input_shape[-1]

    @property
    def action_dim(self):
        return self.output_shape[-1]

    def infer(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)

        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected {self.obs_dim} observations, got {obs.shape[-1]}")

        action = self.session.run([self.output_name], {self.input_name: obs})[0]

        return action.squeeze(0)

    def __call__(self, obs):
        return self.infer(obs)
