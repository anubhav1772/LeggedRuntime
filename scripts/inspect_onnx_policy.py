import numpy as np

from legged_runtime.policies.onnx_policy import ONNXPolicy


policy = ONNXPolicy("checkpoints/g1/flat/policy.onnx")

print("Observation dimension:", policy.obs_dim)
print("Action dimension:", policy.action_dim)

action = policy(np.zeros(policy.obs_dim))

print(action)
