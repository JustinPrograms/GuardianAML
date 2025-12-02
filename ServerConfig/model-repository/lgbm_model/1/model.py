import os
import sys

# Allow vendored deps in ./python (pip install --target ./python ...)
_here = os.path.dirname(__file__)
_vendor_path = os.path.join(_here, "python")
if os.path.isdir(_vendor_path):
    sys.path.insert(0, _vendor_path)

import numpy as np
import joblib
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        repo = args["model_repository"]
        version = args["model_version"]
        name = args["model_name"]
        if os.path.basename(repo) == name:
            repo = os.path.dirname(repo)
        model_path = os.path.join(repo, name, version, "model.joblib")
        self.model = joblib.load(model_path)

    def execute(self, requests):
        responses = []
        for request in requests:
            tensor = pb_utils.get_input_tensor_by_name(request, "input__0")
            inp = tensor.as_numpy()
            if inp.ndim == 1:
                inp = inp.reshape(1, -1)
            preds = self.model.predict(inp)
            outputs = [pb_utils.Tensor("output_label", preds.astype(np.int64).reshape(-1, 1))]
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(inp).astype(np.float32)
                outputs.append(pb_utils.Tensor("output_probability", probs))
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
        return responses
