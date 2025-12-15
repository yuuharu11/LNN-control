# Copyright 2022 Mathias Lechner and Ramin Hasani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
from ..sequence_module import SequenceModule


class LTCCell(SequenceModule):
    def __init__(
        self,
        wiring,
        in_features=None,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        implicit_param_constraints=False,
        digital_RRAM_quantization: Optional[int] = None,
        digital_SRAM_quantization: Optional[int] = None,
        weight_quantization: Optional[int] = None,
        CAM_quantization: Optional[int] = None,
        LUT_quantization: Optional[int] = None,
        log_path: Optional[str] = None,
    ):
        """A `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_ cell.

        .. Note::
            This is an RNNCell that process single time-steps. To get a full RNN that can process sequences see `ncps.torch.LTC`.


        :param wiring:
        :param in_features:
        :param input_mapping:
        :param output_mapping:
        :param ode_unfolds:
        :param epsilon:
        :param implicit_param_constraints:
        """
        super(LTCCell, self).__init__()
        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'in_features' or call the 'wiring.build()'."
            )
        self.make_positive_fn = (
            nn.Softplus() if implicit_param_constraints else nn.Identity()
        )
        self._implicit_param_constraints = implicit_param_constraints
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._clip = torch.nn.ReLU()
        self._allocate_parameters()
        self.digital_RRAM_quantization = digital_RRAM_quantization
        self.digital_SRAM_quantization = digital_SRAM_quantization
        self.quantize_debug: bool = True
        self.weight_quantization = weight_quantization
        self.CAM_quantization = CAM_quantization
        self.LUT_quantization = LUT_quantization
        self.log_path = log_path

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @torch.no_grad()
    def dump_lut_values(
        self,
        activations: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        path: Optional[str] = None,
        bins: int = 64,
        append: bool = True,
    ):
        import json, os
        from json import JSONDecodeError
        path = path or self.log_path
        if path is None:
            return  # 保存先が未指定なら何もしない

        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        def stats(x: torch.Tensor):
            return {
                "min": float(x.min()),
                "max": float(x.max()),
                "mean": float(x.mean()),
                "hist": torch.histc(x, bins=bins, min=0.0, max=1.0).cpu().tolist(),
            }

        record = {
            "mu_minmax": [float(mu.min()), float(mu.max())],
            "sigma_minmax": [float(sigma.min()), float(sigma.max())],
            "activations": stats(activations),
        }

        buf = []
        if append and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    buf = json.load(f)
                if not isinstance(buf, list):
                    buf = [buf]
            except JSONDecodeError:
                buf = []  # 壊れていたら作り直す
        buf.append(record)

        with open(path, "w") as f:
            json.dump(buf, f, indent=2)

    def ptq_weight_symmetric(self, params, n_bits: int = 8, name: Optional[str] = None):
        """
        Symmetric, per-tensor, fake-quantization.
        """
        assert n_bits >= 2, "n_bits must be >= 2"
        qmax = 2 ** (n_bits - 1) - 1
        with torch.no_grad():
            max_val = params.abs().max()
            if max_val.item() == 0.0:
                if self.quantize_debug and name is not None:
                    print(f"[Quantize] {name}: max=0 → skip (bits={n_bits})")
                return params
            scale = max_val / qmax
            p_q = torch.round(params / scale).clamp(-qmax, qmax) * scale
            params.copy_(p_q)
            if self.quantize_debug and name is not None:
                print(f"[Quantize] {name}: bits={n_bits} max={max_val.item():.6f} scale={float(scale):.6e} shape={tuple(params.shape)}")
        return params
    
    def add_weight(self, name, init_value, requires_grad=True):
        param = torch.nn.Parameter(init_value, requires_grad=requires_grad)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size), "sigma"
            ),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value((self.state_size, self.state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value((self.state_size, self.state_size), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(self._wiring.erev_initializer()),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_sigma"
            ),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_mu"
            ),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size), "sensory_w"
            ),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(self._wiring.sensory_erev_initializer()),
        )

        self._params["sparsity_mask"] = self.add_weight(
            "sparsity_mask",
            torch.Tensor(np.abs(self._wiring.adjacency_matrix)),
            requires_grad=False,
        )
        self._params["sensory_sparsity_mask"] = self.add_weight(
            "sensory_sparsity_mask",
            torch.Tensor(np.abs(self._wiring.sensory_adjacency_matrix)),
            requires_grad=False,
        )

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight(
                name="input_w",
                init_value=torch.ones((self.sensory_size,)),
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight(
                name="input_b",
                init_value=torch.zeros((self.sensory_size,)),
            )

        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight(
                name="output_w",
                init_value=torch.ones((self.motor_size,)),
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight(
                name="output_b",
                init_value=torch.zeros((self.motor_size,)),
            )

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state
        # cm/t is loop invariant
        cm_t = self.make_positive_fn(self._params["cm"]) / (elapsed_time / self._ode_unfolds)
        # params
        sensory_w_param = self.make_positive_fn(self._params["sensory_w"])
        w_param = self.make_positive_fn(self._params["w"])
        gleak = self.make_positive_fn(self._params["gleak"])
        vleak = self._params["vleak"]

        # weight quantization
        if self.weight_quantization is not None:
            w_param = self.ptq_weight_symmetric(w_param, n_bits=self.weight_quantization, name="w")
            sensory_w_param = self.ptq_weight_symmetric(sensory_w_param, n_bits=self.weight_quantization, name="sensory_w")
        
        # inputs CAM quantization
        if self.CAM_quantization is not None:
            inputs = self.ptq_weight_symmetric(inputs, n_bits=self.CAM_quantization, name="inputs")
            
        # digital fixed_params quantization
        if self.digital_RRAM_quantization is not None:
            cm_t = self.ptq_weight_symmetric(cm_t, n_bits= self.digital_RRAM_quantization, name="cm_t")
            vleak = self.ptq_weight_symmetric(vleak, n_bits= self.digital_RRAM_quantization, name="vleak")
            gleak = self.ptq_weight_symmetric(gleak, n_bits= self.digital_RRAM_quantization, name="gleak")

        # [LUT] calculate sigmoid activation function for sensory neurons and quantization 
        activate_inputs = self._sigmoid(inputs, self._params["sensory_mu"], self._params["sensory_sigma"])
        if self.LUT_quantization is not None:
            activate_inputs = self.ptq_weight_symmetric(activate_inputs, n_bits=self.LUT_quantization, name="sensory_w_activation")

        # [MVM] We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = sensory_w_param * activate_inputs
        sensory_w_activation = (sensory_w_activation * self._params["sensory_sparsity_mask"])

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)
        
        for t in range(self._ode_unfolds):
            # [CAM/LUT] quantization inside the loop for v_pre
            if self.CAM_quantization is not None:
                v_pre = self.ptq_weight_symmetric(v_pre, n_bits=self.CAM_quantization, name=f"v_pre_step{t}")
            activate_v_pre = self._sigmoid(v_pre, self._params["mu"], self._params["sigma"])
            if self.LUT_quantization is not None:
                activate_v_pre = self.ptq_weight_symmetric(activate_v_pre, n_bits=self.LUT_quantization, name=f"w_activation_step{t}")
                if torch.rand(1).item() < 0.001:
                    self.dump_lut_values(activate_v_pre, self._params["mu"], self._params["sigma"], path=self.log_path, bins=100, append=True)
            w_activation = w_param * activate_v_pre

            if self.digital_SRAM_quantization is not None:
                state = self.ptq_weight_symmetric(
                    state,
                    n_bits=self.digital_SRAM_quantization,
                    name=(f"state"),
                )

            w_activation = w_activation * self._params["sparsity_mask"]

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = cm_t * state + gleak * vleak + w_numerator
            denominator = cm_t + gleak + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)
        self.quantize_debug = False  # Reset after one step
        return v_pre

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0 : self.motor_size]  # slice

        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

    def apply_weight_constraints(self):
        if not self._implicit_param_constraints:
            # In implicit mode, the parameter constraints are implemented via
            # a softplus function at runtime
            self._params["w"].data = self._clip(self._params["w"].data)
            self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
            self._params["cm"].data = self._clip(self._params["cm"].data)
            self._params["gleak"].data = self._clip(self._params["gleak"].data)

    def forward(self, inputs, states, elapsed_time=1.0):
        # Regularly sampled mode (elapsed time = 1 second)
        inputs = self._map_inputs(inputs)

        next_state = self._ode_solver(inputs, states, elapsed_time)

        outputs = self._map_outputs(next_state)

        return outputs, next_state
