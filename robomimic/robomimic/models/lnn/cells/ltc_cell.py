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
        clip_max: Optional[float] = None,
        clip_min: Optional[float] = None,
        implicit_param_constraints=False,
        digital_RRAM_quantization: Optional[int] = None,
        digital_SRAM_quantization: Optional[int] = None,
        weight_quantization: Optional[int] = None,
        CAM_quantization: Optional[int] = None,
        LUT_quantization: Optional[int] = None,
        calibration_path: Optional[str] = None,
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
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._clip = torch.nn.ReLU()
        self.quantize_debug: bool = True
        self.CAM_quantization = CAM_quantization
        self.LUT_quantization = LUT_quantization
        self.digital_RRAM_quantization = digital_RRAM_quantization
        self.digital_SRAM_quantization = digital_SRAM_quantization
        self.weight_quantization = weight_quantization
        self.calibration_path = calibration_path
        self._allocate_parameters()
        
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
        state: torch.Tensor,
        activations: torch.Tensor,
        path: Optional[str] = None,
        bins: int = 100,
        append: bool = True,
    ):
        import json, os
        from json import JSONDecodeError
        if path is None:
            return  

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
            "activations": stats(activations),
            "states": stats(state),
        }

        buf = []
        if append and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    buf = json.load(f)
                if not isinstance(buf, list):
                    buf = [buf]
            except JSONDecodeError:
                buf = [] 
        buf.append(record)

        with open(path, "w") as f:
            json.dump(buf, f, indent=2)
    
    def _make_positive(self):
        self._params["cm"] = self.make_positive_fn(self._params["cm"])
        # params
        self._params["sensory_w"] = self.make_positive_fn(self._params["sensory_w"])
        self._params["w"] = self.make_positive_fn(self._params["w"])
        self._params["gleak"] = self.make_positive_fn(self._params["gleak"])
        print("[Completed] Made cm, sensory_w, w, gleak parameters positive")

    # quantization function
    # for cm, vleak, gleak
    @torch.no_grad()
    def digital_ptq(
        self,
        params: torch.Tensor,
        n_bits: int = 8,
        percentile: float = 0.999,
        name: Optional[str] = None,
        signed: bool = True,
    ):
        """
        Percentile clipping + (signed / unsigned) digital fake quantization.
        No centering, no zero-point shift.
        """

        assert n_bits >= 2, "n_bits must be >= 2"

        if percentile > 1.0:
            percentile /= 100.0

        x = params.to(torch.float32)

        # --------------------------------------------------
        # 1. Percentile-based clipping
        # --------------------------------------------------
        lower = torch.quantile(x, 1.0 - percentile)
        upper = torch.quantile(x, percentile)

        if lower.item() == upper.item():
            return params

        x_clipped = x.clamp(lower, upper)

        # --------------------------------------------------
        # 2. Quantization mode
        # --------------------------------------------------
        if signed:
            qmax = 2 ** (n_bits - 1) - 1
            abs_max = x_clipped.abs().max()
            if abs_max.item() == 0:
                return params
            scale = abs_max / qmax
            x_q = torch.round(x_clipped / scale).clamp(-qmax, qmax) * scale
        else:
            qmax = 2 ** n_bits - 1
            x_clipped = x_clipped.clamp(min=0)
            max_val = torch.quantile(x_clipped, percentile)
            if max_val.item() == 0:
                return params
            scale = max_val / qmax
            x_q = torch.round(x_clipped / scale).clamp(0, qmax) * scale

        params.copy_(x_q.to(params.dtype))

        if getattr(self, "quantize_debug", False) and name is not None:
            mode = "signed" if signed else "unsigned"
            print(
                f"[Quantize-{mode}] {name}: bits={n_bits} "
                f"p{percentile*100:.2f}% scale={scale:.4e}"
            )

        return params

    # for weight
    @torch.no_grad()
    def ptq_weight_nonzero(
        self,
        params: torch.Tensor,
        n_bits: int = 8,
        percentile: float = 0.999,
        name: Optional[str] = None,
        symmetric: bool = True,
    ):
        """
        PTQ for sparse weights.
        - Only non-zero elements are quantized
        - Zero is preserved as a dedicated level
        - symmetric=True  : signed symmetric quantization
        - symmetric=False : unsigned asymmetric quantization
        """
        assert n_bits >= 2
        if percentile > 1.0:
            percentile /= 100.0

        with torch.no_grad():
            x = params.to(torch.float32)

            nz_mask = x != 0
            nz = x[nz_mask]

            if nz.numel() == 0:
                return params

            # percentile clipping (upper-tail only)
            max_clip = torch.quantile(nz, percentile)
            nz = torch.clamp(nz, max=max_clip)

            if symmetric:
                # -----------------------------
                # Symmetric (signed)
                # -----------------------------
                qmax = 2 ** (n_bits - 1) - 1

                abs_max = nz.abs().max()
                if abs_max < 1e-12:
                    return params

                scale = abs_max / qmax
                nz_q = torch.round(nz / scale).clamp(-qmax, qmax) * scale

            else:
                # -----------------------------
                # Asymmetric (unsigned)
                # -----------------------------
                qmax = 2 ** n_bits - 1

                min_val = nz.min()
                max_val = nz.max()
                if max_val - min_val < 1e-12:
                    return params

                scale = (max_val - min_val) / qmax
                nz_q = torch.round((nz - min_val) / scale).clamp(0, qmax) * scale + min_val

            # write back
            out = x.clone()
            out[nz_mask] = nz_q
            params.copy_(out.to(params.dtype))

            if getattr(self, "quantize_debug", False) and name:
                sp = 1.0 - nz.numel() / x.numel()
                mode = "sym" if symmetric else "asym"
                print(
                    f"[Quantize-NZ-{mode}] {name}: bits={n_bits} "
                    f"sp={sp*100:.1f}% scale={scale:.3e}"
                )

        return params
    
    @torch.no_grad()
    def ptq_range(
        self, 
        params: torch.Tensor, 
        n_bits: int = 8, 
        name: Optional[str] = None,
        clip_min: float = -0.5,
        clip_max: Optional[float] = 0.5,   
    ):
        # signed range
        qmax = 2 ** (n_bits) - 1

        x = params.to(torch.float32)
        
        if clip_min is None:    
            clip_min_t = x.min()
        else:
            clip_min_t = torch.tensor(float(clip_min), device=x.device, dtype=x.dtype)
        if clip_max is None:
            clip_max_t = x.max()
        else:
            clip_max_t = torch.tensor(float(clip_max), device=x.device, dtype=x.dtype)
            
        scale = (clip_max_t - clip_min_t) / qmax
        
        x_clipped = x.clamp(min=clip_min_t.item(), max=clip_max_t.item())
        q = torch.round((x_clipped - clip_min_t) / scale).clamp(0, qmax)
        x_q = q * scale + clip_min_t

        params.copy_(x_q.to(params.dtype))

        if getattr(self, "quantize_debug", False) and name:
            print(
                f"[Quantize-minmax] {name}: bits={n_bits}, "
                f"scale={scale.item():.3e}, clip_min={clip_min_t.item():.3e}, clip_max={clip_max_t.item():.3e}"
            )

        return params

    # for LUT sigmoid [0,1]
    @torch.no_grad()
    def ptq_lut_sigmoid(self, params: torch.Tensor, n_bits: int = 8, name: Optional[str] = None):
        qmax = 2 ** n_bits - 1
        scale = 1.0 / qmax
        nz_q = torch.round(params / scale).clamp(0, qmax) * scale
        params.copy_(nz_q.to(params.dtype))

        if getattr(self, "quantize_debug", False) and name:
            print(f"[Quantize] {name}: bits={n_bits}, scale={scale:.3e}")

        return params

    
    def add_weight(self, name, init_value, requires_grad=True):
        param = torch.nn.Parameter(init_value, requires_grad=requires_grad)
        self.register_parameter(name, param)
        return param

    def add_vacant_weight(self, name, shape, requires_grad=False, persistent=True):
        if requires_grad:
            param = nn.Parameter(torch.zeros(shape))
            self.register_parameter(name, param)
        else:
            buf = torch.zeros(shape, device="cuda")
            self.register_buffer(name, buf, persistent=persistent)
            param = buf
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
        self._params["w_mask"] = self.add_vacant_weight("w_mask", self._params["w"].shape, requires_grad=False, persistent=False)
        self._params["sensory_w_mask"] = self.add_vacant_weight("sensory_w_mask", self._params["sensory_w"].shape, requires_grad=False, persistent=False)
        self._params["w_rev"] = self.add_vacant_weight("w_rev", self._params["w"].shape, requires_grad=False, persistent=False)
        self._params["sensory_w_rev"] = self.add_vacant_weight("sensory_w_rev", self._params["sensory_w"].shape, requires_grad=False, persistent=False)

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

    # initial quantization 
    def _fixed_quantization(self, digital_RRAM_quantization=None):
        # digital RRAM quantization
        if digital_RRAM_quantization is not None:
            self._params["gleak"] = self.digital_ptq(self._params["gleak"], n_bits=digital_RRAM_quantization, name="gleak", signed=False) 
            self._params["cm"] = self.digital_ptq(self._params["cm"], n_bits=digital_RRAM_quantization, name="cm", signed=False)
            self._params["vleak"] = self.digital_ptq(self._params["vleak"], n_bits=digital_RRAM_quantization, name="vleak", signed=True)
         
    def _weight_calc(self):
        # weight quantization
        self._params["w_mask"].copy_(self._params["w"] * self._params["sparsity_mask"])
        self._params["sensory_w_mask"].copy_(self._params["sensory_w"] * self._params["sensory_sparsity_mask"])
        self._params["w_rev"].copy_(self._params["w_mask"] * self._params["erev"])
        self._params["sensory_w_rev"].copy_(self._params["sensory_w_mask"] * self._params["sensory_erev"])
        
    def _weight_quantization(self, weight_quantization=None):
        if weight_quantization is not None:
            self._params["w_mask"].copy_(self.ptq_weight_nonzero(self._params["w_mask"], n_bits=weight_quantization, name="w", symmetric=False))
            self._params["sensory_w_mask"].copy_(self.ptq_weight_nonzero(self._params["sensory_w_mask"], n_bits=weight_quantization, name="sensory_w", symmetric=False))
            self._params["w_rev"].copy_(self.ptq_weight_nonzero(self._params["w_rev"], n_bits=weight_quantization, name="w_rev", symmetric=True))
            self._params["sensory_w_rev"].copy_(self.ptq_weight_nonzero(self._params["sensory_w_rev"], n_bits=weight_quantization, name="sensory_w_rev", symmetric=True))
            
    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state
        
        # cm/t is loop invariant
        cm_t = self._params["cm"] / (elapsed_time / self._ode_unfolds)
        # params
        sensory_w_param = self._params["sensory_w_mask"]
        w_param = self._params["w_mask"]
        sensory_w_rev = self._params["sensory_w_rev"]
        w_rev = self._params["w_rev"]
        gleak = self._params["gleak"]
        vleak = self._params["vleak"]
        # inputs CAM quantization
        if self.CAM_quantization is not None:
            inputs = self.ptq_range(inputs, n_bits=self.CAM_quantization, clip_min=self.clip_min, clip_max=self.clip_max, name="inputs")
        
        # [LUT] calculate sigmoid activation function for sensory neurons and quantization 
        activate_inputs = self._sigmoid(inputs, self._params["sensory_mu"], self._params["sensory_sigma"])
        if self.LUT_quantization is not None:
            activate_inputs = self.ptq_lut_sigmoid(activate_inputs, n_bits=self.LUT_quantization, name="sensory_w_activation")

        # [MVM] We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = sensory_w_param * activate_inputs

        sensory_rev_activation = sensory_w_rev * activate_inputs

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)
        
        for t in range(self._ode_unfolds):
            # [CAM/LUT] quantization inside the loop for v_pre
            if self.CAM_quantization is not None:
                v_pre = self.ptq_range(v_pre, n_bits=self.CAM_quantization, clip_min=self.clip_min, clip_max=self.clip_max, name=f"v_pre_step{t}")
            activate_v_pre = self._sigmoid(v_pre, self._params["mu"], self._params["sigma"])
            if self.LUT_quantization is not None:
                activate_v_pre = self.ptq_lut_sigmoid(activate_v_pre, n_bits=self.LUT_quantization, name=f"w_activation_step{t}")
                # for logging LUT activations distribution
            if self.calibration_path is not None:
                self.dump_lut_values(v_pre, activate_v_pre, path=self.calibration_path, bins=100, append=True)

            if self.digital_SRAM_quantization is not None:
                state = self.ptq_range(state, n_bits=self.digital_SRAM_quantization, clip_min=self.clip_min, clip_max=self.clip_max, name=(f"state"))

            w_activation = w_param * activate_v_pre

            rev_activation = w_rev * activate_v_pre

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
