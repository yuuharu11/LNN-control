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
        clip_w_sum_max: Optional[float] = None,
        clip_w_sum_min: Optional[float] = None,
        clip_rev_sum_max: Optional[float] = None,
        clip_rev_sum_min: Optional[float] = None,
        implicit_param_constraints=False,
        digital_RRAM_quantization: Optional[int] = None,
        digital_SRAM_quantization: Optional[int] = None,
        weight_quantization: Optional[int] = None,
        CAM_quantization: Optional[int] = None,
        LUT_quantization: Optional[int] = None,
        ADC_quantization: Optional[int] = None,
        DAC_quantization: Optional[int] = None,
        calibration_path: Optional[str] = None,
        gaussian: Optional[float] = None,
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
        self.clip_w_sum_min = clip_w_sum_min
        self.clip_w_sum_max = clip_w_sum_max
        self.clip_rev_sum_min = clip_rev_sum_min
        self.clip_rev_sum_max = clip_rev_sum_max
        self._clip = torch.nn.ReLU()
        self.quantize_debug: bool = True
        self.CAM_quantization = CAM_quantization
        self.LUT_quantization = LUT_quantization
        self.digital_RRAM_quantization = digital_RRAM_quantization
        self.digital_SRAM_quantization = digital_SRAM_quantization
        self.weight_quantization = weight_quantization
        self.ADC_quantization = ADC_quantization
        self.DAC_quantization = DAC_quantization
        self.calibration_path = calibration_path
        self.gaussian = gaussian
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
        denominator: torch.Tensor,
        numerator: torch.Tensor,
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
                "hist": torch.histc(x, bins=bins, min=float(x.min()), max=float(x.max())).cpu().tolist(),
            }

        record = {
            "activations": stats(activations),
            "states": stats(state),
            "denominator": stats(denominator),
            "numerator": stats(numerator),
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
        PTQ for sparse weights with Noise Injection.
        """
        assert n_bits >= 2
        if percentile > 1.0:
            percentile /= 100.0

        with torch.no_grad():
            x = params.to(torch.float32)

            # 1. Zero-Gating
            nz_mask = x != 0
            nz = x[nz_mask]

            if nz.numel() == 0:
                return params

            # 2. Percentile-based clipping
            if symmetric:
                abs_nz = nz.abs()
                max_val = torch.quantile(abs_nz, percentile)
                nz = torch.clamp(nz, min=-max_val, max=max_val)
            else:
                lower = torch.quantile(nz, 1 - percentile)
                upper = torch.quantile(nz, percentile)
                nz = torch.clamp(nz, min=lower, max=upper)

            # 3. Quantization mode
            if symmetric:
                # Symmetric (signed)
                qmax = 2 ** (n_bits - 1) - 1
                abs_max = nz.abs().max()
                if abs_max < 1e-12: return params # Avoid div/0
                
                scale = abs_max / qmax
                
                nz_q = torch.round(nz / scale).clamp(-qmax, qmax) * scale

            else:
                # Asymmetric (unsigned)
                qmax = 2 ** n_bits - 2
                min_val = nz.min()
                max_val = nz.max()
                if max_val - min_val < 1e-12: return params

                scale = (max_val - min_val) / qmax
                nz_q = torch.round((nz - min_val) / scale).clamp(0, qmax) * scale + min_val

            # write back 
            out = x.clone()
            out[nz_mask] = nz_q
            params.copy_(out.to(params.dtype))

        return params
    
    # noise injection for weight 
    @torch.no_grad() 
    def injection_error( 
        self, 
        params: torch.Tensor, 
        sigma: float = 0.0, 
        shift: float = 0.0, 
        symmetric: bool = True, 
        eps: float = 1e-12, 
        ): 
        if (sigma is None or sigma <= 0.0) and (shift == 0.0): 
            return params 
        
        #------------------------- 
        # # 1. 正規化スケール決定 
        # ------------------------- 
        if symmetric: 
            scale = params.abs().max().clamp(min=eps) 
            w_norm = params / scale 
        else: 
            scale = params.max().clamp(min=eps) 
            w_norm = params / scale 

        # ------------------------- 
        # 2. 0 重みマスク（0の部分には誤差を乗せない） 
        # ------------------------- 
        mask = (w_norm != 0).float() 

        # ------------------------- 
        # 3. 正規化空間で誤差注入 
        # ------------------------- 
        # ガウス誤差（ランダムなバラツキ） 
        noise = torch.randn_like(w_norm) * sigma 

        # シフトエラー（一定方向へのズレ） 
        # すべての重みを一律に shift 分だけ動かす 
        offset = shift * mask 

        # ------------------------- 
        # 4. 正規化空間でエラー注入＆クリップ（範囲外を丸める） 
        # ------------------------- 
        if symmetric: 
            # エラー注入
            
            # 1. 重みの符号を取得 (正: 1, 負: -1, 0: 0)
            w_sign = torch.sign(w_norm)
            
            # 2. ノイズの適用（常に0の方向へずらす）
            #    元の式: w_norm - (大きさ * 符号)
            #    正の時: w - (正の値) -> 小さくなる
            #    負の時: w - (負の値) -> w + 正の値 -> 0に近づく
            decay_amount = (noise + offset) * mask
            w_norm_noisy = w_norm - (decay_amount * w_sign)
            
            # 3. 符号反転（0またぎ）の防止
            #    「元の符号」と「ノイズ後の符号」が異なるとき、行き過ぎたとみなして0にする
            #    (0.1 が -0.05 になったら 0 にする)
            sign_changed = (w_sign * torch.sign(w_norm_noisy)) < 0
            w_norm_noisy[sign_changed] = 0.0

            # 4. 範囲クリッピング
            w_norm_noisy = torch.clamp(w_norm_noisy, -1.0, 1.0)
        else: 
            w_norm_noisy = w_norm + (noise - offset) * mask 
            w_norm_noisy = torch.clamp(w_norm_noisy, 0.0, 1.0) 

        # ------------------------- 
        # 5. 元スケールへ復元 
        # ------------------------- 
        params_noisy = w_norm_noisy * scale 
        print(f"[Injection-Error] sigma={sigma}, shift={shift}, scale={scale.item():.3e}")
        return params_noisy

    @torch.no_grad()
    def injection_error_mlc(
        self,
        params: torch.Tensor,
        sigma: float = 0.0,
        shift: float = 0.0,
        bits: int = 6,
        cell_bits: int = 2,
        symmetric: bool = True,
        eps: float = 1e-12
    ):
        sigma_val = 0.0 if sigma is None else float(sigma)
        shift_val = 0.0 if shift is None else float(shift)
        if (sigma_val <= 0.0) and (shift_val == 0.0):
            return params

        if bits % cell_bits != 0:
            raise ValueError("bits must be divisible by cell_bits")

        # 0重みは不変（floatの厳密比較は避ける）
        weight_nz = (params.abs() > eps).to(torch.float32)

        # ----------------------
        # 1. [0,1] に正規化
        # ----------------------
        if symmetric:
            # sign-magnitude 想定：セルには |w| のみ保存し、最後に符号を戻す
            scale = params.abs().max().clamp(min=eps)
            sign = torch.sign(params)  # -1,0,+1
            w01 = (params.abs() / scale).clamp(0.0, 1.0)  
        else:
            scale = params.max().clamp(min=eps)
            sign = 1.0
            w01 = (params / scale).clamp(0.0, 1.0)

        # ----------------------
        # 2. 整数に変換
        # ----------------------
        # 最大値
        qmax = (1 << bits) - 1  # 2^bits -1 ex) 6bit: 63
        q = torch.round(w01 * qmax).to(torch.int64)

        # セル数を求める
        num_cells = bits // cell_bits

        # セルあたりの最大値
        cell_max = (1 << cell_bits) - 1 # ex) 2bit: 3

        # ノイズ加算用の行列を用意
        q_recon = torch.zeros_like(w01, dtype=torch.float32)

        # ----------------------
        # 3. セルごとにノイズ注入
        # ----------------------
        for i in range(num_cells):
            # セルに位置に合わせたビットシフト量
            shift_bits = i * cell_bits
            bit_sig = 1 << shift_bits  

            # セル値を抽出
            cell_int = (q >> shift_bits) & cell_max
            cell_val = cell_int.to(torch.float32)

            # 元セルが0ならノイズなし
            cell_mask = (cell_int != 0).to(torch.float32) * weight_nz

            # sigma/shiftは「セル値(0..cell_max)空間」で扱う
            cell_sigma = sigma_val * float(cell_max)
            cell_shift = shift_val * float(cell_max)

            noise = torch.randn_like(cell_val) * cell_sigma * cell_mask
            shift = cell_shift * cell_mask
            # shift>0 で 0 方向へ寄せる
            noisy_cell = torch.clamp(cell_val + noise - shift, 0.0, float(cell_max))

            q_recon += noisy_cell * float(bit_sig)

        # ----------------------
        # 4. [0,1] に戻す
        # ----------------------
        w01_noisy = (q_recon / float(qmax)).clamp(0.0, 1.0)

        # ----------------------
        #  デバッグ情報出力
        # ----------------------
        print(f"[Injection-Error-MLC] sigma={sigma_val}, shift={shift_val}, scale={scale.item():.3e}, bits={bits}, cell_bits={cell_bits}")
       
        # ----------------------
        # 5. 元スケールに戻す
        # ----------------------
        if symmetric:
            w_mag_noisy = w01_noisy  # magnitude
            w_norm_noisy = torch.clamp(w_mag_noisy * sign, -1.0, 1.0)
            return w_norm_noisy * scale
        else:
            return w01_noisy * scale
    
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
    
    @torch.no_grad()
    def ptq_cam(
        self, 
        params: torch.Tensor, 
        n_bits: int = 8, 
        name: Optional[str] = None,
        clip_min: float = -0.5,
        clip_max: Optional[float] = 0.5,   
        gaussian: Optional[float] = None
    ):
        # signed range
        qmax = 2 ** (n_bits) - 1

        x = params.to(torch.float32)

        scale = (clip_max - clip_min) / qmax
        
        x_clipped = x.clamp(min=clip_min, max=clip_max)
        q = torch.round((x_clipped - clip_min) / scale).clamp(0, qmax)

        if gaussian is not None and gaussian > 0.0:
            # Gaussian noise injection
            noise = torch.randn_like(q) * gaussian
            q = q + noise
            q = torch.round(q).clamp(0, qmax)            

        x_q = q * scale + clip_min

        params.copy_(x_q.to(params.dtype))

        if getattr(self, "quantize_debug", False) and name:
            print(
                f"[Quantize-minmax] {name}: bits={n_bits}, "
                f"scale={scale:.3e}, clip_min={clip_min:.3e}, clip_max={clip_max:.3e}"
            )

        return params

    # for LUT sigmoid [0,1]
    @torch.no_grad()
    def ptq_lut(self, params: torch.Tensor, n_bits: int = 8, name: Optional[str] = None, gaussian: Optional[float] = None):
        qmax = 2 ** n_bits - 1
        scale = 1.0 / qmax
        nz_q = torch.round(params / scale).clamp(0, qmax) * scale
        if gaussian is not None and gaussian > 0.0:
            # Gaussian noise injection
            noise = torch.randn_like(nz_q) * gaussian * scale
            nz_q = nz_q + noise
            # clamp to [0,1]
            nz_q = nz_q.clamp(0.0, 1.0)
            if getattr(self, "quantize_debug", False) and name:
                print(f"[Quantize-LUT-Gaussian] {name}: bits={n_bits}, scale={scale:.3e}, gaussian={gaussian}")
        else:
            if getattr(self, "quantize_debug", False) and name:
                print(f"[Quantize-LUT] {name}: bits={n_bits}, scale={scale:.3e}")
        params.copy_(nz_q.to(params.dtype))

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
        self._params["concatenated_w"] = self.add_vacant_weight("concatenated_w", (self.sensory_size + self.state_size, self.state_size), requires_grad=False, persistent=False)
        self._params["w_rev"] = self.add_vacant_weight("w_rev", self._params["w"].shape, requires_grad=False, persistent=False)
        self._params["sensory_w_rev"] = self.add_vacant_weight("sensory_w_rev", self._params["sensory_w"].shape, requires_grad=False, persistent=False)
        self._params["concatenated_w_rev"] = self.add_vacant_weight("concatenated_w_rev", (self.sensory_size + self.state_size, self.state_size), requires_grad=False, persistent=False)

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
        self._params["concatenated_w"] = torch.cat((self._params["sensory_w_mask"], self._params["w_mask"]), dim=0)
        self._params["concatenated_w_rev"] = torch.cat((self._params["sensory_w_rev"], self._params["w_rev"]), dim=0)
        
    def _weight_quantization(self, weight_quantization=None, gaussian=0.0, shift=0.0, cell_bits=2):
        if weight_quantization is not None:
            self._params["concatenated_w"].copy_(self.ptq_weight_nonzero(self._params["concatenated_w"], n_bits=weight_quantization, name="w", symmetric=False))
            self._params["concatenated_w_rev"].copy_(self.ptq_weight_nonzero(self._params["concatenated_w_rev"], n_bits=weight_quantization, name="w_rev", symmetric=True))
        if gaussian is not None or shift is not None:
            self._params["concatenated_w"].copy_(self.injection_error_mlc(self._params["concatenated_w"], sigma=gaussian, shift=shift, symmetric=False, cell_bits=cell_bits))
            self._params["concatenated_w_rev"].copy_(self.injection_error_mlc(self._params["concatenated_w_rev"], sigma=gaussian, shift=shift, symmetric=True, cell_bits=cell_bits))

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        # state is fixed
        v_pre = state
        
        # cm/t is loop invariant
        cm_t = self._params["cm"] / (elapsed_time / self._ode_unfolds)
        # params
        concatenated_w = self._params["concatenated_w"]
        concatenated_w_rev = self._params["concatenated_w_rev"]
        gleak = self._params["gleak"]
        vleak = self._params["vleak"]

        # concatenate w and sensory_w, mu, sigma
        concatenated_mu = torch.cat((self._params["sensory_mu"], self._params["mu"]), dim=0)
        concatenated_sigma = torch.cat((self._params["sensory_sigma"], self._params["sigma"]), dim=0)
        
        for t in range(self._ode_unfolds):
            # concatenate state and inputs
            x = torch.cat((inputs, v_pre), dim=1)
            
            # [CAM/LUT] quantization inside the loop for v_pre
            if self.CAM_quantization is not None:
                x = self.ptq_cam(x, n_bits=self.CAM_quantization, clip_min=self.clip_min, clip_max=self.clip_max, name=f"v_pre_step{t}")
            
            # slice x to states for sigmoid calculation
            activate_x = self._sigmoid(x, concatenated_mu, concatenated_sigma)

            if self.LUT_quantization is not None:
                activate_x = self.ptq_lut(activate_x, n_bits=self.LUT_quantization, name=f"w_activation_step{t}")
            
            # DAC quantization for state
            if self.DAC_quantization is not None:
                activate_x = self.ptq_lut(activate_x, n_bits=self.DAC_quantization, name=(f"state_step{t}"))

            if self.digital_SRAM_quantization is not None:
                state = self.ptq_range(state, n_bits=self.digital_SRAM_quantization, clip_min=self.clip_min, clip_max=self.clip_max, name=(f"state"))

            w_activation = concatenated_w * activate_x 
            rev_activation = concatenated_w_rev * activate_x

            # Reduce over dimension 1 (=source neurons)
            w_numerator = torch.sum(rev_activation, dim=1) 
            w_denominator = torch.sum(w_activation, dim=1)

            # for logging LUT activations distribution
            if self.calibration_path is not None:
                if np.random.random() < 0.01:
                    # self.dump_lut_values(x[:,150], activate_x[:,150], w_denominator, w_numerator, path=self.calibration_path, bins=100, append=True)
                    self.dump_lut_values(x, activate_x, w_denominator, w_numerator, path=self.calibration_path, bins=100, append=True)

            if self.ADC_quantization is not None:
                w_numerator = self.ptq_range(w_numerator, n_bits=self.ADC_quantization, clip_min=self.clip_rev_sum_min, clip_max=self.clip_rev_sum_max, name=(f"w_numerator_step{t}"))
                w_denominator = self.ptq_range(w_denominator, n_bits=self.ADC_quantization, clip_min=self.clip_w_sum_min, clip_max=self.clip_w_sum_max, name=(f"w_denominator_step{t}"))

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
