import torch
import torch.nn as nn
from src.models.sequence.base import SequenceModule

class MLP(SequenceModule):
    """
    各タイムステップに独立して適用される
    シンプルなMulti-Layer Perceptron (MLP) レイヤー。
    
    SequenceModel内のSequenceResidualBlockによってラップされて使用されます。
    """
    def __init__(
        self,
        d_input,              # 必須: SequenceResidualBlockから「d_model」の値（例: 256）がこの引数に渡されます。
        
        d_hidden=None,        # MLP内部の隠れ層の次元（例: 256 * 4 = 1024）
        
        d_output=None,        # 通常は使用しません (Noneのままにします)。
                              # Noneの場合、下のロジックで自動的に d_input (d_model) と同じ値に設定されます。
        
        activation='relu',    # 'relu' または 'gelu'
        n_layers=2,           # MLP内部の層の数 (例: 2なら Linear -> Act -> Linear)
        dropout=0.0,
        transposed=False,     # SequenceModelから渡されます (通常はFalse)
    ):
        super().__init__()
        self.transposed = transposed

        # --- ここが「役割分担」の核心部分です ---
        if d_output is None:
            # d_outputが指定されなかった場合（＝通常の利用時）、
            # 出力次元を入力次元（d_model）と同一に設定します。
            # これにより、このMLPブロックは「次元を変えず」、
            # SequenceModel内で残差接続や積層が可能になります。
            d_output = d_input
        
        if d_hidden is None:
            # 隠れ層の次元が未指定なら、d_input (d_model) の4倍を使います。
            d_hidden = d_input * 4

        layers = []
        in_dim = d_input # 最初の層の入力は d_input (d_model) です。
        
        act_fn = nn.ReLU if activation == 'relu' else nn.GELU

        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, d_hidden))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = d_hidden # 2層目以降の入力は d_hidden です。

        # 最終層: d_hidden から d_output (d_model) へマッピングします。
        layers.append(nn.Linear(in_dim, d_output))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)
        
        # d_output プロパティ（SequenceModelが参照する）のために、
        # 最終的な出力次元（d_modelと同じ）を保存します。
        self._d_output = d_output

    def forward(self, inputs, *args, **kwargs):
        """
        入力 (transposed=False の場合): (batch, length, d_input) ※d_input = d_model
        出力: (outputs, state) 
        """
        if self.transposed:
            # (B, D, L) -> (B, L, D)
            inputs = inputs.transpose(1, 2)

        # self.mlp は (B, L, D_in) の最後の次元 D_in に作用します。
        # (B, L, d_input) -> (B, L, d_output)
        outputs = self.mlp(inputs)

        if self.transposed:
            # (B, L, D_out) -> (B, D_out, L)
            outputs = outputs.transpose(1, 2)
            
        # MLPは状態を持たない（ステートレス）ため、state は None を返します。
        return outputs, None

    @property
    def d_output(self):
        """ SequenceModel が参照する出力次元 """
        return self._d_output

    # --- ステートレスなモデルのためのインターフェース実装 ---

    def default_state(self, *args, **kwargs):
        # 状態なし
        return None

    def step(self, x, state, **kwargs):
        """
        ステップ実行 (RNNのように呼び出される可能性があるため実装)
        x shape: (batch, d_input)
        """
        # (B, D_in) -> (B, D_out)
        output = self.mlp(x)
        
        # 状態なし
        return output, None

    @property
    def d_state(self):
        # 状態なし
        return None

    @property
    def state_to_tensor(self):
        # 状態なし
        return lambda state: torch.empty(0)