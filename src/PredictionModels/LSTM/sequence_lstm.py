"""Sequence LSTM モデル定義

過去5走を時系列として LSTM でエンコードし、
現在レース特徴・血統 Embedding と組み合わせて勝ち馬スコアを出力する。

MARコンセプト（タイムと血統重視）:
  - 時系列: 過去5走の走行データ（タイム差・着順・上がり等）を LSTM で処理
  - 血統: father/mother_father/paternal_gf を Embedding に変換
  - 現在: レース条件・騎手成績等の非時系列特徴

モデルサフィックス: _v14lstm_no26
"""

import torch
import torch.nn as nn

SEQ_LEN = 5       # 過去走数
SEQ_FEATURES = 8  # 1走あたりの特徴数: time_df_course, time_df_class, ninki, result, agari, margin, corner_ratio, corner_chase
NUM_CURRENT = 54  # 非時系列・非embed特徴数


class SequenceLSTM(nn.Module):
    """過去5走シーケンス LSTM + 血統 Embedding + 現在特徴 MLP"""

    def __init__(self, vocab_size: int, seq_features: int = SEQ_FEATURES,
                 num_current: int = NUM_CURRENT, embed_dim: int = 16,
                 lstm_hidden: int = 64, lstm_layers: int = 1,
                 hidden: list = None, dropout: float = 0.2):
        """
        Args:
            vocab_size   : 血統 vocab の種類数
            seq_features : 1タイムステップあたりの特徴数（デフォルト 8）
            num_current  : 非時系列・非embed 特徴数（デフォルト 54）
            embed_dim    : 1血統あたりの Embedding 次元数
            lstm_hidden  : LSTM 隠れ層サイズ
            lstm_layers  : LSTM 積層数
            hidden       : MLP 隠れ層サイズリスト
            dropout      : Dropout 率
        """
        super().__init__()
        if hidden is None:
            hidden = [128, 64]

        self.embed_father = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.embed_mf     = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.embed_pgf    = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=seq_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        in_dim = lstm_hidden + embed_dim * 3 + num_current
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, seq, father_ids, mf_ids, pgf_ids, x_current):
        """
        Args:
            seq        : (batch, seq_len, seq_features) FloatTensor  ← 古い走→新しい走の順
            father_ids : (batch,) LongTensor
            mf_ids     : (batch,) LongTensor
            pgf_ids    : (batch,) LongTensor
            x_current  : (batch, num_current) FloatTensor
        Returns:
            (batch,) FloatTensor - 勝ち馬スコア（sigmoid前logit）
        """
        _, (h_n, _) = self.lstm(seq)
        h = h_n[-1]  # 最終層の隠れ状態: (batch, lstm_hidden)
        x = torch.cat([
            h,
            self.embed_father(father_ids),
            self.embed_mf(mf_ids),
            self.embed_pgf(pgf_ids),
            x_current,
        ], dim=1)
        return self.mlp(x).squeeze(1)
