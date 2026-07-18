"""Embedding + MLP モデル定義

血統IDを Embedding ベクトルに変換し、数値特徴量（タイム差・上がり等）と
結合して MLP で勝ち馬スコアを出力する。

MARコンセプト（タイムと血統重視）に則り:
  - 血統: father_cat / mother_father_cat / paternal_gf_cat を
          各 embed_dim 次元の Embedding に変換（血統の潜在的な距離適性を学習）
  - 数値: v8特徴量から血統3列を除いた94列をそのまま入力

モデルサフィックス: _v13mlp_no26
"""

import torch
import torch.nn as nn


class EmbeddingMLP(nn.Module):
    """血統Embedding + 数値特徴量 MLP"""

    def __init__(self, vocab_size: int, embed_dim: int = 16,
                 num_numeric: int = 94, hidden: list = None, dropout: float = 0.2):
        """
        Args:
            vocab_size : 血統vocab の種類数（padding_idx=0 を加えた vocab_size+1 でEmbedding作成）
            embed_dim  : 1血統あたりの Embedding 次元数
            num_numeric: 数値特徴量の列数（race_id・血統3列を除いた v8 列数）
            hidden     : MLP の隠れ層サイズリスト
            dropout    : Dropout 率
        """
        super().__init__()
        if hidden is None:
            hidden = [256, 128, 64]

        # 父・母父・父父は別々の Embedding（コース適性の学習が異なるため）
        self.embed_father = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.embed_mf     = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.embed_pgf    = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        in_dim = embed_dim * 3 + num_numeric
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

    def forward(self, father_ids, mf_ids, pgf_ids, x_num):
        """
        Args:
            father_ids : (batch,) LongTensor
            mf_ids     : (batch,) LongTensor
            pgf_ids    : (batch,) LongTensor
            x_num      : (batch, num_numeric) FloatTensor
        Returns:
            (batch,) FloatTensor - 勝ち馬スコア（sigmoid前のlogit）
        """
        e_f   = self.embed_father(father_ids)
        e_mf  = self.embed_mf(mf_ids)
        e_pgf = self.embed_pgf(pgf_ids)
        x = torch.cat([e_f, e_mf, e_pgf, x_num], dim=1)
        return self.mlp(x).squeeze(1)
