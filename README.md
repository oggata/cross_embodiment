# Cross-Embodiment Training: 人魚への転移実験

> 「形を覚えるな、意図を覚えろ。」
>
> 二足歩行と魚の泳ぎを学んだモデルが、人魚を少量のデータで動かせる——
> π₀と同じ原理をブラウザで動くデモとして実装したプロジェクト。

---

## 概要

Physical AIの研究領域で注目されている **Cross-Embodiment Training** を、
「二足歩行 + 魚 → 人魚への転移」という実験で検証したプロジェクトです。

### 何を証明するか

| 条件 | 事前知識 | 人魚データ量 |
|------|----------|-------------|
| **A（転移あり）** | 二足 + 魚で事前学習済み | 少量（50件） |
| **B（ベースライン）** | ゼロ（ランダム初期化） | 同じ50件 |

**同じデータ量でも、AはBより速く・精度高く人魚の動きを学習できる。**
これが「データ資産が蓄積される」という意味です。

### π₀との共通原理

```
π₀の場合：
  食洗機・電子レンジ・冷蔵庫で学習
    → 「扉を開ける」「ボタンを押す」の意図を抽象化
    → 未知の洗濯機にも転移

本プロジェクトの場合：
  二足歩行・魚の泳ぎで学習
    → 「移動する」「追跡する」の意図を抽象化
    → 人魚（上半身＋尾ひれの混合動作）に転移
```

---

## ファイル構成

```
.
├── README.md                        ← このファイル
├── motion_data_check.html           ← 二足と魚の動きを目視確認するデモ
├── mermaid_demo.html                ← 人魚A/B比較デモ（重み埋め込み対応）
├── cross_embodiment_mermaid.ipynb   ← Colab学習ノートブック
├── weights_A.json                   ← 条件A（転移あり）の学習済み重み
└── weights_B.json                   ← 条件B（ベースライン）の学習済み重み
```

---

## 使い方

### Step 1: 動きのデータを目視確認する

`motion_data_check.html` をブラウザで開く。

タブで切り替え可能：

| タブ | 内容 |
|------|------|
| 二足歩行 | 股関節・膝の位相制御、腕の逆位相スイング |
| 魚 (泳ぎ) | BCPモデルによる頭→尾への伝播波 |

下のパネルに各関節の値がリアルタイム表示されます。
**これがそのまま学習データの数式になっています。**

### Step 2: デモをすぐ動かす

`mermaid_demo.html` をブラウザで開く。

`weights_A.json` と `weights_B.json` が同梱されていれば、
ドラッグ&ドロップするだけで学習済みモデルで比較できます。

**操作方法：**

| UI | 説明 |
|----|------|
| 待機 | アイドルモーション。A/Bの差が小さい状態 |
| 水面移動 | 腕と尾ひれの混合動作。転移の効果が見え始める |
| 水中遊泳 | 尾ひれ推進が主体。カメラが横に移動して側面から比較できる |

水中遊泳モードでは人魚が水平に傾き、カメラが側面に回り込みます。
尾ひれの動きの**なめらかさの差**がAとBで確認できます。

### Step 3: 自分でモデルを学習させる

1. `cross_embodiment_mermaid.ipynb` をGoogle Colabで開く
2. ランタイム → すべてのセルを実行
3. 最後のセルで `weights_A.json` と `weights_B.json` がダウンロードされる
4. `mermaid_demo.html` を開き、それぞれドラッグ&ドロップ

```
Colab（GPU推奨）：
  学習時間 ≈ 10〜15分（Tesla T4）
  出力ファイル：weights_A.json / weights_B.json（各 ~100KB）
```

---

## 技術的な仕組み

### モデルアーキテクチャ

```
入力
  State（10次元）
    [dist, dir_x, dir_z, sin(phase), cos(phase),
     speed, water_depth, height, task_id, obstacle]
  Spec（5次元）← Embodiment Token
    [move_speed, leg_count_norm, has_tail, water_ability, body_length_norm]

         ↓
┌──────────────────────────────────┐
│  EmbodimentEncoder               │
│  Linear(5→32)→ReLU→Linear(32→32)│
│  身体スペックを潜在ベクトル化      │
└──────────────┬───────────────────┘
               │ [state + emb] = 42次元
┌──────────────▼───────────────────┐
│  SharedBackbone（全キャラ共通）   │
│  Linear(42→128)→...→Linear(→64) │
│  意図・状況理解を共有学習          │
└──────────────┬───────────────────┘
               │ 64次元の潜在表現（latent）
    ┌──────────┴──────────┐
    ▼                     ▼
┌──────────────┐   ┌──────────────┐
│ ActionExpert │   │ ActionExpert │
│  (biped/fish)│   │  (mermaid)   │
│  64→32→8    │   │  64→32→8    │
└─────┬────────┘   └─────┬────────┘
      │                  │
      ▼                  ▼
  既存キャラ出力      人魚出力（8次元）
                  [arm_l, arm_r, torso, bob,
                   tail_mid, tail_tip, hip_pitch, roll]
```

### 転移のポイント

**条件A（転移あり）：**
- SharedBackboneを凍結（`requires_grad = False`）
- ActionExpertだけ50件の人魚データで学習
- Backboneが持つ「移動の意図表現」をそのまま活用

**条件B（ベースライン）：**
- 全パラメータをランダム初期化（Kaiming uniform）
- 同じ50件から全パラメータを学習
- 意図表現もゼロから構築しなければならない

### 学習データの生成方法

リアルのモーションキャプチャデータではなく、**生体力学の研究に基づいた数式**から合成データを生成しています。

**二足歩行：**
```python
lHip  =  np.sin(phase) * amp        # 股関節：正弦波
lKnee =  max(0, -np.sin(phase)) * amp * 1.2  # 膝：片側整流
lArm  = -np.sin(phase) * 0.28       # 腕：脚と逆位相
```

**魚の泳ぎ（BCPモデル）：**
```python
# 頭→尾への伝播波、振幅は二次関数的に増大
def seg_angle(s):  # s: 0(頭)→1(尾)
    amp    = amp_base + (amp_tail - amp_base) * s**2
    offset = waveK * s * np.pi
    return np.sin(phase - offset) * amp
```

**人魚（水深でブレンド）：**
```python
blend = water_depth  # 0=陸上, 1=完全水中
arm_l    = -np.sin(phase) * 0.28 * (1 - blend * 0.6)  # 水中では腕は小さく
tail_tip =  seg_angle(1.0) * blend                      # 水中では尾ひれが主体
```

### ブラウザ上でのニューラルネット推論

ONNXランタイム不要。純粋なJS行列演算で実装：

```javascript
function mv(W, b, x, out_dim, in_dim) {
  const out = new Float32Array(out_dim);
  for (let i = 0; i < out_dim; i++) {
    let s = b[i];
    for (let j = 0; j < in_dim; j++) s += W[i * in_dim + j] * x[j];
    out[i] = s;
  }
  return out;
}
```

---

## 何が言えて、何が言えないか

### 言えること ✓

- 同じデータ量でも、転移ありの方が低い損失を達成できる
- 特にデータが少ない（N=10〜50）場面で効果が大きい
- 身体の多様性がデータ資産になる（二足+魚→人魚）

### 言えないこと ✗

| 誤解されやすい主張 | 正確な言い方 |
|---|---|
| 「二足のデータだけで人魚が作れる」 | 人魚のデータは必要。ただし少量でよい |
| 「羽・飛行など全く異なる身体にも転移できる」 | 同じ物理ドメイン（地上・水中移動）内での転移 |
| 「合成データで証明された」 | 合成データでの有効性。実データでの検証は別途必要 |

---

## 今後の拡張アイデア

### 視覚入力化
Three.jsのcanvasを`getImageData()`でCNN encoderに入力する。
数値Specを廃止することで、形状定義なしに転移できる方向へ。

### GNN Morphology Encoder
身体スペックをグラフ構造で渡す。
「足が増えた」「尾ひれが2本」に設計変更なしで対応可能。

### Flow Matching Head
1ステップ出力 → Tステップのトラジェクトリ生成。
π₀により近い実装へ。

### トビウオへの拡張
水中モード → 飛行モード → 水中モードの**モード遷移**を持つキャラクター。
Backboneが文脈を理解しているかどうかが試される、最も技術的に面白いケース。

---

## 参考

- [Open X-Embodiment (Google DeepMind)](https://robotics-transformer-x.github.io/)
- [π₀ (Physical Intelligence)](https://www.physicalintelligence.company/blog/pi0)
- [RT-2 (Google DeepMind)](https://robotics-transformer2.github.io/)
- [Body Caudal Propulsion モデル — Lighthill (1960)](https://royalsocietypublishing.org/doi/10.1098/rspa.1960.0133)


