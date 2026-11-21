"""Typed Hydra config schema. Defaults are Q-JEPA Table 5 (MovingMNIST)."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class EncoderCfg:
    name: str = "cnn"
    res: int = 16  # encoder input resolution; data/readout/metrics stay 64x64
    d: int = 64
    input_scale: float = 1.0  # l2 only: constant multiplier on the unit-RMS frame
    input_norm: str = "none"  # "l2": unit-RMS frames, what amplitude encoding keeps; use for fair HQViT comparisons
    # vit / hqvit only (HQViT §5: 4x4 patches, L = 3 blocks, FFN hidden = patch dim)
    patch: int = 4
    depth: int = 3
    ffn_hidden: int = 0  # 0 = patch * patch
    pool: str = "mean"  # "mean" (HQViT, ignores patch positions), "softargmax" (mean + per-channel x, y), "flatten"
    ansatz_depth: int = 2  # hqvit only: repeated PQC units D per Q/K/V circuit (HQViT §5)
    tokens: str = "frame"  # "frame" (encode frames separately) or "spacetime" (attention across frames)
    pos_emb: str = "none"  # "none" (HQViT paper), "sincos" (fixed 2-D), "learned"


@dataclass
class PriorCfg:
    name: str = "softmax_categorical"
    n_bits: int = 3
    hidden: int = 48
    components: int = 2  # mixture only
    offset_init: float = 0.1  # QCBM: std of the body-angle offsets; Q-JEPA §3.2 says only "small"
    layers: int = 2  # QCBM body layers (brickwork CNOTs; MovingMNIST: CNOT(0,1), CNOT(1,2))
    layout: str = "brickwork"  # QCBM CNOTs: "brickwork" (App. B, C) or "hep18_grid" (App. C, 4 rows of 3 qubits)


@dataclass
class PredictorCfg:
    width: int = 96
    kind: str = "shared"  # "shared" (Eqs. 5-6, LN output), "agent_local" (HEP18), "scaled" (D3IL, Eq. 15)


@dataclass
class QuantumCfg:
    target: str = "qpp-cpu"
    precision: str = "default"  # nvidia only: "fp32" (its default) or "fp64"; see the 2026-11-20 GPU gate spec
    shots: int = 256
    readout_flip: float = 0.005
    shift_obs: int = 8
    shift_coords: int = 9
    grad: str = "shift"  # QCBM prior update: "shift" (Eq. 14, reported) or "ideal" (reference, App. A.4)
    hqvit_train: str = "cudaq"  # HQViT training simulator: "cudaq" or "torch" (exact, faster); eval is always cudaq


@dataclass
class TrainerCfg:
    iters: int = 4000
    batch: int = 64
    k_train: int = 32
    warmup_repr: int = 200
    warmup_pred: int = 300
    lr_enc: float = 3e-4
    lr_enc_later: float = 1e-5
    lr_pred: float = 1e-3
    lr_prior: float = 1e-3
    weight_decay: float = 1e-5
    clip: float = 5.0
    lambda_reg: float = 4 / 3
    lambda_change: float = 0.0  # HEP18: weight of R(t - h) (App. F.3: 2/3)
    energy_beta: float = 1.0  # distance rho_{beta,eps} (Eq. 18); 1, 0 = Euclidean / sqrt(d)
    energy_eps: float = 0.0
    finetune_iters: int = 0  # D3IL: final predictor/prior stage on real segment windows (App. B.3: 1000)
    lr_finetune: float = 1e-4
    finetune_windows: int = 16  # windows of 3 consecutive pairs per batch
    device: str = "auto"  # cuda if available, else cpu
    ckpt_every: int = 500
    stop_after: int = 0  # end this session after N world-model iterations (Colab chunking); 0 = run to the end


@dataclass
class VISRegCfg:
    n_proj: int = 128
    w_center: float = 1.0
    w_scale: float = 10.0
    shape_ramp: int = 100


@dataclass
class ReadoutCfg:
    iters: int = 4000
    batch: int = 64
    lr: float = 2e-3
    weight_decay: float = 0.01
    clip: float = 5.0
    iters_low: int = 0  # D3IL: further steps at lr_low after `iters` (App. B.3: 2000 at 2e-4)
    lr_low: float = 2e-4


@dataclass
class EvalCfg:
    n_obs: int = 80
    k_eval: int = 128
    split: str = "test"  # "dev" for tuning/selection (phase-2 F5); writes metrics_dev.json instead of metrics.json
    rollouts: int = 384  # D3IL: recursive continuations from the common start (App. B.4: 3 x 128)
    max_calls: int = 16  # D3IL: predictions per rollout
    subset: int = 60  # D3IL: accepted-path budget for expected route coverage and TV


@dataclass
class DatasetCfg:
    name: str = "mmnist"
    root: str = "data"
    n_train: int = 800
    n_dev: int = 80
    n_test: int = 80
    fold: int = 0  # d3il only: which of the three folds
    agents: int = 1  # encoded separately and concatenated into h (HEP18: 4 cars, d = 4 x encoder.d)
    layout: str = "column"  # HEP18 rendering: "column" (one column, ~2.1 m/px) or "snake" (wrapped lane)
    bands: int = 8  # HEP18 snake layout: horizontal runs (~0.25 m/px at 8)


@dataclass
class WandbCfg:
    mode: str = "offline"
    project: str = "q2jepa"
    entity: str = ""  # team; "" = the API key's default
    name: str = ""  # run name; also the resume id, so a resumed run continues its curves (scripts/run_grid.py sets it)
    group: str = ""


@dataclass
class Config:
    seed: int = 0
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    prior: PriorCfg = field(default_factory=PriorCfg)
    predictor: PredictorCfg = field(default_factory=PredictorCfg)
    quantum: QuantumCfg = field(default_factory=QuantumCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    visreg: VISRegCfg = field(default_factory=VISRegCfg)
    readout: ReadoutCfg = field(default_factory=ReadoutCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)


ConfigStore.instance().store(name="base_config", node=Config)
