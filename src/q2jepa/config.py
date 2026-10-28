"""Typed Hydra config schema. Defaults are Q-JEPA Table 5 (MovingMNIST)."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class EncoderCfg:
    name: str = "cnn"
    res: int = 16  # encoder input resolution; data/readout/metrics stay 64x64
    d: int = 64


@dataclass
class PriorCfg:
    name: str = "softmax_categorical"
    n_bits: int = 3
    hidden: int = 48
    components: int = 2  # mixture only
    layers: int = 2  # QCBM body layers


@dataclass
class PredictorCfg:
    width: int = 96


@dataclass
class QuantumCfg:
    target: str = "qpp-cpu"
    shots: int = 256
    readout_flip: float = 0.005
    shift_obs: int = 8
    shift_coords: int = 9


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


@dataclass
class EvalCfg:
    n_obs: int = 80
    k_eval: int = 128


@dataclass
class DatasetCfg:
    name: str = "mmnist"
    root: str = "data"
    n_train: int = 800
    n_dev: int = 80
    n_test: int = 80


@dataclass
class WandbCfg:
    mode: str = "offline"
    project: str = "q2jepa"


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
