"""Cenário A — From Scratch, sem Data Augmentation.

Treina RTMPose-X do zero (W₀ aleatório) no 3DSP sem augmentation adicional.
É o baseline experimental: define a performance mínima atingível sem
transfer learning nem augmentation.

Uso via train.py:
    python scripts/train.py --cenario A [--epochs 50]
"""

custom_imports = dict(
    imports=["mmpose", "football_orient_pose.finetuning.dataset"],
    allow_failed_imports=False,
)

# ─── Codec SimCC ────────────────────────────────────────────────────────────
# Entrada: 288×384 (largura × altura). Split ratio 2.0 → 576 bins-x, 768 bins-y.
# Sigma calibrado para crops 100×100 escalados 2.88× com padding 48px top/bottom.
codec = dict(
    type="SimCCLabel",
    input_size=(288, 384),
    sigma=(6.0, 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

# ─── Modelo ──────────────────────────────────────────────────────────────────
# RTMPose-X: CSPNeXt-X backbone (sem neck) + RTMCCHead (SimCC).
# from_scratch: load_from=None, frozen_stages=0.
# O train.py sobrescreve frozen_stages e load_from para cada fase do TL.
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=1.33,
        widen_factor=1.25,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="SiLU", inplace=True),
        frozen_stages=0,  # from scratch: nada congelado
        init_cfg=None,    # pesos aleatórios (sem pretrained)
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=1280,
        out_channels=17,
        input_size=codec["input_size"],
        in_featuremap_size=(9, 12),  # 288/32 × 384/32
        simcc_split_ratio=codec["simcc_split_ratio"],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            num_token=17,
            in_token_dims=256,
            out_token_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
        ),
        loss=dict(
            type="KLDiscretLoss",
            use_target_weight=True,
            beta=10.0,
            label_softmax=True,
        ),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True),
)

# ─── Dataset ─────────────────────────────────────────────────────────────────
dataset_type = "DSP3Dataset"
data_root = "data"
split_file = "configs/split.json"

# Pipeline de treino (Cenário A: sem MotionBlur/RandomErasing)
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

# Pipeline de validação
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split_file=split_file,
        split="train",
        pipeline=train_pipeline,
        metainfo=dict(dataset_name="3dsp"),
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split_file=split_file,
        split="val",
        pipeline=val_pipeline,
        metainfo=dict(dataset_name="3dsp"),
    ),
)

test_dataloader = val_dataloader

# ─── Métricas ────────────────────────────────────────────────────────────────
val_evaluator = dict(
    type="PCKAccuracy",
    thr=0.2,
    norm_item="bbox",
)
test_evaluator = val_evaluator

# ─── Treinamento ─────────────────────────────────────────────────────────────
# O train.py sobrescreve max_epochs via argumento --epochs.
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=5)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# LR uniforme para from scratch — sem paramwise_cfg.
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-3, weight_decay=0.05),
)

# LR scheduler: warmup + cosine decay
param_scheduler = [
    dict(
        type="LinearLR",
        begin=0,
        end=5,
        start_factor=1e-3,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        begin=5,
        end=50,  # sobrescrito pelo train.py
        eta_min=1e-6,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# ─── Hooks ───────────────────────────────────────────────────────────────────
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=5,
        save_best="PCK",
        rule="greater",
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="PoseVisualizationHook", enable=False),
)

# ─── Logging / Visualização ──────────────────────────────────────────────────
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="PoseLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
