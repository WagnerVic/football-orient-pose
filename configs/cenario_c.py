"""Cenário C — Transfer Learning, sem Data Augmentation.

Treina RTMPose-X partindo dos pesos COCO (W₀ pré-treinado) com progressive
unfreezing em 3 fases. É o experimento de alta prioridade do trabalho:
valida se o TL melhora a precisão de keypoints em crops de futebol.

O train.py gerencia as 3 fases sequencialmente:
  Fase 1 — frozen_stages=4 (backbone completo), LR head = 1e-3, epochs=15
  Fase 2 — frozen_stages=2 (stages 0-1 congelados), LR 1e-4/1e-5, epochs=20
  Fase 3 — frozen_stages=1 (stage 0 congelado), LR 1e-4/1e-6, epochs=15
            (só executa se ΔPCK fase2−fase1 > 5pp)

Este arquivo contém as definições compartilhadas. O train.py sobrescreve
frozen_stages, load_from, max_epochs e optim_wrapper antes de cada Runner.

Uso via train.py:
    python scripts/train.py --cenario C [--epochs-fase1 15] [--epochs-fase2 20] [--epochs-fase3 15]
"""

custom_imports = dict(
    imports=["football_orient_pose.finetuning.dataset"],
    allow_failed_imports=False,
)

default_scope = "mmpose"

codec = dict(
    type="SimCCLabel",
    input_size=(288, 384),
    sigma=(6.0, 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

# Pesos COCO PyTorch para transfer learning.
# O train.py usa este path como load_from na Fase 1.
COCO_CHECKPOINT = "checkpoints/rtmpose-x_coco.pth"

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
        # frozen_stages e init_cfg são sobrescritos pelo train.py por fase.
        frozen_stages=4,
        init_cfg=None,
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=1280,
        out_channels=17,
        input_size=codec["input_size"],
        in_featuremap_size=(9, 12),
        simcc_split_ratio=codec["simcc_split_ratio"],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
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

dataset_type = "DSP3Dataset"
data_root = "data/3dsp"
split_file = "configs/split.json"

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

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

val_evaluator = dict(
    type="PCKAccuracy",
    thr=0.2,
    norm_item="bbox",
)
test_evaluator = val_evaluator

# train_cfg.max_epochs é sobrescrito pelo train.py por fase.
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=15, val_interval=5)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# optim_wrapper é sobrescrito pelo train.py por fase (LR diferenciado por camada).
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-3, weight_decay=0.05),
)

param_scheduler = [
    dict(
        type="LinearLR",
        begin=0,
        end=3,
        start_factor=1e-3,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        begin=3,
        end=15,  # sobrescrito pelo train.py
        eta_min=1e-6,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

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

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="Visualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = COCO_CHECKPOINT
resume = False
