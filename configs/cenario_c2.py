"""Cenário C2 — Transfer Learning em FASE ÚNICA (ablação do progressive unfreezing).

Mesma inicialização do Cenário C (pesos COCO), mas SEM as 3 fases de progressive
unfreezing: treina numa única fase com ``frozen_stages=2`` (o estágio que rendeu o
melhor C) e LR discriminativo (cabeça alta, backbone baixo). O warmup de LR substitui
o papel da fase 1 (proteger os features do COCO da cabeça crua).

Não substitui o Cenário C — é uma ablação para responder: "o progressive unfreezing é
necessário, ou um fine-tune de fase única no estágio certo basta?".

Uso via train.py (rota de fase única):
    python scripts/train.py --cenario C2 [--epochs 150]
"""

custom_imports = dict(
    imports=["football_orient_pose.finetuning.dataset", "football_orient_pose.finetuning.metric"],
    allow_failed_imports=False,
)

default_scope = "mmpose"

# train.py usa este path como load_from (pesos COCO).
COCO_CHECKPOINT = "checkpoints/rtmpose-x_coco.pth"

# ─── Codec SimCC ────────────────────────────────────────────────────────────
codec = dict(
    type="SimCCLabel",
    input_size=(288, 384),
    sigma=(6.0, 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

# ─── Modelo ──────────────────────────────────────────────────────────────────
# RTMPose-X com pesos COCO; backbone parcialmente congelado (frozen_stages=2):
# stages 0-2 congelados, stages 3-4 + cabeça treinam.
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
        frozen_stages=2,  # fase única: stages 0-2 congelados
        init_cfg=None,    # pesos vêm do load_from (COCO)
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

# ─── Dataset ─────────────────────────────────────────────────────────────────
dataset_type = "DSP3Dataset"
data_root = "data/3dsp"
split_file = "configs/split.json"

# Sem augmentation (comparável ao Cenário C).
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

val_evaluator = dict(type="StrictPCKMetric", pck_thr=0.2, pdj_thr=0.5)
test_evaluator = val_evaluator

# ─── Treinamento ─────────────────────────────────────────────────────────────
# train.py sobrescreve max_epochs (--epochs) e param_scheduler (proporcional).
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=150, val_interval=5)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# LR discriminativo (1 fase): cabeça 1e-3, backbone 1e-5 (lr_mult 0.01).
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-3, weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.01)}),
)

param_scheduler = [
    dict(type="LinearLR", begin=0, end=15, start_factor=1e-3,
         by_epoch=True, convert_to_iter_based=True),
    dict(type="CosineAnnealingLR", begin=15, end=150, eta_min=1e-6,
         by_epoch=True, convert_to_iter_based=True),
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

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="Visualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = COCO_CHECKPOINT
resume = False
