import os
import torch
import torch.nn as nn
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss, mse_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.ckd import build_ostrack_ckd

# forward propagation related
from lib.train.actors import OSTrack_CKD_Actor
try:
    from lib.train.actors.ckd_high2low_attn import OSTrack_CKD_Attn_Actor
    ATTN_ACTOR_AVAILABLE = True
except ImportError:
    ATTN_ACTOR_AVAILABLE = False
    print("[Warning] OSTrack_CKD_Attn_Actor not available, using standard actor")

try:
    from lib.train.actors.ckd_vitkd import OSTrack_CKD_Actor as OSTrack_CKD_ViTKD_Actor
    VITKD_ACTOR_AVAILABLE = True
except ImportError:
    VITKD_ACTOR_AVAILABLE = False
    print("[Warning] OSTrack_CKD_ViTKD_Actor not available")

try:
    from lib.train.actors.ckd_vitkd_resp import OSTrack_CKD_ViTKD_Resp_Actor
    VITKD_RESP_ACTOR_AVAILABLE = True
except ImportError:
    VITKD_RESP_ACTOR_AVAILABLE = False
    print("[Warning] OSTrack_CKD_ViTKD_Resp_Actor not available")

try:
    from lib.train.actors.ckd_vitkd_sls import OSTrack_CKD_ViTKD_SLS_Actor
    VITKD_SLS_ACTOR_AVAILABLE = True
except ImportError:
    VITKD_SLS_ACTOR_AVAILABLE = False
    print("[Warning] OSTrack_CKD_ViTKD_SLS_Actor not available")

try:
    from lib.train.actors.ckd_vitkd_sls_dynamic import OSTrack_CKD_ViTKD_SLS_Dynamic_Actor
    VITKD_SLS_DYNAMIC_ACTOR_AVAILABLE = True
except ImportError:
    VITKD_SLS_DYNAMIC_ACTOR_AVAILABLE = False
    print("[Warning] OSTrack_CKD_ViTKD_SLS_Dynamic_Actor not available")

try:
    from lib.train.actors.ckd_dfl import OSTrack_CKD_DFL_Actor
    DFL_ACTOR_AVAILABLE = True
except ImportError:
    DFL_ACTOR_AVAILABLE = False
    print("[Warning] OSTrack_CKD_DFL_Actor not available")
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "ckd":
        net = build_ostrack_ckd(cfg)    

    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name in ["ckd"]:  # 
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'style': cfg.TRAIN.STYLE_DISTILL_WEIGHT, 
                       'content': cfg.TRAIN.CONTENT_DISTILL_WEIGHT, 'focal': 1., 'cls': 1.0}
        # actor = OSTrack_CKD_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        
        # 根据配置选择使用不同的actor版本
        ckd_loss_type = getattr(cfg.TRAIN, 'CKD_LOSS', 'CKD')
        enable_response_kd = getattr(cfg.TRAIN, 'ENABLE_RESPONSE_KD', False)
        enable_sls_loss = getattr(cfg.TRAIN, 'ENABLE_SLS_LOSS', False)  # SLS损失开关
        enable_dynamic_sls = getattr(cfg.TRAIN, 'ENABLE_DYNAMIC_SLS', False)  # 动态SLS开关
        actor_type = getattr(cfg.TRAIN, 'ACTOR_TYPE', None)  # 显式指定 actor 类型
        use_attn_actor = False  # 标记是否使用了带attention的actor
        
        # 优先使用 ACTOR_TYPE 配置
        if actor_type == 'CKD_DFL' and DFL_ACTOR_AVAILABLE:
            print("[Train] Using OSTrack_CKD_DFL_Actor - Training with DFL (Distribution Focal Loss)")
            actor = OSTrack_CKD_DFL_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
            use_attn_actor = True
        elif enable_sls_loss and enable_dynamic_sls and ckd_loss_type == 'CKD_vitkd' and VITKD_SLS_DYNAMIC_ACTOR_AVAILABLE:
            print("[Train] Using OSTrack_CKD_ViTKD_SLS_Dynamic_Actor - ViTKD + Dynamic SLS Loss")
            actor = OSTrack_CKD_ViTKD_SLS_Dynamic_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
            use_attn_actor = True
        elif enable_sls_loss and ckd_loss_type == 'CKD_vitkd' and VITKD_SLS_ACTOR_AVAILABLE:
            print("[Train] Using OSTrack_CKD_ViTKD_SLS_Actor - ViTKD + SLS Loss (standard)")
            actor = OSTrack_CKD_ViTKD_SLS_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
            use_attn_actor = True
        elif enable_response_kd and VITKD_RESP_ACTOR_AVAILABLE:
            print("[Train] Using OSTrack_CKD_ViTKD_Resp_Actor - ViTKD + Response Map Distillation")
            actor = OSTrack_CKD_ViTKD_Resp_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
            use_attn_actor = True
        elif ckd_loss_type == 'CKD_vitkd' and VITKD_ACTOR_AVAILABLE:
            print("[Train] Using OSTrack_CKD_ViTKD_Actor - Student upsamples to teacher resolution (ViTKD style)")
            actor = OSTrack_CKD_ViTKD_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
            use_attn_actor = True
        elif ckd_loss_type == 'CKD_ADV' and ATTN_ACTOR_AVAILABLE:
            print("[Train] Using OSTrack_CKD_Attn_Actor with layer-wise AttentionDownsample")
            actor = OSTrack_CKD_Attn_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
            use_attn_actor = True
        else:
            print("[Train] Using standard OSTrack_CKD_Actor")
            actor = OSTrack_CKD_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        
        # 将 ckd_loss 移动到 GPU（因为它是 actor 的子模块，不会自动移动）
        if hasattr(actor, 'ckd_loss') and isinstance(actor.ckd_loss, torch.nn.Module):
            actor.ckd_loss = actor.ckd_loss.cuda()
            print(f"[Train] Moved actor.ckd_loss to GPU")
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # 如果使用了PRETRAIN_FILE，需要手动加载actor.ckd_loss和response_loss的权重（因为它们不在net中）
    if use_attn_actor and cfg.MODEL.PRETRAIN_FILE != "":
        try:
            checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
            
            # 加载ckd_loss
            if hasattr(actor, 'ckd_loss'):
                # 方法1: 从checkpoint根级别加载ckd_loss（新checkpoint格式）
                if 'ckd_loss' in checkpoint:
                    actor.ckd_loss.load_state_dict(checkpoint['ckd_loss'], strict=False)
                    print(f"[Checkpoint] Loaded ckd_loss from checkpoint root ({len(checkpoint['ckd_loss'])} keys)")
                else:
                    # 方法2: 从net中提取ckd_loss参数（旧格式，如果ckd_loss在net中）
                    ckd_loss_state = {k.replace('ckd_loss.', ''): v for k, v in checkpoint['net'].items() if 'ckd_loss' in k}
                    if len(ckd_loss_state) > 0:
                        actor.ckd_loss.load_state_dict(ckd_loss_state, strict=False)
                        print(f"[Checkpoint] Loaded {len(ckd_loss_state)} parameters for actor.ckd_loss from net")
                    else:
                        print("[Checkpoint] Warning: No ckd_loss parameters found in checkpoint, using random initialization")
                        print("[Info] This is expected if the checkpoint was saved before ckd_loss saving was implemented")
            
            # 加载response_loss（如果actor有这个模块）
            if hasattr(actor, 'response_loss'):
                if 'response_loss' in checkpoint:
                    actor.response_loss.load_state_dict(checkpoint['response_loss'], strict=False)
                    print(f"[Checkpoint] Loaded response_loss from checkpoint root ({len(checkpoint['response_loss'])} keys)")
                else:
                    print("[Checkpoint] No response_loss found in checkpoint, using random initialization")
                    print("[Info] This is expected - response_loss is a new module")
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to load loss module weights: {e}")
            import traceback
            traceback.print_exc()
    
    # 如果配置了冻结，在optimizer之前先冻结网络参数
    freeze_epochs = getattr(cfg.TRAIN, 'FREEZE_BACKBONE_EPOCHS', 0)
    if freeze_epochs > 0:
        print(f"[Freeze] Will freeze backbone for first {freeze_epochs} epochs")
        print(f"[Freeze] This protects pretrained weights while Generator/Response Map warm up")
        # 冻结网络参数（但不冻结ckd_loss和response_loss）
        frozen_count = 0
        for name, param in net.named_parameters():
            # 不要冻结loss模块的参数
            if 'ckd_loss' not in name and 'response_loss' not in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"[Freeze] Frozen {frozen_count} network parameters")
    
    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    
    # Check if we should freeze generator (Phase 2: LoRA fine-tuning)
    freeze_generator = getattr(cfg.TRAIN, 'FREEZE_GENERATOR', False)
    
    # 如果使用AttentionDownsample，需要将其参数加入优化器
    # 注意：只有 nn.Module 类型的 ckd_loss 才有 parameters() 方法（例如 ViTKD Generator）
    if use_attn_actor and hasattr(actor, 'ckd_loss') and isinstance(actor.ckd_loss, nn.Module):
        # 调试：打印所有ckd_loss参数
        all_ckd_params = list(actor.ckd_loss.parameters())
        ckd_loss_params = [p for p in all_ckd_params if p.requires_grad]
        print(f"[Debug] ckd_loss total parameters: {len(all_ckd_params)}")
        print(f"[Debug] ckd_loss trainable parameters: {len(ckd_loss_params)}")
        
        # Phase 2: Freeze generator if specified
        if freeze_generator:
            print(f"[Freeze] Freezing generator (ckd_loss) - FREEZE_GENERATOR=True")
            for p in all_ckd_params:
                p.requires_grad = False
            ckd_loss_params = []

        if len(all_ckd_params) > 0 and len(ckd_loss_params) == 0 and not freeze_generator:
            print("[Warning] ckd_loss has parameters but all are frozen! Setting requires_grad=True...")
            for p in all_ckd_params:
                p.requires_grad = True
            ckd_loss_params = all_ckd_params

        if len(ckd_loss_params) > 0:
            # 添加ckd_loss参数到优化器（ViTKD Generator或其他loss模块）
            optimizer.add_param_group({
                'params': ckd_loss_params,
                'lr': cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            })
            print(f"[Optimizer] Added {len(ckd_loss_params)} parameters from ckd_loss (Generator/Loss modules) to optimizer")
            print(f"[Optimizer] Learning rate for ckd_loss: {cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER}")
        else:
            if freeze_generator:
                print("[Info] Generator frozen - no ckd_loss parameters added to optimizer")
            else:
                print("[Warning] No parameters found in ckd_loss module!")
    
    # 添加response_loss参数到优化器
    if use_attn_actor and hasattr(actor, 'response_loss') and actor.response_loss is not None:
        response_loss_params = [p for p in actor.response_loss.parameters() if p.requires_grad]
        print(f"[Debug] response_loss trainable parameters: {len(response_loss_params)}")
        
        if len(response_loss_params) > 0:
            optimizer.add_param_group({
                'params': response_loss_params,
                'lr': cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            })
            print(f"[Optimizer] Added {len(response_loss_params)} parameters from response_loss to optimizer")
        else:
            print("[Warning] No trainable parameters found in response_loss module!")
    
    # Print LoRA information if present (Phase 2)
    # Note: LoRA parameters are already included in optimizer via PARAM_KEY: [head, ...]
    # because LoRA params are named as "box_head.conv5_xxx.lora_A/lora_B"
    from lib.models.layers.lora import get_lora_parameters, print_lora_info
    lora_params = get_lora_parameters(net)
    if len(lora_params) > 0:
        print_lora_info(net)
        print(f"[Info] LoRA parameters are included in optimizer via PARAM_KEY ['head'] matching")
    
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)