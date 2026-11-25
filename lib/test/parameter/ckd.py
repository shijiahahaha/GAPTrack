import os

from lib.config.ckd.config import cfg as default_cfg
from lib.config.ckd.config import clone_config, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.utils import TrackerParams


def parameters(yaml_name: str, checkpoint_path=None, *, experiment="ckd"):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    yaml_file = os.path.join(prj_dir, "experiments", experiment, f"{yaml_name}.yaml")
    cfg = clone_config(default_cfg)
    update_config_from_file(yaml_file, base_cfg=cfg)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            prj_dir, "output", "checkpoints", "train", experiment, yaml_name,
            f"OSTrack_CKD_ep{cfg.TEST.EPOCH:04d}.pth.tar",
        )
    params.checkpoint = checkpoint_path

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
