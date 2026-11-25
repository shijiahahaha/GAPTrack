import os


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.gtot_dir =r'E:\GTOT\Multi_Modal_RGBT_dataset_CSR'
        self.rgbt210_dir = r'E:\RGBT210'
        self.rgbt234_dir = os.environ.get("GAPTRACK_RGBT234_ROOT", os.path.join(self.workspace_dir, "data", "RGBT234"))
        self.LasHeR_dir = os.environ.get("GAPTRACK_LASHER_ROOT", os.path.join(self.workspace_dir, "data", "LasHeR"))
        self.lasher_dir = self.LasHeR_dir
        self.lasher_trainingset_dir = os.path.join(self.lasher_dir, "train", "trainingset")
        self.lasher_testingset_dir = os.path.join(self.lasher_dir, "test", "testingset")
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
