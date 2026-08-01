class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = r'D:\CKD-ACMMM2024-master'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.gtot_dir =r'E:\GTOT\Multi_Modal_RGBT_dataset_CSR'
        self.rgbt210_dir = r'E:\RGBT210'
        self.rgbt234_dir = r'E:\RGBT234'
        self.LasHeR_dir = r'E:\LasHeR0327'
        self.lasher_trainingset_dir = r'E:\LasHeR0327\train\trainingset'
        self.lasher_testingset_dir = r'E:\LasHeR0327\test\testingset'
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
