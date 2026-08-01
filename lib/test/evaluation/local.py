from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.gtot_path = '/data1/andong.lu/data/RGBT_DATA/GTOT/'
    settings.lasher_path = '/data1/andong.lu/data/RGBT_DATA/LasHeR/'
    settings.lashertestingset_path = '/data1/andong.lu/data/RGBT_DATA/LasHeR/'
    settings.network_path = r'D:\CKD-ACMMM2024-master\lib\test\networks'    # Where tracking networks are stored.
    settings.prj_dir = './'
    settings.result_plot_path = r'D:\CKD-ACMMM2024-master\result_plot'
    settings.results_path = r'D:\CKD-ACMMM2024-master\tracking_result'    # Where to store tracking results (改回项目根目录)
    settings.rgbt210_path = '/data1/andong.lu/data/RGBT_DATA/RGBT210/'
    settings.rgbt234_path = r'E:\RGBT234'
    settings.save_dir = './'
    settings.segmentation_path = r'D:\CKD-ACMMM2024-master\segmentation_result'

    return settings


