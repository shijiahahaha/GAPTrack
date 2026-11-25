import os

from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.gtot_path = '/data1/andong.lu/data/RGBT_DATA/GTOT/'
    settings.prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    settings.lasher_path = os.environ.get("GAPTRACK_LASHER_ROOT", os.path.join(settings.prj_dir, "data", "LasHeR"))
    settings.lashertestingset_path = os.path.join(settings.lasher_path, "test", "testingset")
    settings.network_path = os.path.join(settings.prj_dir, "lib", "test", "networks")
    settings.result_plot_path = os.path.join(settings.prj_dir, "result_plot")
    settings.results_path = os.path.join(settings.prj_dir, "tracking_result")
    settings.rgbt210_path = '/data1/andong.lu/data/RGBT_DATA/RGBT210/'
    settings.rgbt234_path = os.environ.get("GAPTRACK_RGBT234_ROOT", os.path.join(settings.prj_dir, "data", "RGBT234"))
    settings.save_dir = './'
    settings.segmentation_path = os.path.join(settings.prj_dir, "segmentation_result")

    return settings


