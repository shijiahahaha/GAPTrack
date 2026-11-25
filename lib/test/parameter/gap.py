from .ckd import parameters as ckd_parameters


def parameters(yaml_name: str, checkpoint_path=None):
    return ckd_parameters(yaml_name, checkpoint_path, experiment="gap")
