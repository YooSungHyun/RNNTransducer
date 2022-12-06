from dataclasses import dataclass, field


@dataclass
class InferenceArguments:
    """Help string for this group of command-line arguments"""

    pl_data_dir: str = "../datasets"
    vocab_path: str = "../config/vocab.json"
    model_dir: str = "../model/model.ckpt"
    model_config: str = "../config/config.json"
    val_on_cpu: bool = False
    precision: int = 16
