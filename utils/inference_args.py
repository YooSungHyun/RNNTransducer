from dataclasses import dataclass, field


@dataclass
class InferenceArguments:
    """Help string for this group of command-line arguments"""

    pl_data_dir: list[str] = field(default_factory=list)
    vocab_path: str = "../config/vocab.json"
    model_dir: str = "../model/model.ckpt"
    model_config: str = "../config/config.json"
    val_on_cpu: bool = False
