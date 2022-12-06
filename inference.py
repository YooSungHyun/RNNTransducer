import pytorch_lightning as pl
from model import RNNTransducer
from utils import read_json, InferenceArguments, get_concat_dataset, dataclass_to_namespace
import torch
from torch.nn.utils.rnn import pack_sequence
from simple_parsing import ArgumentParser


def main(hparams):
    # prednet_params, transnet_params, jointnet_params, args
    # prednet_params: dict, transnet_params: dict, jointnet_params: dict, args: Namespace
    datasets = get_concat_dataset(hparams.pl_data_dir, "eval_clean")
    datasets.set_format("torch", ["input_values", "input_ids"])
    model_config_dict = read_json(hparams.model_config)["model"]
    model = RNNTransducer.load_from_checkpoint(
        hparams.model_dir,
        prednet_params=model_config_dict["prednet"],
        transnet_params=model_config_dict["transnet"],
        jointnet_params=model_config_dict["jointnet"],
        args=hparams,
    )
    model.eval()
    input_audios = [datasets[0]["input_values"]]
    input_audios = pack_sequence(input_audios)
    with torch.no_grad():
        y_hat_logits = model.jointnet.recognize(input_audios, model.tokenizer.bos_token_id).detach()
    print(y_hat_logits)
    print(model.tokenizer.decode(datasets[0]["input_ids"]))
    print(model.tokenizer.batch_decode(y_hat_logits))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_arguments(InferenceArguments, dest="inference_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "inference_args")
    main(args)
