import pytorch_lightning as pl
from model import RNNTransducer
from utils import read_json, InferenceArguments, get_concat_dataset, dataclass_to_namespace
import torch
from simple_parsing import ArgumentParser
import kenlm


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
    lm = kenlm.LanguageModel(hparams.lm_path)
    model.eval()
    test_target = datasets[0]
    input_audios = torch.stack([test_target["input_values"]], dim=0)
    input_lengths = [s.size(0) for s in input_audios]
    with torch.no_grad():
        pred_sentences = model.jointnet.recognize_improved_beams(
            input_audios, input_lengths, model.tokenizer.pad_token_id, 100, lm=lm, tokenizer=model.tokenizer
        )
    print(model.tokenizer.decode(test_target["input_ids"]))
    for pred_sentence in pred_sentences:
        print(model.tokenizer.decode(pred_sentence))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_arguments(InferenceArguments, dest="inference_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "inference_args")
    main(args)
