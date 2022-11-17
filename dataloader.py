import torch
from torch.nn.utils.rnn import pad_sequence


class AudioDataLoader(torch.utils.data.DataLoader):
    def __init__(self, pad_token_id, n_mels, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.pad_token_id = pad_token_id
        self.n_mels = n_mels

    def _collate_fn(self, batch):
        # batch : input_values: log_melspect, ["grapheme_labels"]["input_ids"]: tokenized labels
        # input_values shape: (seq, mel_cnt)
        input_values = [s["input_values"] for s in batch]
        seq_lengths = [s["input_values"].size(0) for s in batch]
        # input_ids: (,token)
        labels = [s["input_ids"] for s in batch]
        target_lengths = [len(s["input_ids"]) for s in batch]

        assert self.n_mels == batch[0]["input_values"].size(-1), "config의 feature shape과 실제 데이터의 feature가 다름"

        input_values = pad_sequence(input_values, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_token_id)

        return input_values, labels, seq_lengths, target_lengths
