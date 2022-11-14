import torch


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
        seq_lengths = [s["input_values"].size(0) for s in batch]
        # input_ids: (,token)
        target_lengths = [len(s["grapheme_labels"]["input_ids"]) for s in batch]

        max_seq_size = max(seq_lengths)
        max_target_size = max(target_lengths)

        assert self.n_mels == batch[0]["input_values"].size(-1), "config의 feature shape과 실제 데이터의 feature가 다름"
        feat_size = self.n_mels
        batch_size = len(batch)

        input_values = torch.zeros(batch_size, max_seq_size, feat_size)

        labels = torch.zeros(batch_size, max_target_size).to(torch.long)
        labels.fill_(self.pad_token_id)

        for batch_idx in range(batch_size):
            sample = batch[batch_idx]
            feature = sample["input_values"]
            target = sample["grapheme_labels"]["input_ids"]
            seq_length = feature.size(0)
            input_values[batch_idx].narrow(0, 0, seq_length).copy_(feature)
            labels[batch_idx].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

        return input_values, labels, seq_lengths, target_lengths
