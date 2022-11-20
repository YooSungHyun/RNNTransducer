import torch
from torch.nn.utils.rnn import pad_sequence


class AudioDataLoader(torch.utils.data.DataLoader):
    def __init__(self, pad_token_id, bos_token_id, n_mels, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.n_mels = n_mels

    def _collate_fn(self, batch):
        # batch : input_values: log_melspect, ["grapheme_labels"]["input_ids"]: tokenized labels
        # input_values shape: (seq, mel_cnt)
        input_audios = [s["input_values"] for s in batch]
        audio_lengths = [s["input_values"].size(0) for s in batch]
        targets = [torch.as_tensor(s["input_ids"], dtype=torch.int32) for s in batch]
        target_lengths = [len(s["input_ids"]) for s in batch]
        audio_lengths = torch.IntTensor(audio_lengths)
        target_lengths = torch.IntTensor(target_lengths)

        # input_ids: (,token)
        input_texts = list()
        for s in batch:
            input_texts.append(
                torch.cat(
                    [
                        torch.full(size=[1], fill_value=self.bos_token_id),
                        torch.as_tensor(s["input_ids"], dtype=torch.int32),
                    ]
                )
            )
        text_lengths = [len(s) for s in input_texts]
        assert self.n_mels == batch[0]["input_values"].size(-1), "config의 feature shape과 실제 데이터의 feature가 다름"
        for s in range(len(target_lengths)):
            assert text_lengths[s] == target_lengths[s] + 1, "prednet의 Input은 targets_lengts에 +1(blank)여야 합니다. 데이터 오류!"
        input_texts = pad_sequence(input_texts, batch_first=True, padding_value=self.pad_token_id)
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_token_id)

        return input_audios, audio_lengths, input_texts, text_lengths, targets, target_lengths
