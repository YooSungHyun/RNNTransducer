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
        tensor_audio_lengths = torch.IntTensor(audio_lengths)
        target_lengths = torch.IntTensor(target_lengths)

        # input_ids: (,token)
        input_texts = list()
        for s in batch:
            input_texts.append(
                torch.cat(
                    [
                        torch.full(size=[1], fill_value=self.pad_token_id),
                        torch.as_tensor(s["input_ids"], dtype=torch.int32),
                    ]
                )
            )
        text_lengths = [len(s) for s in input_texts]
        assert self.n_mels == batch[0]["input_values"].size(-1), "config의 feature shape과 실제 데이터의 feature가 다름"
        for s in range(len(target_lengths)):
            assert text_lengths[s] == target_lengths[s] + 1, "prednet의 Input은 targets_lengts에 +1(blank)여야 합니다. 데이터 오류!"
        input_audios = pad_sequence(input_audios, batch_first=True, padding_value=self.pad_token_id)
        input_texts = pad_sequence(input_texts, batch_first=True, padding_value=self.pad_token_id)
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_token_id)

        # input_audios, tensor_audio_lengths, input_texts, targets, audio_lengths, target_lengths: on_cuda
        # audio_lengths, text_lengths : on_cpu or not tensor
        # pack_pad를 진행하기위한 lengths계산은 무조건 CPU에서 동작해야하는데, rnntloss를 계산하기위한 lengths는 무조건 Tensor여야 한다.
        # 여기서 Tensor로 집어넣으면 무조건 gpu로 가기때문에, .cpu() .cuda()로 옮겨다니면 오버헤드가 발생할 여지가 있으므로, 애시당초에 여러 버전의 lengths를 주기로 했다.
        return input_audios, audio_lengths, tensor_audio_lengths, input_texts, text_lengths, targets, target_lengths
