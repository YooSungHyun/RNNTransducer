import torch


class AudioDataLoader(torch.utils.data.DataLoader):
    def __init__(self, pad_token_id, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.pad_token_id = pad_token_id

    def _collate_fn(self, batch):
        def seq_length_(p):
            return len(p[0])

        def target_length_(p):
            return len(p[1])

        seq_lengths = [len(s[0]) for s in batch]
        target_lengths = [len(s[1]) for s in batch]

        max_seq_sample = max(batch, key=seq_length_)[0]
        max_target_sample = max(batch, key=target_length_)[1]

        max_seq_size = max_seq_sample.size(0)
        max_target_size = len(max_target_sample)

        feat_size = max_seq_sample.size(1)
        batch_size = len(batch)

        seqs = torch.zeros(batch_size, max_seq_size, feat_size)

        targets = torch.zeros(batch_size, max_target_size).to(torch.long)
        targets.fill_(self.pad_token_id)

        for x in range(batch_size):
            sample = batch[x]
            tensor = sample[0]
            target = sample[1]
            seq_length = tensor.size(0)
            seqs[x].narrow(0, 0, seq_length).copy_(tensor)
            targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

        return seqs, targets, seq_lengths, target_lengths


# torch.Size([16, 1192, 240])
# torch.Size([16, 31])
# [919, 474, 649, 365, 544, 225, 407, 590, 627, 468, 473, 450, 304, 436, 406, 1192]
# [24, 16, 15, 12, 31, 10, 18, 16, 21, 17, 15, 20, 11, 14, 13, 17]
