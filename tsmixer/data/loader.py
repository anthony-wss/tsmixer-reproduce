from torch.utils.data import DataLoader, Dataset
from .ett import ETDataset

def get_data(L, T, name='ETTh1', flag='train'):
    if name[:3] == 'ETT':
        dataset = ETDataset(L, T, flag, name)
    else:
        raise NotImplementedError
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True if flag=='train' else False,
        num_workers=8
    )
    print(flag, len(dataset))
    return dataset, dataloader