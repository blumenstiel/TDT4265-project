from torch.utils.data import ConcatDataset
from ssd.config.path_catlog import DatasetCatalog
from .voc import VOCDataset
from .mnist import MNISTDetection
from .tdt4265 import TDT4265Dataset 
from .rdd2020 import RDDDataset 
_DATASETS = {
    'VOCDataset': VOCDataset,
    'MNISTDetection': MNISTDetection,
    'TDT4265Dataset': TDT4265Dataset, 
    "RDDDataset": RDDDataset 
}


def build_dataset(base_path: str, dataset_list, transform=None, target_transform=None, is_train=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(base_path, dataset_name)
        args = data['args']
        factory = _DATASETS[data['factory']]
        args['transform'] = transform
        args['target_transform'] = target_transform
        if factory == VOCDataset:
            args['keep_difficult'] = not is_train
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    return [dataset]
