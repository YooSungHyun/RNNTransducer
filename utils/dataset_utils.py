import os
from typing import List
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
import json
from .comfy import mkdir
from collections import defaultdict


def get_dataset_paths(dataset_dir: os.PathLike, train_type: str) -> List[os.PathLike]:
    """_summary_
    arrow파일이 저장된 datasets의 dir에 0 ~ n번 까지의 arrow데이터를 불러오는 함수 입니다.

    _dir_structure_
    .
    └── Arrow_data_dir/
        ├── sharded_train/
        │   ├── arrow_01/
        │   │   └── arrow_file
        │   ├── arrow_02/
        │   │   └── arrow_file
        │   └── ...
        ├── sharded_eval
        └── sharded_dev

    Args:
        dataset_dir (os.PathLike): arrow데이터가 저장되어 있는 경로를 전달받습니다.
        train_type (str): 불러올 데이터의 폴더 이름을 지정합니다. 이때 폴더 이름은 train, dev, eval과 같은 이름을 사용합니다.

    Returns:
        List[os.PathLike]: train_type아래에 저장되어 있는 arrow파일을 list로 반환합니다.
        ex: [kspon/train/0, kspon/train/1 ...]
    """

    dataset_dir = os.path.join(dataset_dir, train_type)
    sharding_dataset_paths = [os.path.join(dataset_dir, shard_num) for shard_num in os.listdir(dataset_dir)]
    sharding_dataset_paths.sort()
    return sharding_dataset_paths


def get_concat_dataset(dataset_dirs: List[os.PathLike], train_type: str) -> Dataset:
    """_summary_
    dir에 있는 분할된 arrow파일을 불러온 다음 하나로 합친뒤 반환하는 함수 입니다.

    _dir_structure_
    .
    └── Arrow_data_dir/
        ├── sharded_train/
        │   ├── arrow_01/
        │   │   └── arrow_file
        │   ├── arrow_02/
        │   │   └── arrow_file
        │   └── ...
        ├── sharded_eval
        └── sharded_dev

    Args:
        dataset_dir (os.PathLike): arrow데이터가 저장되어 있는 경로를 전달받습니다.
        train_type (str): 불러올 데이터의 폴더 이름을 지정합니다. 이때 폴더 이름은 train, dev, eval과 같은 이름을 사용합니다.

    Returns:
        Dataset: HuggingFace Dataset를 반환합니다.
    """
    dataset_lists = []
    for dataset_dir in dataset_dirs:
        sharding_dataset_paths = get_dataset_paths(dataset_dir, train_type)
        sharding_datasets = [load_from_disk(p) for p in tqdm(sharding_dataset_paths)]
        postprocess_dataset = concatenate_datasets(sharding_datasets)
        postprocess_log_path = os.path.join(dataset_dir, "postprocess_log.json")
        if os.path.isfile(postprocess_log_path):
            with open(postprocess_log_path, "r") as log_file:
                postprocess_log = json.load(log_file)
            print(postprocess_log)
            if "filter" in postprocess_log.keys():
                if train_type in postprocess_log["filter"].keys():
                    postprocess_dataset = postprocess_dataset.filter(
                        cache_file_name=os.path.join(dataset_dir, postprocess_log["filter"][train_type]["path"]),
                        num_proc=int(postprocess_log["filter"][train_type]["num_proc"]),
                    )
        dataset_lists.append(postprocess_dataset)
    concat_dataset = concatenate_datasets(dataset_lists)
    return concat_dataset


def get_cache_file_path(cache_dir: str, cache_task_func: callable, train_type: str) -> str:
    if cache_dir is None:
        return None
    else:
        mkdir(cache_dir)
        mkdir(os.path.join(cache_dir, cache_task_func.__name__))
        mkdir(os.path.join(cache_dir, cache_task_func.__name__, train_type))
        return os.path.join(cache_dir, cache_task_func.__name__, train_type, "cache.arrow")


def set_cache_log(dataset_dir: str, num_proc: int, cache_task_func: callable, train_type: str):
    postprocess_log_path = os.path.join(dataset_dir, "postprocess_log.json")
    result_dict = defaultdict()
    history_dict = dict()
    if os.path.isfile(postprocess_log_path):
        # 이미 진행됐던 캐시가 있다면, postprocess_log.json이 존재할 것이다. 읽어서 만약 중복된다면 더 이상 진행 안함
        with open(postprocess_log_path, "r") as history_file:
            history_dict = json.load(history_file)
        if cache_task_func.__name__ in history_dict.keys():
            print(f"이미 {cache_task_func.__name__} 캐시가 존재합니다!! 스킵됩니다!")
            return
    # 파일은 있는데 중복되는게 없거나, 아예 파일이 없는경우
    result_dict[train_type] = dict()
    result_dict[train_type]["num_proc"] = num_proc
    result_dict[train_type]["path"] = cache_task_func.__name__ + "/" + train_type + "/" + "cache.arrow"
    history_dict[cache_task_func.__name__] = result_dict
    with open(postprocess_log_path, "w") as history_file:
        json.dump(history_dict, history_file, indent=4)
