# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the nq dataset to parquet format
Modified from Search-R1 by adding flag --data_source. default is nq, which is equivalent to original Search-R1.
Otherwise, can choose a HF dataset instead. It must have a train and test splits; must have columns 'question' and 'golden_answers'.
e.g. 
    python -m ipdb scripts/data_process/nq_search.py --data_source jmhb/bioasq_trainv0_n1609_test100
"""

import re
import os
import datasets
import ipdb

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument(
        '--data_source', type=str,
        default='nq')  # 'nq' for default, or provide HF dataset
    parser.add_argument('--num_train_samples',
                        type=int,
                        default=None,
                        help="Number of training samples to process")
    parser.add_argument('--num_test_samples',
                        type=int,
                        default=None,
                        help="Number of test samples to process")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="Random seed for shuffling")

    args = parser.parse_args()
    data_source = args.data_source

    if data_source == 'nq':
        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')
        args.local_dir = "./data/nq_search"
    else:
        dataset = datasets.load_dataset(data_source)
        args.local_dir = f"./data/{data_source}_search"

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    if args.num_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(
            range(args.num_train_samples))

    if args.num_test_samples is not None:
        test_dataset = test_dataset.shuffle(seed=args.seed).select(
            range(args.num_test_samples))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'),
                                      with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'),
                                    with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(
        f"Processed {len(train_dataset)} training samples and {len(test_dataset)} test samples"
    )
    print(f"Local directory: {local_dir}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
