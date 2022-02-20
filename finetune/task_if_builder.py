# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Returns task instances given the task name."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import configure_if_finetuning
from finetune.classification import classification_if_tasks
from model import tokenization


def get_tasks(config: configure_if_finetuning.FinetuningIFConfig):
    tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file,
                                           do_lower_case=config.do_lower_case)
    return [get_task(config, task_name, tokenizer)
            for task_name in config.task_names]


def get_task(config: configure_if_finetuning.FinetuningIFConfig, task_name,
             tokenizer):
    """Get an instance of a task based on its name."""
    if task_name == "fn":
        return classification_if_tasks.FN(config, tokenizer)
    elif task_name == "npc":
        return classification_if_tasks.NPC(config, tokenizer)
    elif task_name == "vn":
        return classification_if_tasks.VN(config, tokenizer)
    elif task_name == "wn":
        return classification_if_tasks.WN(config, tokenizer)
    elif task_name == "fn_full":
        return classification_if_tasks.FNFull(config, tokenizer)
    elif task_name == "npc_full":
        return classification_if_tasks.NPCFull(config, tokenizer)
    elif task_name == "vn_full":
        return classification_if_tasks.VNFull(config, tokenizer)
    elif task_name == "wn_full":
        return classification_if_tasks.WNFull(config, tokenizer)
    else:
        raise ValueError("Unknown task " + task_name)
