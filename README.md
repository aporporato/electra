# ELECTRA FOR IF

## Introduction

Experiment on the usefulness of Large Language Model in the context of Interactive Fiction.

Inspired by
the [CIS 700-001 - Interactive Fiction and Text Generation](https://interactive-fiction-class.org/index.html)
Course.

## ELECTRA

[**ELECTRA**](https://github.com/google-research/electra) is a method for self-supervised language representation
learning. It can be used to pre-train transformer networks using relatively little compute. ELECTRA models are trained
to distinguish "real" input tokens vs "fake" input tokens generated by another neural network, similar to the
discriminator of a [GAN](https://arxiv.org/pdf/1406.2661.pdf). For a detailed description and experimental results,
refer to the original ICLR 2020 paper
[ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/pdf?id=r1xMH1BtvB)
(Clark et al.). Aureliano Porporato did not take part in the original project in any way.

### Original Project Requirements

* Python 3.7
* [TensorFlow](https://www.tensorflow.org/) _1.15_
* [NumPy](https://numpy.org/) (1.21.2)
* [scikit-learn](https://scikit-learn.org/stable/) (1.0.1) and [SciPy](https://www.scipy.org/) (1.7.3) for computing
  some evaluation metrics only.

## Fine-tuning

The original small model has been finetuned on a subset of the [GLUE](https://gluebenchmark.com/) classification tasks (
i.e.: [CoLA](https://nyu-mll.github.io/CoLA/), [SST](https://nlp.stanford.edu/sentiment/index.html),
MRPC, [STS](http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark), QQP, [MNLI](https://cims.nyu.edu/~sbowman/)
, [RTE](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment)
, [WNLI](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html) and [AX](https://gluebenchmark.com/diagnostics)
[included in MNLI]), the [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) QA task and the
[CoNLL](https://www.clips.uantwerpen.be/conll2000/chunking/) text chunking task. The data were prepared as
shown [here](https://github.com/google-research/electra#setup-1). The model was finetuned on the data from all the tasks
at once, for 10 epochs (`$DATA_DIR` = `"D:\ELECTRAIFFinetuned\electra"`).

`python run_finetuning.py --data-dir $DATA_DIR --model-name electra_small --hparams finetuning_config.json`

That command took ~40 hours on a single NVIDIA GeForce GTX 1050 Ti.

### Results

The results (test set for chunking, dev set for the other tasks) for the model finetuned in the way described above are
reported (ELECTRA-Small-Finetuned). For references, also the result expected for the finetuned original model, as
reported [here](https://github.com/google-research/electra#expected-results), are shown (ELECTRA-Small).

|  | MNLI | WNLI | STS | MRPC | Chunking | RTE | QQP | SST | SQuAD 2.0 | CoLA |
| --- | --- | --- | --- | --- | ---  | --- | --- | --- | --- | --- |
| Metrics | Acc | Acc | Pear/Spear | Acc  | F1 | Acc | Acc | Acc | EM | MCC |
| ELECTRA-Small | 81.3 | - | -/87.5 |  88.0 | 96.5  | 66.7 | 89.0 | 91.2 | 70.1 | 57.0 |
| ELECTRA-Small-Finetuned | 82.6 | 32.4 | 86.5/86.5 |  86.5 | 96.4 | 71.1 | 90.3 | 89.8 | 70.1 |  49.7 |

## Citation

Original paper:

```
@inproceedings{clark2020electra,
  title = {{ELECTRA}: Pre-training Text Encoders as Discriminators Rather Than Generators},
  author = {Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning},
  booktitle = {ICLR},
  year = {2020},
  url = {https://openreview.net/pdf?id=r1xMH1BtvB}
}
```

## Finetune ELECTRA on IF dataset

The finetuned small model has been in turn finetuned on various classification tasks. Details on the datasets can be
fount [here](https://github.com/aporporato/jericho-corpora). Move each folder in the `IF` one into
your `finetuning_data` directory (refer to
the [original project](https://github.com/google-research/electra#finetune-electra-on-a-glue--task) for detailed
instructions). The model was finetuned on the data from all the tasks at once, for 10 epochs.

```
python run_if_finetuning.py --data-dir $DATA_DIR --model-name electra_small_finetuned --hparams '{"model_size": "small", "task_names": ["npc", "fn", "vn", "wn"]}'
```

That command took less than 1 hours on a single NVIDIA GeForce GTX 1050 Ti.

### Results

The results on test set for the tasks are reported for the newly finetuned model (ELECTRA-Small-IF). For comparison, the
result for the original model and for the model finetuned on GLUE tasks are also reported (ELECTRA-Small and
ELECTRA-Small-Finetuned respectively, obtained with `"do_train": false` option).

|  | WordNet | VerbNet | FrameNet | Third Person Command |
| --- | --- | --- | --- | --- |
| Metrics | Acc | Acc | Acc | Acc |
| ELECTRA-Small | 0.1 | 0.3 | 0.3 | 0.8 |
| ELECTRA-Small-Finetuned | 0.1 | 0.9 | 0.0 | 0.0 |
| ELECTRA-Small-IF | 92.9 | 52.7 | 80.8 | 99.4 |