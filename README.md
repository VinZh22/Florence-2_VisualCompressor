# Visual Compressor — Florence-2 Finetuning & Compression

This repository provides tools to finetune and evaluate a Florence-2 model with optional visual-token compression (average pooling or a learnable pooling operator).

It is a personal project that aims to re-implement the paper **Efficient Large Multi-modal Models via Visual Context Compression** on the VLM Florence-2, because it is smaller than the Llava 7B used in the paper. It will be tested on the benchmark VQAv2.

Due to my limited computing power (single RTX4070 laptop GPU), we will work on subsets of the VQA dataset.

## Quick links
- Project entry: [run.py](run.py) — contains the [`main`](run.py) CLI and example training/evaluation flow.  
- Data & dataset: [data.py](data.py) — defines the [`VQADataset`](data.py) used for VQA-style training.  
- Training helper: [train_compressor.py](train_compressor.py) — contains the [`train`](train_compressor.py) routine used by [`main`](run.py).  
- Inference & evaluation: [inference.py](inference.py) — contains [`evaluate`](inference.py) and [`run_example`](inference.py).  
- Optimizer helper: [soap.py](soap.py) — contains the [`SOAP`](soap.py) optimizer.  
- Compression utilities: [Florence-2-base/compression_fct.py](Florence-2-base/compression_fct.py) — includes [`compress_average_pooling`](Florence-2-base/compression_fct.py) and [`Learnable_pooling`](Florence-2-base/compression_fct.py).  
- Model implementation & config: [Florence-2-base/modeling_florence2.py](Florence-2-base/modeling_florence2.py) — includes `Florence2` model classes (e.g. [`Florence2ForConditionalGeneration`](Florence-2-base/modeling_florence2.py)).  
- Model configuration: [Florence-2-base/configuration_florence2.py](Florence-2-base/configuration_florence2.py) — includes [`Florence2LanguageConfig`](Florence-2-base/configuration_florence2.py) (compression-related fields are added here).
- Downsampling of VQA dataset: [Data/preprocess.ipynb](Data/preprocess.ipynb) — select $1\%$, $5\%$ and $10\%$ for faster training and testing (only a proof-of-concept for now). 

## Requirements
- Python 3.8+
- PyTorch
- transformers (for AutoProcessor, AutoModelForCausalLM, AutoConfig)
- datasets, Pillow, tqdm
- CUDA if training on GPU

Install typical dependencies (example):
```sh
pip install torch transformers datasets pillow tqdm
```

## Usage

1. Place/download the Florence-2 model folder (example: `Florence-2-base/`) at repository root. This repo expects a local model folder like `./Florence-2-base`.

2. Prepare your dataset under `Data/` (VQA-style). The [`VQADataset`](data.py) constructor is used by [run.py](run.py).

3. Run finetuning / evaluation via the CLI in [run.py](run.py). Example commands:

- Run training with learnable pooling compression:
```sh
python run.py --model_name Florence-2-base --compression_mode learnable_pool --compression_factor 4 --compression_stage 6 --compression_sorted
```

- Run with average pooling compression:
```sh
python run.py --model_name Florence-2-base --compression_mode avg_pool --compression_factor 4 --compression_stage 6
```

- Run without compression:
```sh
python run.py --model_name Florence-2-base --compression_mode none
```

CLI options and defaults are defined in [run.py](run.py). The script sets these values on the loaded config (see [`Florence2LanguageConfig`](Florence-2-base/configuration_florence2.py)) before model construction.

## Compression modes
- Average pooling: implemented as [`compress_average_pooling`](Florence-2-base/compression_fct.py).  
- Learnable pooling: implemented as [`Learnable_pooling`](Florence-2-base/compression_fct.py). When using learnable pooling, [run.py](run.py) will call [`train`](train_compressor.py) to train pooling modules if configured.

Compression parameters are passed to the model config in [run.py](run.py) and consumed by the language/decoder classes in [Florence-2-base/modeling_florence2.py](Florence-2-base/modeling_florence2.py).

The idea behind the learnable pooling is to study the choice of average pooling used in the original paper. Notably comparing it with max pooling. Which is the reason why there is a sorted parameter, in order for the 1D pooling to be independant on the position of the tokens.

## File overview
- [run.py](run.py) — CLI; sets config flags and coordinates dataset, training and evaluation (uses [`main`](run.py)).  
- [data.py](data.py) — dataset classes (`VQADataset`) and utility functions.  
- [train_compressor.py](train_compressor.py) — training loop used to fine-tune pooling modules.  
- [inference.py](inference.py) — evaluation and example inference routines (`evaluate`, `run_example`).  
- [soap.py](soap.py) — custom optimizer implementation (`SOAP`).  
- [Florence-2-base/](Florence-2-base/) — local model weights & implementation (model code, config, compression helpers). Key files:
  - [Florence-2-base/modeling_florence2.py](Florence-2-base/modeling_florence2.py) — model classes (vision + language, e.g. [`Florence2ForConditionalGeneration`](Florence-2-base/modeling_florence2.py)).  
  - [Florence-2-base/configuration_florence2.py](Florence-2-base/configuration_florence2.py) — model configuration (`Florence2LanguageConfig`) including compression fields.  
  - [Florence-2-base/compression_fct.py](Florence-2-base/compression_fct.py) — pooling implementations.

- [Data/preprocess.ipynb](Data/preprocess.ipynb) — Jupyter notebook to select $1\%$, $5\%$ and $10\%$ for faster training and testing (only a proof-of-concept for now). Please note that the $1\%$ is strictly included in the $5\%$ dataset, same for $10\%$, and are the first elements of the bigger dataset (reason being the usage of the same random state).

## Notes & tips
- The script uses `AutoConfig.from_pretrained(.., trust_remote_code=True)` and `AutoModelForCausalLM.from_pretrained(.., trust_remote_code=True)` in [run.py](run.py) to load the local model implementation.  
- Adjust training hyperparameters in [run.py](run.py) or wire them to CLI args if needed. The default training flow will call [`train`](train_compressor.py) when `--compression_mode learnable_pool` is used.  
- Evaluate on a held-out split via the `evaluate` routine in [inference.py](inference.py).

## License & support
See [Florence-2-base/LICENSE](Florence-2-base/LICENSE) and [Florence-2-base/SUPPORT.md](Florence-2-base/SUPPORT.md) for licensing and support notes.

If you need help running examples, inspect:
- the CLI and flow in [run.py](run.py) (`main`),  
- dataset layout expected by [data.py](data.py) (`VQADataset`), and  
- the compression implementations in [Florence-2-base/compression_fct.py](Florence-2-base/compression_fct.py).
