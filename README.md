# SA-EBAMA

Official implementation for the manuscript:

**Separation-Aware Energy-Based Attention Map Alignment (SA-EBAMA): Object-Level Attention Separation and Adaptive Inference Guidance for Compositional Text-to-Image Generation**


SA-EBAMA adds object separation regularization and adaptive timestep guidance to EBAMA during inference. The text encoder, U-Net and VAE remain frozen.

## Repository contents

| Path | Contents |
| --- | --- |
| `inference.py`, `diffusion_pipeline.py`, `compute_loss.py` | Single-prompt generation, diffusion pipeline and EBAMA alignment losses |
| `losses/`, `utils/` | Separation regularization, attention-map diagnostics and timestep scheduling |
| `dataset/` | 400 released prompts from AnE, DVMP, ABC-6K and T2I-Spatial |
| `scripts/` | Benchmark generation and evaluation, parameter analysis, runtime reporting, figures and human evaluation |
| `metrics/` | CLIP image–text and BLIP caption–text evaluation utilities |
| `result/table/`, `result/paper_figure/` | Archived experimental tables, image index and manuscript figures |

## Setup

From the repository root, use Python 3.10 and a CUDA-capable machine. The environment installs PyTorch 2.0.1 with CUDA 11.8 and Diffusers 0.21.1. The first run needs access to the model weights, or a local Stable Diffusion v1.4 directory supplied through `--model_path`.

```bash
conda env create -f environment.yaml
conda activate sa-ebama
python -m spacy download en_core_web_trf
```

## Generation

```bash
python inference.py --prompt "a yellow sliced banana and a green horse" --prompt_id demo_0001 --seed 1 --gamma_sep 0.05 --sep_percentile 0.70 --adaptive_overlap_threshold 0.45
python scripts/main400_all_methods/run_formal_internal.py --model_path CompVis/stable-diffusion-v1-4 --methods sa_ebama --end_idx 1 --dry_run
```

The first command generates one example with the manuscript settings. The second prints the four seed commands for the first benchmark prompt. For the full 400-prompt run, omit `--end_idx 1 --dry_run`. Use `--resume` to skip prompt-seed pairs with complete outputs. Omit `--methods` to run all four configurations under `outputs/formal_main400_internal_locked/`.

| Configuration | Separation | Adaptive guidance | Schedule |
| --- | --- | --- | --- |
| EBAMA | off | off | fixed25 |
| EBAMA + Separation | on | off | fixed25 |
| EBAMA + Adaptive | off | on | sparse10 |
| SA-EBAMA | on | on | sparse10 |



| Parameter                      | Value                                                    |
| ------------------------------ | -------------------------------------------------------- |
| Base model                     | Stable Diffusion v1.4                                    |
| Scheduler                      | DDIM                                                     |
| Sampling steps                 | 50                                                       |
| Classifier-free guidance scale | 7.5                                                      |
| Image size                     | 512 × 512                                                |
| Seeds                          | `1`, `42`, `1234`, `70690`                               |
| EBAMA step size                | `20`                                                     |
| EBAMA distance                 | `cos`                                                    |
| `lambda_ours`                  | `0.5`                                                    |
| Attention resolution           | `attn_res=256`, corresponding to a 16 × 16 attention map |
| `gamma_sep`                    | `0.05`                                                   |
| `sep_tau`                      | `percentile`                                             |
| `sep_percentile`               | `0.70`                                                   |
| `sep_temp`                     | `0.15`                                                   |
| `adaptive_mode`                | `sparse10`                                               |
| `adaptive_overlap_threshold`   | `0.45`                                                   |
| `adaptive_dynamic_step`        | `True`                                                   |
| `max_guidance_step`            | `25`                                                     |


Each prompt directory contains per-seed images and, when enabled, guidance traces. Add `--save_overlap_vis` to save attention diagnostics.

## Reproduce the benchmark

Run the following commands from the repository root. Use a local model directory or a Hugging Face model ID for `--model_path`. The bundled prompt CSV is the default for both dispatchers.

```bash
python scripts/main400_all_methods/run_formal_external.py --model_path CompVis/stable-diffusion-v1-4 --methods sd attend_excite syngen conform storm --end_idx 1 --dry_run
python scripts/main400_all_methods/build_all_methods_index.py --csv dataset/formal_prompts_main_400.csv
```

External generation requires compatible upstream checkouts and their dependencies. Supply the checkout paths using `--repo_attend_excite`, `--repo_syngen`, `--repo_conform` and `--repo_storm`. The wrappers expect the upstream entry points named in their source. Attend-and-Excite and SynGen accept different argument layouts through the corresponding `--*_cmd_template` options. CONFORM and STORM can fall back to ordinary Stable Diffusion for unsupported prompt structures. Their metadata record the fallback. The archive does not identify exact upstream revisions.

The indexer reads the internal and external output directories and writes `results/formal_main400_all_methods/all_images_index_with_storm.csv`. For the paper protocol, use this index with the following scripts. Evaluation uses the original prompt and its recorded component phrases. CLIP Min is the weakest component similarity. The weak-object risk proxy is the gap between mean and minimum component similarity.

| Script under `scripts/main400_all_methods/` | Inputs and outputs |
| --- | --- |
| `eval_clip_from_index.py` | `--index_csv` and `--out_csv`. Save as `results/formal_main400_all_methods/metrics/clip_metrics_with_storm.csv`. |
| `eval_blip_tc_from_index.py` | Same arguments. Save as `results/formal_main400_all_methods/metrics/blip_tc_metrics_with_storm.csv`. Downloads the default CLIP and BLIP checkpoints on first use. |
| `collect_runtime_trace_metrics.py` | Same arguments. Save as `results/formal_main400_all_methods/metrics/runtime_trace_metrics_with_storm.csv`. |
| `merge_metrics_with_storm.py` | Join the three outputs on method, prompt ID and seed. |
| `merge_storm_t2i_into_eval_metrics.py` | Join per-image scores from the separate T2I-Spatial evaluator. Use `--help` for the paths and supply a score CSV with method, prompt ID, seed index and spatial score. The bundled helper `prepare_t2i_spatial_storm.py` stages the STORM samples. |
| `make_paper_table_with_storm_t2i.py` | Generate the main comparison table from the merged evaluation CSV. |
| `make_sci_tables_with_storm.py` | Generate intermediate per-dataset and paired summaries using `--eval_csv` and `--out_dir`. Its T2I-Spatial column is reserved for the separate evaluation. |

For quick evaluation of a single generation directory, the modules in `metrics/` retain EBAMA's two-part scoring convention. They are separate from the index-based paper protocol. Use `python -m metrics.compute_clip_similarity --help` for CLIP and `python -m metrics.blip_captioning_and_clip_similarity --help` for the optional LAVIS evaluation. LAVIS may require a separate environment.

## Additional experiments

| Directory under `scripts/` | Use |
| --- | --- |
| `attention_overlap_vis/` | Screen matched cases, regenerate attention diagnostics and draw the overlap figure. |
| `param_scheduler_analysis/` | Run gamma and mask sweeps and compare schedules with `bash scripts/param_scheduler_analysis/run_param_schedule_analysis_all.sh all`. Run the separate threshold sweep with `bash scripts/param_scheduler_analysis/run_expB_eta_threshold.sh full`, followed by `summarize_expB_eta_threshold.py`. Both launchers accept the `MODEL_PATH`, `FORMAL_CSV` and `GPU` environment variables. |
| `qualitative_analysis/` | Build qualitative grids, failure-case manifests and plots from manuscript tables. These tools take result CSVs and manifests as inputs. |
| `manual_fusion_missing/` | Prepare blinded tasks, serve the local annotation interface, resolve labels and analyze ratings. The interface retains the Chinese labels used in the study. Analysis requires the corresponding annotation records. |
| `runtime_report/` | Summarize per-image runtimes and update counts from a collected runtime CSV. |

## Data and code availability

The source archive includes the prompt CSVs, archived summary tables, figures, image index and analysis code. Paths in the archived image index identify experimental outputs. Obtain model weights separately and regenerate images before running analyses that require those images. External evaluator code, private annotation records and full historical generation traces are outside this archive.

## Provenance and license

SA-EBAMA builds on [EBAMA](https://github.com/YasminZhang/EBAMA) by Yasi Zhang, Peiyu Yu and Ying Nian Wu. EBAMA credits [Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite) and [SynGen](https://github.com/RoyiRa/Linguistic-Binding-in-Diffusion-Models). The diffusion pipeline also uses Hugging Face Diffusers. The SA-EBAMA contributions use the MIT License. The consolidated `LICENSE` includes upstream notices and the Apache 2.0 terms for the relevant third-party portions. Model weights and third-party benchmark materials retain their original terms. The prompt CSVs record the source for each example. Cite the manuscript named above when using SA-EBAMA and the [EBAMA paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05822.pdf) for the baseline.
