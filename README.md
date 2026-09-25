# ReSight-SMC

Code for reproducing the results in *Two-Stage Power Sampling via Island SMC with Visual Scouts*. ReSight-SMC combines sequence-power island SMC, routed visual-scout proposals with importance correction, and an answer-marginal power readout.

The paper evaluates four Qwen-VL backbones on LogicVista, MathVista, MMStar-R, MMStar-P, and RealWorldQA. The main tables use four independent seeds (0–3), report pass@1 as mean ± **sample standard deviation**, and weight the all-data average by question count.

## Install

The experiments use Linux, Python 3.11, CUDA 12.8, and NVIDIA GPUs; the reported system-cost measurements use one RTX 5090. The pinned environment includes Transformers for SMC and vLLM for the ordinary sampling baselines.

~~~bash
conda env create -f environment.yml
conda activate resight-smc
~~~

Models and datasets are downloaded from Hugging Face on first use. To prefetch them, for example:

~~~bash
python download.py --model Qwen/Qwen2.5-VL-7B-Instruct
python download.py --dataset logicvista
python download.py --dataset mathvista
python download.py --dataset mmstar_r
python download.py --dataset mmstar_p
python download.py --dataset realworldqa
~~~

The other main backbones are Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen3-VL-4B-Instruct, and Qwen/Qwen3-VL-8B-Instruct. The post-trained references are maveryn/trace-qwen2.5-vl-3b, maveryn/trace-qwen2.5-vl-7b, and OpenMOSS-Team/Game-RL-Qwen2.5-VL-7B.

| Benchmark | Split | Questions | Prompt |
|---|---:|---:|---|
| LogicVista | test | 448 | Chain of thought |
| MathVista | testmini | 1,000 | Chain of thought |
| MMStar-R | val, four reasoning categories | 1,000 | Chain of thought |
| MMStar-P | val, two perception categories | 500 | Direct answer |
| RealWorldQA | test | 765 | Released direct-answer instruction |

The same deterministic answer extraction and dataset-specific scoring are used for all methods. No LLM judge is required. See the paper appendix for the exact prompt and evaluation protocol.

## Run ReSight-SMC

`configs/default.yaml` contains the paper's ReSight-SMC method settings. The sole shell launcher runs one model, one benchmark, and one seed per invocation; its defaults are Qwen2.5-VL-7B, LogicVista, and seed 0.

| Setting | Paper value |
|---|---|
| Proposal routing | biased attention route |
| Region overlap penalty | global scope, coefficient 1.0 |
| Region-size exponent | 0.75 |
| Islands × particles | 4 × 8 (32 total) |
| Sequence / answer powers | alpha = 2, gamma = 2 |
| Bridge | linear ramp through token 128 |
| Resampling | island-local stratified resampling, ESS < 0.5M, checked every 32 tokens |
| Visual scouts | fraction 0.25, checkpoint 40, 16 scout tokens |
| Attention biases | ln(2) for image tokens, another ln(4) for routed-region tokens |
| Decoder / numerical path | Transformers SDPA, BF16 model, FP32 full-vocabulary log-softmax, no top-k or nucleus truncation |
| Response horizon | 1,024 tokens |

Run the default experiment:

~~~bash
bash scripts/run_resight.sh
~~~

Choose a different model, benchmark, seed, or GPU with environment variables:

~~~bash
MODEL=Qwen/Qwen3-VL-4B-Instruct DATASET=mmstar_p SEED=1 GPU=0 \
  bash scripts/run_resight.sh
~~~

Supported `DATASET` values are `logicvista`, `mathvista`, `mmstar_r`, `mmstar_p`, and `realworldqa`. The launcher selects the paper's chain-of-thought or direct-answer prompt for each benchmark. Repeat the single-run command for each desired model–dataset–seed combination; the paper uses seeds 0–3. Each run gets an ID containing the model, dataset, seed, and UTC timestamp; set `RUN_ID` to choose one explicitly. Additional `run.py` options can be passed after the script name, for example `--output-dir /path/to/results`.

## Outputs and aggregation

Each run writes to `outputs/<run_id>/`:

| File | Contents |
|---|---|
| config.yaml | Effective model, dataset, method, seed, and hyperparameters |
| predictions.jsonl | Per-question response, terminal particles and weights, extracted answer, evaluation, and diagnostics |
| metrics.json | Current accuracy, example count, latency, memory, and available coverage/genealogy summaries |

Results are written incrementally. Before aggregating, check that metrics.json reports the full benchmark count from the dataset table above and that there is exactly one chosen complete run per model–dataset–seed.

For four complete run directories of one benchmark:

~~~bash
python summarize.py \
  --runs "$RUN_SEED0" "$RUN_SEED1" "$RUN_SEED2" "$RUN_SEED3" \
  --output outputs/summary.json
~~~

Set `RUN_SEED0` through `RUN_SEED3` to the four complete run-directory paths, in seed order.

The summary contains accuracy_by_seed, mean_accuracy, pass_at_4, and Coverage@32 when available. For the paper’s pass@1 “mean ± SD,” multiply the four accuracy_by_seed fractions by 100 and compute their **sample** standard deviation (ddof=1). Compute each seed’s all-data accuracy from the five dataset accuracies weighted by their question counts, then take mean ± sample SD across the four seed-level all-data values. Coverage@32 uses the terminal particle set from one execution and is reported separately from pass@4.

The paper contains the target tables and the complete experimental protocol.
