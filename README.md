# Qwen3.8-Flash-Next-GSQ-RCO-Abliterated (IQ3_S) @ 131k context (12k reasoning budget)
# ~20-25 tok/sec decode (~39-45 tok/sec Q2), 300-100k+ tok/sec prefill
# on just 12GB VRAM + 32GB RAM + NVME

This fork is very experimental and behind upstream.

| metric | measured |
|---|---|
| Decode | 22.6 tok/s live. Window mean 20.6, median 20.5, p95 25.3. |
| Prefill | 389 tok/s fresh (engine, 16k chunks). 4.5k-91.5k on prefix-cache hits; session reuses 92.1% |
| Decode (Q2_0) | 39-44 tok/s at 1-4k context (bench), 33-40 live agentic at ~25k, 30.4 at 120k. |
| Prefill (Q2_0) | 260-744 tok/s fresh at 1-4k, 1090 at 120k. |
| Context | 131072 tokens. |

---

## Models

Same engine, same context for both:

| build | repo | files | size |
|---|---|---|---:|
| Abliterated `IQ3_S` (what this box runs) | [SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) | `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf` + `...-00002-of-00002.gguf` | 55063448064 + 28800138432 bytes (51.3 + 26.8 GiB) |
| Abliterated `Q2_0` | same repo | `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0-00001-of-00002.gguf` + `...-00002-of-00002.gguf` | 38021379872 + 28800138432 bytes (35.4 + 26.8 GiB) |
| Stock | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001/00002-of-00002.gguf` | 83.6 GB |

The SC117 build is the ISTA quant with 144 write-to-residual tensors transplanted from [orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) (`ssm_out` 36, `attn_output` 12, `ffn_down_shexp` 48, `ffn_down_exps` 48, all 48 layers). No GSQ value recomputed; per-tensor blake2b confirms the other 1079 tensors match upstream byte for byte. Cost is +0.25 GiB on IQ3_S (95 tensors moved to `Q8_0`, 49 kept their type). Shard 2 (28800138432 bytes) is the shared n-gram table, byte-identical across all four tiers and upstream.

Q2_0 config is checked in as `strata-sc117-q2.json` (port 8127, cache 3000, hot tier 21.0 GiB, prompt-cache 6, workers 12). Measured above. Smaller arena (664 vs 982 MB served per token), ~4 points weaker on task average — see Quality.

---

## Hardware

Reference box: Ryzen 9 9900X 12C/24T, RTX 4070 SUPER 12 GB (12282 MiB, driver 615.71.09), 30.4 GiB RAM + 30 GiB zram swap, SPCC 1 TB NVMe (DRAM-less, PCIe 4.0 x4), Linux, CUDA 13.3 toolkit.

---

## Why it fits

### 1. Only 10 of 512 experts fire per layer

Qwen3.8-Flash-Next is MoE (`general.architecture = qwen4exp`):

| quantity | value |
|---|---:|
| Blocks | 48 |
| Experts per MoE layer | 512, top-10 routed |
| Embedding | 2560 |
| Attention | 24 heads, 2 KV heads, key/value 256, full attention every 4th layer |
| Arch context | 262144 (recipe sizes 131072 for the KV + RAM budget) |

Per decode token the engine touches 10 experts x 48 layers = 480 blobs. Blob and arena sizes computed from the released RCO tensor-allocation files:

| quant | arena | blob | MB served / token |
|---|---:|---:|---:|
| `IQ3_S` (this recipe) | 46.8 GiB | 2.046 MB | 982 |
| `Q2_0` | 31.6 GiB | 1.382 MB | 664 |

### 2. The 46.8 GiB arena lives on SSD + a 24 GiB RAM tier

`--mmap-experts` keeps the packed arena (`packs/sc117-iq3s/experts-native.bin`, 47 GB + 1.5 GB `dense.bin`) on SSD under the OS page cache. `--hot-ram-gib 24.0` pins the profile's hottest 24 GiB in an mlocked host tier (`profile-sc117-r2.bin`), so steady-state misses are the tail of the routing distribution. `--expert-cache 1500 --expert-cache-per-layer` keeps ~1500 hot blobs resident in VRAM and computes them on the GPU. Q2_0 uses 3000 slots against its smaller arena.

### 3. KV is 4-bit, prefix cache does the rest

`--kv q4_0 --max-context 131072` sizes KV for the full window. `--prompt-cache 2` checkpoints conversations, so follow-up turns re-read only fresh tokens: 92.1% reused this session, 17k-91k tok/s logical prefill on hits vs 389 tok/s fresh. `--prefill 16384` chunks new prompts; `--short-read` equivalent stays default.

### 4. Speculation is MTP + suffix lookup

`--spec 6 --spec-min-p 0.8 --suffix-draft 3 --mtp <rt>` with the packed MTP runtime (`mtp/rt`: 116 MB dense + 708 MB experts). Sampler runs per-request server side. Monitor renders this as `MTP 8 · min-p 0.8 · lookup 3`; the config values are the source of truth.

---

## Quality

BF16 is 354 GB. On 12 GB VRAM you quantize regardless. Question is which quant.

GSQ-RCO ([GSQ](https://arxiv.org/abs/2604.18556), [RCO](https://arxiv.org/abs/2605.00649), ISTA DASLab) is non-uniform: each tensor gets its own quant type from a gradient search under a size budget.

Published numbers, xhigh reasoning effort, vs BF16 (ISTA's published benchmarks; SWE-bench only reported for Coder):

| variant | size | LCB v6 | AIME25 | GPQA-D | task avg |
|---|---:|---:|---:|---:|---:|
| BF16 | 354 GB | 87.43 | 100.00 | 91.92 | 93.12 |
| GSQ-RCO IQ3_S (this recipe) | 83.6 GB | 86.86 | 100.00 | 92.93 | 93.26 |
| GSQ-RCO IQ3_XXS | 75.8 GB | 86.29 | 100.00 | 91.41 | 92.57 |
| GSQ-RCO IQ2_XS | 68.0 GB | 83.43 | 96.67 | 87.37 | 89.16 |
| GSQ-RCO Q2_0 | 66.4 GB | 81.14 | 96.67 | 89.39 | 89.07 |

Task average 93.26 vs 93.12. Ties AIME25, +1.01 on GPQA-Diamond, -0.57 on LiveCodeBench.

---

## Reproduce

### 1. Model

```bash
# Abliterated IQ3_S (what this box runs; 83.86 GB, 2 shards)
hf download SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF \
  --include "IQ3_S/*" --local-dir ~/models/sc117-abliterated

# Abliterated Q2_0 (66.82 GB, 2 shards)
hf download SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF \
  --include "Q2_0/*" --local-dir ~/models/sc117-abliterated

# Stock IQ3_S instead (83.6 GB)
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  --local-dir ~/models/qwen3.8-flash-next-gsq-rco
```

Name the folders exactly (`IQ3_S/`, `Q2_0/`). Any other `qwen4exp` GSQ-RCO GGUF pair with the same 2-shard layout is a drop-in; point `--native` at the shard holding `token_embd.weight` (shard 1 here) and `--ple-gguf` at the n-gram shard (shard 2).

### 2. Build

```bash
./build.sh        # pins orca-port 860f339, flags in the script
```

That commit is the measured one. Pin matters: kernels, tiering, and the server protocol change between snapshots, so retune `--expert-cache` / `--hot-ram-gib` if you move. Needs CUDA 13.3 (`~/deps/cuda-13.3` or `/opt/cuda`), cmake, ninja, and a Python 3.10+ venv in the Strata checkout (`setup.sh` creates it). `setcap cap_ipc_lock,cap_sys_nice` needs root once — without it the hot tier stays reclaimable and decode collapses under pressure.

### 3. Pack

The engine does not read GGUFs directly for experts. One-time prep per quant (measured box's paths; adjust to where you put the files):

```bash
# pack: native experts + dense + tokenizer (47 + 1.5 GB for IQ3_S)
venv/bin/python tools/iq_pack.py \
  --gguf ~/models/qwen3.8-flash-next-gsq-rco-iq3_s-abliterated/IQ3_S/*-00001-of-00002.gguf \
  --out ~/models/strata/packs/sc117-iq3s

# head sidecar (arch keys in shard 1, output.weight in shard 1 here)
venv/bin/python tools/make_native_head.py \
  --arch-shard   ...-00001-of-00002.gguf \
  --tensor-shard ...-00001-of-00002.gguf \
  --out ~/models/strata/ple/head-native-sc117.gguf   # 498 MB

# MTP draft runtime (shared by both configs)
venv/bin/python tools/mtp_rt.py \
  --gguf ~/models/strata/mtp/mtp-q2_0.gguf \
  --out  ~/models/qwen3.8-flash-next-gsq-rco-iq3_s-abliterated/strata/rt
```

Profile `data/profile-sc117-r2.bin` ships with the Strata checkout. Regenerate over a few thousand of your own tokens before trusting coverage on a new workload.

### 4. Serve

```bash
./serve.sh                                       # defaults, or:
CONFIG=./strata-sc117-q2.json PORT=8127 ./serve.sh
```

`serve.sh` swaps `/home/USER` for `$HOME` inside the checked-in JSON, checks the paths exist, and execs:

```bash
venv/bin/python serve/server.py \
  --engine strata --config /tmp/strata-recipe-strata-sc117-iq3s-8126.json \
  --port 8126 --host 127.0.0.1
```

Verbatim engine args (`strata-sc117-iq3s.json`):

```bash
run-engine-mlock.sh \
  --pack         ~/models/strata/packs/sc117-iq3s \
  --native       ~/models/qwen3.8-flash-next-gsq-rco-iq3_s-abliterated/IQ3_S/*-00001-of-00002.gguf \
  --native-head-gguf ~/models/strata/ple/head-native-sc117.gguf \
  --ple-gguf     ~/models/qwen3.8-flash-next-gsq-rco-iq3_s-abliterated/IQ3_S/*-00002-of-00002.gguf \
  --native-dense-gguf ...-00001-of-00002.gguf \
  --native-dense-gguf ...-00002-of-00002.gguf \
  --expert-profile ~/models/strata/data/profile-sc117-r2.bin \
  --expert-cache 1500 --expert-cache-per-layer \
  --hot-ram-gib 24.0 --mmap-experts --pool-workers 20 \
  --prefill 16384 --spec 6 --spec-min-p 0.8 \
  --mtp ~/models/qwen3.8-flash-next-gsq-rco-iq3_s-abliterated/strata/rt \
  --max-context 131072 --kv q4_0 \
  --suffix-draft 3 --prompt-cache 2
# env: STRATA_RSPLIT=1, STRATA_STATIC_TIER=1, STRATA_NO_HITS=1
# server: fit_max_tokens=true, sampler per-request (server side)
```

## License

MIT for these scripts and configs, see [LICENSE](LICENSE). SC117 card states Apache-2.0 for the model files. Strata is its own repo and license; this recipe only pins it.
