# Serve Qwen3.8-Flash-Next (125B) on 12 GB VRAM + 32 GB RAM

One 12 GB card and a full RAM box. Strata at `orca-port` commit `860f339`, not stock main.

Result (IQ3_S, the config this box runs):

| metric | measured |
|---|---|
| Decode | 22.6 tok/s live. Window mean 20.6, median 20.5, p95 25.3. Engine 22.7, session 22.7. History 14.8-21.0 across the last 11 requests |
| Prefill | 389 tok/s fresh (engine, 16k chunks). 4.5k-91.5k on prefix-cache hits; session reuses 92.1% (5.6M of 6.1M tokens skipped, 12.6x logical/eval work, ~04:01:40 saved) |
| Context | 131072 tokens. Live 60,878 in use (46.4%), 70,194 free. Peak ctx 111,076. One active slot |
| RAM | Server RSS 24.6 GiB. System 29.4/30.4 GiB at load (1.0 free, swap 11.9%). Hot tier 24.0 GiB mlocked. Budget the whole 30 GB |
| GPU | RTX 4070 SUPER at 100%, 80.5 W during decode. Expert cache 1500 slots, pool workers 20 |
| Load | 99 turns, 69,062 generated, 909 requests in the window (56.9M in / 478k out / 07:46:56 wall). Uptime 11:30:46 |

This is not BF16. Base is 354 GB and does not fit. This runs the GSQ-RCO `IQ3_S` GGUFs at ~3.5 bits/weight in a 2-shard split (79 GB on disk). Authors report task average 93.26 vs 93.12 BF16. Details and caveats under Quality.

---

## Models

Same engine, same context for both. Pick one:

| build | repo | files | size |
|---|---|---|---:|
| Stock | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001/00002-of-00002.gguf` | 79 GB |
| Abliterated (what this box runs) | same layout, local rename | `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001/00002-of-00002.gguf` | 52 + 27 GB |
| Faster / dumber | same repo | `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001/00002-of-00002.gguf` (or `-abliterated-` twin) | 36 + 27 GB |

The abliterated files are the ISTA quant with refusal directions edited in place. GGUF metadata still reads `quantized_by = ISTA DASLab`, `repo_url = .../Qwen3.8-Flash-Next-GSQ-RCO-GGUF`. There is no separate public URL for them; treat stock as the reproducible base and swap `-m` / `--native` to the abliterated shards if you have them. Shard 2 (27-28.8 GB) is the shared PLE/n-gram blob, byte-identical across quants.

Q2_0 config is checked in as `strata-sc117-q2.json` (port 8127, cache 3000, hot tier 21.0 GiB, prompt-cache 6, workers 12). It is not the measured config here. Expect roughly 2x the decode of IQ3_S from the smaller arena (664 vs 982 MB served per token), at a clear quality cost — see Quality. If you want speed, start there; if you want answers, stay on IQ3_S.

---

## Hardware

Reference box: Ryzen 9 9900X 12C/24T, RTX 4070 SUPER 12 GB (12282 MiB, driver 615.71.09), 30.4 GiB RAM + 30 GiB zram swap (nominal 32 GB box), SPCC 1 TB NVMe (DRAM-less, PCIe 4.0 x4), Linux, CUDA 13.3 toolkit.

Throughput follows SSD random-read bandwidth plus DRAM bandwidth for the hot tier. The old Kingston SNV2S1000G did 816 MB/s QD1 / ~969 saturated on `experts-native.bin`; the SPCC does 3227-3341 QD1 / ~5050 saturated at 0.63-0.71 ms per 2 MiB read. That is what moved this box from ~7-17 to ~20-23 tok/s on IQ3_S. A slower drive still runs, just slower.

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

Per decode token the engine touches 10 experts x 48 layers = 480 blobs. Blob and arena sizes come from the released RCO allocation (REPORT-GSQ §4):

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

Published numbers, xhigh reasoning effort, vs BF16 (from REPORT-GSQ §9; SWE-bench only reported for Coder):

| variant | size | LCB v6 | AIME25 | GPQA-D | task avg |
|---|---:|---:|---:|---:|---:|
| BF16 | 354 GB | 87.43 | 100.00 | 91.92 | 93.12 |
| GSQ-RCO IQ3_S (this recipe) | 83.6 GB | 86.86 | 100.00 | 92.93 | 93.26 |
| GSQ-RCO IQ3_XXS | 75.8 GB | 86.29 | 100.00 | 91.41 | 92.57 |
| GSQ-RCO IQ2_XS | 68.0 GB | 83.43 | 96.67 | 87.37 | 89.16 |
| GSQ-RCO Q2_0 | 66.4 GB | 81.14 | 96.67 | 89.39 | 89.07 |

Task average 93.26 vs 93.12. Ties AIME25, +1.01 on GPQA-Diamond, -0.57 on LiveCodeBench.

Caveats:

- These are the release authors' benchmarks. Little independent verification.
- The abliteration is not in these numbers. It edits residual writers, not the router or the n-gram table, but its cost is unmeasured here — expect a small alignment tax, not a capability jump.
- Q2_0 is ~4 points off the base on task average (89.07 vs 93.12). Faster, visibly dumber. That matches the "Q2 is kinda stupid" field report.
- The pruned Coder build ("IQ1_M", actually 3.5 bpw over 256 experts) was evaluated and declined: same per-token work as IQ3_S, 91.3% on SWE-bench, breaks vision. See REPORT-GSQ §2-9.

---

## Reproduce

### 1. Model

```bash
# Stock IQ3_S (79 GB, 2 shards)
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  --local-dir ~/models/qwen3.8-flash-next-gsq-rco

# Stock Q2_0 (63 GB, 2 shards)
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --local-dir ~/models/qwen3.8-flash-next-gsq-rco
```

This box serves the `-abliterated-` renames of the same files (52+27 GB IQ3_S, 36+27 GB Q2_0). Any other `qwen4exp` GSQ-RCO GGUF pair with the same 2-shard layout is a drop-in; point `--native` at the shard holding `token_embd.weight` (shard 1 here) and `--ple-gguf` at the n-gram shard (shard 2).

### 2. Build

```bash
./build.sh        # pins orca-port 860f339, flags in the script
```

That commit is the measured one. Pin matters: kernels, tiering, and the server protocol change between snapshots, so retune `--expert-cache` / `--hot-ram-gib` if you move. Needs CUDA 13.3 (`~/deps/cuda-13.3` or `/opt/cuda`), cmake, ninja, and a Python 3.10+ venv in the Strata checkout (`setup.sh` creates it). `setcap cap_ipc_lock,cap_sys_nice` needs root once — without it the hot tier stays reclaimable and decode collapses under pressure.

### 3. Pack

The engine does not read GGUFs directly for experts. One-time prep per quant (paths are the measured box's; swap `$HOME` as needed):

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

| flag | reason |
|---|---|
| `--expert-cache 1500 --expert-cache-per-layer` | Per-layer GPU-resident experts. 1500 for IQ3_S; 3000 for Q2_0's smaller arena |
| `--hot-ram-gib 24.0` | mlocked host tier from the profile. 24.0 IQ3_S / 21.0 Q2_0. Lower it if the box swaps |
| `--mmap-experts` | Arena stays on SSD under page cache. Without it the 47 GB pack wants resident RAM that is not there |
| `--pool-workers 20` | CPU expert threads. 20 here (12 for Q2_0); sweep on your box, SMT siblings measured worse |
| `--kv q4_0` | KV for the full 131k window. `f16`/`q8_0` do not fit it |
| `--max-context 131072` | Sized to that KV + tier budget. Arch allows 262144; RAM does not |
| `--prompt-cache 2` | Conversation checkpoints. The 92.1% hit rate; 6 for Q2_0 |
| `--prefill 16384` | Chunk size for fresh prompts. Larger is faster, less headroom |
| `--spec 6 --spec-min-p 0.8 --suffix-draft 3 --mtp` | DraftGuesses checked 6-8 at a time plus suffix repeats. Same answer, fewer rounds |
| `--host 127.0.0.1` | Loopback only. The reference `srv.sh` binds Tailscale + API key; add `--api-key` and set CORS if you expose it |

Reasoning budgets are "unlimited" in the sense that `fit_max_tokens` is true and the server clamps `max_tokens` to the room left instead of 400ing — a 53,834-token ask runs (563 done, 53,271 to go, ETA 00:39:17 in the screenshot). It still decodes one request at a time; extra clients queue.

---

## What did not fit

- The full 262144 arch context. Sized to 131072 for KV + 24 GiB tier + 1500 VRAM slots. Longer needs a smaller cache, lower tier, or more RAM.
- A second slot. One active, queue 0 is the config — KV and tier are allocated once.
- `f16` / `q8_0` KV at 131k. Twice to 4x the VRAM for no measured quality gain here.
- Vision. Text-only; `mmproj` + `strata-vision` are a separate build.
- The Coder pruned build. Evaluated, declined (§Quality). Same per-token work, worse SWE-bench, dead vision path.
- Headroom. System sits at 1.0 GiB free with 11.9% swap in use during decode. Close browsers. The server's memory governor (`STRATA_MEM_FLOOR_MIB`, default 1536) sheds cold tier slices between requests; `STRATA_MEM_FLOOR_MIB=0` disables it. Watchdog kills a 300 s-silent engine.

---

## Measuring it yourself

- Decode/prefill: the server log and `logs/sc117-iq3s-final.log`, e.g. per-request `decode 22.6 tok/s`, `prefill 91,501.9 tok/s` on cache hits vs `389.3 tok/s` fresh-engine. Session ledger prints seen / evaluated / reused and the saved estimate.
- Cache: `cache 92.1% · 12.6x logical/eval work` line. If reuse drops under ~80% on your workload, raise `--prompt-cache` and check turn-token chunking.
- VRAM: sample peak during a request, not idle: `nvidia-smi --query-gpu=memory.used,memory.free --format=csv` once per second.
- SSD: `build/qdbench <file> 2097152 1 4` (QD1) and `... 2097152 8 4` (saturated). Expect ~3200 / ~5050 MB/s on the SPCC; under ~1000 the recipe still works but decode tracks the miss rate.
- Tensor sizes: read the GGUF header. Keys: `qwen4exp.block_count`, `qwen4exp.expert_count`, `qwen4exp.expert_used_count`, `qwen4exp.ple.*`, `split.*`.

---

## License

MIT for these scripts and configs, see [LICENSE](LICENSE). Models are Apache-2.0 per Hugging Face metadata (abliterated twins carry their own terms — check the file you actually serve). Strata is its own repo and license; this recipe only pins it.
