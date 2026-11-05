# HeartMuLa training contract

What native HeartMuLa LoRA training computes, the reference it is checked
against, the tolerances it must meet, the GGML backward support it relies on,
and the Python numbers it has to beat. The native prepare and trainer build
against this page; where an implementation has to differ, change this page in
the same commit.

Status: `engine/tools/heartmula-train-reference.py` produces the fixtures and
the baseline below. The native trainer is `ace-train heartmula-train`
(`engine/src/train/heartmula-train-*.h`); its `--selftest <fixture dir>`
checks section 5 against the fixture set.

Python is an offline oracle. Nothing here runs Python at training time.

## 1. Model and adapter

The inference model is described in [HEARTMULA_CONTRACT.md](HEARTMULA_CONTRACT.md).
Training touches it as follows.

- **Trained:** LoRA on the global (backbone) attention projections q, k, v and
  output, all 28 layers, 112 sites. Rank 64, alpha 64, so the delta scale
  alpha / rank is 1. A is `[64, in]`, B is `[out, 64]`; q and output are
  3072 x 3072, k and v are 3072 -> 1024. 36,700,160 trainable parameters.
- **Frozen:** everything else, including the local decoder, the embeddings,
  `muq_linear`, the projection, the codebook-0 head and `audio_head`. The local
  decoder is frozen but sits on the gradient path: the loss of codebooks 1 to 7
  reaches the global LoRA through it (section 2).
- **Init:** A uniform Kaiming (a = sqrt 5, torchtune's `_lora_a_init_params`),
  B zero, so a fresh adapter is the base model. Values come from the trainer's
  own seeded RNG; only the distribution has to match.
- **Masters and optimizer:** F32 A and B, AdamW (lr 1e-4, betas 0.9 / 0.999,
  eps 1e-8, decoupled weight decay 0.01 on every A and B), constant LR. The
  Python recipe keeps A and B in BF16, which loses most of a 1e-4 update
  (section 5.3 measures it). Native keeps F32 masters; this is a deliberate
  improvement, not a parity target.
- **Batch:** one sample per micro-batch, four micro-batches per optimizer step
  (the proven recipe). Each micro-batch loss is divided by the number of
  micro-batches actually accumulated in that step. The Python script divides by
  4 even for a short end-of-epoch step; native does not copy that.
- **Dropout:** the recipe applies p = 0.05 to the input of every LoRA branch,
  independently per site. The fixtures use p = 0 because no two RNGs agree.
  Native implements it as `--lora-dropout p` (default 0); section 8.1, lever 6
  says how and what it costs.

## 2. Loss (frozen)

For one sample with prompt length S and T audio frames, codes `c[t][k]`
(t in 0..T-1, k in 0..7, values 0..8191):

1. **Backbone input**, length S + T - 1. Rows 0..S-1 are the prompt: the text
   embedding of the prompt id. Row `muq_idx` (the MuQ slot) is replaced by
   `muq_linear(0)`, which is `muq_linear.bias`, as inference does with no
   reference audio. Row S + t - 1 for t in 1..T-1 is the sum of
   `audio_embeddings[c[t-1][k] + k * 8197]` over the 8 codebooks. The last
   frame is predicted and never fed. Causal attention, positions 0..S+T-2.
2. **Codebook 0:** backbone output (after the final norm) at position
   S - 1 + t predicts frame t: `logits0[t] = codebook0_head(h[S-1+t])`.
   Positions S-1..S+T-2, T rows.
3. **Codebooks 1 to 7:** for every frame t the local decoder runs on 8
   positions `[h[S-1+t], emb_0(c[t][0]), ..., emb_6(c[t][6])]`, where
   `emb_k(x) = audio_embeddings[x + k * 8197]`, after `projection`, causal over
   the 8 positions, positions 0..7. Output position i (1..7) after the decoder
   norm gives `logits_i[t] = out_i @ audio_head[i-1]`. That is heartlib's
   `generate_frame` with teacher-forced codes, all T frames as one batch.
4. **Loss** = (1/8) x sum over k of mean over t of `CE(logits_k[t], c[t][k])`.
   Every frame, every codebook, no sampling, no masking, no label smoothing.
   Logits are 8197 wide; codes never reach the special ids above 8191.
   `--cb0-weighted` (off by default, outside the frozen fixtures) replaces the
   1/8 with `w_k / 4.8`, w = (2.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1), the
   paper's SFT weighting normalised to sum to 1.

By default no audio EOS frame is appended: the reference fixtures are windows cut
from longer songs, so their last frame is not the end of a song. With `--eos`, a
window that reaches its song's real end gets one more frame of 8193 in all eight
codebooks, the frame the base model itself samples when it ends a song (logged as
`end of song at frame N` by the engine). The labels are 8197 wide, so only the data
changes. `--crop-frames N` trains frames [0, N) of longer songs (the loss on a prefix
of a causal model equals that prefix's loss inside the whole song) and every fourth
sample whole, with its EOS frame. `--max-frames` keeps random windows for the
selftest. The EOS frame is added only while S + T stays within the context
(`hmt_eos_fits`), in training and in `--eval` scoring alike.

### Prompt

Prompt ids are built exactly as heartlib's `HeartMuLaGenPipeline.preprocess`
builds them at inference with no CFG: tags lower-cased and wrapped in
`<tag>...</tag>`, tokenized, BOS 128000 prepended and EOS 128001 appended if
missing; lyrics lower-cased, tokenized, same BOS/EOS rule. Layout:
`tags_ids + [0] + lyrics_ids`, `muq_idx = len(tags_ids)`. The id 0 at the MuQ
slot is overwritten in step 1, so its value never matters. Native prepare must
reproduce `tokenizer/prompt_ids` (section 4) from the text recorded in the
manifest.

## 3. Context limit

The backbone context is 8192 positions (torchtune `max_seq_len`). A sample
needs S + T - 1 positions. Prepare reports a sample over the limit as invalid;
the trainer never truncates silently.

## 4. Fixtures

Produced by:

```
python engine/tools/heartmula-train-reference.py --src <checkpoint-root> \
    --sample clip-a.pt --sample clip-b.pt --out <new dir> --device cuda
```

Environment: the oracle checkout's Python 3.10 venv (torch 2.10.0+cu128,
torchtune 0.4.0, transformers 4.57.0, tokenizers 0.22.1, heartlib 0.1.0,
numpy 2.0.2), `TORCHDYNAMO_DISABLE=1`, TF32 off. RTX 5090. Seed 776.

Location (RobWorkPC, not in the repository; 2.29 GiB, 164 arrays):

| Item | Path | SHA-256 |
|---|---|---|
| Fixture set | `M:\heartmula-train-fixtures\hm776-v2\` | manifest `92167955ff3078e981251b120ee19967de6b5288f050fd4d88403627ae79955c` |
| Input clip A | `M:\heartmula-train-fixtures\inputs\clip-a.pt` | `92626bfb8c84cc3a4273d62cd059e054faa32c8f04865b6d2295047ecf6cb7d7` |
| Input clip B | `M:\heartmula-train-fixtures\inputs\clip-b.pt` | `54269b0bb0f82822eef216e594401c97b6d9b425f2b3d5c835e1640faecc4b91` |
| Script at the run | `engine/tools/heartmula-train-reference.py` | `031b8bd939bb84d4003e334843999291c0538ff5bafb60ed969219b46b377851` with CRLF line ends, as run; the committed LF blob is `c91e3cf1f2b99dcdff35734db2c4ea052f94708c8c9a8ea11abe2817922f2e67` |
| Base checkpoint | HeartMuLa-oss-3B, HF revision `d12ac79c6b3387d5c9eee456323495d8f08bb09d` | per shard in the manifest |

`fixture-manifest.json` records every array's file, shape, dtype, bytes and
SHA-256, the package versions, the checkpoint shard hashes and the metrics
quoted below. The clips are codec tokens of two roughly 90 s training windows
from the oracle run's own data (real codec encoder, not the mel workaround):
clip A has S = 148, T = 1112; clip B S = 131, T = 1059.

Contents (names are paths under the fixture set, `.npy`):

| Name | Shape | Meaning |
|---|---|---|
| `input/prompt_ids`, `input/muq_idx`, `input/codes` | [148], [1], [1112, 8] | clip A |
| `input/crops` | [2, 2] | the two micro-batches: frames 0..63 and 256..319 of clip A, same prompt |
| `tokenizer/prompt_ids`, `tokenizer/muq_idx` | [36], [1] | neutral text (in the manifest) through heartlib preprocess |
| `lora/init/{A,B}_{q,k,v,o}` | [28, 64, in], [28, out, 64] | seeded torchtune init (B zero) |
| `lora/perturbed/B_{q,k,v,o}` | [28, out, 64] | seeded N(0, 2e-3) B; A as in init |
| `fp32/taps/*` | | perturbed state, crop 1: `input_embedding` [211, 3072], `muq_slot` [3072], `backbone_last` [64, 3072], `decoder_out` [64, 8, 3072], `logits` [64, 8, 8197] |
| `fp32/forward/{ce,loss}`, `fp32/forward_base/ce` | [8], [1] | perturbed and base losses, crop 1 |
| `fp32/init/grad/*`, `fp32/perturbed/grad/*` | as the factors | dL/dA, dL/dB, crop 1 (A grads are zero at init) |
| `fp32/adamw/step{1,2,3}/grad/*` | | accumulated gradient of each step (both crops, each loss / 2) |
| `fp32/adamw/step{1,3}/*`, `fp32/adamw/step3/exp_avg*/*` | | factors after steps 1 and 3; AdamW moments after step 3 |
| `fp32/adamw/micro_losses` | [3, 2] | loss per step per micro-batch |
| `bf16/...` | | the same graph in plain BF16 (the recipe's precision), no per-step grads |

Measured at FP32: crop-1 loss 5.795520 (base 5.794338); per codebook
3.485, 4.841, 5.785, 5.373, 5.974, 6.479, 7.136, 7.290. AdamW micro-losses
[5.7955, 6.7733], [5.7231, 6.7127], [5.6599, 6.6532].

## 5. Tolerances (declared before porting)

The reference is the FP32 set. Native runs with the shipped F16 matrices, or a
quant of them, and mixed-precision arithmetic, so it cannot match FP32 bit for
bit. The bound is set from the measured drift of the Python recipe's own BF16
graph against FP32 (BF16 weights round harder than F16), at about twice the
measured value. A native result outside these bounds is a bug until shown
otherwise.

### 5.1 Forward (perturbed state, crop 1)

| Quantity | Measured BF16 drift | Native bound |
|---|---|---|
| per-codebook CE, absolute | max 7.3e-3 | 1.5e-2 |
| loss, absolute | 2.4e-4 | 5e-3 |
| `input_embedding` relative L2 | 1.5e-3 | 5e-3 |
| `muq_slot` relative L2 | 0 | 1e-3 |
| `backbone_last` relative L2 | 1.3e-2 | 3e-2 |
| `decoder_out` relative L2 | 9.0e-3 | 2e-2 |
| `logits` relative L2 | 8.9e-3 | 2e-2 |

### 5.2 Gradients

Per stacked tensor (`A_q` ... `B_o`), at both the init and the perturbed state:
relative L2 at most 1e-1 (measured max 4.5e-2) and cosine at least 0.995
(measured min 0.99903). Init-state A gradients are exactly zero in the
reference; native must give zero there too (B = 0 makes dL/dA vanish).

### 5.3 Optimizer

- **Replay (exact maths):** start from `lora/init/A_*` and `lora/perturbed/B_*`,
  feed `fp32/adamw/step{1,2,3}/grad/*` as the gradients. Factors after steps 1
  and 3 and the first moment after step 3 within relative L2 1e-5; the second
  moment within 3e-5. This checks the optimizer alone, with no model in the
  loop. The second moment has its own bound because ggml's AdamW kernel forms
  1 - beta2 in F32 (1.0f - 0.999f = 0.0010000467) where torch passes 0.001, a
  4.7e-5 relative difference in every increment. A numpy replay of the fixture
  gradients gives 1.29e-5 with the kernel's constant and 4.6e-8 with torch's;
  native measures 1.29e-5. The factors stay within 1e-7.
- **End to end:** three steps of two micro-batches from the same start.
  Micro-losses within 1e-2 absolute of `fp32/adamw/micro_losses`. The update
  (step-3 factors minus start) has cosine at least 0.98 and relative L2 at
  most 0.25 against FP32. That is the Python recipe's own result (0.984 and
  0.217, dominated by its BF16 masters), so native with F32 masters has to be
  at least as close as the recipe is to its own FP32 maths.

### 5.4 Resume

A run checkpointed and resumed (optimizer, RNG and data cursor restored) gives
the same factors and moments as an uninterrupted run with the same seed: bit
identical on the same build and device, or relative L2 at most 1e-6 if the
trainer documents a non-deterministic kernel.

## 6. Op inventory and CUDA backward audit

Checked against the pinned HOT-ggml commit `8eb519bd` (engine/ggml clean) on
the RTX 5090. The probe built the training graph at clip A's real shapes
(S = 148, T = 1112, L = 1259): one global block with the four LoRA sites, the
final norm, the codebook-0 head and its CE, the projection, one local block and
the seven local heads with their CEs. It ran `ggml_build_backward_expand` and
asked `ggml_backend_supports_op` on the CUDA device for every forward and
backward node. One block of each kind covers every op and shape the 28 + 3
layers use. The probe source is kept outside the tree (it is not a deliverable
of this slice); the trainer's own init probe (section 6.3) replaces it.

### 6.1 Forward ops on the gradient path

| Site | Ops | Backward needed |
|---|---|---|
| input embedding sum, MuQ slot | GET_ROWS, MUL (gates), ADD, REPEAT | no: frozen data, upstream of every parameter |
| global RMSNorm x2 per layer, final norm | RMS_NORM, MUL (F32 weight) | yes: RMS_NORM_BACK, MUL |
| global q/k/v/o | MUL_MAT (frozen weight), MUL_MAT A, SCALE, MUL_MAT B, ADD | yes: activation grad through the frozen weight; OUT_PROD for dA, dB |
| RoPE (Llama-3 scaled, NORMAL mode, freq factors) | ROPE | yes: ROPE_BACK |
| global attention, head dim 128, 24 q / 8 kv heads, causal | FLASH_ATTN_TRAIN, or MUL_MAT, SOFT_MAX (mask, scale), MUL_MAT | yes |
| SwiGLU MLP (global and local) | MUL_MAT x3, SILU, MUL, ADD | yes: SILU_BACK, MUL |
| frame slice of the backbone output | VIEW, CONT | yes |
| codebook-0 head and CE | MUL_MAT, CROSS_ENTROPY_LOSS | yes: CROSS_ENTROPY_LOSS_BACK |
| local input `[last_h, emb(c0..c6)]` | ACC into a frozen embedding canvas | yes (ACC); see 6.2 for CONCAT |
| global mlp w1/w2/w3, with `--lora-mlp` | as global q/k/v/o; w2's A branch reads the SwiGLU output | yes |
| projection, local q/k/v/o (no LoRA) | MUL_MAT (frozen) | yes: activation grad only |
| local attention, head dim 384, 8 q / 4 kv heads, 8 positions | MUL_MAT, SOFT_MAX, MUL_MAT | yes |
| local heads 1..7 and CEs, loss mean | VIEW, CONT, MUL_MAT, CROSS_ENTROPY_LOSS, ADD, SCALE | yes |

### 6.2 Result

With BF16 frozen weights, every forward and backward node is supported on
CUDA in all three attention modes (exact, FLASH_ATTN_TRAIN TF32, and
FLASH_ATTN_TRAIN BF16 Q/K/V), with either backward arm. With F16, Q8_0 or Q4_K
frozen weights, every node is supported with the `GGML_BACKWARD_MM=1` arm
(ace-train's `--bwd mm`).

**No GGML fork change is needed, so slice 3a is not required.** The rules
below are graph-construction rules for the trainer. Each one was found by the
probe failing first and passing after the change.

| # | Gap (pinned GGML) | Where it bites | Rule |
|---|---|---|---|
| G1 | CUDA OUT_PROD accepts only F32 or BF16 `src0`; the default MUL_MAT backward emits the activation gradient as OUT_PROD(weight, ...) | every frozen-weight GEMM whose input carries a gradient (20 in the probe's two blocks; only the first block's q/k/v input is gradient-free) is refused with F16, Q8_0 or Q4_K weights, so the scheduler would run them on the CPU | Shipped GGUFs are F16 (`convert-heartmula.py` `storage_dtype`). Train with `GGML_BACKWARD_MM=1`, set through the process environment before the first graph build (the `--bwd mm` path the other trainers use), or with BF16 weights. Probe at init (6.3). |
| G2 | `ggml_compute_backward` has no CONCAT rule (aborts, `ggml.c:7834`) | building the local input `[last_h, emb(c0..c6)]` with CONCAT | Build it as ACC of `last_h` into row 0 of a frozen `[H, 8, T]` embedding canvas (the YuE2 graph's `join_rows` idea). |
| G3 | Broadcast MUL_MAT backward asserts `ne3 == 1` (`ggml.c:7345`) | GQA by broadcast in the local decoder, where frames sit in `ne3` | Fold (heads, frames) into `ne2`, heads fastest: q index `t*Nh + h` then reads kv index `t*Nkv + h/2`, torchtune's GQA order. |
| G4 | CUDA ROPE_BACK needs a `ggml_is_contiguous_2` source; the gradient arriving through the attention permute is a 4-D permuted view | the two local RoPE sites (global passes, its `ne3` is 1) | Lay local q and k out as `[D, heads*frames, 8, 1]` before RoPE (one CONT), so the gradient view is 3-D as in the global block. |
| G5 | CONT backward asserts a contiguous gradient (`ggml.c:7523`) | `cont(transpose(cont(v)))` | One CONT over the combined permute and transpose view. |
| G6 | FLASH_ATTN_TRAIN kernels take head dim 64 or 128 | the local decoder's head dim 384 | Local attention stays exact. It is 8 x 8 per frame, so the retained softmax is T x 8 heads x 64 values, about 2.3 MiB per layer at T = 1112. Global attention (head dim 128) uses FLASH_ATTN_TRAIN since slice 4b (section 8.1). |
| G7 | Autodiff accepts BF16 sources only as MUL_MAT `src0` or FLASH_ATTN_TRAIN q/k/v (`ggml.c:8026`) | casting activations to BF16 | Keep activations F32 (use BF16 rounding, as the YuE2 graph does), cast weights or q/k/v only. |

Supported is not the same as correct. Slice 4's selftest still checks the
numbers against section 5.

### 6.3 Trainer obligations from the audit

- At init, build the real graph shapes (both attention sites, effective kv
  heads, the weight types actually loaded) and fail with a named error if any
  node is unsupported on the training device. `backend_sched_new` registers the
  CPU backend beside CUDA, so an unsupported node is silently run on the CPU:
  correct, very slow, and invisible in every number the run reports.
- Log the resolved attention precision and the backward arm.

## 7. Python baseline

The proven recipe exactly as the oracle run used it: BF16 weights, BF16 LoRA
factors, no autocast (its Accelerator has no mixed precision), LoRA dropout
0.05, AdamW lr 1e-4 wd 0.01, batch 1, four micro-batches per step, no
recompute in the backbone (torchtune has no per-layer switch, so the recipe
never had one), recompute on the local decoder only. PyTorch SDPA with a
boolean causal mask. RTX 5090, GPU otherwise idle (no other GPU job, the app
idle), 2026-10-25. Median of 3 timed optimizer steps after one warm-up step.

| Case | S | T | L | ms per micro-batch | ms per optimizer step | peak allocated GiB | peak reserved GiB |
|---|---|---|---|---|---|---|---|
| fixture crop | 148 | 64 | 211 | 104 | 426 | 8.72 | 8.83 |
| clip A, 250 frames | 148 | 250 | 397 | 120 | 476 | 10.34 | 10.80 |
| clip A, 500 frames | 148 | 500 | 647 | 181 | 726 | 12.50 | 13.34 |
| clip A, 750 frames | 148 | 750 | 897 | 263 | 1056 | 14.68 | 15.77 |
| **clip A, whole (real recipe)** | 148 | 1112 | 1259 | **343** | **1376** | **17.81** | 19.37 |
| clip B, whole (real recipe) | 131 | 1059 | 1189 | 333 | 1336 | 17.28 | 19.37 |

Weights alone take 7.43 GiB allocated. The oracle's own training script, run on
the same two clips (8 samples, epochs 1 and 5), gave 334 ms per micro-batch
from the difference of the two wall times and 19.81 and 22.62 GiB peak
reserved in two runs; the reserved figure moves with allocator history, so
compare allocated. The overnight log's 15.00 and 16.16 GiB reserved came from
its own, shorter datasets.

**Native targets:** under 343 ms per micro-batch and under 17.81 GiB allocated
on clip A, with the same recipe, the same GPU and an idle GPU. Every engine
slice reports both on this case.

Work per micro-batch at clip A (forward multiply-adds, from the shapes; an
estimate, not a profile): backbone 3.55 T, local decoder 2.77 T, local heads
and projection 0.28 T, codebook-0 head 0.03 T. The local decoder is about 45%
of the forward because it runs 8 positions for each of 1112 frames, 8896 rows
through layers as wide as the backbone's. With frozen weights the backward is
about one more forward, plus the recompute.

## 8. Levers for slice 4b

Ordered by expected effect on step time first, then VRAM. Each one is measured
on clip A against the baseline above, one change at a time, and reported with
ms per micro-batch and peak allocated.

1. **Recompute granularity.** Python stores every backbone activation (most of
   its 10.4 GiB above the weights) and recomputes the local decoder. Options:
   no recompute when it fits, per-block checkpoints (block inputs are
   L x 3072 F32, 15 MiB per layer at clip A), or every k blocks. The local
   decoder is 45% of the forward, so recomputing it costs more than it does in
   most models; try keeping it and recomputing backbone blocks instead.
2. **Flash-attention backward** on the global attention (FLASH_ATTN_TRAIN, BF16
   Q/K/V at head dim 128, causal hint). Removes the retained L x L x 24
   softmax, 145 MiB per layer F32 at clip A. The local decoder stays exact (G6).
3. **Fused cross-entropy.** Eight CEs over 8197-wide logits with dense one-hot
   labels: logits, gradients and labels are about 35 MiB each per codebook at
   clip A. An index-label CE that writes the gradient in place saves most of it
   and two passes over the logits.
4. **Frozen-weight quant.** The whole model is 7.43 GiB at 16 bits; Q8_0 is
   about half that, Q4_K about a quarter. Global and local are separate GGUFs and can differ. Needs the
   `--bwd mm` arm (G1), which transposes each weight per use; measure that cost
   against the VRAM saved, and the loss drift against section 5.
5. **GEMM precision.** BF16 tensor-core GEMMs with F32 accumulation, as the YuE2
   trainer does, against the F16 weights' own path. Check against section 5.
6. **LoRA dropout cost.** 112 independent masks of L x 3072 must be kept for
   the backward; at clip A that is 1.6 GiB as F32 or 0.8 GiB as F16. Options:
   p = 0, a mask shared by q, k and v (they read the same input; a deviation
   from the recipe), or regenerating masks from a seed in the backward.
   Whether dropout matters to the result is a listening question.
7. **Sequence bucketing and batch packing.** Buckets by L let one built graph
   serve many samples (no rebuild per sample) and allow batch > 1 for short
   clips; packing several short clips into one sequence needs a block-diagonal
   mask and per-clip positions.

### 8.1 Measured (slice 4b)

RTX 5090, Unreal not running, the app up and idle, 2026-10-26. Clip B whole
(S 131, T 1059), rank 64, F16 weights. "Step" is four micro-batches of clip B
(a four-copy cache) including AdamW; median of the timed steps after two
warm-up steps. Allocated is the trainer's own figure (weights, factors,
optimizer, checkpoints, compute buffer), comparable with Python's peak
allocated.

| Configuration | ms per step | allocated GiB |
|---|---|---|
| Python recipe (section 7) | 1336 | 17.28 |
| Slice 4: exact attention, every block recomputed | 1796 | 10.41 |
| flash-bf16, every block recomputed (`--keep-layers -1`) | 1373 | 10.41 |
| flash-bf16, 14 blocks kept | 1060 | 13.62 |
| flash (TF32), 28 blocks kept | 1102 | 15.67 |
| **Default: flash-bf16, 28 blocks kept** | **908** | **15.36** |
| default + `--weights bf16` | 924 | 15.36 |
| default + `--lora-dropout 0.05` | 982 | 16.65 |

Clip A whole, through `--selftest` (no optimizer in the figure): 239 ms per
micro-batch against Python's 343, 15.76 GiB against 17.81. Every section 5
bound holds at the default (gradients rel L2 at most 0.0094, cos at least
0.99995; resume bit-identical).

Per lever, one micro-batch per step on clip B, each measured on the build of
that point:

1. **Recompute.** The profile of the slice-4 graph (per micro-batch: global
   forward 90 ms, global backward 208, local forward 40, local backward 90,
   tail 11, graph build and allocation under 5 in total) put the cost in the
   recompute, not in host work. `--keep-layers N` builds blocks 28-N..27, the
   head, the local decoder and the tail as one graph with its backward, and
   recomputes only the blocks below. With exact attention each kept block saved
   about 3.7 ms and held 321 MiB; the local decoder kept saved 45 ms for
   2.6 GiB (465 ms at N = -1, 420 at 0, 316 at 28 and 21.5 GiB). Losses are
   bit-identical in every mode. `auto` (the default) takes the most of 28, 24,
   ..., 0 that fits the free VRAM, then -1: with 14 GiB held by another process
   it chose 8 and trained at 285 to 301 ms.
2. **Flash-attention backward** (global attention only, G6). Faster here, not
   just smaller: at N = -1, 465 ms exact, 410 TF32, 361 with BF16 q/k/v. With
   BF16 q/k/v a kept block holds about 165 MiB (316 ms at N = 0, 244 at 28 and
   17.1 GiB). `--attn auto` takes flash-bf16, then flash, then exact, by the
   device's `supports_op` on the real global block (BF16 needs CUDA sm_80+).
3. **Fused cross-entropy: not done.** It needs a new ggml op through the fork
   (an index-label CE). The tail is 11 ms of 460 in the slice-4 profile, and the
   one-hot labels take 278 MiB at clip B; that is the ceiling of the saving.
4. **Q8_0 frozen weights: rejected for the default.** 3.4 GiB less, but
   443 ms against 244 at N = 28 (536 at N = -1, 7.4 GiB). `--type Q8_0` stays
   available for a small card.
5. **GEMM precision: BF16 weights rejected.** 924 against 908 ms per step,
   and slice 4 measured worse gradients with them (rel L2 0.011 against 0.004).
6. **LoRA dropout** (section 1): implemented. Masks are windows of one
   persistent pool of Bernoulli(1 - p) values (0 or 1 / (1 - p)), drawn once
   from the seed, at an offset hashed from (seed, step, micro-batch, layer,
   site). A recompute reads the same window, so p = 0.05 gives bit-identical
   losses at N = -1, 14 and 28. Cost at the default: +74 ms per step (+8%) and
   +1.3 GiB, the masked LoRA inputs autodiff keeps for dA. Windows of different
   sites overlap in the pool, so a value can recur at a shifted position in
   another site's mask; each mask on its own is an i.i.d. Bernoulli field.
7. **Bucketing and packing: not done.** Graph build and allocation take under
   2 ms per micro-batch in the kept mode, so there is nothing to amortise.

Also kept: split SwiGLU in both MLPs (its backward keeps the two GEMM outputs
and recomputes the silu: 1.8 GiB less at N = 28, same speed), checkpoint
buffers allocated only for the blocks a mode recomputes (1 GiB at N = 28), the
causal masks uploaded once per sample shape, and the NVML query behind the
step event's `device_used_mib` refreshed at most every 5 s (it cost about 9 ms
per step). Wall time per one-micro-batch step went from 253 to 232 ms with the
last two.

Placement: the admission log prints every segment's node count and scheduler
splits. At the default all are one split on CUDA0 (the kept graph has 4484
nodes), and the optimizer graph is all CUDA0. `HMT_PROFILE=1` prints the
per-segment build, allocation and compute times per step.

## 9. Open

- Dropout default for native: the CLI default stays p = 0; the recipe's 0.05
  costs 8% per step and 1.3 GiB (8.1, lever 6). The choice is Rob's.
- Fused cross-entropy (8.1, lever 3) would need a fork change; worth at most
  about 2% of a step and 0.3 GiB.
