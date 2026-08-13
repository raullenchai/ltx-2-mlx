# ESPECIFICACIÓN DEL PORT LTX-2.5 → MLX (video-lab / ltx-2-mlx)

Fecha: análisis completo de fuentes oficiales. Artefactos de referencia en `/tmp/ltx25-ref/`
(REAMEs, headers safetensors vía Range, código ComfyUI sparse-cloneado, código oficial Lightricks/LTX-2).

---

## 1. ACCESO OFICIAL (gate)

| Repo | gated | README | Archivos |
|---|---|---|---|
| `Lightricks/LTX-2.5` | `auto` | **200 OK** | **OK (Range funciona)** — gate ACEPTADA con el token local |
| `Lightricks/LTX-2.5-Diffusers` | `auto` | **403** | **403 todo** (configs incluidos) — gate NO aceptada aún |
| `dummy9996/LTX-2.5-ungate` (mirror) | no | OK | OK — mismos blobs LFS (sha256 idénticos) |

**Decisión**: los configs Diffusers se reconstruyen de los configs embebidos en los single-files
(autoritativos, mismos valores). El pack Diffusers solo aporta el empaquetado `transformer.*`/`text_encoder.*`
que NO necesitamos (el port sigue el formato split tipo Comfy, igual que el 2.3 local).

Licencia: **LTX-2.x Community License Agreement** (fecha 2026-08-11), texto completo embebido en
`__metadata__.license` de cada safetensors. Libre <$10M revenue; requiere aceptación del gate.

---

## 2. ARCHIVOS OFICIALES + sha256 (verificación)

Repo principal `Lightricks/LTX-2.5` (sha `8a4ff96f…`):

| Archivo | Size | sha256 |
|---|---|---|
| diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors | 42 018 190 584 | `31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4` |
| diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors | 42 018 190 584 | `792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584` |
| diffusion_models/…-distilled-transformer-comfy-int8-convrot.safetensors | 21 504 034 224 | `c4279eeff115cbeaca494bd2183e7d768c38fe85a184dc6afbb7159157c44334` |
| diffusion_models/…-dev-transformer-comfy-int8-convrot.safetensors | 21 504 034 224 | `2edbdb4465cd6c3b532cd67a31ddb38a63e97dcad20be3729675e2a4e8caf92b` |
| diffusion_models/…-distilled-transformer-nvfp4.safetensors | 18 721 432 024 | `f9c4c2ae9a6aa8f732eb02a1c4c3b34888caad3dd35bb65deaf3b5043cda78fa` |
| text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors | 26 263 860 594 | `1c647a94c0e902fb87f9a403cbca36a8b6d8e5867094442df1b41ae557cfd1c6` |
| text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors | 15 372 971 786 | `09a89e084de1a149c3de60cfe9dfd3e5161967eb09eea39e806fcdeffdd568de` |
| vae/ltx-2.5-video-vae-bf16.safetensors | 1 472 223 346 | `847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3` |
| vae/ltx-2.5-video-vae-conv-bf16.safetensors | 1 452 269 922 | `685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d` |
| vae/ltx-2.5-audio-vae-bf16.safetensors | 364 866 540 | `c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5` |
| loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors | 8 899 889 568 | `86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5` |
| model_patches/ltx-2.5-duration-head-bf16.safetensors | 3 843 690 | `2ec71e4206ed365d015f00c05a48caccfb0ee862986809d06ae376c09f5d9190` |
| latent_upscale_models/…-spatial-upscaler-x2-bf16-1.0.safetensors | 995 778 752 | `eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8` |
| latent_upscale_models/…-temporal-upscaler-x2-bf16-1.0.safetensors | 261 944 000 | `2bc3300f2b3c3c1834d72164fbf13a3b9fd73e5a741e8a2c3f4035f89a75c3fe` |

Mirror `dummy9996/LTX-2.5-22b-ungate` (sha `70efb480…`): todos los archivos con **sha256 idénticos** a los oficiales
(mismos punteros LFS). Añade `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors`
(327 322 640 B, `984851b769ea2bcb4c9e0a239a7676239e42c6a6001ddc69943b41ff0b283c1d`) y NO tiene los
latent_upscale_models ni el duration-head. El mirror NO tiene configs/JSON.

Pack `Lightricks/LTX-2.5-Diffusers` (sha `a6de4b53…`, descargable solo tras aceptar gate — hashes capturados de la API):
`model_index.json` (859 B), `modular_model_index.json` (3553 B), `transformer/config.json` (1440 B),
`text_encoder/config.json` (4594 B), `connectors/config.json` (817 B), `vae/config.json` (1480 B),
`scheduler/scheduler_config.json` (489 B), `duration_head/config.json` (251 B), `latent_upsampler/config.json`
(301 B), `diffusion_decoder/config.json` (1089 B), `audio_vae/config.json` (505 B), `vocoder/config.json`
(1478 B), `tokenizer/tokenizer.json` (32 169 626 B, `cc8d3a0c…`), `tokenizer/tokenizer_config.json` (3735 B),
`text_encoder/generation_config.json` (255 B); pesos shardeados con index.json. El transformer_full/ es
idéntico a transformer/ (mismos sha de shards). Los sha de los pesos del pack Diffusers están en
`/tmp/ltx25_blobs.json` (los shards de transformer difieren del single-file: re-empaquetado con prefijos diffusers).

---

## 3. CONFIGS (embebidos en `__metadata__` de los single-files — extraídos vía Range)

### 3.1 Transformer (`config.transformer`, AVTransformer3DModel) — EXACTO
```json
{"_class_name": "AVTransformer3DModel", "activation_fn": "gelu-approximate", "attention_bias": true,
 "attention_head_dim": 128, "attention_type": "default", "caption_channels": 3840,
 "cross_attention_dim": 4096, "double_self_attention": false, "dropout": 0.0, "in_channels": 128,
 "norm_elementwise_affine": false, "norm_eps": 1e-06, "norm_num_groups": 32, "num_attention_heads": 32,
 "num_embeds_ada_norm": 1000, "num_layers": 48, "num_vector_embeds": null, "only_cross_attention": false,
 "cross_attention_norm": true, "out_channels": 128, "upcast_attention": false, "use_linear_projection": false,
 "qk_norm": "rms_norm", "standardization_norm": "rms_norm", "positional_embedding_type": "rope",
 "positional_embedding_theta": 10000.0, "positional_embedding_max_pos": [20, 2048, 2048],
 "timestep_scale_multiplier": 1000, "av_ca_timestep_scale_multiplier": 1000.0,
 "causal_temporal_positioning": true, "audio_num_attention_heads": 32, "audio_attention_head_dim": 64,
 "use_audio_video_cross_attention": true, "ff_bias": false, "share_ff": false, "audio_out_channels": 128,
 "audio_cross_attention_dim": 2048, "audio_positional_embedding_max_pos": [20], "av_cross_ada_norm": true,
 "use_embeddings_connector": true, "connector_attention_head_dim": 128, "connector_num_attention_heads": 32,
 "connector_num_layers": 8, "connector_positional_embedding_max_pos": [4096],
 "connector_num_learnable_registers": 128, "connector_norm_output": true, "use_middle_indices_grid": true,
 "apply_gated_attention": true, "connector_apply_gated_attention": true,
 "caption_projection_first_linear": false, "caption_projection_second_linear": false,
 "caption_proj_input_norm": false, "connector_learnable_registers_std": 1,
 "caption_proj_before_connector": true, "audio_connector_attention_head_dim": 64,
 "audio_connector_num_attention_heads": 32, "cross_attention_adaln": true, "rope_type": "split",
 "frequencies_precision": "float64", "text_encoder_norm_type": "PER_TOKEN_RMS",
 "use_keyframes_abs_pos_embedding": true}
```
`scheduler`: `RectifiedFlowScheduler`, num_train_timesteps 1000, sampler `LinearQuadratic` (no shifting).

### 3.2 Text encoder (embebido en `__metadata__.gemma_config` del TE file)
- `architectures`: **`Gemma4UnifiedForConditionalGeneration`** (model_type `gemma4_unified`). Confirmado.
- `text_config` (gemma4_unified_text, EXACTO):
  hidden_size **3840**, num_hidden_layers **48**, num_attention_heads **16**, num_key_value_heads **8**,
  num_global_key_value_heads **1**, head_dim **256**, global_head_dim **512**, intermediate_size **15360**,
  vocab_size **262144**, sliding_window **1024**, attention_k_eq_v **true**, attention_bias false,
  final_logit_softcapping 30.0, hidden_activation `gelu_pytorch_tanh`, tie_word_embeddings true,
  max_position_embeddings 262144, rms_norm_eps 1e-6, bos 2 / eos 1 / pad 0,
  `layer_types` explícito: `sliding_attention`×5, `full_attention`, repitiendo (full en índices 5,11,17,23,29,35,41,47),
  `rope_parameters`: full_attention → `{"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"}`;
  sliding_attention → `{"rope_theta": 1e4, "rope_type": "default"}`,
  use_double_wide_mlp false, enable_moe_block false, num_kv_shared_layers 0, hidden_size_per_layer_input 0,
  use_bidirectional_attention "vision".
- `audio_config` (gemma4_unified_audio, para prompt-enhancer multimodal; no se usa para encoding de texto).
- Tokens especiales: image_token_id 258880, audio 258881, boi 255999, boa 256000, eoi 258882, eoa 258883.
- **`text_embedding_projection`** (4 tensores, en el MISMO archivo):
  - `video_aggregate_embed.weight` BF16 **[4096, 188160]** + bias [4096]
  - `audio_aggregate_embed.weight` BF16 **[2048, 188160]** + bias [2048]
  - 188160 = 3840 × **49** (48 capas + embedding).
- Otros grupos top-level del archivo: `model.*` (666), `vision_model.*` (9), `audio_projector.embedding_projection.weight`
  [3840, 640], `multi_modal_projector.embedding_projection.weight` [3840, 3840], `hf_asset__*` (chat_template,
  generation_config, processor_config, tokenizer_config) y `tokenizer_json` (32 MB, U8). Los grupos
  vision_model/audio_projector/multi_modal_projector NO se usan para LTX (encoding de texto puro).

### 3.3 VAE video (2 variantes)
- **`ltx-2.5-video-vae-bf16`** = **DiffVAE** (`CausalDiffusionVAE`, 397 tensores): encoder estilo causal
  (blocks res_x/compress_space_res×2/compress_time_res/compress_all_res, patch 4, base_channels 128,
  latent_log_var constant −7.824, pixel_norm, out 128 ch) + **decoder `NADiffusionDecoder`** (310 tensores:
  conv_in [2048,128], conv_in_x_t [256,48], 5 det_stages con attn (head_dim 64) y kernels
  [3,7,7]/[3,7,7]/[3,5,5]/[3,5,5]/[11,11,11], stage_channels [2048,1024,512,512,256],
  stage_depths [4,6,4,2,8], x_t conditioning 48 canales, conv_out [48,256]) + `per_channel_statistics` (2).
- **`ltx-2.5-video-vae-conv-bf16`** = `CausalVideoAutoencoder` clásico (171 tensores, MISMA arquitectura que el
  VAE 2.3 ya portado: encoder/decoder res_x/compress_*, scaling_factor 1.0, patch 4, latent 128 ch).
  **Recomendado para el primer port** (código ya existe en ltx-core-mlx).

### 3.4 Audio VAE + vocoder (archivo combinado)
`audio_vae` (102 tensores) = misma arch que el 2.3 ya portado (ddconfig: mel 64 bins, z 8 ch, ch 128,
ch_mult [1,2,4], 2 res_blocks, pixel norm, sampling_rate 16000, stereo, causal_padding 3; stft 1024/160;
mel fmin 0 fmax 8000, max_wav_value 32768, duration 5.12). `vocoder` (1227 tensores) = mismo que 2.3
(AMP1 resblocks, upsample_rates [5,2,2,2,2,2], snakebeta, stereo, incluye BWE generator).
→ **sin código nuevo**: solo sustituir pesos por los 2.5.

### 3.5 Duration head (16 tensores, prefijo `duration_head.`)
Arch = `AttentionPooler` (query 1×256, MultiheadAttention 256/4 heads) + `video_input_proj` [256,4096] +
`audio_input_proj` [256,2048] + modality_embs [256] + `mlp_hidden` [256,256] + `mlp_out` [1,256] →
salida `exp()`, **segundos**. `seconds_to_num_frames`: `(raw-1)//8*8+1` (grid 8k+1).

### 3.6 Latent upscalers
- spatial x2: `LatentUpsampler` in 128 / mid 1024 / 4 blocks por stage, spatial_upsample, rational_resampler **false**
  → **config idéntica al x2-v1.1 del pack 2.3 local** (código ya portado).
- temporal x2: in 128 / mid 512, temporal_upsample, rational_resampler **true** → igual al temporal x2 2.3 local.

---

## 4. TRANSFORMER — INVENTARIO DE PESOS EXACTO (single-file, 4349 tensores)

Prefijo en archivo: `model.diffusion_model.*` → tras strip queda `transformer.*` (mismo esquema de nombres que el 2.3 MLX).

Top-level (1 c/u salvo nota):
- `patchify_proj` [4096,128]+b, `proj_out` [128,4096]+b; `audio_patchify_proj` [2048,128]+b, `audio_proj_out` [128,2048]+b
- `scale_shift_table` F32 (9,4096); `audio_scale_shift_table` F32 (9,2048); `prompt_scale_shift_table` (2,4096);
  `audio_prompt_scale_shift_table` (2,2048)
- `keyframes_abs_pos_embedding` BF16 **(1, 4096)** ← NUEVO en 2.5
- `adaln_single`: timestep_embedder linear_1 [4096,256]+b, linear_2 [4096,4096]+b, `linear` [36864,4096]+b
  (36864 = 9×4096); `audio_adaln_single` idem con 2048/18432; `prompt_adaln_single` [8192,4096]+b;
  `audio_prompt_adaln_single` [4096,2048]+b
- AV-CA adaln (todos emb→linear 256→dim→linear N×dim): `av_ca_video_scale_shift_adaln_single` (16384,4096),
  `av_ca_audio_scale_shift_adaln_single` (8192,2048), `av_ca_a2v_gate_adaln_single` (4096,4096),
  `av_ca_v2a_gate_adaln_single` (2048,2048)
- `video_embeddings_connector` / `audio_embeddings_connector`: `learnable_registers` [128,4096] / [128,2048] +
  8× `transformer_1d_blocks.N.{attn1.{q,k,v,to_gate_logits,to_out.0} (+bias), q_norm, k_norm, ff.net.{0.proj,2} (+bias)}`
  — CON bias en ff (connector_ff_bias=true), q/k_norm RMS, RoPE 1D split, theta 1e4, max_pos [4096], float64 grid.

Por transformer_blocks.N (48 capas), video (4096) / audio (2048):
- `attn1` (self video), `attn2` (cross text video), `audio_attn1`, `audio_attn2` (cross text audio),
  `audio_to_video_attn` (Q 2048 ← K,V 4096 con to_q [2048,4096], to_k [2048,4096], to_v [2048,4096], to_out [4096,2048]),
  `video_to_audio_attn` (Q 4096 ← K,V 2048: to_q [2048,4096], to_k [2048,4096], to_v [2048,4096], to_out [2048,2048])
- TODAS las attention con `to_gate_logits` [32, dim]+b (gated attention: `out *= 2·sigmoid(gate_logits)`) y
  `q_norm`/`k_norm` RMSNorm(dim)
- `ff.net.0.proj` [16384,4096] / `audio_ff.net.0.proj` [8192,2048] + `ff.net.2` inverso — **SIN bias** (ff_bias=false)
- tablas por capa: `scale_shift_table` (9,4096), `prompt_scale_shift_table` (2,4096), `audio_scale_shift_table`
  (9,2048), `audio_prompt_scale_shift_table` (2,2048), `scale_shift_table_a2v_ca_video` (5,4096),
  `scale_shift_table_a2v_ca_audio` (5,2048)

Semántica (referencia ComfyUI `av_model.py` + `model.py`, fiel al oficial ltx-core):
- adaLN: `timestep_scaled = t·1000`; `adaln_single` (Timesteps 256 sin/cos → SiLU → Linear) → por-token
  [B,T,9D]; tablas: [0:2] shift/scale MSA, [2:3] gate MSA, [3:5] shift/scale MLP, [5:6] gate MLP, [6:9]
  shift/scale/gate Q del cross-attn (cross_attention_adaln). `prompt_timestep` = max sobre tokens → prompt_adaln_single → (2,D) shift/scale KV.
- cross text attention: `rms_norm(x)·(1+q_scale)+q_shift` → attn2(Q) contra context·(1+scale_kv)+shift_kv → ×gate_q.
- AV cross: `av_ca_*_scale_shift` (4 params: shift/scale de ambas modalidades) + gates a2v/v2a (1 param);
  RoPE de cruce con `use_middle_indices_grid=True` (posición media del token) en dim audio_cross_attention_dim (2048).
- RoPE split: freqs `theta**(linspace(log1, log_theta, dim/(2·n_pos)))·π/2`, fraccionales
  `(coord/max_pos)·2−1`, pad a dim/2, float64 grid (`frequencies_precision`). Video coords:
  pixel→latent coords con vae_scale_factors (8,32,32) y causal_fix, `t·(1/frame_rate)`.
- **keyframes_abs_pos_embedding**: se SUMA a los tokens cuyo `temporal_start == 0` (frames individuales /
  keyframes de multishot): `x + mask·emb`. No-op si el tensor no existe (compat).
- Patchifier: SymmetricPatchifier(patch 1, start_end=True): token = frame completo; latentes [B,128,F,H,W]
  → [B, F·H·W, 4096]. Audio: [B,8,T,16] → [B, T, 128·16=2048]→[B,T,2048].

---

## 5. TEXT ENCODER → MLX (veredicto)

**mlx-lm 0.31.3 (release) YA SOPORTA gemma4**: `mlx_lm/models/gemma4.py` (model_type `"gemma4"`, wrapper
`language_model` = gemma4_text) y `gemma4_text.py`. El venv local tiene **0.31.1 (sin gemma4)** →
**bloqueo resuelto con `uv pip install -U mlx-lm` (0.31.3)** (verificado: el tag v0.31.3 contiene gemma4.py).

Mapeo verificado:
- El archivo LTX usa claves `model.embed_tokens.weight`, `model.layers.N.{input_layernorm,
  post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm, layer_scalar,
  mlp.{gate,up,down}_proj, self_attn.{q,k,v,o}_proj, q_norm, k_norm}`, `model.norm.weight` — coincide 1:1 con
  `Gemma4TextModel` de mlx-lm (`sanitize` de gemma4.Model: quita `model.` y `vision_tower`/`multi_modal_projector`).
- `attention_k_eq_v=true` → mlx-lm usa `use_k_eq_v = k_eq_v and not sliding` (v = k) **solo en las 8 capas
  full-attention** — coincide EXACTO con el archivo (40 capas sliding con v_proj [2048,3840]; 8 full con
  q [8192,3840] = 16×512, k [512,3840] = 1×512 global, SIN v_proj).
- Config de carga: `{"model_type": "gemma4", "text_config": <text_config embebido>}` (ModelArgs.from_dict filtra
  campos extra). layer_types explícito del config; rope por tipo (proportional 0.25/1e6 vs default/1e4).
- Hidden states por capa: mismo mecanismo que el 2.3 (`get_all_hidden_states`): emb (×sqrt(3840)) + 48 capas
  con mask causal completa (ventana 1024 ≡ causal para secuencias ≤1024; TOKENIZER_MAX_LENGTH=1024, left-pad).
  Para capas full-attention pasar mask causal completa; sliding idéntico (longitud ≤ ventana). Atención:
  `layer(h, mask, cache=None)` devuelve tupla — usar `h[0]`.

Tokenizer: `tokenizer_json` embebido en el TE file (32 MB, extraíble con safetensors); backend tokenizers
(gemma3-style, sin sentencepiece); specials: bos `<bos>`=2, eos `<eos>`=1 (y 106/50), pad 0, `<turn|>` etc.
Equivalente al de gemma-3-12b (vocab 262144) — reutilizable `mlx-community/gemma-3-12b-it-4bit` tokenizer
solo si verifica vocab igual; lo correcto es extraer `tokenizer_json` del archivo TE (ya lo hace el 2.3? no:
el 2.3 usa el tokenizer del repo MLX; para 2.5 el archivo TE lo trae).

Feature extraction (V2, ya portado en `GemmaFeaturesExtractorV2` de ltx-core-mlx para 2.3 — MISMO código):
stack de 49 capas → [B,T,3840,49] → RMS por token sobre dim 3840 (`rsqrt(mean(x²)+1e-6)`) → flatten → [B,T,188160]
→ zero en pads → `video_aggregate_embed(x·sqrt(4096/3840))` → [B,T,4096]; `audio_aggregate_embed(x·sqrt(2048/3840))`
→ [B,T,2048]. (ltx-core oficial `FeatureExtractorV2` y ComfyUI `DualLinearProjection` coinciden.)

Connectors (ídem 2.3, ya portados): permutación right-pad (válidos primero), `video_embeddings_connector`
(8 bloques, 4096, 32 heads × 128, gated, RoPE 1D split float64 theta 1e4 max_pos 4096) y `audio_embeddings_connector`
(2048, 32×64). Registers: `num_dups = ceil(max(1024, T)/128)`; tile de `learnable_registers`; en ltx-core se
**reemplazan las posiciones de padding** (seq % 128 == 0 ⇒ 1024 posiciones); ComfyUI los concatena — equivalentes
tras right-pad. Salida: video [B,1024,4096] + audio [B,1024,2048] (mask de atencion a cero = full attention).
Contexto final del DiT: **concat([video, audio], dim=-1) = [B, 1024, 6144]**, split [4096|2048] en
`_prepare_context`.

---

## 6. PIPELINE (valores exactos)

- Distilled: **stage 1 a media res** (ej. 960×544×121 @24fps), **8 pasos**, sigmas
  `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]` (9 valores), **CFG=1** (sin CFG),
  guidance_scale 1.0, audio_guidance_scale 1.0, stg 0, modality_scale 1.0 (valores del ejemplo Diffusers
  oficial; idénticos a las sigmas del scheduler MLX 2.3 ya implementado).
- Stage 2 (super-res): latents upscaled ×2 → mismo DiT, 3 pasos, sigmas `[0.909375, 0.725, 0.421875, 0.0]`,
  noise_scale = sigma[0].
- Restricciones: `num_frames % 8 == 1` (1,9,…,121); W,H divisibles por 32; fps 24.
- Condicionamiento imagen: `--image PATH FRAME_IDX STRENGTH` (keyframes con pixel_coords de imagen; strength
  atenúa la self-attention noisy↔guía — opcional para port 1).
- Duration head (opcional): si no se pasa num_frames, predice segundos desde los tokens del connector
  (video y/o audio) → frames en grid 8k+1.
- Prompt enhancer (opcional): gemma4 instruct con system prompt (gemma4_t2v_system_prompt.txt en ltx-core),
  greedy, no_repeat_ngram 5, max_new_tokens 600.
- Upscaler espacial x2 del pipeline distilled: el README oficial usa el de LTX-2.3
  (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`), pero el 2.5 trae el suyo con config idéntica.
- VAE decode: DiffVAE con tiling (`enable_tiling`) en stage 2; alternativa conv-VAE.
- Audio: latente [B,8,T,16] @ 25 tokens/s (16000 Hz, hop 160, downsample 4); el DiT genera video+audio
  simultáneamente; decode con audio-VAE (mel 64) + vocoder (hiFiGAN-AMP1 + BWE, snakebeta, stereo).

---

## 7. TABLA DE DELTAS 2.3 → 2.5 (port)

| Componente | 2.3 (estado actual MLX) | 2.5 | Trabajo |
|---|---|---|---|
| DiT transformer (48×4096/2048, gated, adaLN cross, connectors 8×128 reg) | portado | **misma arquitectura**; pesos nuevos | cambiar pesos + `keyframes_abs_pos_embedding` (1,4096): suma a tokens t=0 (tensor nuevo; código ~10 líneas) |
| ff_bias | false (ya) | false | — |
| Text encoder | gemma3-12b (mlx-lm) | **gemma4-12b** (mlx-lm ≥0.31.3, arch nueva) | upgrade mlx-lm; loader config `gemma4`+text_config; tokenizer del TE file; mismo collector de hidden states |
| Feature extractor (aggregates 188160→4096/2048) | portado | idéntico | pesos nuevos en TE file (`text_embedding_projection.*`) |
| Connectors video/audio | portado | idéntico | pesos nuevos en el single-file transformer (`model.diffusion_model.*_embeddings_connector.*`) |
| Video VAE | causal (encoder+decoder) | DiffVAE (encoder causal + **NADiffusionDecoder** 5 stages con attn) o conv-VAE (idéntico al 2.3) | **nuevo**: NADiffusionDecoder (310 tensores; conv_in_x_t, det_stages con atención, x_t 48ch) — o port 1 con conv-VAE |
| Audio VAE + vocoder | portado | idéntico (archivo combinado) | solo pesos |
| Upscalers x2 esp/temp | portado (x2 v1.1, x1.5, temporal) | idéntico | solo pesos |
| Duration head | — | **nuevo** (16 tensores, trivial) | portar `DurationHead` + `seconds_to_num_frames` |
| Sigmas/CFG/scheduler | idénticos | idénticos | — |
| Tokenizer/prompts | gemma3 | gemma4 (tokenizer.json embebido; system prompts nuevos) | extraer tokenizer; prompts |
| Empaquetado | `transformer-*.safetensors` (`transformer.*`), `connector.safetensors` (`connector.*`) | misma estructura: strip `model.diffusion_model.`→`transformer.`; TE file: strip `model.`→gemma4 (mlx-lm), `text_embedding_projection.*`+`*_embeddings_connector.*`→`connector.` | scripts de conversión (iguales al 2.3) |

**Empaquetado propuesto (split MLX)**:
1. `text_encoder/` (mlx-lm gemma4): del TE file, claves `model.*` (strip `model.`) + config.json
   (`model_type: "gemma4"`, `text_config: <embebido>`) + tokenizer.json/tokenizer_config.json extraídos.
2. `connector.safetensors` (prefijo `connector.`): `text_embedding_projection.*` + `video_embeddings_connector.*`
   + `audio_embeddings_connector.*` del single-file transformer (strip `model.diffusion_model.`).
3. `transformer-distilled.safetensors` / `transformer-dev.safetensors` (prefijo `transformer.`): resto del
   single-file (incluye keyframes_abs_pos_embedding; rename `linear_1`→`linear1` igual que el convert 2.3).
4. `vae_decoder.safetensors`+`vae_encoder.safetensors` (conv) o `diffusion_decoder.safetensors` (DiffVAE).
5. `audio_vae.safetensors` (102) + `vocoder.safetensors` (1227) del archivo combinado.
6. `duration_head.safetensors` (strip `duration_head.` / `model.diffusion_model.duration_head.`).
7. `spatial_upscaler_x2.safetensors`, `temporal_upscaler_x2.safetensors`.

---

## 8. BLOQUEOS REALES Y DECISIONES

1. **Gate Diffusers NO aceptada** (403 en todo). No bloquea: configs reconstruidos de los metadata embebidos
   (mismos valores) — guardados en `/tmp/ltx25-ref/headers/*.metadata.json`.
2. **mlx-lm 0.31.1 local sin gemma4** → resolver con upgrade a **0.31.3** (release que incluye
   `gemma4.py`/`gemma4_text.py`; verificado en el tag v0.31.3). Sin port de arch a mlx-lm necesario.
3. El text encoder necesita ~26 GB bf16 (12B) — cuantificar (el 2.3 usa gemma-3-12b-it-4bit; para gemma4 no
   hay aún conversión MLX en el cache local; el archivo bf16 oficial se puede cuantizar con `mlx_lm.convert`
   o cargar bf16 y dejar que MLX lo maneje; las capas full-attention (8) con k_eq_v y kv heads globales están
   soportadas en 0.31.3).
4. DiffVAE decoder (NADiffusionDecoder) = componente nuevo más grande del port; alternativa inmediata:
   `ltx-2.5-video-vae-conv` (arquitectura 2.3 ya portada).
5. Mirror `dummy9996/LTX-2.5-22b-ungate` sirve como fuente alternativa sin gate (sha256 idénticos verificados).

Archivos de referencia clave guardados: `/tmp/ltx25-ref/headers/ALL_HEADERS.json` (headers completos de los 9
archivos), `distilled_transformer.metadata.json` (config+licencia), `gemma4_te.header.json`,
`gemma4_mlxlm.py`/`gemma4_text_mlxlm.py` (mlx-lm main), `ComfyUI/comfy/ldm/lightricks/*`,
`LTX-2/packages/ltx-core/...` (gemma/feature_extractor.py, embeddings_processor.py, encoder_configurator.py),
`diffusers_ltx2_utils.py` (DEFAULT_NEGATIVE_PROMPT + sigmas).
