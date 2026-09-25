from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from .cache import copy_cache_rows, crop_cache, reorder_cache, repeat_cache
from .config import Config


def prepare_processor_compatibility(model: str) -> bool:
    """Work around processor metadata errors in the paper's Game-RL checkpoint."""

    if model != "OpenMOSS-Team/Game-RL-Qwen2.5-VL-7B":
        return False
    import transformers
    import transformers.models.qwen2_5_vl as qwen25_module
    from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES

    qwen25_module.Qwen2_5_VLImageProcessor = transformers.Qwen2VLImageProcessor
    qwen25_module.Qwen2_5_VLImageProcessorFast = transformers.Qwen2VLImageProcessorFast
    IMAGE_PROCESSOR_MAPPING_NAMES["qwen2_5_vl"] = (
        "Qwen2_5_VLImageProcessor",
        "Qwen2_5_VLImageProcessorFast",
    )
    return True


def _move(value: Any, device: torch.device) -> Any:
    """Recursively move processor outputs to the LVLM device."""

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _repeat_rows(value: torch.Tensor | None, count: int) -> torch.Tensor | None:
    """Replicate single-example model state across the particle batch."""

    if value is None:
        return None
    return value.repeat_interleave(count, dim=0)


@dataclass
class PreparedBatch:
    """Hold text, images, and processor tensors for a full-image prefill."""

    images: list[Image.Image]
    prompt_texts: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    model_kwargs: dict[str, Any]


@dataclass
class DecoderState:
    """Hold particle prefixes, KV cache, and next-token logits."""

    prepared: PreparedBatch
    generated_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Any
    next_logits: torch.Tensor
    rope_deltas: torch.Tensor | None


class TransformersVLM:
    """Provide full-image target and attention-scout states for Qwen-VL models."""

    def __init__(self, cfg: Config):
        """Load the frozen LVLM and locate routing and attention-bias layers."""

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[cfg.dtype]
        use_fast = prepare_processor_compatibility(cfg.model)
        self.processor = AutoProcessor.from_pretrained(
            cfg.model,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
            **({"use_fast": True} if use_fast else {}),
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        model_kwargs: dict[str, Any] = {
            "dtype": self.dtype,
            "attn_implementation": cfg.attention_backend,
        }
        if cfg.device_map:
            model_kwargs["device_map"] = cfg.device_map
        self.model = AutoModelForImageTextToText.from_pretrained(cfg.model, **model_kwargs)
        if not cfg.device_map:
            self.model.to(self.device)
        else:
            self.device = next(self.model.parameters()).device
        self.model.eval()
        self.model.config.use_cache = True
        if cfg.compile:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        config = getattr(self.model, "config", None)
        ids: list[int] = []
        for value in (
            getattr(config, "eos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
        ):
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                if item is not None and int(item) not in ids:
                    ids.append(int(item))
        self.terminal_ids = tuple(ids)
        pad_id = self.tokenizer.pad_token_id
        self.pad_token_id = int(self.terminal_ids[0] if pad_id is None else pad_id)

        self.model_type = str(getattr(config, "model_type", ""))
        self._attention_layers = self._find_attention_layers()
        self._attention, self._attention_layer = self._find_attention(cfg.attention_layer)
        self._visual_positions: torch.Tensor | None = None
        self._visual_keys: torch.Tensor | None = None
        self._response_query: torch.Tensor | None = None
        self.qk_projection_seconds = 0.0

    def _messages(self, question: str) -> list[dict[str, Any]]:
        """Build a user message with the image and question, without a system prompt."""

        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

    def render_prompt(self, question: str) -> str:
        """Render the full-image prompt with the backbone's native chat template."""

        return self.processor.apply_chat_template(
            self._messages(question), tokenize=False, add_generation_prompt=True
        )

    def prepare(
        self,
        images: Sequence[Image.Image],
        prompt_texts: Sequence[str],
        *,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> PreparedBatch:
        """Encode images and rendered prompts, retaining visual grid parameters."""

        kwargs: dict[str, Any] = {
            "text": list(prompt_texts),
            "images": [image.convert("RGB") for image in images],
            "padding": True,
            "return_tensors": "pt",
        }
        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        batch = dict(self.processor(**kwargs))
        input_ids = batch.pop("input_ids").to(self.device)
        attention_mask = batch.pop("attention_mask", torch.ones_like(input_ids)).to(self.device)
        return PreparedBatch(
            images=[image.convert("RGB") for image in images],
            prompt_texts=list(prompt_texts),
            input_ids=input_ids,
            attention_mask=attention_mask,
            model_kwargs=_move(batch, self.device),
        )

    def prepare_example(self, image: Image.Image, question: str) -> PreparedBatch:
        """Convert one ``(I, x)`` pair to full-image prefill inputs."""

        return self.prepare([image], [self.render_prompt(question)])

    def _find_attention_layers(self) -> list[Any]:
        """Collect decoder self-attention layers for scout visual-logit bias."""

        root = getattr(self.model, "_orig_mod", self.model)
        multimodal = getattr(root, "model", None)
        language_model = getattr(multimodal, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers is None:
            return []
        return [layer.self_attn for layer in layers]

    def _find_attention(self, layer: float) -> tuple[Any, int]:
        """Resolve the layer used for prefix Q--K routing scores."""

        if not self._attention_layers:
            return None, int(layer)

        # A fractional layer denotes relative depth across different backbones.
        if 0.0 < layer < 1.0:
            index = int(layer * len(self._attention_layers))
        else:
            index = int(layer)
            if float(index) != float(layer):
                raise ValueError("attention-layer must be a relative depth in (0, 1) or an integer index.")
            if index < 0:
                index += len(self._attention_layers)
        if not 0 <= index < len(self._attention_layers):
            raise ValueError(
                f"attention-layer={layer} is out of range for a {len(self._attention_layers)}-layer model."
            )
        return self._attention_layers[index], index

    def _image_positions(self, prepared: PreparedBatch) -> torch.Tensor:
        """Return original-image token positions ``V_I`` in the multimodal prompt."""

        root = getattr(self.model, "_orig_mod", self.model)
        image_token_id = getattr(getattr(root, "config", None), "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(self.processor, "image_token_id", None)
        return torch.nonzero(
            prepared.input_ids[0].eq(int(image_token_id)), as_tuple=False
        ).flatten()

    def _project_qk(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and rotate query/key vectors for visual routing."""

        batch, length, _ = hidden_states.shape
        if self.model_type == "qwen2_5_vl":
            query = (
                module.q_proj(hidden_states)
                .view(batch, length, -1, module.head_dim)
                .transpose(1, 2)
            )
            key = (
                module.k_proj(hidden_states)
                .view(batch, length, -1, module.head_dim)
                .transpose(1, 2)
            )
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                apply_multimodal_rotary_pos_emb,
            )

            cosine, sine = position_embeddings
            query, key = apply_multimodal_rotary_pos_emb(
                query, key, cosine, sine, module.rope_scaling["mrope_section"]
            )
        elif self.model_type == "qwen3_vl":
            shape = (batch, length, -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden_states).view(shape)).transpose(1, 2)
            key = module.k_norm(module.k_proj(hidden_states).view(shape)).transpose(1, 2)
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

            query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        else:
            raise RuntimeError("ReSight Q--K routing supports the paper's Qwen2.5-VL/Qwen3-VL models.")
        if query.shape[1] != key.shape[1]:
            key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        return query, key

    @contextlib.contextmanager
    def _capture_qk(self, mode: str) -> Iterator[None]:
        """Temporarily capture original-image keys and the current last-token query."""

        if self._attention is None:
            yield
            return

        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            """Compute and cache routing Q--K state from the selected attention layer."""

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            projection_start = time.perf_counter()
            hidden = kwargs.get("hidden_states")
            if hidden is None:
                hidden = args[0]
            query, key = self._project_qk(module, hidden, kwargs["position_embeddings"])
            if mode == "prompt":
                assert self._visual_positions is not None
                self._visual_keys = key[0, :, self._visual_positions, :].detach().float()
            else:
                self._response_query = query[:, :, -1, :].detach().float()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.qk_projection_seconds += time.perf_counter() - projection_start

        handle = self._attention.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()

    def _prefill(self, prepared: PreparedBatch, *, capture_prompt: bool = False) -> Any:
        """Run native multimodal prefill, optionally capturing visual keys."""

        kwargs = {
            "input_ids": prepared.input_ids,
            "attention_mask": prepared.attention_mask,
            "cache_position": torch.arange(prepared.input_ids.shape[1], device=self.device),
            "use_cache": True,
            "return_dict": True,
            "logits_to_keep": 1,
            **prepared.model_kwargs,
        }
        context = self._capture_qk("prompt") if capture_prompt else contextlib.nullcontext()
        with torch.inference_mode(), context:
            return self.model(**kwargs)

    @staticmethod
    def _rope(out: Any) -> torch.Tensor | None:
        """Read the multimodal RoPE offset returned by a Qwen-VL forward pass."""

        value = getattr(out, "rope_deltas", None)
        return value if torch.is_tensor(value) else None

    def _set_rope(self, value: torch.Tensor | None) -> None:
        """Restore the RoPE offset aligned with the current particle batch."""

        if value is None:
            return
        root = getattr(self.model, "_orig_mod", self.model)
        for target in (getattr(root, "model", None), root):
            if target is not None and hasattr(target, "rope_deltas"):
                target.rope_deltas = value
                return

    def init_full(
        self, prepared: PreparedBatch, count: int, *, capture_prompt: bool = False
    ) -> DecoderState:
        """Run one full-image prefill and replicate its cache for ``K*M`` particles."""

        if capture_prompt:
            self._visual_positions = self._image_positions(prepared)
        out = self._prefill(prepared, capture_prompt=capture_prompt)
        return DecoderState(
            prepared=prepared,
            generated_ids=torch.empty((count, 0), dtype=torch.long, device=self.device),
            attention_mask=prepared.attention_mask.repeat_interleave(count, dim=0),
            past_key_values=repeat_cache(self.model, out.past_key_values, count),
            next_logits=out.logits[:, -1, :].repeat_interleave(count, dim=0).contiguous(),
            rope_deltas=_repeat_rows(self._rope(out), count),
        )

    @contextlib.contextmanager
    def _visual_attention_reactivation(
        self, visual_bias: torch.Tensor, key_length: int
    ) -> Iterator[None]:
        """Raise original-image attention logits only during scout forward passes."""

        if not self._attention_layers or self._visual_positions is None:
            raise RuntimeError("This model has no text-attention layers for visual reactivation.")
        if visual_bias.shape[1] != self._visual_positions.numel():
            raise ValueError("Visual attention bias does not match the image-token count.")

        def hook(
            _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            """Add ``b_j(R_g)`` to the last query's image-key mask at each layer."""

            hidden = kwargs.get("hidden_states")
            if hidden is None:
                hidden = args[0]
            batch, query_length = hidden.shape[:2]
            if batch != visual_bias.shape[0]:
                raise ValueError("Scout attention-bias batch size does not match decoder state.")

            attention_mask = kwargs.get("attention_mask")
            mask_key_length = (
                int(attention_mask.shape[-1]) if torch.is_tensor(attention_mask) else key_length
            )
            additive = torch.zeros(
                (batch, 1, query_length, mask_key_length),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            positions = self._visual_positions.to(hidden.device)
            valid = positions < mask_key_length
            additive[:, 0, -1, positions[valid]] = visual_bias[:, valid].to(hidden.dtype)
            kwargs["attention_mask"] = (
                additive if attention_mask is None else attention_mask + additive
            )
            return args, kwargs

        handles = [
            attention.register_forward_pre_hook(hook, with_kwargs=True)
            for attention in self._attention_layers
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def init_attention_scout(
        self,
        state: DecoderState,
        particle_indices: torch.Tensor,
        visual_bias: torch.Tensor,
    ) -> DecoderState:
        """Fork the full-image prefix and replay its last token with attention bias."""

        indices = particle_indices.to(self.device).long()
        if state.generated_ids.shape[1] == 0:
            raise ValueError("An attention-reactivated scout needs at least one generated token.")
        selected = DecoderState(
            prepared=state.prepared,
            generated_ids=state.generated_ids.index_select(0, indices),
            attention_mask=state.attention_mask.index_select(0, indices),
            past_key_values=copy_cache_rows(state.past_key_values, indices),
            next_logits=state.next_logits.index_select(0, indices),
            rope_deltas=(
                state.rope_deltas.index_select(0, indices)
                if state.rope_deltas is not None
                and state.rope_deltas.shape[0] == state.generated_ids.shape[0]
                else state.rope_deltas
            ),
        )
        last_tokens = selected.generated_ids[:, -1].clone()
        selected.generated_ids = selected.generated_ids[:, :-1]
        selected.attention_mask = selected.attention_mask[:, :-1]
        selected.past_key_values = crop_cache(
            selected.past_key_values, selected.attention_mask.shape[1]
        )
        return self.advance(selected, last_tokens, visual_attention_bias=visual_bias)

    def _prompt_rows(self, prepared: PreparedBatch, count: int) -> torch.Tensor:
        """Reuse or replicate prompt-token rows to match the decoder batch."""

        if prepared.input_ids.shape[0] == count:
            return prepared.input_ids
        return prepared.input_ids.repeat_interleave(count, dim=0)

    def _response_mask(self, tokens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Mask post-EOS padding so it does not affect conditional probabilities."""

        terminal = torch.zeros_like(tokens, dtype=torch.bool)
        for token_id in self.terminal_ids:
            terminal |= tokens.eq(token_id)
        before = terminal.int().cumsum(dim=1) - terminal.int()
        return before.eq(0).to(dtype)

    def advance(
        self,
        state: DecoderState,
        tokens: torch.Tensor,
        *,
        capture_query: bool = False,
        visual_attention_bias: torch.Tensor | None = None,
    ) -> DecoderState:
        """Advance the decoder with sampled tokens and return next-token logits.

        The persistent target state omits ``visual_attention_bias`` and yields
        ``p_F``; a temporary scout state uses the bias and yields ``p_A``.
        """

        tokens = tokens.to(self.device).long().view(-1, 1)
        generated = torch.cat([state.generated_ids, tokens], dim=1)
        prompt_length = state.attention_mask.shape[1] - state.generated_ids.shape[1]
        prompt_mask = state.attention_mask[:, :prompt_length]
        attention_mask = torch.cat(
            [prompt_mask, self._response_mask(generated, state.attention_mask.dtype)], dim=1
        )
        full_ids = torch.cat([self._prompt_rows(state.prepared, tokens.shape[0]), generated], dim=1)
        start = state.attention_mask.shape[1]
        cache_position = torch.tensor([start], dtype=torch.long, device=self.device)
        self._set_rope(state.rope_deltas)
        inputs = self.model.prepare_inputs_for_generation(
            input_ids=full_ids,
            past_key_values=state.past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=True,
        )
        inputs["logits_to_keep"] = 1
        capture_context = (
            self._capture_qk("response") if capture_query else contextlib.nullcontext()
        )
        attention_context = (
            self._visual_attention_reactivation(visual_attention_bias, attention_mask.shape[1])
            if visual_attention_bias is not None
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), capture_context, attention_context:
            out = self.model(**inputs, return_dict=True)
        return DecoderState(
            prepared=state.prepared,
            generated_ids=generated,
            attention_mask=attention_mask,
            past_key_values=out.past_key_values,
            next_logits=out.logits[:, -1, :],
            rope_deltas=self._rope(out) if self._rope(out) is not None else state.rope_deltas,
        )

    def reorder(self, state: DecoderState, indices: torch.Tensor) -> DecoderState:
        """Reorder trajectories, KV cache, logits, and RoPE state by ancestor."""

        indices = indices.to(self.device).long()
        rope = state.rope_deltas
        if rope is not None and rope.shape[0] == state.generated_ids.shape[0]:
            rope = rope.index_select(0, indices)
        return DecoderState(
            prepared=state.prepared,
            generated_ids=state.generated_ids.index_select(0, indices),
            attention_mask=state.attention_mask.index_select(0, indices),
            past_key_values=reorder_cache(self.model, state.past_key_values, indices),
            next_logits=state.next_logits.index_select(0, indices),
            rope_deltas=rope,
        )

    def route_attention(self, count: int) -> torch.Tensor:
        """Compute visual Q--K attention relevance used for routing."""

        query = self._response_query
        if query is None or self._visual_keys is None:
            raise RuntimeError("No Q--K state was captured at the visual checkpoint.")
        if query.shape[0] == 1 and count > 1:
            query = query.repeat_interleave(count, dim=0)
        scaling = float(getattr(self._attention, "scaling", self._attention.head_dim**-0.5))
        logits = torch.einsum("bhd,hnd->bhn", query, self._visual_keys) * scaling
        return torch.softmax(logits, dim=-1)

    def visual_grid(self, prepared: PreparedBatch) -> tuple[int, int, int]:
        """Return the ``(T,H,W)`` grid aligned with post-merge visual tokens."""

        grid = prepared.model_kwargs["image_grid_thw"][0].long()
        merge = int(
            getattr(
                getattr(getattr(self.model, "model", None), "visual", None), "spatial_merge_size", 2
            )
        )
        return int(grid[0]), int(grid[1] // merge), int(grid[2] // merge)

    def visual_token_count(self, prepared: PreparedBatch) -> int:
        """Count language-side visual tokens produced by image prefill."""

        grid = prepared.model_kwargs.get("image_grid_thw")
        if grid is None:
            return 0
        merge = int(
            getattr(
                getattr(getattr(self.model, "model", None), "visual", None), "spatial_merge_size", 2
            )
        )
        return int(grid.long().prod(dim=1).sum().item() // (merge * merge))

    def decode(self, tokens: Sequence[int]) -> str:
        """Truncate particle tokens at the first EOS and decode response text."""

        result: list[int] = []
        for token in tokens:
            token = int(token)
            if token in self.terminal_ids:
                break
            result.append(token)
        return self.tokenizer.decode(result, skip_special_tokens=True)
