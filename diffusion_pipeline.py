"""Stable Diffusion pipeline adapted from EBAMA for SA-EBAMA inference."""
import itertools
from typing import Any, Callable, Dict, Optional, Union, List, Tuple

import spacy
import torch
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_attend_and_excite import (
    AttentionStore,
    AttendExciteAttnProcessor,
)


import numpy as np
import math
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    logging,
)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from compute_loss import (
    get_attention_map_index_to_wordpiece,
    split_indices,
    calculate_positive_loss,
    calculate_negative_loss,
    get_indices,
    start_token,
    end_token,
    align_wordpieces_indices,
    extract_attribution_indices,
    extract_attribution_indices_with_verbs,
    extract_attribution_indices_with_verb_root,
)

from losses.separation_loss import object_separation_loss, attention_overlap_ratio
from utils.adaptive_scheduler import (
    get_active_steps,
    should_attempt_guidance,
    should_update_latent,
    dynamic_step_size,
    update_moving_average,
)
from utils.overlap_vis_utils import (
    build_overlap_snapshot,
    save_overlap_summary,
    parse_vis_steps,
)

from PIL import Image

logger = logging.get_logger(__name__)


class EbamaDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )

        self.parser = spacy.load("en_core_web_trf")
        self.subtrees_indices = None
        self.doc = None

        self.use_separation = True
        self.gamma_sep = 0.10
        self.sep_tau = "percentile"
        self.sep_temp = 0.15
        self.sep_percentile = 0.65

        self.use_adaptive_guidance = True
        self.adaptive_mode = "sparse10"
        self.max_guidance_step = 25
        self.adaptive_dynamic_step = True
        self.adaptive_thresholds = {"loss": None, "overlap": 0.35}
        self._active_steps = set(range(26))
        self._last_loss_dict = {}
        self._loss_moving_avg = None
        self.guidance_trace = []
        self.log_guidance = False

    def _aggregate_and_get_attention_maps_per_token(self):
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        attention_maps_list = _get_attention_maps_list(attention_maps=attention_maps)
        return attention_maps_list

    @staticmethod
    def _update_latent(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        attn_res: Optional[Tuple[int]] = (16, 16),
        step_size: float = 20.0,
        parsed_prompt: str = None,
        use_separation: bool = True,
        gamma_sep: float = 0.10,
        sep_tau: Union[str, float] = "percentile",
        sep_temp: float = 0.15,
        sep_percentile: float = 0.65,
        use_adaptive_guidance: bool = True,
        adaptive_mode: str = "sparse10",
        max_guidance_step: int = 25,
        adaptive_loss_threshold: Optional[float] = None,
        adaptive_overlap_threshold: Optional[float] = 0.35,
        adaptive_dynamic_step: bool = True,
        log_guidance: bool = False,

        save_overlap_vis: bool = False,
        overlap_vis_dir: Optional[str] = None,
        overlap_vis_steps: Optional[Union[str, List[int], Tuple[int, ...]]] = None,
        overlap_vis_method: Optional[str] = None,
        overlap_vis_prompt_id: Optional[str] = None,
        overlap_vis_seed: Optional[int] = None,
    ):
        """Generate images with EBAMA alignment and optional separation and adaptive guidance."""

        if parsed_prompt:
            self.doc = parsed_prompt
        else:
            self.doc = self.parser(prompt)
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )

        negative_prompt_embeds, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.stack([negative_prompt_embeds, prompt_embeds], dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attention_store = AttentionStore(attn_res)
        self.register_attention_control()

        self.use_separation = bool(use_separation)
        self.gamma_sep = float(gamma_sep)
        self.sep_tau = sep_tau
        self.sep_temp = float(sep_temp)
        self.sep_percentile = float(sep_percentile)
        self.use_adaptive_guidance = bool(use_adaptive_guidance)
        self.adaptive_mode = adaptive_mode
        self.max_guidance_step = int(max_guidance_step)
        self.adaptive_dynamic_step = bool(adaptive_dynamic_step) and self.use_adaptive_guidance
        self.adaptive_thresholds = {
            "loss": adaptive_loss_threshold if self.use_adaptive_guidance else None,
            "overlap": adaptive_overlap_threshold if self.use_adaptive_guidance else None,
        }
        self._active_steps = get_active_steps(
            mode=adaptive_mode if use_adaptive_guidance else "fixed25",
            num_inference_steps=num_inference_steps,
            max_guidance_step=self.max_guidance_step,
        )
        self._last_loss_dict = {}
        self._loss_moving_avg = None
        self.guidance_trace = []
        self.log_guidance = bool(log_guidance)

        self.save_overlap_vis = bool(save_overlap_vis)
        self.overlap_vis_dir = overlap_vis_dir
        self.overlap_vis_steps = set(parse_vis_steps(overlap_vis_steps))
        self.overlap_vis_records = []
        self.overlap_vis_meta = {
            "method": overlap_vis_method,
            "prompt_id": overlap_vis_prompt_id,
            "seed": int(overlap_vis_seed) if overlap_vis_seed is not None else None,
            "prompt": prompt if isinstance(prompt, str) else str(prompt),
            "vis_steps": sorted(list(self.overlap_vis_steps)),
        }

        text_embeddings = [prompt_embeds[1][None, ...]]

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                self.i = i

                if should_attempt_guidance(
                    i,
                    active_steps=self._active_steps,
                    max_guidance_step=self.max_guidance_step,
                    last_loss_dict=self._last_loss_dict,
                    thresholds=self.adaptive_thresholds,
                ):
                    latents = self._step(
                        latents,
                        text_embeddings,
                        t,
                        i,
                        step_size,
                        cross_attention_kwargs,
                        prompt,
                    )
                elif self.log_guidance:
                    self.guidance_trace.append(
                        {"step": int(i), "timestep": int(t), "attempted": False, "updated": False}
                    )

                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text[0], guidance_rescale=guidance_rescale
                )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        if (
            getattr(self, "save_overlap_vis", False)
            and getattr(self, "overlap_vis_dir", None) is not None
        ):
            image_for_vis = None
            if output_type == "pil" and isinstance(image, list) and len(image) > 0:
                image_for_vis = image[0]

            save_overlap_summary(
                records=getattr(self, "overlap_vis_records", []),
                output_dir=self.overlap_vis_dir,
                prompt=prompt if isinstance(prompt, str) else str(prompt),
                method=self.overlap_vis_meta.get("method", "unknown"),
                prompt_id=self.overlap_vis_meta.get("prompt_id", None),
                seed=self.overlap_vis_meta.get("seed", None),
                vis_steps=sorted(list(getattr(self, "overlap_vis_steps", []))),
                sep_tau=self.sep_tau,
                sep_temp=self.sep_temp,
                sep_percentile=self.sep_percentile,
                image=image_for_vis,
            )

        if not return_dict:
            return (image, has_nsfw_concept)

        return (
            StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=has_nsfw_concept
            ),
            None,
        )

    def _step(
        self,
        latents,
        text_embeddings,
        t,
        i,
        step_size,
        cross_attention_kwargs,
        prompt,

    ):
        with torch.enable_grad():

            max_iter_to_alter = 1
            updated_latents = []
            for latent, text_embedding in zip(latents, text_embeddings):
                latent = latent[None, ...]

                for k in range(max_iter_to_alter):
                    latent = latent.clone().detach().requires_grad_(True)
                    self.unet(
                        latent,
                        t,
                        encoder_hidden_states=text_embedding,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )

                    self.unet.zero_grad()
                    attention_maps = self._aggregate_and_get_attention_maps_per_token()
                    loss, loss_dict = self._compute_loss(
                        attention_maps=attention_maps, prompt=prompt, return_dict=True
                    )

                    if (
                        getattr(self, "save_overlap_vis", False)
                        and int(i) in getattr(self, "overlap_vis_steps", set())
                    ):
                        object_indices = self._collect_object_indices_for_separation()

                        if len(object_indices) >= 2:
                            timestep_value = (
                                int(t.detach().cpu().item())
                                if hasattr(t, "detach")
                                else int(t)
                            )

                            snapshot = build_overlap_snapshot(
                                attention_maps=attention_maps,
                                object_indices=object_indices,
                                step=int(i),
                                timestep=timestep_value,
                                tau=self.sep_tau,
                                temp=self.sep_temp,
                                percentile=self.sep_percentile,
                            )

                            if snapshot is not None:
                                self.overlap_vis_records.append(snapshot)

                    raw_loss_value = float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else float(loss)

                    adaptive_loss_value = abs(raw_loss_value)
                    previous_ma = self._loss_moving_avg
                    self._loss_moving_avg = update_moving_average(previous_ma, adaptive_loss_value)

                    do_update = should_update_latent(
                        i,
                        loss_dict=loss_dict,
                        thresholds=self.adaptive_thresholds,
                        active_steps=self._active_steps,
                        max_guidance_step=self.max_guidance_step,
                        always_update_active=True,
                    )

                    actual_step_size = float(step_size)
                    if do_update and self.adaptive_dynamic_step:
                        actual_step_size = dynamic_step_size(
                            base_step_size=step_size,
                            step=i,
                            loss_value=adaptive_loss_value,
                            loss_ma=previous_ma,
                            max_guidance_step=self.max_guidance_step,
                        )

                    if do_update and isinstance(loss, torch.Tensor) and loss.requires_grad and torch.isfinite(loss):
                        latent = self._update_latent(
                            latents=latent, loss=loss, step_size=actual_step_size
                        )

                    self._last_loss_dict = dict(loss_dict)
                    if self.log_guidance:
                        trace_entry = dict(loss_dict)
                        trace_entry.update(
                            {
                                "step": int(i),
                                "timestep": int(t),
                                "attempted": True,
                                "updated": bool(do_update),
                                "step_size": float(actual_step_size),
                            }
                        )
                        self.guidance_trace.append(trace_entry)

                updated_latents.append(latent)

        latents = torch.cat(updated_latents, dim=0)

        return latents

    def _compute_max_attention_per_index(
        self,
        attention_maps: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Computes the maximum attention value for each of the tokens we wish to alter."""
        attention_for_text = torch.stack(attention_maps, dim=-1)[:, :, 1:-1]
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        indices = [
            index[-1] - 1 if isinstance(index[-1], int) else index[-1][0] - 1
            for index in self.subtrees_indices
        ]

        max_indices_list = []
        for i in indices:
            image = attention_for_text[:, :, i]
            smoothing = GaussianSmoothing().to(attention_for_text.device)
            input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
            image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        return max_indices_list

    @staticmethod
    def _compute_excite_loss(
        max_attention_per_index: List[torch.Tensor],
    ) -> torch.Tensor:
        """Computes the attend-and-excite loss using the maximum attention value for each token."""
        losses = [max(0, 1.0 - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        return loss

    def _excitation_loss(self, attention_maps, prompt, attn_map_idx_to_wp):

        max_attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
        )

        excite_loss = self._compute_excite_loss(max_attention_per_index)

        return excite_loss

    def _excitation_loss_ours(self, attention_maps, prompt, attn_map_idx_to_wp):

        max_attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
        )

        excite_loss = self._compute_excite_loss_ours(max_attention_per_index)

        return excite_loss

    @staticmethod
    def _compute_excite_loss_ours(
        max_attention_per_index: List[torch.Tensor],
    ) -> torch.Tensor:
        """Return the negative sum of the token attention maxima."""
        loss = -sum(max_attention_per_index)
        return loss

    def _compute_volumn_attention_per_index(self, attention_maps):
        attention_for_text = torch.stack(attention_maps, dim=-1)[:, :, 1:-1]
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        indices = [
            index[-1] - 1 if isinstance(index[-1], int) else index[-1][0] - 1
            for index in self.subtrees_indices
        ]

        weight_per_token = attention_for_text.sum(dim=[0, 1]) / attention_for_text.sum()

        return weight_per_token[indices]

    def _sum_loss(self, attention_maps, prompt, attn_map_idx_to_wp):

        volumn_attention_per_index = self._compute_volumn_attention_per_index(
            attention_maps=attention_maps,
        )
        sum_loss = self._compute_excite_loss(volumn_attention_per_index)

        return sum_loss

    def _compute_loss(
        self,
        attention_maps: List[torch.Tensor],
        prompt: Union[str, List[str]],
        return_dict: bool = False,
    ) -> torch.Tensor:
        attn_map_idx_to_wp = get_attention_map_index_to_wordpiece(
            self.tokenizer, prompt
        )
        zero = torch.zeros((), device=attention_maps[0].device)
        if self.i > self.max_guidance_step:
            loss_dict = {
                "total": 0.0,
                "ebama": 0.0,
                "attr": 0.0,
                "intensity": 0.0,
                "sep": 0.0,
                "overlap": 0.0,
            }
            return (zero, loss_dict) if return_dict else zero

        loss_s = self._attribution_loss_ours(attention_maps, prompt, attn_map_idx_to_wp)
        loss_t = self._excitation_loss_ours(attention_maps, prompt, attn_map_idx_to_wp)
        loss_ebama = loss_s + self.lambda_ours * loss_t

        object_indices = self._collect_object_indices_for_separation()

        loss_sep = zero
        overlap = 0.0

        if len(object_indices) >= 2:
            overlap = attention_overlap_ratio(
                attention_maps,
                object_indices,
                tau=self.sep_tau,
                temp=self.sep_temp,
                percentile=self.sep_percentile,
            )

        if self.use_separation and self.gamma_sep > 0 and len(object_indices) >= 2:
            loss_sep = object_separation_loss(
                attention_maps,
                object_indices,
                tau=self.sep_tau,
                temp=self.sep_temp,
                percentile=self.sep_percentile,
            )

        loss = loss_ebama + self.gamma_sep * loss_sep

        loss_dict = {
            "total": abs(float(loss.detach().cpu())),
            "raw_total": float(loss.detach().cpu()),
            "ebama": float(loss_ebama.detach().cpu()),
            "attr": float(loss_s.detach().cpu()) if isinstance(loss_s, torch.Tensor) else float(loss_s),
            "intensity": float(loss_t.detach().cpu()) if isinstance(loss_t, torch.Tensor) else float(loss_t),
            "sep": float(loss_sep.detach().cpu()) if isinstance(loss_sep, torch.Tensor) else float(loss_sep),
            "overlap": float(overlap),
            "num_objects": int(len(object_indices)),
        }
        return (loss, loss_dict) if return_dict else loss

    def _collect_object_indices_for_separation(self):
        """Collect one token-index group per object from parsed EBAMA subtrees."""
        object_indices = []
        seen = set()
        for subtree_indices in self.subtrees_indices or []:
            noun, _ = split_indices(subtree_indices)
            key_items = []
            for item in noun:
                if isinstance(item, list):
                    key_items.extend(item)
                elif item is not None:
                    key_items.append(item)
            key = tuple(key_items)
            if key and key not in seen:
                object_indices.append(noun)
                seen.add(key)
        return object_indices

    def _attribution_loss_ours(
        self,
        attention_maps: List[torch.Tensor],
        prompt: Union[str, List[str]],
        attn_map_idx_to_wp,
    ):
        self.subtrees_indices = self._extract_attribution_indices_ours(prompt)
        subtrees_indices = self.subtrees_indices
        loss = 0

        for subtree_indices in subtrees_indices:
            noun, modifier = split_indices(subtree_indices)
            all_subtree_pairs = list(itertools.product(noun, modifier))
            positive_loss, negative_loss = self._calculate_losses(
                attention_maps,
                all_subtree_pairs,
                subtree_indices,
                attn_map_idx_to_wp,
            )
            loss += positive_loss

            loss += negative_loss

        return loss

    def _calculate_losses(
        self,
        attention_maps,
        all_subtree_pairs,
        subtree_indices,
        attn_map_idx_to_wp,
    ):
        positive_loss = []
        negative_loss = []
        for pair in all_subtree_pairs:
            noun, modifier = pair
            if modifier:
                positive_loss.append(
                    calculate_positive_loss(
                        attention_maps, modifier, noun, dist=self.dist
                    )
                )
            negative_loss.append(
                calculate_negative_loss(
                    attention_maps,
                    modifier,
                    noun,
                    subtree_indices,
                    attn_map_idx_to_wp,
                    dist=self.dist,
                    ours=True,
                )
            )

        positive_loss = sum(positive_loss)
        negative_loss = sum(negative_loss)

        return positive_loss, negative_loss

    def _align_indices(self, prompt, spacy_pairs):
        wordpieces2indices = get_indices(self.tokenizer, prompt)
        paired_indices = []
        collected_spacy_indices = (
            set()
        )
        for pair in spacy_pairs:
            curr_collected_wp_indices = (
                []
            )
            for member in pair:
                for idx, wp in wordpieces2indices.items():
                    if wp in [start_token, end_token]:
                        continue
                    wp = wp.replace("</w>", "")
                    if member.text == wp:
                        if idx not in curr_collected_wp_indices and idx not in collected_spacy_indices:
                            curr_collected_wp_indices.append(idx)
                            break
                    elif member.text.startswith(wp) and wp != member.text:
                        wp_indices = align_wordpieces_indices(
                            wordpieces2indices, idx, member.text
                        )
                        if wp_indices and (wp_indices not in curr_collected_wp_indices) and all([wp_idx not in collected_spacy_indices for wp_idx in wp_indices]):
                            curr_collected_wp_indices.append(wp_indices)
                            break
            for collected_idx in curr_collected_wp_indices:
                if isinstance(collected_idx, list):
                    for idx in collected_idx:
                        collected_spacy_indices.add(idx)
                else:
                    collected_spacy_indices.add(collected_idx)
            paired_indices.append(curr_collected_wp_indices)
        return paired_indices

    def _align_indices_ours(self, prompt, spacy_pairs):
        wordpieces2indices = get_indices(self.tokenizer, prompt)
        paired_indices = []
        for pair in spacy_pairs:
            noun = pair[0]
            paired_indices.append(wordpieces2indices[noun])

        return paired_indices

    def _extract_attribution_indices_ours(self, prompt):

        pairs = extract_attribution_indices(self.doc)

        pairs_2 = extract_attribution_indices_with_verb_root(self.doc)
        pairs_3 = extract_attribution_indices_with_verbs(self.doc)
        pairs = unify_lists(pairs, pairs_2, pairs_3)

        paired_indices = self._align_indices(prompt, pairs)

        nouns_already_extracted = []
        attributes_already_extracted = []
        for indices in paired_indices:
            temp = indices[-1]
            temp1 = indices[:-1]
            if isinstance(temp, list):
                nouns_already_extracted += temp
            else:
                nouns_already_extracted.append(temp)
            if isinstance(temp1, list):
                attributes_already_extracted += temp1
            else:
                attributes_already_extracted.append(temp1)
        nouns = extract_noun_indices(self.doc)
        noun_indices = self._align_indices(prompt, [[noun] for noun in nouns])

        for noun in noun_indices:
            if noun:
                if (
                    noun[0] not in nouns_already_extracted
                    and noun[0] not in attributes_already_extracted
                ):
                    paired_indices += [[None, noun[0]]]

        print(f"Final pairs collected: {paired_indices}")

        return paired_indices


def extract_noun_indices(doc):
    noun_indices = []
    for k, token in enumerate(doc):
        if token.pos_ == "NOUN" or token.pos_ == "PROPN":
            noun_indices.append(token)
    return noun_indices


def _get_attention_maps_list(attention_maps: torch.Tensor) -> List[torch.Tensor]:
    attention_maps *= 100
    attention_maps_list = [
        attention_maps[:, :, i] for i in range(attention_maps.shape[2])
    ]

    return attention_maps_list


def is_sublist(sub, main):
    return len(sub) < len(main) and all(item in main for item in sub)


def unify_lists(lists_1, lists_2, lists_3):
    unified_list = lists_1 + lists_2 + lists_3
    sorted_list = sorted(unified_list, key=len)
    seen = set()

    result = []

    for i in range(len(sorted_list)):
        if tuple(sorted_list[i]) in seen:
            continue

        sublist_to_add = True
        for j in range(i + 1, len(sorted_list)):
            if is_sublist(sorted_list[i], sorted_list[j]):
                sublist_to_add = False
                break

        if sublist_to_add:
            result.append(sorted_list[i])
            seen.add(tuple(sorted_list[i]))

    return result


class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        kernel = kernel / torch.sum(kernel)

        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(
            input,
            weight=self.weight.to(input.dtype).to(input.device),
            groups=self.groups,
        )
