import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from transformers import AutoProcessor, CLIPVisionModelWithProjection


# Research Note: These UNet models (UNetGarm2DConditionModel, UNetVton2DConditionModel)
# various Nepali traditional garments. This involves architectural tweaks to make the drape more in fusion with the western ones
# like specialized attention mechanisms for intricate patterns (e.g., Dhaka fabric)
# or different residual blocks.
from models.unet_Nepgarm_condition import UNetGarm2DConditionModel # Renamed for clarity
from models.unet_Nepvton_condition import UNetVton2DConditionModel # Renamed for clarity

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def preprocess_image_for_vae(image): # Renamed for specificity
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    deprecate("preprocess_image_for_vae", "1.0.0", deprecation_message, standard_warn=False) # Updated deprecation message
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        # Research Note: The image resizing to a multiple of 8 is standard for VAEs.
        # For Nepali wear, ensuring high-resolution input might be crucial for
        # preserving intricate patterns like those on a Sari or Daura Suruwal.
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class NepaliWearVtonPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin): # Renamed pipeline class
    r"""
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet_garm ([`UNetGarm2DConditionModelNepali`]): # Updated UNet name
            A `UNetGarm2DConditionModelNepali` to denoise the encoded garment latents.
        unet_vton ([`UNetVton2DConditionModelNepali`]): # Updated UNet name
            A `UNetVton2DConditionModelNepali` to denoise the encoded image latents for try-on.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """
    model_cpu_offload_seq = "text_encoder->unet_garm->unet_vton->vae" # Adjusted offload sequence
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "vton_latents"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet_garm: UNetGarm2DConditionModel, # Updated UNet type
        unet_vton: UNetVton2DConditionModel, # Updated UNet type
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet_garm_model=unet_garm, # Renamed attribute for clarity
            unet_vton_model=unet_vton, # Renamed attribute for clarity
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        garment_image: PipelineImageInput = None, # Renamed input
        person_image: PipelineImageInput = None, # Renamed input
        mask_image: PipelineImageInput = None, # Renamed input
        original_person_image: PipelineImageInput = None, # Renamed input
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generating virtual try-on images of Nepali wear. # Updated docstring

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
                # Research Note: Specific prompts tailored for Nepali wear (e.g., "traditional Nepali sari,"
                # "Daura Suruwal with Dhaka topi") could significantly improve results.
            garment_image (`torch.FloatTensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`): # Updated arg name
                The garment image to be virtually tried on.
            person_image (`torch.FloatTensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`): # Updated arg name
                The person image onto whom the garment will be tried.
            mask_image (`torch.FloatTensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`): # Updated arg name
                The mask indicating the region of the person's body where the garment should be placed.
                # Research Note: The quality of the mask is paramount for realistic try-ons. For traditional
                # Nepali attire, masks might need to be more precise to handle intricate draping or layers.
            original_person_image (`torch.FloatTensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`): # Updated arg name
                The original person image, used for repainting.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            image_guidance_scale (`float`, *optional!, defaults to 1.5):
                Push the generated image towards the initial `person_image`. Image guidance scale is enabled by setting
                `image_guidance_scale > 1`. Higher image guidance scale encourages generated images that are closely
                linked to the source `person_image`, usually at the expense of lower image quality. This pipeline requires a
                value of at least `1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Check inputs
        self.check_pipeline_inputs( # Renamed method
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        if (person_image is None) or (garment_image is None): # Updated variable names
            raise ValueError("`person_image` and `garment_image` inputs cannot be undefined.") # Updated error message

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # check if scheduler is in sigmas space
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

        # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 3. Preprocess images and mask
        # Research Note: For Nepali wear, ensuring consistent preprocessing across diverse
        # garment types (sari, daura suruwal, lehenga, etc.) is important.
        processed_garment_image = self.image_processor.preprocess(garment_image) # Renamed variable
        processed_person_image = self.image_processor.preprocess(person_image) # Renamed variable
        processed_original_person_image = self.image_processor.preprocess(original_person_image) # Renamed variable

        processed_mask = np.array(mask_image) # Renamed variable
        processed_mask[processed_mask < 127] = 0
        processed_mask[processed_mask >= 127] = 255
        processed_mask = torch.tensor(processed_mask)
        processed_mask = processed_mask / 255
        processed_mask = processed_mask.reshape(-1, 1, processed_mask.size(-2), processed_mask.size(-1))

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare Garment and VTON latents
        garment_latents = self.prepare_garment_latents( 
            processed_garment_image, 
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
            generator,
        )

        vton_latents, mask_latents, original_person_latents = self.prepare_tryon_latents( # Renamed method and variable
            processed_person_image, 
            processed_mask, 
            processed_original_person_image, 
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
            generator,
        )

        height, width = vton_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        # 6. Prepare latent variables for diffusion
        num_channels_latents = self.vae.config.latent_channels
        diffusion_latents = self.prepare_diffusion_latents( # Renamed method and variable
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        noise = diffusion_latents.clone() # Renamed variable

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # Compute garment features using unet_garm_model
        # Research Note: The quality of garment feature extraction is crucial.
        # For Nepali wear, this needs to accurately capture intricate embroidery,
        # fabric textures (e.g., Dhaka), and unique garment silhouettes.
        _, garment_spatial_attention_outputs = self.unet_garm_model( # Updated attribute name
            garment_latents,
            0, # Timestep 0 for garment feature extraction
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([diffusion_latents] * 2) if self.do_classifier_free_guidance else diffusion_latents

                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # Research Note: The concatenation strategy here (channel dimension) is standard.
                # Explore if other fusion methods (e.g., adaptive instance normalization,
                # more complex cross-attention in UNetVton2DConditionModelNepali) could better
                # blend Nepali garment features.
                latent_vton_model_input = torch.cat([scaled_latent_model_input, vton_latents], dim=1)

                # Pass garment features to the VTON UNet
                vton_spatial_attention_inputs = garment_spatial_attention_outputs.copy() # Renamed variable

                # predict the noise residual
                noise_pred = self.unet_vton_model( # Updated attribute name
                    latent_vton_model_input,
                    vton_spatial_attention_inputs, # Updated variable
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. So we need to compute the
                # predicted_original_sample here if we are using a karras style scheduler.
                if scheduler_is_in_sigma_space:
                    step_index = (self.scheduler.timesteps == t).nonzero()[0].item()
                    sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_text_image, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = (
                        noise_pred_text
                        + self.image_guidance_scale * (noise_pred_text_image - noise_pred_text)
                    )

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. But the scheduler.step function
                # expects the noise_pred and computes the predicted_original_sample internally. So we
                # need to overwrite the noise_pred here such that the value of the computed
                # predicted_original_sample is correct.
                if scheduler_is_in_sigma_space:
                    noise_pred = (noise_pred - diffusion_latents) / (-sigma) # Updated variable

                # compute the previous noisy sample x_t -> x_t-1
                diffusion_latents = self.scheduler.step(noise_pred, t, diffusion_latents, **extra_step_kwargs, return_dict=False)[0] # Updated variable

                init_latents_proper = original_person_latents * self.vae.config.scaling_factor # Updated variable

                # repainting
                # Research Note: The repainting step is critical for seamless integration,
                # especially for areas where the original body parts should show through
                # or where the garment needs to interact with the body's contours.
                # For Nepali wear, this might require careful handling of exposed skin
                # areas (e.g., midriff for a sari, neck for a cholo).
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([noise_timestep])
                    )

                diffusion_latents = (1 - mask_latents) * init_latents_proper + mask_latents * diffusion_latents # Updated variable

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        # Safely retrieve local variables
                        if k in locals():
                            callback_kwargs[k] = locals()[k]
                        else:
                            logger.warning(f"Callback tensor input '{k}' not found in local variables.")

                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    diffusion_latents = callback_outputs.pop("latents", diffusion_latents) # Updated variable
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    vton_latents = callback_outputs.pop("vton_latents", vton_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, diffusion_latents) # Updated variable

        if not output_type == "latent":
            final_image = self.vae.decode(diffusion_latents / self.vae.config.scaling_factor, return_dict=False)[0] # Renamed variable
            final_image, has_nsfw_concept = self.run_safety_checker(final_image, device, prompt_embeds.dtype) # Renamed variable
        else:
            final_image = diffusion_latents # Renamed variable
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * final_image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        final_image = self.image_processor.postprocess(final_image, output_type=output_type, do_denormalize=do_denormalize) # Renamed variable

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (final_image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=final_image, nsfw_content_detected=has_nsfw_concept)


    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def check_pipeline_inputs( # Renamed method
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_diffusion_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None): # Renamed method
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_garment_latents( # Renamed method
        self, image_garm, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        # Research Note: For diverse Nepali garments, robust encoding of varied
        # textures (e.g., silk, cotton, wool, Dhaka), patterns, and fabric stiffness
        # is critical for realistic try-on results.
        if not isinstance(image_garm, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image_garm` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image_garm)}"
            )

        image_garm = image_garm.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image_garm.shape[1] == 4:
            garment_latents = image_garm # Renamed variable
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                garment_latents = [self.vae.encode(image_garm[i : i + 1]).latent_dist.mode() for i in range(batch_size)] # Renamed variable
                garment_latents = torch.cat(garment_latents, dim=0)
            else:
                garment_latents = self.vae.encode(image_garm).latent_dist.mode() # Renamed variable

        if batch_size > garment_latents.shape[0] and batch_size % garment_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // garment_latents.shape[0]
            garment_latents = torch.cat([garment_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > garment_latents.shape[0] and batch_size % garment_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image_garm` of batch size {garment_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            garment_latents = torch.cat([garment_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_garment_latents = torch.zeros_like(garment_latents) # Renamed variable
            garment_latents = torch.cat([garment_latents, uncond_garment_latents], dim=0) # Renamed variable

        return garment_latents

    def prepare_tryon_latents( # Renamed method
        self, person_image, mask, original_person_image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        # Research Note: For accurate try-on of Nepali garments, the pose and body
        # shape encoding from the person image is crucial for realistic drape and fit.
        # This might require specialized pose estimation or shape representations
        # if cultural poses are common in the dataset.
        if not isinstance(person_image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`person_image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(person_image)}"
            )

        person_image = person_image.to(device=device, dtype=dtype) # Renamed variable
        original_person_image = original_person_image.to(device=device, dtype=dtype) # Renamed variable

        batch_size = batch_size * num_images_per_prompt

        if person_image.shape[1] == 4:
            person_latents = person_image # Renamed variable
            original_person_latents = original_person_image # Renamed variable
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                person_latents = [self.vae.encode(person_image[i : i + 1]).latent_dist.mode() for i in range(batch_size)] # Renamed variable
                person_latents = torch.cat(person_latents, dim=0)
                original_person_latents = [self.vae.encode(original_person_image[i : i + 1]).latent_dist.mode() for i in range(batch_size)] # Renamed variable
                original_person_latents = torch.cat(original_person_latents, dim=0)
            else:
                person_latents = self.vae.encode(person_image).latent_dist.mode() # Renamed variable
                original_person_latents = self.vae.encode(original_person_image).latent_dist.mode() # Renamed variable

        mask = torch.nn.functional.interpolate(
            mask, size=(person_latents.size(-2), person_latents.size(-1)) # Using person_latents size
        )
        mask_latents = mask.to(device=device, dtype=dtype) # Renamed variable

        if batch_size > person_latents.shape[0] and batch_size % person_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // person_latents.shape[0]
            person_latents = torch.cat([person_latents] * additional_image_per_prompt, dim=0)
            mask_latents = torch.cat([mask_latents] * additional_image_per_prompt, dim=0)
            original_person_latents = torch.cat([original_person_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > person_latents.shape[0] and batch_size % person_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `person_image` of batch size {person_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            person_latents = torch.cat([person_latents], dim=0)
            mask_latents = torch.cat([mask_latents], dim=0)
            original_person_latents = torch.cat([original_person_latents], dim=0)

        if do_classifier_free_guidance:
            # uncond_image_latents = torch.zeros_like(person_latents) # No need for explicit uncond_image_latents if cat operation handles it
            person_latents = torch.cat([person_latents] * 2, dim=0)
            # Research Note: The mask and original person latents are not concatenated with uncond,
            # implying they are consistent across conditional/unconditional paths. This might need
            # careful consideration if the mask itself varies for conditional generation.

        return person_latents, mask_latents, original_person_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_freeu
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet_vton_model"): # Changed from self.unet
            raise ValueError("The pipeline must have `unet_vton_model` for using FreeU.") # Updated error message
        self.unet_vton_model.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2) # Changed from self.unet

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet_vton_model.disable_freeu() # Changed from self.unet

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def do_classifier_free_guidance(self):
        return self.image_guidance_scale >= 1.0