import torch
from stable_diffusion_videos import StableDiffusionWalkPipeline

from typing import Callable, List, Optional, Union, Tuple
import inspect
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import numpy as np
from random import randint
from stable_diffusion_videos import get_timesteps_arr, make_video_pyav
import PIL
from transformers import pipeline
import json 
import time

from pathlib import Path

class Image2ImageWalkPipeline(StableDiffusionWalkPipeline):

    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    def image_to_caption(self, image):
        return self.captioner(image, max_new_tokens=70)[0]['generated_text']
    
    def prompt_to_embedding(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = text_inputs.input_ids.to(self.text_encoder.device)
        text_embeddings = self.text_encoder(text_inputs)[0]

        return text_embeddings.detach()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        init_latent,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        noise: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        batch_size = prompt.shape[0]

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        text_embeddings = prompt

        # duplicate text embeddings for each generation per prompt
        text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError("The length of `negative_prompt` should be equal to batch_size.")
            else:
                uncond_tokens = negative_prompt

            max_length = self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt
            uncond_embeddings = uncond_embeddings.repeat_interleave(batch_size * num_images_per_prompt, dim=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # encode the init image into latents and scale the latents
        latents_dtype = text_embeddings.dtype
        init_latents = 0.18215 * init_latent

        if isinstance(prompt, str):
            prompt = [prompt]
        if len(prompt) > init_latents.shape[0] and len(prompt) % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            pass
        elif len(prompt) > init_latents.shape[0] and len(prompt) % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `init_image` of batch size {init_latents.shape[0]} to {len(prompt)} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)

        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size * num_images_per_prompt, device=self.device)

        # add noise to latents using the timesteps
        if noise is None:
            noise = torch.randn(init_latents.shape, generator=generator, device=self.device, dtype=latents_dtype)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)
        
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents

        t_start = max(num_inference_steps - init_timestep + offset, 0)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                self.device
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


    def slerp(self, t, v0, v1, DOT_THRESHOLD=0.9995):
        """helper function to spherically interpolate two arrays v1 v2"""

        inputs_are_torch = isinstance(v0, torch.Tensor)
        if inputs_are_torch:
            input_device = v0.device
            v0 = v0.cpu().numpy()
            v1 = v1.cpu().numpy()

        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        if inputs_are_torch:
            v2 = torch.from_numpy(v2).to(input_device)

        return v2

    def init_noise(self, seed, noise_shape, dtype):
        """Helper to initialize noise"""
        # randn does not exist on mps, so we create noise on CPU here and move it to the device after initialization
        if self.device.type == "mps":
            noise = torch.randn(
                noise_shape,
                device="cpu",
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).to(self.device)
        else:
            noise = torch.randn(
                noise_shape,
                device=self.device,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                dtype=dtype,
            )
        return noise

    def random_seed(self, n=8):
        range_start = 10**(n-1)
        range_end = (10**n)-1
        return randint(range_start, range_end)

    def pil_preprocess(self, image, dtype):
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return (2.0 * image - 1.0).detach().to(dtype).to(self.device)

    
    def generate_inputs(self, image_a, image_b, prompt_a, prompt_b, seed_a, seed_b, T, batch_size):
        embeds_a = self.prompt_to_embedding(prompt_a)
        embeds_b = self.prompt_to_embedding(prompt_b)
        latents_dtype = embeds_a.dtype

        latents_a = self.vae.encode(self.pil_preprocess(image_a, latents_dtype)).latent_dist.sample().detach().to(self.device)
        latents_b = self.vae.encode(self.pil_preprocess(image_b, latents_dtype)).latent_dist.sample().detach().to(self.device)
        noise_shape = latents_a.shape

        noise_a = self.init_noise(seed_a, noise_shape, latents_dtype)
        noise_b = self.init_noise(seed_b, noise_shape, latents_dtype)

        batch_idx = 0
        embeds_batch, noise_batch, latents_batch = None, None, None

        for i, t in enumerate(T):
            embeds = torch.lerp(embeds_a, embeds_b, t)
            # embeds = self.slerp(float(t), embeds_a, embeds_b)
            noise = self.slerp(float(t), noise_a, noise_b)
            latents = self.slerp(float(t), latents_a, latents_b)

            embeds_batch = embeds if embeds_batch is None else torch.cat([embeds_batch, embeds])
            noise_batch = noise if noise_batch is None else torch.cat([noise_batch, noise])
            latents_batch = latents if latents_batch is None else torch.cat([latents_batch, latents])
            batch_is_ready = embeds_batch.shape[0] == batch_size or i + 1 == T.shape[0]
            if not batch_is_ready:
                continue
            yield batch_idx, embeds_batch, noise_batch, latents_batch
            batch_idx += 1
            del embeds_batch, noise_batch, latents_batch
            torch.cuda.empty_cache()
            embeds_batch, noise_batch, latents_batch = None, None, None

    def make_clip_frames(
        self,
        image_a,
        image_b,
        prompt_a,
        prompt_b,
        seed_a,
        seed_b,
        num_interpolation_steps: int = 5,
        save_path: Union[str, Path] = "outputs/",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        upsample: bool = False,
        batch_size: int = 1,
        image_file_ext: str = ".png",
        T: np.ndarray = None,
        skip: int = 0,
        negative_prompt: str = None,
        step: Optional[Tuple[int, int]] = None,
    ):

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        T = T if T is not None else np.linspace(0.0, 1.0, num_interpolation_steps)
        if T.shape[0] != num_interpolation_steps:
            raise ValueError(f"Unexpected T shape, got {T.shape}, expected dim 0 to be {num_interpolation_steps}")

        batch_generator = self.generate_inputs(
            image_a,
            image_b,
            prompt_a,
            prompt_b,
            seed_a,
            seed_b,
            # (1, self.unet.in_channels, height // 8, width // 8),
            T[skip:],
            batch_size,
        )

        frame_index = skip
        for batch_idx, embeds_batch, noise_batch, latents_batch in batch_generator:
            outputs = self(
                prompt=embeds_batch,
                init_latent=latents_batch,
                strength=0.75,
                guidance_scale=guidance_scale,
                noise=noise_batch,
                num_inference_steps = num_inference_steps
            )['images']

            for image_idx, image in enumerate(outputs):
                if frame_index == skip and image_idx == 0:
                    frame_filepath = save_path / (f"frame%06d{image_file_ext}" % frame_index)
                    image_a.save(frame_filepath)
                    frame_index += 1
                elif (batch_idx + 1) * batch_size == len(T) and image_idx + 1 == len(outputs):
                    frame_filepath = save_path / (f"frame%06d{image_file_ext}" % (frame_index))
                    image_b.save(frame_filepath)
                    frame_index += 1
                else:
                    frame_filepath = save_path / (f"frame%06d{image_file_ext}" % frame_index)
                    image = image if not upsample else self.upsampler(image)
                    image.save(frame_filepath)
                    frame_index += 1
    
    def walk(
        self,
        images,
        prompts = None,
        num_interpolation_steps: Optional[Union[int, List[int]]] = 5,  # int or list of int
        output_dir: Optional[str] = "./dreams",
        name: Optional[str] = None,
        image_file_ext: Optional[str] = ".png",
        fps: Optional[int] = 30,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        upsample: Optional[bool] = False,
        batch_size: Optional[int] = 1,
        resume: Optional[bool] = False,
        audio_filepath: str = None,
        audio_start_sec: Optional[Union[int, float]] = None,
        margin: Optional[float] = 1.0,
        smooth: Optional[float] = 0.0,
        negative_prompt: Optional[str] = None,
        make_video: Optional[bool] = True,
    ):
        """Generate a video from a sequence of prompts and seeds. Optionally, add audio to the
        video to interpolate to the intensity of the audio.

        Args:
            prompts (Optional[List[str]], optional):
                list of text prompts. Defaults to None.
            seeds (Optional[List[int]], optional):
                list of random seeds corresponding to prompts. Defaults to None.
            num_interpolation_steps (Union[int, List[int]], *optional*):
                How many interpolation steps between each prompt. Defaults to None.
            output_dir (Optional[str], optional):
                Where to save the video. Defaults to './dreams'.
            name (Optional[str], optional):
                Name of the subdirectory of output_dir. Defaults to None.
            image_file_ext (Optional[str], *optional*, defaults to '.png'):
                The extension to use when writing video frames.
            fps (Optional[int], *optional*, defaults to 30):
                The frames per second in the resulting output videos.
            num_inference_steps (Optional[int], *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (Optional[float], *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (Optional[float], *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            height (Optional[int], *optional*, defaults to None):
                height of the images to generate.
            width (Optional[int], *optional*, defaults to None):
                width of the images to generate.
            upsample (Optional[bool], *optional*, defaults to False):
                When True, upsamples images with realesrgan.
            batch_size (Optional[int], *optional*, defaults to 1):
                Number of images to generate at once.
            resume (Optional[bool], *optional*, defaults to False):
                When True, resumes from the last frame in the output directory based
                on available prompt config. Requires you to provide the `name` argument.
            audio_filepath (str, *optional*, defaults to None):
                Optional path to an audio file to influence the interpolation rate.
            audio_start_sec (Optional[Union[int, float]], *optional*, defaults to 0):
                Global start time of the provided audio_filepath.
            margin (Optional[float], *optional*, defaults to 1.0):
                Margin from librosa hpss to use for audio interpolation.
            smooth (Optional[float], *optional*, defaults to 0.0):
                Smoothness of the audio interpolation. 1.0 means linear interpolation.
            negative_prompt (Optional[str], *optional*, defaults to None):
                Optional negative prompt to use. Same across all prompts.
            make_video (Optional[bool], *optional*, defaults to True):
                When True, makes a video from the generated frames. If False, only
                generates the frames.

        This function will create sub directories for each prompt and seed pair.

        For example, if you provide the following prompts and seeds:

        ```
        prompts = ['a dog', 'a cat', 'a bird']
        seeds = [1, 2, 3]
        num_interpolation_steps = 5
        output_dir = 'output_dir'
        name = 'name'
        fps = 5
        ```

        Then the following directories will be created:

        ```
        output_dir
        ├── name
        │   ├── name_000000
        │   │   ├── frame000000.png
        │   │   ├── ...
        │   │   ├── frame000004.png
        │   │   ├── name_000000.mp4
        │   ├── name_000001
        │   │   ├── frame000000.png
        │   │   ├── ...
        │   │   ├── frame000004.png
        │   │   ├── name_000001.mp4
        │   ├── ...
        │   ├── name.mp4
        |   |── prompt_config.json
        ```

        Returns:
            str: The resulting video filepath. This video includes all sub directories' video clips.
        """
        # 0. Default height and width to unet
        first_image_height = images[0].size[1]
        first_image_width = images[0].size[0]
        # scale to a multiple of 32
        height = first_image_height - first_image_height % 32 or self.unet.config.sample_size * self.vae_scale_factor
        width = first_image_width - first_image_width % 32 or self.unet.config.sample_size * self.vae_scale_factor

        output_path = Path(output_dir)

        name = name or time.strftime("%Y%m%d-%H%M%S")
        save_path_root = output_path / name
        save_path_root.mkdir(parents=True, exist_ok=True)

        # Where the final video of all the clips combined will be saved
        output_filepath = save_path_root / f"{name}.mp4"

        # If using same number of interpolation steps between, we turn into list
        if not resume and isinstance(num_interpolation_steps, int):
            num_interpolation_steps = [num_interpolation_steps] * (len(images) - 1)

        if not resume:
            audio_start_sec = audio_start_sec or 0

        # Save/reload prompt config
        prompt_config_path = save_path_root / "prompt_config.json"
        if not resume:
            prompt_config_path.write_text(
                json.dumps(
                    dict(
                        num_interpolation_steps=num_interpolation_steps,
                        fps=fps,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        eta=eta,
                        upsample=upsample,
                        height=height,
                        width=width,
                        audio_filepath=audio_filepath,
                        audio_start_sec=audio_start_sec,
                        negative_prompt=negative_prompt,
                    ),
                    indent=2,
                    sort_keys=False,
                )
            )
        else:
            data = json.load(open(prompt_config_path))
            num_interpolation_steps = data["num_interpolation_steps"]
            fps = data["fps"]
            num_inference_steps = data["num_inference_steps"]
            guidance_scale = data["guidance_scale"]
            eta = data["eta"]
            upsample = data["upsample"]
            height = data["height"]
            width = data["width"]
            audio_filepath = data["audio_filepath"]
            audio_start_sec = data["audio_start_sec"]
            negative_prompt = data.get("negative_prompt", None)
        
        prompt_a = None
        prompt_b = None
        seed_a = None
        seed_b = None

        for i, (image_a, image_b, num_step) in enumerate(
            zip(images, images[1:], num_interpolation_steps)
        ):

            # {name}_000000 / {name}_000001 / ...
            save_path = save_path_root / f"{name}_{i:06d}"

            # Where the individual clips will be saved
            step_output_filepath = save_path / f"{name}_{i:06d}.mp4"

            # Determine if we need to resume from a previous run
            skip = 0
            if resume:
                if step_output_filepath.exists():
                    print(f"Skipping {save_path} because frames already exist")
                    continue

                existing_frames = sorted(save_path.glob(f"*{image_file_ext}"))
                if existing_frames:
                    skip = int(existing_frames[-1].stem[-6:]) + 1
                    if skip + 1 >= num_step:
                        print(f"Skipping {save_path} because frames already exist")
                        continue
                    print(f"Resuming {save_path.name} from frame {skip}")

            audio_offset = audio_start_sec + sum(num_interpolation_steps[:i]) / fps
            audio_duration = num_step / fps

            # resize images
            image_a_re = image_a.resize((width, height))
            image_b_re = image_b.resize((width, height))

            # get prompts
            if prompts is not None:
                start_index = i
                if start_index < len(prompts):
                    prompt_a = prompts[start_index]
                if start_index + 1 < len(prompts):
                    prompt_b = prompts[start_index + 1]
                
            # generate if not found
            if prompt_a is None or len(prompt_a) == 0:
                prompt_a = self.image_to_caption(image_a_re)
            if prompt_b is None or len(prompt_b) == 0:
                prompt_b = self.image_to_caption(image_b_re)
            
            # generate seeds
            seed_a = self.random_seed()
            seed_b = self.random_seed()

            self.make_clip_frames(
                image_a_re,
                image_b_re,
                prompt_a,
                prompt_b,
                seed_a,
                seed_b,
                num_interpolation_steps=num_step,
                save_path=save_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                height=height,
                width=width,
                upsample=upsample,
                batch_size=batch_size,
                T=get_timesteps_arr(
                    audio_filepath,
                    offset=audio_offset,
                    duration=audio_duration,
                    fps=fps,
                    margin=margin,
                    smooth=smooth,
                )
                if audio_filepath
                else None,
                skip=skip,
                negative_prompt=negative_prompt,
                step=(i, len(images) - 1),
            )

            # update prompts
            prompt_a = prompt_b
            prompt_b = None

            # update seeds
            seed_a = seed_b
            seed_b = None
    
            if make_video:
                make_video_pyav(
                    save_path,
                    audio_filepath=audio_filepath,
                    fps=fps,
                    output_filepath=step_output_filepath,
                    glob_pattern=f"*{image_file_ext}",
                    audio_offset=audio_offset,
                    audio_duration=audio_duration,
                    sr=44100,
                )
        if make_video:
            return make_video_pyav(
                save_path_root,
                audio_filepath=audio_filepath,
                fps=fps,
                audio_offset=audio_start_sec,
                audio_duration=sum(num_interpolation_steps) / fps,
                output_filepath=output_filepath,
                glob_pattern=f"**/*{image_file_ext}",
                sr=44100,
            )