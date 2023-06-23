import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"

import torch
torch.use_deterministic_algorithms(True)
import torchaudio

from audiocraft.models.musicgen import MusicGen
from audiocraft.utils.utils import dict_from_config
from audiocraft.data.audio_utils import normalize_audio, i16_pcm
from audiocraft.models.loaders import _get_state_dict, load_compression_model, HF_MODEL_CHECKPOINTS_MAP
from audiocraft.models.builders import (
    get_condition_fuser, 
    get_codebooks_pattern_provider, 
    get_debug_compression_model, 
    get_debug_lm_model
)
from audiocraft.models.lm import LMModel
from audiocraft.modules.conditioners import (
    ConditioningProvider,
    T5Conditioner,
    LUTConditioner,
    ChromaStemConditioner,
    BaseConditioner
)
import omegaconf
from omegaconf import OmegaConf
import typing as tp
import random
from pathlib import Path
from huggingface_hub import hf_hub_download

import io



class CustomT5Conditioner(T5Conditioner):
    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        # if current sample doesn't have a certain attribute, replace with empty string
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        if self.normalize_text:
            _, _, entries = self.text_normalizer(entries, return_text=True)
        if self.word_dropout > 0. and self.training:
            new_entries = []
            for entry in entries:
                words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(entries, return_tensors="pt", padding=True).to(self.device)
        mask = inputs["attention_mask"]

        # replacement for below
        for i in empty_idx:
            row = i.item()
            mask[row, :] = 0

        # incompatible code
        # mask[empty_idx, :] = 0  # zero-out index where the input is non-existant

        return inputs
    
def get_conditioner_provider(output_dim: int, cfg: omegaconf.DictConfig) -> ConditioningProvider:
    """Instantiate a conditioning model.
    """
    device = cfg.device
    duration = cfg.dataset.segment_duration
    cfg = getattr(cfg, "conditioners")
    cfg = omegaconf.OmegaConf.create({}) if cfg is None else cfg
    conditioners: tp.Dict[str, BaseConditioner] = {}
    with omegaconf.open_dict(cfg):
        condition_provider_args = cfg.pop('args', {})
    for cond, cond_cfg in cfg.items():
        model_type = cond_cfg["model"]
        model_args = cond_cfg[model_type]
        if model_type == "t5":
            conditioners[str(cond)] = CustomT5Conditioner(output_dim=output_dim, device=device, **model_args)
        elif model_type == "lut":
            conditioners[str(cond)] = LUTConditioner(output_dim=output_dim, **model_args)
        elif model_type == "chroma_stem":
            model_args.pop('cache_path', None)
            conditioners[str(cond)] = ChromaStemConditioner(
                output_dim=output_dim,
                duration=duration,
                device=device,
                **model_args
            )
        else:
            raise ValueError(f"unrecognized conditioning model: {model_type}")
    conditioner = ConditioningProvider(conditioners, device=device, **condition_provider_args)
    return conditioner

def get_lm_model(cfg: omegaconf.DictConfig) -> LMModel:
    """Instantiate a transformer LM.
    """
    if cfg.lm_model == 'transformer_lm':
        kwargs = dict_from_config(getattr(cfg, 'transformer_lm'))
        n_q = kwargs['n_q']
        q_modeling = kwargs.pop('q_modeling', None)
        codebooks_pattern_cfg = getattr(cfg, 'codebooks_pattern')
        attribute_dropout = dict_from_config(getattr(cfg, 'attribute_dropout'))
        cls_free_guidance = dict_from_config(getattr(cfg, 'classifier_free_guidance'))
        cfg_prob, cfg_coef = cls_free_guidance["training_dropout"], cls_free_guidance["inference_coef"]
        fuser = get_condition_fuser(cfg)
        condition_provider = get_conditioner_provider(kwargs["dim"], cfg).to(cfg.device)
        if len(fuser.fuse2cond['cross']) > 0:  # enforce cross-att programatically
            kwargs['cross_attention'] = True
        if codebooks_pattern_cfg.modeling is None:
            assert q_modeling is not None, \
                'LM model should either have a codebook pattern defined or transformer_lm.q_modeling'
            codebooks_pattern_cfg = omegaconf.OmegaConf.create(
                {'modeling': q_modeling, 'delay': {'delays': list(range(n_q))}}
            )
        pattern_provider = get_codebooks_pattern_provider(n_q, codebooks_pattern_cfg)
        return LMModel(
            pattern_provider=pattern_provider,
            condition_provider=condition_provider,
            fuser=fuser,
            cfg_dropout=cfg_prob,
            cfg_coef=cfg_coef,
            attribute_dropout=attribute_dropout,
            dtype=getattr(torch, cfg.dtype),
            device=cfg.device,
            **kwargs
        ).to(cfg.device)
    else:
        raise KeyError(f'Unexpected LM model {cfg.lm_model}')

def load_lm_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    model = get_lm_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model

class CustomMusicGen(MusicGen):
    @staticmethod
    def get_pretrained(name: str = 'melody', device=None):
        """Return pretrained model, we provide four models:
        - small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-medium
        - melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-melody
        - large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-large
        """

        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm)

        if name not in HF_MODEL_CHECKPOINTS_MAP:
            if not os.path.isfile(name) and not os.path.isdir(name):
                raise ValueError(
                    f"{name} is not a valid checkpoint name. "
                    f"Choose one of {', '.join(HF_MODEL_CHECKPOINTS_MAP.keys())}"
                )

        cache_dir = os.environ.get('MUSICGEN_ROOT', None)
        compression_model = load_compression_model(name, device=device, cache_dir=cache_dir)
        lm = load_lm_model(name, device=device, cache_dir=cache_dir)
        if name == 'melody':
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)


def tensor_to_audio_bytes(
    buffer,
    wav: torch.Tensor, sample_rate: int,
    stem_name: tp.Union[str, Path] = '',
    format: str = 'wav', mp3_rate: int = 320, normalize: bool = True,
    strategy: str = 'peak', peak_clip_headroom_db: float = 1,
    rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
    log_clipping: bool = True,   
):
    assert wav.dtype.is_floating_point, "wav is not floating point"
    if wav.dim() == 1:
        wav = wav[None]
    elif wav.dim() > 2:
        raise ValueError("Input wav should be at most 2 dimension.")
    assert wav.isfinite().all()
    wav = normalize_audio(wav, normalize, strategy, peak_clip_headroom_db,
                          rms_headroom_db, loudness_headroom_db, log_clipping=log_clipping,
                          sample_rate=sample_rate, stem_name=str(stem_name))
    kwargs: dict = {}
    if format == 'mp3':
        kwargs.update({"compression": mp3_rate})
    elif format == 'wav':
        wav = i16_pcm(wav)
        kwargs.update({"encoding": "PCM_S", "bits_per_sample": 16})
    else:
        raise RuntimeError(f"Invalid format {format}. Only wav or mp3 are supported.")

    torchaudio.save(buffer, wav, sample_rate, **kwargs)