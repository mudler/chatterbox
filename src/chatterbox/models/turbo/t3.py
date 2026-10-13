# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import Union, Optional, List

logger = logging.getLogger(__name__)

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig, GPT2Config, GPT2Model
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
)
from ..t3.modules.learned_pos_emb import LearnedPositionEmbeddings

from ..t3.modules.cond_enc import T3CondEnc, T3Cond
from .t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from ..utils import AttrDict


logger = logging.getLogger(__name__)


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones,
        * tokenization, including start / stop tokens are always added externally to this class
        * conditioning data like CLAP, emotion, etc are all in a separate file for more modularity
        * careful! this class assumes relative positional encoding -- with absolute PE, we would at
            least want to reset the position to 0 when speech tokens begin, and optionally use a
            different PE embedding space for speech.
    """

    def __init__(self, hp=None):
        if hp is None:
            hp = T3Config.english_only()
        super().__init__()
        self.hp = hp

        config_dict = LLAMA_CONFIGS[hp.llama_config_name]
        self.is_gpt = config_dict.get("model_type") == "gpt2"

        if self.is_gpt:
            self.cfg = GPT2Config(**config_dict)
            self.tfmr = GPT2Model(self.cfg)
        else:
            self.cfg = LlamaConfig(**config_dict)
            self.tfmr = LlamaModel(self.cfg)

        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        self.text_pos_emb = None
        self.speech_pos_emb = None
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=self.is_gpt)
        self.compiled = False

    @property
    def device(self):
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens)
            if not self.is_gpt:
                t3_cond.cond_prompt_speech_emb += self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        # prepare input embeddings (skip backbone tranformer embeddings)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        if cfg_weight > 0.0 and not self.is_gpt:
            text_emb[1].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
             cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        embeds = torch.stack([
            torch.cat((ce, te, se))
            for ce, te, se in zip(cond_emb, text_emb, speech_emb)
        ])  # (B, length, dim)
        return embeds, len_cond

    @torch.inference_mode()
    def inference_turbo(self, t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2,
                        max_gen_len=1000):

        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))


        speech_start_token = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

        generated_speech_tokens = []

        llm_outputs = self.tfmr(
            inputs_embeds=embeds,
            use_cache=True
        )

        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values

        speech_hidden = hidden_states[:, -1:]
        speech_logits = self.speech_head(speech_hidden)

        processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)

        generated_speech_tokens.append(next_speech_token)
        current_speech_token = next_speech_token

        for _ in tqdm(range(max_gen_len)):
            current_speech_embed = self.speech_emb(current_speech_token)

            llm_outputs = self.tfmr(
                inputs_embeds=current_speech_embed,
                past_key_values=past_key_values,
                use_cache=True
            )

            hidden_states = llm_outputs[0]
            past_key_values = llm_outputs.past_key_values
            speech_logits = self.speech_head(hidden_states)

            input_ids = torch.cat(generated_speech_tokens, dim=1)
            processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])
            if torch.all(processed_logits == -float("inf")):
                print("Warning: All logits are -inf")
                break

            probs = F.softmax(processed_logits, dim=-1)
            next_speech_token = torch.multinomial(probs, num_samples=1)

            generated_speech_tokens.append(next_speech_token)
            current_speech_token = next_speech_token
            if torch.all(next_speech_token == self.hp.stop_speech_token):
                break

        all_tokens = torch.cat(generated_speech_tokens, dim=1)

        # Remove EOS token if present
        if all_tokens.size(1) > 0 and all_tokens[0, -1] == self.hp.stop_speech_token:
            all_tokens = all_tokens[:, :-1]

        return all_tokens
