from typing import List, Optional, Tuple, Union
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM, Qwen2Model
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.generation.logits_process import TemperatureLogitsWarper

from ..vispeak_arch import ViSpeakMetaForCausalLM, ViSpeakMetaModel


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()



def custom_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    user_inputs_embedding: Optional[torch.LongTensor] = None,
    informative_labels: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

    >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        user_inputs_embedding=user_inputs_embedding,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    # logits = logits.float()

    informative_logits = None
    if hasattr(self, 'informative_head'):
        informative_logits = self.informative_head(hidden_states).float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
    if informative_labels is not None:
        # loss_fct = CrossEntropyLoss()
        mask = informative_labels != -100
        if torch.any(mask):
            loss += sigmoid_focal_loss(informative_logits[mask].squeeze(), informative_labels[mask].squeeze().to(torch.float))
        else:
            loss += torch.sum(informative_logits) * 0

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    #import pdb; pdb.set_trace()
    new_output = CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    new_output.informative_logits = informative_logits
    return new_output

def custom_forward2(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    user_inputs_embedding: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    use_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache) and not self.training:
        use_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
        if self.combination_type == "linear_combine":
            inputs_embeds = self.combination(torch.cat([inputs_embeds, user_inputs_embedding], dim=-1))
        elif self.combination_type == "adaptive_combine":
            adaptive_weight = self.combination(torch.cat([inputs_embeds, user_inputs_embedding], dim=-1))
            inputs_embeds = inputs_embeds * torch.sigmoid(adaptive_weight) + user_inputs_embedding * (1 - torch.sigmoid(adaptive_weight))
        elif self.combination_type == "add":
            inputs_embeds = (inputs_embeds + user_inputs_embedding) / 2
        else:
            raise NotImplementedError

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


Qwen2Model.forward = custom_forward2
Qwen2ForCausalLM.forward = custom_forward


class ViSpeakQwen2Config(Qwen2Config):
    model_type = "vispeak-Qwen2"


class ViSpeakQwen2Model(ViSpeakMetaModel, Qwen2Model):
    config_class = ViSpeakQwen2Config

    def __init__(self, config: Qwen2Config):
        super(ViSpeakQwen2Model, self).__init__(config)


class ViSpeakQwen2ForCausalLM(Qwen2ForCausalLM, ViSpeakMetaForCausalLM):
    config_class = ViSpeakQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = ViSpeakQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        if getattr(self.model.config, 'use_informative_head', False):
            self.bulid_informative_head()

    def bulid_informative_head(self, ckpt=None):
        if not hasattr(self, 'informative_head'):
            self.informative_head = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size // 2, 1, bias=False)
            )
        self.model.use_informative_head = True
        if ckpt is not None:
            ckpt = torch.load(ckpt)
            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}
            self.informative_head.load_state_dict(get_w(ckpt, 'informative_head'))
        self.informative_head.to(self.lm_head.weight.device).to(self.lm_head.weight.dtype)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        user_input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        audios: Optional[dict] = None,
        video_audios: Optional[dict] = None,
        pad_token_id: int = None,
        sf_masks: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        user_inputs_embedding: Optional[torch.LongTensor] = None,
        informative_labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                informative_labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, user_input_ids, position_ids, attention_mask, past_key_values, labels, images, audios, video_audios, pad_token_id, sf_masks, informative_labels=informative_labels
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            user_inputs_embedding=user_inputs_embedding,
            informative_labels=informative_labels,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        user_input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        video_audios: Optional[torch.Tensor] = None,
        pad_token_id: int = None,
        sf_masks: Optional[torch.Tensor] = None,
        shared_v_pid_stride: Optional[int] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or audios is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                user_input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                audios,
                video_audios,
                pad_token_id,
                sf_masks,
                shared_v_pid_stride,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        audios = kwargs.pop("audios", None)
        video_audios = kwargs.pop("video_audios", None)
        pad_token_id = kwargs.pop("pad_token_id", None)
        sf_masks = kwargs.pop("sf_masks", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

#        import pdb; pdb.set_trace()
        position_ids = _inputs["position_ids"]
        cache_position = _inputs["cache_position"]
        if cache_position.shape[-1] == 1 and position_ids.shape[-1] > 1:
            new_position_ids = torch.zeros((position_ids.shape[0],1), dtype=position_ids.dtype, device=position_ids.device)
            new_position_ids[:, 0] = position_ids[0,-1] + cache_position[-1] + 1 - position_ids.shape[-1]
            position_ids = new_position_ids
            _inputs["position_ids"] = position_ids
#        import pdb; pdb.set_trace()

        if images is not None:
            _inputs["images"] = images
        if audios is not None:
            _inputs["audios"] = audios
        if video_audios is not None:
            _inputs["video_audios"] = video_audios
        if pad_token_id is not None:
            _inputs["pad_token_id"] = pad_token_id
        if sf_masks is not None:
            _inputs["sf_masks"] = sf_masks
        return _inputs
    
    # only support bs 1 inference
    @torch.no_grad()
    def streaming_generate(self, 
        user_input_ids,
        start_inference_seg,
        timestamps,
        seg_token_id,
        agent_input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        video_audios: Optional[torch.Tensor] = None,
        pad_token_id: int = None,
        temperature=0.01,
        max_new_tokens=2048,
        padding_size=16,
        stopping_criteria=None,
        proactive=False,
        interrupt=False,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        (
            position_ids,
            attention_mask,
            inputs_embeds,
        ) = self.prepare_inputs_labels_for_multimodal_inference(
            user_input_ids,
            position_ids,
            attention_mask,
            images,
            audios,
            video_audios,
            pad_token_id,
            padding_size,
        )
        user_input_ids = torch.cat([user_input_ids, torch.full((len(user_input_ids), padding_size,), pad_token_id, device=user_input_ids.device, dtype=user_input_ids.dtype)], dim=1)
        previous_inputs_embeds = inputs_embeds[:, :start_inference_seg[1]]
        if agent_input_ids is not None:
            remained_agent_input_ids = agent_input_ids[:, start_inference_seg[1]:]
            agent_input_ids = agent_input_ids[:, :start_inference_seg[1]]
        else:
            agent_input_ids = torch.full((len(user_input_ids), start_inference_seg[1]), pad_token_id, device=user_input_ids.device, dtype=user_input_ids.dtype)
        previous_agent_inputs_embeds = self.get_model().embed_tokens(agent_input_ids)
        if self.model.combination_type == "linear_combine":
            previous_inputs_embeds = self.model.combination(torch.cat([previous_inputs_embeds, previous_agent_inputs_embeds], dim=-1))
        elif self.model.combination_type == "adaptive_combine":
            adaptive_weight = self.model.combination(torch.cat([previous_inputs_embeds, previous_agent_inputs_embeds], dim=-1))
            previous_inputs_embeds = previous_inputs_embeds * torch.sigmoid(adaptive_weight) + previous_agent_inputs_embeds * (1 - torch.sigmoid(adaptive_weight))
        elif self.model.combination_type == "add":
            previous_inputs_embeds = (previous_inputs_embeds + previous_agent_inputs_embeds) / 2
        else:
            raise NotImplementedError

        inputs_embeds = inputs_embeds[:, start_inference_seg[1]-1:] # the first pass do not need embedding
        user_input_ids = user_input_ids[:, start_inference_seg[1]-1:]
        position_ids = position_ids[:, :start_inference_seg[1]]
        attention_mask = attention_mask[:, :start_inference_seg[1]]

        logits_processor = TemperatureLogitsWarper(temperature)


        # keep track of which sequences are already finished
        input_ids = torch.zeros((1,0), device=user_input_ids.device, dtype=user_input_ids.dtype)
        batch_size, cur_len = input_ids.shape
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, {'inputs_embeds': previous_inputs_embeds, 'position_ids': position_ids, 'attention_mask': attention_mask})
        start_prediction = False
        current_seg = start_inference_seg[0] - 1
        if hasattr(self, 'informative_head'):
            queue = deque(maxlen=3)

        previous_informative_logits = torch.tensor([[-100]], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        output_timestamp = timestamps[current_seg]
        use_input_agent_ids = False
        if interrupt:
            if len(remained_agent_input_ids) > 0:
                use_input_agent_ids = True
                agent_input_num = 0

        for i in range(inputs_embeds.shape[1]):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs["user_inputs_embedding"] = inputs_embeds[:, i: i+1]

            # forward pass to get next token
            outputs = self.forward(**model_inputs, return_dict=True)

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()
            if hasattr(self, 'informative_head'):
                informative_logits = outputs.informative_logits[:, -1, :].clone()
            else:
                informative_logits = torch.tensor([[0]], device=next_token_logits.device, dtype=next_token_logits.dtype)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # only start to predict at seg token
            if user_input_ids[0, i] == seg_token_id:
                current_seg += 1
                if not start_prediction and hasattr(self, 'informative_head'):
                    # token_scores, topk_tokens = torch.topk(next_token_scores, 2, dim=-1)
                    previous_informative_logits = previous_informative_logits.sigmoid()
                    queue.append(previous_informative_logits[0][0])

                    # token_scores[0, 0] *= 0.85
                    # print(timestamps[current_seg], token_scores, token_scores[0, 1] - token_scores[0, 0], previous_informative_logits, sum(queue))
                    # next_token_scores[..., pad_token_id] *= 0.85
            
            # token selection
            next_tokens = torch.argmax(next_token_scores, dim=-1)

            if use_input_agent_ids:
                next_tokens[:] = remained_agent_input_ids[:, agent_input_num]
                agent_input_num += 1
                if agent_input_num >= len(remained_agent_input_ids):
                    use_input_agent_ids = False

            if not proactive:
                if not start_prediction and user_input_ids[0, i] == seg_token_id and not torch.all(next_tokens == pad_token_id):
                    start_prediction = True
                    print("start to speak at ", timestamps[current_seg])
                    output_timestamp = timestamps[current_seg]
            else:
                if not start_prediction and user_input_ids[0, i] == seg_token_id and sum(queue) > 0.35:
                    start_prediction = True
                    print("start to speak at ", timestamps[current_seg])
                    next_tokens[:] = 146164
                    output_timestamp = timestamps[current_seg]
                    use_input_agent_ids = False

            # update generated ids, model inputs, and length for next step
            if not start_prediction:
                next_tokens[:] = pad_token_id

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
            )

            if start_prediction:
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
                if unfinished_sequences.max() == 0:
                    break
            cur_len += 1
            previous_informative_logits = informative_logits


            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        print(timestamps)
        print(output_timestamp)

        return input_ids[:, :max_new_tokens], output_timestamp
    
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        num_new_tokens: int = 1,
    ):
        # update past_key_values keeping its naming used in model code
        cache_name, cache = self._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], num_new_tokens))],
                dim=-1,
            )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def process_images(self, images, model_cfg):
        vision_tower = self.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = self.expand2square(
                    image, tuple(int(x * 255) for x in image_processor.image_mean)
                )
                image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images


AutoConfig.register("vispeak-Qwen2", ViSpeakQwen2Config)
AutoModelForCausalLM.register(ViSpeakQwen2Config, ViSpeakQwen2ForCausalLM)



