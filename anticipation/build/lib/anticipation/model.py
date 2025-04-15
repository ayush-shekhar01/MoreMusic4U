from transformers import GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
from torch import nn
from typing import Callable, Optional, Tuple, Union

# number of feature values 
NUM_SONG_FEATURES = 20
NUM_SONGS = 5
DEVICE = "cuda"

def truncate_prompt_to_fit(input_ids, prefix_len, model):
    max_len = model.config.n_positions  # e.g., 1024 for GPT-2
    prompt_len = input_ids.shape[1]
    allowed_prompt_len = max_len - prefix_len

    if prompt_len > allowed_prompt_len:
        # Truncate from the left (oldest tokens)
        input_ids = input_ids[:, -allowed_prompt_len:]

    return input_ids

class PrefixGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = GPT2Config

    def __init__(self, config, prefix_dim=NUM_SONGS, prefix_len=NUM_SONG_FEATURES):
        super().__init__(config)
        self.prefix_len = prefix_len
        self.prefix_dim = prefix_dim
        # num_songs feature values -> feature vector
        # only 1 linear layer (for now)
        self.prefix_proj = nn.Linear(prefix_dim, config.n_embd, device=DEVICE)
        # Add encoder here ...

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        prefix: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Standard token embeddings
        inputs_embeds = self.transformer.wte(input_ids)
        batch_size, seq_len, hidden_dim = inputs_embeds.shape

        if not prefix is None:
          # Get prefix
          prefix = self.prefix_proj(prefix).view(batch_size, self.prefix_len, hidden_dim)  # (B, prefix_len, D)

          # Concatenate prefix embeddings with input embeddings
          inputs_embeds = truncate_prompt_to_fit(inputs_embeds, self.prefix_len, self)
          inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)  # (B, prefix_len + T, D)

          transformer_outputs = self.transformer(
              past_key_values=past_key_values,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids,
              position_ids=position_ids,
              head_mask=head_mask,
              inputs_embeds=inputs_embeds,
              encoder_hidden_states=encoder_hidden_states,
              encoder_attention_mask=encoder_attention_mask,
              use_cache=use_cache,
              output_attentions=output_attentions,
              output_hidden_states=output_hidden_states,
              return_dict=return_dict,
          )
        else:
          transformer_outputs = self.transformer(
              input_ids=input_ids,
              past_key_values=past_key_values,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids,
              position_ids=position_ids,
              head_mask=head_mask,
              encoder_hidden_states=encoder_hidden_states,
              encoder_attention_mask=encoder_attention_mask,
              use_cache=use_cache,
              output_attentions=output_attentions,
              output_hidden_states=output_hidden_states,
              return_dict=return_dict,
          )


        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )