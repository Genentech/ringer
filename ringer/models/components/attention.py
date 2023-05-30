import logging
import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.bert import configuration_bert, modeling_bert

# Monkey-patch BertSelfAttention to enable cyclic relative positional encoding
# This means that the output of BertSelfAttention.forward transforms equivariantly under
# a cyclic shift of the input sequences
if modeling_bert.BertSelfAttention.__name__ == "BertSelfAttention":

    class BertSelfAttentionWithCyclicEncoding(modeling_bert.BertSelfAttention):
        def __init__(self, config: configuration_bert.BertConfig, *args, **kwargs) -> None:
            super().__init__(config, *args, **kwargs)
            if self.position_embedding_type == "cyclic_relative_key":
                logging.info("Using cyclic positional encoding")
                assert not hasattr(self, "max_position_embeddings")
                assert not hasattr(self, "distance_embedding")
                self.max_position_embeddings = config.max_position_embeddings
                self.distance_embedding = nn.Embedding(
                    2 * config.max_position_embeddings - 1, self.attention_head_size
                )

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:
            mixed_query_layer = self.query(hidden_states)

            # If this is instantiated as a cross-attention module, the keys
            # and values come from an encoder; the attention mask needs to be
            # such that the encoder's padding tokens are not attended to.
            is_cross_attention = encoder_hidden_states is not None

            if is_cross_attention and past_key_value is not None:
                # reuse k,v, cross_attentions
                key_layer = past_key_value[0]
                value_layer = past_key_value[1]
                attention_mask = encoder_attention_mask
            elif is_cross_attention:
                key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
                value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
                attention_mask = encoder_attention_mask
            elif past_key_value is not None:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            else:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))

            query_layer = self.transpose_for_scores(mixed_query_layer)

            use_cache = past_key_value is not None
            if self.is_decoder:
                # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
                # Further calls to cross_attention layer can then reuse all cross-attention
                # key/value_states (first "if" case)
                # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
                # all previous decoder key/value_states. Further calls to uni-directional self-attention
                # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
                # if encoder bi-directional self-attention `past_key_value` is always `None`
                past_key_value = (key_layer, value_layer)

            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            if self.position_embedding_type in {
                "relative_key",
                "relative_key_query",
                "cyclic_relative_key",
            }:
                query_length, key_length = query_layer.shape[2], key_layer.shape[2]
                if use_cache:
                    position_ids_l = torch.tensor(
                        key_length - 1, dtype=torch.long, device=hidden_states.device
                    ).view(-1, 1)
                else:
                    position_ids_l = torch.arange(
                        query_length, dtype=torch.long, device=hidden_states.device
                    ).view(-1, 1)
                position_ids_r = torch.arange(
                    key_length, dtype=torch.long, device=hidden_states.device
                ).view(1, -1)

                if self.position_embedding_type in {
                    "relative_key",
                    "relative_key_query",
                }:
                    distance = position_ids_l - position_ids_r

                    positional_embedding = self.distance_embedding(
                        distance + self.max_position_embeddings - 1
                    )
                    positional_embedding = positional_embedding.to(
                        dtype=query_layer.dtype
                    )  # fp16 compatibility

                    if self.position_embedding_type == "relative_key":
                        relative_position_scores = torch.einsum(
                            "bhld,lrd->bhlr", query_layer, positional_embedding
                        )
                        attention_scores = attention_scores + relative_position_scores
                    elif self.position_embedding_type == "relative_key_query":
                        relative_position_scores_query = torch.einsum(
                            "bhld,lrd->bhlr", query_layer, positional_embedding
                        )
                        relative_position_scores_key = torch.einsum(
                            "bhrd,lrd->bhlr", key_layer, positional_embedding
                        )
                        attention_scores = (
                            attention_scores
                            + relative_position_scores_query
                            + relative_position_scores_key
                        )
                elif self.position_embedding_type == "cyclic_relative_key":
                    distance = position_ids_l - position_ids_r

                    # Attention mask at this point is already expanded and has zeros in the locations that should be attended to
                    seq_lengths = (attention_mask == 0).sum(-1)
                    forward_distance = distance % seq_lengths
                    reverse_distance = distance % -seq_lengths

                    forward_positional_embedding = self.distance_embedding(
                        forward_distance + self.max_position_embeddings - 1
                    )
                    reverse_positional_embedding = self.distance_embedding(
                        reverse_distance + self.max_position_embeddings - 1
                    )
                    forward_positional_embedding = forward_positional_embedding.to(
                        dtype=query_layer.dtype
                    )  # fp16 compatibility
                    reverse_positional_embedding = reverse_positional_embedding.to(
                        dtype=query_layer.dtype
                    )  # fp16 compatibility

                    # Need batch dimension in einsum for positional embeddings because they are different for each sequence in the batch
                    relative_position_scores_forward = torch.einsum(
                        "bhld,blrd->bhlr", query_layer, forward_positional_embedding
                    )
                    relative_position_scores_reverse = torch.einsum(
                        "bhld,blrd->bhlr", query_layer, reverse_positional_embedding
                    )
                    attention_scores = (
                        attention_scores
                        + relative_position_scores_forward
                        + relative_position_scores_reverse
                    )

            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            if self.is_decoder:
                outputs = outputs + (past_key_value,)
            return outputs

    modeling_bert.BertSelfAttention = BertSelfAttentionWithCyclicEncoding  # Patch
