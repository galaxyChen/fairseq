# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

from collections import OrderedDict

from fairseq import utils
from fairseq.models import FairseqMultiModel, register_model, register_model_architecture, BaseFairseqModel

from fairseq.models.transformer import (
    base_architecture,
    Embedding,
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
)

import torch.nn as nn
import torch.nn.functional as F
import torch

class XTransformerEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.mask_idx = dictionary.mask_index
        self.dropout = args.dropout

    def forward(self, src_tokens, src_lengths):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx) | src_tokens.eq(self.mask_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }


class XTransformerDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.dropout = args.dropout

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, positions=None):
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
            positions=positions,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn, _ = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}


@register_model('xtransformer_lm')
class XTransformerModel(BaseFairseqModel):
    def __init__(self, encoders, decoders, eval_lang_pair=None):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.tgt_key = None
        if eval_lang_pair is not None:
            self.source_lang = eval_lang_pair.split('-')[0]
            self.target_lang = eval_lang_pair.split('-')[1]

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif hasattr(self, 'decoders'):
            return self.decoders[self.tgt_key].get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def max_positions(self):
        return None

    def max_decoder_positions(self):
        return min(decoder.max_positions() for decoder in self.decoders.values())

    def forward(self, src_tokens, src_lengths, prev_output_tokens, src_key, tgt_key, positions=None):
        encoder_out = self.encoders[src_key](src_tokens, src_lengths)
        decoder_out = self.decoders[tgt_key](prev_output_tokens, encoder_out, positions=positions)
        self.tgt_key = tgt_key
        return decoder_out

    @classmethod
    def build_model(cls, args, task):
        langs = [lang for lang in args.langs]

        embed_tokens = {}
        for lang in langs:
            if len(embed_tokens) == 0 or args.share_all_embeddings is False:
                embed_token = build_embedding(
                    task.dicts[lang], args.encoder_embed_dim, args.encoder_embed_path        
                )
                embed_tokens[lang] = embed_token
            else:
                embed_tokens[lang] = embed_tokens[langs[0]]

        args.share_decoder_input_output_embed = True
        encoders, decoders = {}, {}

        for lang in langs:
            encoder_embed_tokens = embed_tokens[lang]
            decoder_embed_tokens = encoder_embed_tokens
            if lang in args.source_langs:
                encoder = XTransformerEncoder(args, task.dicts[lang], encoder_embed_tokens)
                encoders[lang] = encoder
            if lang in args.target_langs:
                decoder = XTransformerDecoder(args, task.dicts[lang], decoder_embed_tokens)
                decoders[lang] = decoder
        return XTransformerModel(encoders, decoders, args.eval_lang_pair)
    
    @property
    def decoder(self):
        return self.decoders[self.target_lang]

    @property
    def encoder(self):
        return self.encoders[self.source_lang]


@register_model_architecture('xtransformer_lm', 'xtransformer_lm')
def base_x_transformer(args):
    base_architecture(args)


def build_embedding(dictionary, embed_dim, path=None):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)
    # if provided, load from preloaded dictionaries
    if path:
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, emb)
    return emb



class Args:
    def __init__(self):
        self.langs = ["en", "ja"]
        self.share_all_embeddings = False
        self.encoder_embed_dim = 1024
        self.encoder_embed_path = None
        self.dropout = 0.1
        self.max_source_positions = 1024
        self.no_token_positional_embeddings = False
        self.encoder_learned_pos = False
        self.encoder_layers = 10
        self.decoder_layers = 6
        self.encoder_attention_heads = 16
        self.attention_dropout = 0.1
        self.encoder_normalize_before = True
        self.encoder_ffn_embed_dim = 4096
        self.decoder_embed_dim = 1024
        self.decoder_output_dim = 1024
        self.max_target_positions = 1024
        self.decoder_learned_pos = False
        self.decoder_attention_heads = 16
        self.decoder_normalize_before = True
        self.decoder_ffn_embed_dim = 4096
        self.encoder_layerdrop = 0
        self.decoder_layerdrop = 0
        self.init_type = "default"
        self.fp16 = False
        self.adaptive_input = False
        self.quant_noise_pq = 0
        self.no_scale_embedding = False
        self.adaptive_softmax_cutoff = None
        self.share_decoder_input_output_embed = True
        self.source_langs = "en"
        self.target_langs = "ja"
        self.eval_lang_pair = "en-ja"

class Task:
    def __init__(self, dicts):
        self.dicts = dicts



if __name__ == "__main__":
    from fairseq import checkpoint_utils
    from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary

    en_dict = MaskedLMDictionary.load(
        "/mnt/lustre/sjtu/home/xyc30/code/nmt/mass_a10/data/training/processed/dict.en.txt")
    ja_dict = MaskedLMDictionary.load(
        "/mnt/lustre/sjtu/home/xyc30/code/nmt/mass_a10/data/training/processed/dict.ja.txt")
    args = Args()
    dicts = {"ja": ja_dict, "en": en_dict}
    task = Task(dicts)
    model = XTransformerModel.build_model(args, task)
    state = torch.load(
        "/mnt/lustre/sjtu/home/xyc30/code/nmt/mass_a10/checkpoints/mass/pretraining/checkpoint_last.pt")
    model.load_state_dict(state['model'], strict=False)
    print(model)
