# coding: utf-8
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import groupby
# from signjoey.initialization import initialize_model
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from signjoey.decoders import Decoder, RecurrentDecoder, TransformerDecoder
# from signjoey.search import beam_search, greedy
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from signjoey.batch import FeatBatch
from signjoey.helpers import freeze_params
from signjoey.search import beam_search_feat, greedy_feat
from signjoey.initialization import initialize_feat_model
from torch import Tensor
from typing import Union


# Clément
class FeatModel(nn.Module):
    """
    Base Model class
    """

    def __init__(
        self,
        encoder_body: Encoder,
        encoder_face: Encoder,
        encoder_hand: Encoder,
        gloss_output_layer: nn.Module,
        decoder_body: Decoder,
        decoder_face: Decoder,
        decoder_hand: Decoder,
        body_embed: SpatialEmbeddings,
        face_embed: SpatialEmbeddings,
        hand_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        do_recognition: bool = False,
        do_translation: bool = True,
        do_anchoring: bool = True
    ):
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param pose_embed: list of pose embeddings
        :param txt_embed: spoken language word embedding
        :param gls_vocab: gls vocabulary
        :param txt_vocab: spoken language vocabulary
        :param do_recognition: flag to build the model with recognition output.
        :param do_translation: flag to build the model with translation decoder.
        """
        super().__init__()

        self.encoder_body = encoder_body
        self.encoder_face = encoder_face
        self.encoder_hand = encoder_hand
        self.decoder_body = decoder_body
        self.decoder_face = decoder_face
        self.decoder_hand = decoder_hand

        self.body_embed = body_embed
        self.face_embed = face_embed
        self.hand_embed = hand_embed

        self.txt_embed = txt_embed

        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

        self.gloss_output_layer = gloss_output_layer
        self.do_recognition = do_recognition
        self.do_translation = do_translation
        self.do_anchoring = do_anchoring

        self.output_layer = nn.Linear(4*self.decoder_body._hidden_size, self.decoder_body._output_size, bias=False)

        if self.do_anchoring:
            self.dope_predictor = DopePredictor()
        else:
            self.dope_predictor = None

    # pylint: disable=arguments-differ
    def forward(
        self,
        # sgn: Tensor,
        body_feat: Tensor,
        face_feat: Tensor,
        hand_feat_1: Tensor,
        hand_feat_2: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param sgn: source input
        :param sgn_mask: source mask
        :param sgn_lengths: length of source inputs
        :param txt_input: target input
        :param txt_mask: target mask
        :return: decoder outputs
        """

        encoder_body, encoder_face, encoder_hand_1, encoder_hand_2 = self.encode(
            body_feat=body_feat,
            face_feat=face_feat,
            hand_feat_1=hand_feat_1,
            hand_feat_2=hand_feat_2,
            sgn_mask=sgn_mask, 
            sgn_length=sgn_lengths
        )
        encoder_body, encoder_body_hidden = encoder_body
        encoder_face, encoder_face_hidden = encoder_face
        encoder_hand_1, encoder_hand_hidden_1 = encoder_hand_1
        encoder_hand_2, encoder_hand_hidden_2 = encoder_hand_2
        
        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
        else:
            gloss_probabilities = None

        if self.do_translation:
            unroll_steps = txt_input.size(1)
            decoder_outputs = self.decode(
                encoder_body=encoder_body,
                encoder_face=encoder_face,
                encoder_hand_1=encoder_hand_1,
                encoder_hand_2=encoder_hand_2,
                encoder_body_hidden=encoder_body_hidden,
                encoder_face_hidden=encoder_face_hidden,
                encoder_hand_hidden_1=encoder_hand_hidden_1,
                encoder_hand_hidden_2=encoder_hand_hidden_2,
                sgn_mask=sgn_mask,
                txt_input=txt_input,
                unroll_steps=unroll_steps,
                txt_mask=txt_mask,
            )
        else:
            decoder_outputs = None

        if self.do_anchoring:
            dope_outputs = self.dope_predictor(encoder_body, encoder_face, encoder_hand_1, encoder_hand_2)
        else:
            dope_outputs = None

        return decoder_outputs, gloss_probabilities, dope_outputs

    def encode(
        self, body_feat: Tensor, face_feat: Tensor, hand_feat_1: Tensor, hand_feat_2: Tensor, sgn_mask: Tensor, sgn_length: Tensor
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param feat:
        :param sgn_mask:
        :param sgn_length:
        :return: encoder outputs (output, hidden_concat)
        """
        mask = body_feat.new(body_feat.shape[0], 1, body_feat.shape[1]).fill_(1)

        embed_body = self.body_embed(x=body_feat, mask=mask)
        embed_face = self.face_embed(x=face_feat, mask=mask)
        embed_hand_1 = self.hand_embed(x=hand_feat_1, mask=mask)
        embed_hand_2 = self.hand_embed(x=hand_feat_2, mask=mask)

        encoded_body = self.encoder_body(
            embed_src=embed_body,
            src_length=sgn_length,
            mask=sgn_mask,
        )
        encoded_face = self.encoder_face(
            embed_src=embed_face,
            src_length=sgn_length,
            mask=sgn_mask,
        )
        encoded_hand_1 = self.encoder_hand(
            embed_src=embed_hand_1,
            src_length=sgn_length,
            mask=sgn_mask,
        )
        encoded_hand_2 = self.encoder_hand(
            embed_src=embed_hand_2,
            src_length=sgn_length,
            mask=sgn_mask,
        )

        return [encoded_body, encoded_face, encoded_hand_1, encoded_hand_2]

    def decode(
        self,
        encoder_body: Tensor,
        encoder_face: Tensor,
        encoder_hand_1: Tensor,
        encoder_hand_2: Tensor,
        encoder_body_hidden: Tensor,
        encoder_face_hidden: Tensor,
        encoder_hand_hidden_1: Tensor,
        encoder_hand_hidden_2: Tensor,
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param sgn_mask: sign sequence mask, 1 at valid tokens
        :param txt_input: spoken language sentence inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param txt_mask: mask for spoken language words
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """

        decoder_body = self.decoder_body(
            encoder_output=encoder_body,
            encoder_hidden=encoder_body_hidden,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )
        decoder_face = self.decoder_face(
            encoder_output=encoder_face,
            encoder_hidden=encoder_face_hidden,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )
        decoder_hand_1 = self.decoder_hand(
            encoder_output=encoder_hand_1,
            encoder_hidden=encoder_hand_hidden_1,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )
        decoder_hand_2 = self.decoder_hand(
            encoder_output=encoder_hand_2,
            encoder_hidden=encoder_hand_hidden_2,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )

        # concatenate decoder outputs
        concat_decoder = torch.cat([decoder_body[1], decoder_face[1], decoder_hand_1[1], decoder_hand_2[1]], dim=2)
        output = self.output_layer(concat_decoder)

        return output, concat_decoder, None, None

    def get_loss_for_batch(  # TODO: ANCHOR LOSS
        self,
        batch: FeatBatch,
        recognition_loss_function: nn.Module,
        translation_loss_function: nn.Module,
        anchoring_loss_function: nn.Module,
        recognition_loss_weight: float,
        translation_loss_weight: float,
        anchoring_loss_cls_weight: float,
        anchoring_loss_reg_weight: float,
    ) -> (Tensor, Tensor):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param recognition_loss_function: Sign Language Recognition Loss Function (CTC)
        :param translation_loss_function: Sign Language Translation Loss Function (XEntropy)
        :param recognition_loss_weight: Weight for recognition loss
        :param translation_loss_weight: Weight for translation loss
        :return: recognition_loss: sum of losses over sequences in the batch
        :return: translation_loss: sum of losses over non-pad elements in the batch
        """
        # pylint: disable=unused-variable

        # Do a forward pass
        decoder_outputs, gloss_probabilities, dope_outputs = self.forward(
            # sgn=batch.sgn,
            body_feat=batch.body_feat,
            face_feat=batch.face_feat,
            hand_feat_1=batch.hand_feat_1,
            hand_feat_2=batch.hand_feat_2,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
        )

        if self.do_recognition:
            assert gloss_probabilities is not None
            # Calculate Recognition Loss
            recognition_loss = (
                recognition_loss_function(
                    gloss_probabilities,
                    batch.gls,
                    batch.sgn_lengths.long(),
                    batch.gls_lengths.long(),
                )
                * recognition_loss_weight
            )
        else:
            recognition_loss = None

        if self.do_translation:
            assert decoder_outputs is not None
            word_outputs, _, _, _ = decoder_outputs
            # Calculate Translation Loss
            txt_log_probs = F.log_softmax(word_outputs, dim=-1)
            translation_loss = (
                translation_loss_function(txt_log_probs, batch.txt)
                * translation_loss_weight
            )
        else:
            translation_loss = None

        if self.do_anchoring:
            assert dope_outputs is not None
            scores, pose_deltas = dope_outputs
            # Calculate Anchoring Loss
            anchoring_loss = anchoring_loss_function(
                    scores,
                    pose_deltas,
                    batch.body_scores,
                    batch.body_deltas,
                    batch.face_scores,
                    batch.face_deltas,
                    batch.hand_scores_1,
                    batch.hand_deltas_1,
                    batch.hand_scores_2,
                    batch.hand_deltas_2
                )
            anchoring_loss = (anchoring_loss[0]*anchoring_loss_cls_weight + anchoring_loss[1]*anchoring_loss_reg_weight)
        else:
            anchoring_loss = None

        return recognition_loss, translation_loss, anchoring_loss

    def run_batch(
        self,
        batch: FeatBatch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param recognition_beam_size: size of the beam for CTC beam search
            if 1 use greedy
        :param translation_beam_size: size of the beam for translation beam search
            if 1 use greedy
        :param translation_beam_alpha: alpha value for beam search
        :param translation_max_output_length: maximum length of translation hypotheses
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """

        encoder_body, encoder_face, encoder_hand_1, encoder_hand_2 = self.encode(
            body_feat=batch.body_feat,
            face_feat=batch.face_feat,
            hand_feat_1=batch.hand_feat_1,
            hand_feat_2=batch.hand_feat_2,
            sgn_mask=batch.sgn_mask,
            sgn_length=batch.sgn_lengths
        )
        encoder_body, encoder_body_hidden = encoder_body
        encoder_face, encoder_face_hidden = encoder_face
        encoder_hand_1, encoder_hand_hidden_1 = encoder_hand_1
        encoder_hand_2, encoder_hand_hidden_2 = encoder_hand_2

#        if self.do_recognition:
#            # Gloss Recognition Part
#            # N x T x C
#            gloss_scores = self.gloss_output_layer(encoder_output)
#            # N x T x C
#            gloss_probabilities = gloss_scores.log_softmax(2)
#            # Turn it into T x N x C
#            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
#            gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
#            tf_gloss_probabilities = np.concatenate(
#                (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
#                axis=-1,
#            )
#
#            assert recognition_beam_size > 0
#            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
#                inputs=tf_gloss_probabilities,
#                sequence_length=batch.sgn_lengths.cpu().detach().numpy(),
#                beam_width=recognition_beam_size,
#                top_paths=1,
#            )
#            ctc_decode = ctc_decode[0]
#            # Create a decoded gloss list for each sample
#            tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]
#            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
#                tmp_gloss_sequences[dense_idx[0]].append(
#                    ctc_decode.values[value_idx].numpy() + 1
#                )
#            decoded_gloss_sequences = []
#            for seq_idx in range(0, len(tmp_gloss_sequences)):
#                decoded_gloss_sequences.append(
#                    [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
#                )
#        else:
#            decoded_gloss_sequences = None
        decoded_gloss_sequences = None

        # Mask (should be in defined in PoseBatch)
        sgn_mask = (batch.body_feat != (batch.body_feat).new_zeros(batch.body_feat.shape))[..., 0].unsqueeze(1)

        if self.do_translation:
            # greedy decoding
            if translation_beam_size < 2:
                stacked_txt_output, stacked_attention_scores = greedy_feat(
                    decoder_body=self.decoder_body,
                    decoder_face=self.decoder_face,
                    decoder_hand=self.decoder_hand,
                    encoder_body=encoder_body,
                    encoder_face=encoder_face,
                    encoder_hand_1=encoder_hand_1,
                    encoder_hand_2=encoder_hand_2,
                    encoder_body_hidden=encoder_body_hidden,
                    encoder_face_hidden=encoder_face_hidden,
                    encoder_hand_hidden_1=encoder_hand_hidden_1,
                    encoder_hand_hidden_2=encoder_hand_hidden_2,
                    output_layer=self.output_layer,
                    src_mask=sgn_mask,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    max_output_length=translation_max_output_length,
                )
                # batch, time, max_sgn_length
            else:  # beam size
                stacked_txt_output, stacked_attention_scores = beam_search_feat(
                    size=translation_beam_size,
                    encoder_body=encoder_body,
                    encoder_face=encoder_face,
                    encoder_hand_1=encoder_hand_1,
                    encoder_hand_2=encoder_hand_2,
                    encoder_body_hidden=encoder_body_hidden,
                    encoder_face_hidden=encoder_face_hidden,
                    encoder_hand_hidden_1=encoder_hand_hidden_1,
                    encoder_hand_hidden_2=encoder_hand_hidden_2,
                    output_layer=self.output_layer,
                    src_mask=sgn_mask,
                    embed=self.txt_embed,
                    max_output_length=translation_max_output_length,
                    alpha=translation_beam_alpha,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    bos_index=self.txt_bos_index,
                    decoder_body=self.decoder_body,
                    decoder_face=self.decoder_face,
                    decoder_hand=self.decoder_hand
                )
        else:
            stacked_txt_output = stacked_attention_scores = None

        return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return (
            "%s(\n"
            "\tencoder_body=%s,\n"
            "\tencoder_face=%s,\n"
            "\tencoder_hand=%s,\n"
            "\tdecoder_body=%s,\n"
            "\tdecoder_face=%s,\n"
            "\tdecoder_hand=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder_body,
                self.encoder_face,
                self.encoder_hand,
                self.decoder_body,
                self.decoder_face,
                self.decoder_hand,
                self.txt_embed,
            )
        )

# Clément
def build_feat_model(
    cfg: dict,
    sgn_dim: int,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    do_anchoring: bool,
    do_recognition: bool = False,
    do_translation: bool = True,
) -> FeatModel:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param sgn_dim: feature dimension of the sign frame representation, i.e. 2560 for EfficientNet-7.
    :param gls_vocab: sign gloss vocabulary
    :param txt_vocab: spoken language word vocabulary
    :return: built and initialized model
    :param do_recognition: flag to build the model with recognition output.
    :param do_translation: flag to build the model with translation decoder.
    """

    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    body_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
    )
    face_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
    )
    hand_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
    )

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert (
        cfg["encoder"]["embeddings"]["embedding_dim"]
        == cfg["encoder"]["hidden_size"]
    ), "for transformer, emb_size must be hidden_size"

    encoder_body = TransformerEncoder(
        **cfg["encoder"],
        emb_size=body_embed.embedding_dim,
        emb_dropout=enc_emb_dropout,
    )
    encoder_face = TransformerEncoder(
        **cfg["encoder"],
        emb_size=face_embed.embedding_dim,
        emb_dropout=enc_emb_dropout,
    )
    encoder_hand = TransformerEncoder(
        **cfg["encoder"],
        emb_size=hand_embed.embedding_dim,
        emb_dropout=enc_emb_dropout,
    )

    if do_recognition:
        gloss_output_layer = nn.Linear(encoder_body.output_size, len(gls_vocab))
        if cfg["encoder"].get("freeze", False):
            freeze_params(gloss_output_layer)
    else:
        gloss_output_layer = None

    # build decoder and word embeddings
    if do_translation:
        txt_embed: Union[Embeddings, None] = Embeddings(
            **cfg["decoder"]["embeddings"],
            num_heads=cfg["decoder"]["num_heads"],
            vocab_size=len(txt_vocab),
            padding_idx=txt_padding_idx,
        )
        dec_dropout = cfg["decoder"].get("dropout", 0.0)
        dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)

        decoder_body = TransformerDecoder(
            **cfg["decoder"],
            encoder=encoder_body,
            vocab_size=len(txt_vocab),
            emb_size=txt_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
        )
        decoder_face = TransformerDecoder(
            **cfg["decoder"],
            encoder=encoder_face,
            vocab_size=len(txt_vocab),
            emb_size=txt_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
        )
        decoder_hand = TransformerDecoder(
            **cfg["decoder"],
            encoder=encoder_hand,
            vocab_size=len(txt_vocab),
            emb_size=txt_embed.embedding_dim,
            emb_dropout=dec_emb_dropout,
        )
    else:
        txt_embed = None
        decoder_body = None
        decoder_face = None
        decoder_hand = None

    model: FeatModel = FeatModel(
        encoder_body=encoder_body,
        encoder_face=encoder_face,
        encoder_hand=encoder_hand,
        gloss_output_layer=gloss_output_layer,
        decoder_body=decoder_body,
        decoder_face=decoder_face,
        decoder_hand=decoder_hand,
        body_embed=body_embed,
        face_embed=face_embed,
        hand_embed=hand_embed,
        txt_embed=txt_embed,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        do_recognition=do_recognition,
        do_translation=do_translation,
        do_anchoring=do_anchoring
    )

    if do_translation:
        # tie softmax layer with txt embeddings
        if cfg.get("tied_softmax", False):
            # noinspection PyUnresolvedReferences
            if txt_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
                # (also) share txt embeddings and softmax layer:
                # noinspection PyUnresolvedReferences
                model.decoder_body.output_layer.weight = txt_embed.lut.weight
            else:
                raise ValueError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer."
                )

    # custom initialization of model parameters
    initialize_feat_model(model, cfg, txt_padding_idx)

    return model


class DopePredictor(nn.Module):

    def __init__(self):
        super(self.__class__, self).__init__()

        in_channels = 512
        num_joints = {'body': 13, 'hand': 21, 'face': 84}
        num_anchor_poses = {'body': 20, 'hand': 10, 'face': 10}
        dict_num_classes = {k: v + 1 for k, v in num_anchor_poses.items()}
        dict_num_posereg = {k: num_anchor_poses[k] * num_joints[k] * 5 for k in num_joints.keys()}

        self.body_cls_score = nn.Linear(in_channels, dict_num_classes['body'])
        self.body_pose_pred = nn.Linear(in_channels, dict_num_posereg['body'])
        self.hand_cls_score = nn.Linear(in_channels, dict_num_classes['hand'])
        self.hand_pose_pred = nn.Linear(in_channels, dict_num_posereg['hand'])
        self.face_cls_score = nn.Linear(in_channels, dict_num_classes['face'])
        self.face_pose_pred = nn.Linear(in_channels, dict_num_posereg['face'])

    def forward(self, encoder_out_body, encoder_out_face, encoder_out_hand_1, encoder_out_hand_2):

        scores = {}
        pose_deltas = {}

        scores['body'] = self.body_cls_score(encoder_out_body)
        pose_deltas['body'] = self.body_pose_pred(encoder_out_body)
        scores['hand_1'] = self.hand_cls_score(encoder_out_hand_1)
        pose_deltas['hand_1'] = self.hand_pose_pred(encoder_out_hand_1)
        scores['hand_2'] = self.hand_cls_score(encoder_out_hand_2)
        pose_deltas['hand_2'] = self.hand_pose_pred(encoder_out_hand_2)
        scores['face'] = self.face_cls_score(encoder_out_face)
        pose_deltas['face'] = self.face_pose_pred(encoder_out_face)

        return scores, pose_deltas