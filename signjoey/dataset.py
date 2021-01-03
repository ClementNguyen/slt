# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch
import os
import random


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)



# Clément
class PoseTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation.
    
    Fields: sequence
            signer
            gls
            txt
            body_2d
            body_3d
            face_2d
            face_3d
            left_hand_2d
            left_hand_3d
            right_hand_2d
            right_hand_3d
    """

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field, Field, Field, Field, Field, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("gls", fields[2]),
                ("txt", fields[3]),
                ("body_2d", fields[4]),
                ("body_3d", fields[5]),
                ("face_2d", fields[6]),
                ("face_3d", fields[7]),
                ("left_hand_2d", fields[8]),
                ("left_hand_3d", fields[9]),
                ("right_hand_2d", fields[10]),
                ("right_hand_3d", fields[11])           
            ]

#        if not isinstance(path, list):
#            path = [path]

        path = os.listdir(path)

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    for field_idx in range(4,12): 
                      field_name = fields[field_idx][0]
                      samples[seq_id][field_name] = torch.cat(
                          [samples[seq_id][field_name], s[field_name]], axis=1
                      )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "body_2d": s["body_2d"],
                        "body_3d": s["body_3d"],
                        "face_2d": s["face_2d"],
                        "face_3d": s["face_3d"],
                        "left_hand_2d": s["left_hand_2d"],
                        "left_hand_3d": s["left_hand_3d"],
                        "right_hand_2d": s["right_hand_2d"],
                        "right_hand_3d": s["right_hand_3d"],
                    }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                        # This is for numerical stability
                        sample["body_2d"] + 1e-8,
                        sample["body_3d"] + 1e-8,
                        sample["face_2d"] + 1e-8,
                        sample["face_3d"] + 1e-8,
                        sample["left_hand_2d"] + 1e-8,
                        sample["left_hand_3d"] + 1e-8,
                        sample["right_hand_2d"] + 1e-8,
                        sample["right_hand_3d"] + 1e-8
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)





class FeatTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation.
    
    Fields: sequence
            signer
            gls
            txt
            body_feat
            body_scores
            body_deltas
            face_feat
            face_scores
            face_deltas
            hand_feat_1
            hand_scores_1
            hand_deltas_1
            hand_feat_2
            hand_scores_2
            hand_deltas_2
    """

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field, Field, Field, Field, Field, Field, Field, Field],
        do_anchoring: bool,
        size=1,
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("gls", fields[2]),
                ("txt", fields[3]),
                ("body_feat", fields[4]),
                ("body_scores", fields[5]),
                ("body_deltas", fields[6]),
                ("face_feat", fields[7]),
                ("face_scores", fields[8]),
                ("face_deltas", fields[9]),
                ("hand_feat_1", fields[10]),
                ("hand_scores_1", fields[11]),
                ("hand_deltas_1", fields[12]),
                ("hand_feat_2", fields[13]),
                ("hand_scores_2", fields[14]),
                ("hand_deltas_2", fields[15])
            ]

        feat_fields = [4, 7, 10, 13]
        loss_fields = [5, 6, 8, 9, 11, 12, 14, 15]

#        if not isinstance(path, list):
#            path = [path]

        path = os.listdir(path)
        path = random.shuffle(path)[:int(len(path)*size)]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    for field_idx in range(4,16):
                        field_name = fields[field_idx][0]
                        if field_idx in feat_fields:
                            samples[seq_id][field_name] = torch.cat(
                                [samples[seq_id][field_name], s[field_name]], axis=1
                            )
                        else:
                            if do_anchoring:
                                samples[seq_id][field_name] = torch.cat(
                                    [samples[seq_id][field_name], s[field_name]], axis=1
                                )
                            else:
                                samples[seq_id][field_name] = None
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "body_feat": s["body_feat"],
                        "face_feat": s["face_feat"],
                        "hand_feat_1": s["hand_feat_1"],
                        "hand_feat_2": s["hand_feat_2"],
                    }
                    if do_anchoring:
                        samples[seq_id] = {
                            "body_scores": s["body_scores"],
                            "body_deltas": s["body_deltas"],
                            "face_scores": s["face_scores"],
                            "face_deltas": s["face_deltas"],
                            "hand_scores_1": s["hand_scores_1"],
                            "hand_deltas_1": s["hand_deltas_1"],
                            "hand_scores_2": s["hand_scores_2"],
                            "hand_deltas_2": s["hand_deltas_2"]
                        }
                    else:
                        samples[seq_id] = {
                            "body_scores": None,
                            "body_deltas": None,
                            "face_scores": None,
                            "face_deltas": None,
                            "hand_scores_1": None,
                            "hand_deltas_1": None,
                            "hand_scores_2": None,
                            "hand_deltas_2": None
                        },

        examples = []
        for s in samples:
            sample = samples[s]
            if do_anchoring:
                examples.append(
                    data.Example.fromlist(
                        [
                            sample["name"],
                            sample["signer"],
                            sample["gloss"].strip(),
                            sample["text"].strip(),
                            # This is for numerical stability
                            sample["body_feat"] + 1e-8,
                            sample["body_scores"] + 1e-8,
                            sample["body_deltas"] + 1e-8,
                            sample["face_feat"] + 1e-8,
                            sample["face_scores"] + 1e-8,
                            sample["face_deltas"] + 1e-8,
                            sample["hand_feat_1"] + 1e-8,
                            sample["hand_scores_1"] + 1e-8,
                            sample["hand_deltas_1"] + 1e-8,
                            sample["hand_feat_2"] + 1e-8,
                            sample["hand_scores_2"] + 1e-8,
                            sample["hand_deltas_2"] + 1e-8
                        ],
                        fields,
                    )
                )
            else:
                examples.append(
                    data.Example.fromlist(
                        [
                            sample["name"],
                            sample["signer"],
                            sample["gloss"].strip(),
                            sample["text"].strip(),
                            # This is for numerical stability
                            sample["body_feat"] + 1e-8,
                            sample["body_scores"],
                            sample["body_deltas"],
                            sample["face_feat"] + 1e-8,
                            sample["face_scores"],
                            sample["face_deltas"],
                            sample["hand_feat_1"] + 1e-8,
                            sample["hand_scores_1"],
                            sample["hand_deltas_1"],
                            sample["hand_feat_2"] + 1e-8,
                            sample["hand_scores_2"],
                            sample["hand_deltas_2"]
                        ],
                        fields,
                    )
                )
        super().__init__(examples, fields, **kwargs)


# def load_dataset_from_dir(dir_name):
#     dataset = []
#     file_list = os.listdir(dir_name)
#     for file in file_list:
#         with open(os.path.join(dir_name, file), "rb") as f:
#             loaded_object = pickle.load(f)
#             dataset.append(loaded_object)
#     return dataset
