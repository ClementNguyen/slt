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
            body_2d
            body_3d
            face_feat
            face_scores
            face_2d
            face_3d
            hand_feat_1
            hand_scores_1
            hand_2d_1
            hand_3d_1
            hand_feat_2
            hand_scores_2
            hand_2d_2
            hand_3d_2
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
                ("body_feat", fields[4]),
                ("body_scores", fields[5]),
                ("body_2d", fields[6]),
                ("body_3d", fields[7]),
                ("face_feat", fields[8]),
                ("face_scores", fields[9]),
                ("face_2d", fields[10]),
                ("face_3d", fields[11]),
                ("hand_feat_1", fields[12]),
                ("hand_scores_1", fields[13]),
                ("hand_2d_1", fields[14]),
                ("hand_3d_1", fields[15]),
                ("hand_feat_2", fields[16]),
                ("hand_scores_2", fields[17]),
                ("hand_2d_2", fields[18]),
                ("hand_3d_2", fields[19])   
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
                    for field_idx in range(4,20): 
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
                        "body_feat": s["body_feat"],
                        "body_scores": s["body_scores"],
                        "body_2d": s["body_2d"],
                        "body_3d": s["body_3d"],
                        "face_feat": s["face_feat"],
                        "face_scores": s["face_scores"],
                        "face_2d": s["face_2d"],
                        "face_3d": s["face_3d"],
                        "hand_feat_1": s["hand_feat_1"],
                        "hand_scores_1": s["hand_scores_1"],
                        "hand_2d_1": s["hand_2d_1"],
                        "hand_3d_1": s["hand_3d_1"],
                        "hand_feat_2": s["hand_feat_2"],
                        "hand_scores_2": s["hand_scores_2"],
                        "hand_2d_2": s["hand_2d_2"],
                        "hand_3d_2": s["hand_3d_2"]
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
                        sample["body_feat"] + 1e-8,
                        sample["body_scores"] + 1e-8,
                        sample["body_2d"] + 1e-8,
                        sample["body_3d"] + 1e-8,
                        sample["face_feat"] + 1e-8,
                        sample["face_scores"] + 1e-8,
                        sample["face_2d"] + 1e-8,
                        sample["face_3d"] + 1e-8,
                        sample["hand_feat_1"] + 1e-8,
                        sample["hand_scores_1"] + 1e-8,
                        sample["hand_2d_1"] + 1e-8,
                        sample["hand_3d_1"] + 1e-8,
                        sample["hand_feat_2"] + 1e-8,
                        sample["hand_scores_2"] + 1e-8,
                        sample["hand_2d_2"] + 1e-8,
                        sample["hand_3d_2"] + 1e-8
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
