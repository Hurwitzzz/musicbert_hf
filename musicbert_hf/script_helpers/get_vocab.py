"""
This module contains functions used by the `data_preprocessing.py` script for handling
vocabularies.
"""

import glob
import json
import logging
import os
from typing import Iterable, Literal

import pandas as pd
from tqdm import tqdm


def get_vocab(
    csv_folder=None,
    feature=None,
    path: str | None = None,
    sort: Literal["lexical", "frequency", "none"] = "lexical",
    specials: Iterable[str] = ("<unk>", "<pad>", "<s>", "</s>"),
):
    """
    Retrieves or creates a vocabulary list for a specific feature.

    Logic flow:
    1. If path exists, load vocabulary from that file.
    2. If path doesn't exist but is provided, infer vocabulary from CSV and save it to the path.
    3. If path is None, infer vocabulary from CSV but don't save it.

    Parameters:
    -----------
    csv_folder : str, optional
        Directory containing CSV files to analyze for vocabulary inference
    feature : str, optional
        Column name in CSV files to extract tokens from
    path : str, optional
        Path to an existing vocabulary file or where to save a new vocabulary.
        If None, the vocabulary is not saved.
    sort : {"lexical", "frequency", "none"}, default="lexical"
        Method to sort vocabulary tokens
    specials : Iterable[str], default=("<unk>", "<pad>", "<s>", "</s>")
        Special tokens to include at the beginning of vocabulary

    Returns:
    --------
    list
        List of vocabulary tokens
    """
    # Case 1: Load from existing file
    if path is not None and os.path.exists(path):
        if path.endswith(".json"):
            logging.info(f"Loading JSON vocab from {path}")
            with open(path, "r") as f:
                return json.load(f)
        elif path.endswith(".txt"):
            logging.info(f"Loading FairSEQ formatted vocab from {path}")
            special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]

            with open(path, "r") as f:
                file_tokens = [
                    line.split()[0].strip()
                    for line in f.readlines()
                    if not line.startswith("madeupword")
                ]

            return special_tokens + [
                token for token in file_tokens if token not in set(special_tokens)
            ]
        else:
            logging.info(f"Loading plaintext vocab from {path}")
            with open(path, "r") as f:
                return [line.strip() for line in f.readlines()]

    # Cases 2 & 3: Infer from CSV data
    assert (
        feature is not None and csv_folder is not None
    ), "Both 'feature' and 'csv_folder' must be provided to infer vocabulary"

    logging.info(f"Inferring {feature} vocab from {csv_folder}")
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Extract tokens from all CSV files
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    unique_tokens = set(specials)
    for csv_file in tqdm(csv_files, total=len(csv_files)):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            unique_tokens.update(row[feature].split())

    # Remove specials to put them first after sorting
    unique_tokens = list(unique_tokens - set(specials))

    if sort == "lexical":
        # TODO: (Malcolm 2025-01-13) This won't work for MusicBERT input because
        #    <0-100> comes before <0-25>
        unique_tokens = sorted(unique_tokens)
    elif sort == "frequency":
        unique_tokens = sorted(
            unique_tokens, key=lambda x: df[feature].str.count(x).sum()
        )

    vocab = list(specials) + unique_tokens

    if path is not None:
        logging.info(f"Saving vocabulary to {path}")
        with open(path, "w") as f:
            if path.endswith(".json"):
                json.dump(vocab, f)
            else:
                for token in vocab:
                    f.write(token + "\n")

    return vocab


def handle_vocab(csv_folder=None, feature=None, path=None):
    itos = get_vocab(csv_folder, feature, path)
    stoi = {token: i for i, token in enumerate(itos)}

    # pad is -100 in huggingface, 1 in musicbert
    # For itos, we want to support both (therefore we need a dict rather
    #     than the simpler list implementation)
    # for stoi, we use -100 to be compatible with huggingface
    itos = {i: token for i, token in enumerate(itos)} | {-100: "<pad>"}
    stoi["<pad>"] = -100

    return itos, stoi
