# -*- coding: utf-8 -*-
"""UTF-8 byte-level BPE tokenizer used by the mini GPT assignment."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE tokenizer.

    ID layout:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: raw byte 0~255
    - 260+: tokens created by BPE merge rules
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token: dict[int, str | bytes | tuple[int, int]] = {}
        self.token_to_id: dict[str | bytes | tuple[int, int], int] = {}
        self.merges: list[tuple[int, int]] = []

    def _init_special_tokens(self):
        self.id_to_token = {}
        self.token_to_id = {}

        for token, token_id in SPECIAL_IDS.items():
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
            token = bytes([byte_value])
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def get_pad_id(self):
        """Return the padding token ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """Return the unknown token ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """Return the beginning-of-sequence token ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """Return the end-of-sequence token ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """Learn BPE merge rules from a UTF-8 corpus."""
        self._init_special_tokens()
        self.merges = []

        ids = [byte_value + BYTE_OFFSET for byte_value in corpus.encode("utf-8")]
        while len(self.id_to_token) < self.vocab_size and len(ids) >= 2:
            pair_counts = Counter(zip(ids, ids[1:]))
            if not pair_counts:
                break

            best_pair, _ = pair_counts.most_common(1)[0]
            new_id = len(self.id_to_token)
            self.merges.append(best_pair)
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id
            ids = self._replace_pair(ids, best_pair, new_id)

    @staticmethod
    def _replace_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        replaced: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                replaced.append(new_id)
                i += 2
            else:
                replaced.append(ids[i])
                i += 1
        return replaced

    def save(self, path: str | Path):
        """Save vocabulary and merge rules as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serialized_vocab = {}
        for token_id, token in self.id_to_token.items():
            if isinstance(token, str):
                value = {"type": "str", "value": token}
            elif isinstance(token, bytes):
                value = {"type": "bytes", "value": list(token)}
            else:
                value = {"type": "tuple", "value": list(token)}
            serialized_vocab[str(token_id)] = value

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": serialized_vocab,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path):
        """Load vocabulary and merge rules saved by save()."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        self.vocab_size = int(data.get("vocab_size", self.vocab_size))
        self.id_to_token = {}
        self.token_to_id = {}

        for token_id_str, saved_token in data["id_to_token"].items():
            token_id = int(token_id_str)
            token_type = saved_token["type"]
            value = saved_token["value"]
            if token_type == "str":
                token = value
            elif token_type == "bytes":
                token = bytes(value)
            elif token_type == "tuple":
                token = tuple(int(x) for x in value)
            else:
                raise ValueError(f"Unknown token type: {token_type}")
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        self.merges = [tuple(int(x) for x in pair) for pair in data.get("merges", [])]

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """Convert text to token IDs."""
        if not self.id_to_token:
            self._init_special_tokens()

        ids = [byte_value + BYTE_OFFSET for byte_value in text.encode("utf-8")]
        for pair in self.merges:
            new_id = self.token_to_id.get(pair)
            if new_id is not None:
                ids = self._replace_pair(ids, pair, new_id)

        if add_bos_eos:
            return [self.get_bos_id()] + ids + [self.get_eos_id()]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convert token IDs back to text."""
        byte_values: list[int] = []

        def expand(token_id: int) -> None:
            token = self.id_to_token.get(int(token_id))
            if token is None:
                return
            if isinstance(token, str):
                if not skip_special:
                    byte_values.extend(token.encode("utf-8"))
                return
            if isinstance(token, bytes):
                byte_values.extend(token)
                return
            left, right = token
            expand(left)
            expand(right)

        for token_id in ids:
            if skip_special and int(token_id) in SPECIAL_IDS.values():
                continue
            expand(int(token_id))

        return bytes(byte_values).decode("utf-8", errors="replace")
