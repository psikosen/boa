"""Byte-level tokenizer for Bash-MANTIS.

Uses raw byte values (0-255) plus special tokens. This naturally handles
shell punctuation, quoting, flags, and unusual characters without any
out-of-vocabulary issues.
"""

from __future__ import annotations


class ByteTokenizer:
    """Byte-level tokenizer with special tokens for structured prompts."""

    # Special tokens
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3  # separates sections in structured prompts

    # Tag tokens for structured prompts
    TASK_TAG = 4
    INTENT_TAG = 5
    CONSTRAINTS_TAG = 6
    OUTPUT_TAG = 7
    INPUT_SCRIPT_TAG = 8
    CANDIDATE_TAG = 9

    NUM_SPECIAL = 64  # reserve 0-63 for special tokens
    BYTE_OFFSET = NUM_SPECIAL  # bytes 0-255 map to tokens 64-319

    def __init__(self, vocab_size: int = 320):
        self.vocab_size = vocab_size
        self._tag_map = {
            "[TASK]": self.TASK_TAG,
            "[INTENT]": self.INTENT_TAG,
            "[CONSTRAINTS]": self.CONSTRAINTS_TAG,
            "[OUTPUT]": self.OUTPUT_TAG,
            "[INPUT_SCRIPT]": self.INPUT_SCRIPT_TAG,
            "[CANDIDATE_1]": self.CANDIDATE_TAG,
            "[CANDIDATE_2]": self.CANDIDATE_TAG,
            "[CANDIDATE_3]": self.CANDIDATE_TAG,
        }

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Encode a string into a list of token IDs.

        Recognizes structured prompt tags and replaces them with special tokens.
        All other characters are encoded as raw bytes.
        """
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.BOS)

        i = 0
        while i < len(text):
            # Check for structured tags
            matched = False
            if text[i] == "[":
                for tag, tok_id in self._tag_map.items():
                    if text[i : i + len(tag)] == tag:
                        tokens.append(tok_id)
                        i += len(tag)
                        matched = True
                        break
            if not matched:
                byte_val = ord(text[i]) if ord(text[i]) < 256 else ord("?")
                tokens.append(byte_val + self.BYTE_OFFSET)
                i += 1

        if add_eos:
            tokens.append(self.EOS)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        # Reverse tag map for decoding
        tag_strings = {
            self.TASK_TAG: "[TASK]",
            self.INTENT_TAG: "[INTENT]",
            self.CONSTRAINTS_TAG: "[CONSTRAINTS]",
            self.OUTPUT_TAG: "[OUTPUT]",
            self.INPUT_SCRIPT_TAG: "[INPUT_SCRIPT]",
            self.CANDIDATE_TAG: "[CANDIDATE]",
        }

        chars: list[str] = []
        for tid in token_ids:
            if tid in (self.PAD, self.BOS, self.EOS):
                continue
            if tid == self.SEP:
                chars.append("\n")
            elif tid in tag_strings:
                chars.append(tag_strings[tid])
            elif self.BYTE_OFFSET <= tid < self.BYTE_OFFSET + 256:
                chars.append(chr(tid - self.BYTE_OFFSET))
        return "".join(chars)

    def pad(
        self, token_ids: list[int], max_len: int, pad_id: int | None = None
    ) -> list[int]:
        """Pad or truncate a token list to max_len."""
        pid = pad_id if pad_id is not None else self.PAD
        if len(token_ids) >= max_len:
            return token_ids[:max_len]
        return token_ids + [pid] * (max_len - len(token_ids))

    def batch_encode(
        self,
        texts: list[str],
        max_len: int = 256,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[list[int]]:
        """Encode and pad a batch of strings."""
        return [
            self.pad(self.encode(t, add_bos=add_bos, add_eos=add_eos), max_len)
            for t in texts
        ]
