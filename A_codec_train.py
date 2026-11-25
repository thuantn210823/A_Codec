"""Multi-speaker speech-activity codec (conv autoencoder + RVQ bottleneck).

Each example is the speech activity of ALL speakers in a chunk:
    activity: (max_speakers, T),  T = chunk_seconds * fps  (default 30 s * 100 fps = 3000)
Rows are ordered by first onset (arrival order); absent speakers are zero rows.

Data: real AMI / NOTSOFAR Lhotse cut manifests (mt-asr chunking) and/or a
turn-taking conversation simulator.
Loss: reconstruction (BCE-with-logits or MSE) + RVQ commitment loss.
"""
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from lhotse import CutSet
from transformers import get_linear_schedule_with_warmup
from vector_quantize_pytorch import ResidualVQ


# ============================================================
# Real data: AMI / NOTSOFAR cut manifests -> chunks (mt-asr chunking)
# ============================================================
# Chunking constants ported from mt-asr-data-prep's
# ``pre_segment_using_alignments.py``.
ALIGNMENT_PUNCTUATION = "!\"#$%&()*+,./:;<=>?@[\\]^_`'{|}~"
ALIGNMENT_TRANSLATOR = str.maketrans("", "", ALIGNMENT_PUNCTUATION)
CHUNK_EPS = 1e-5


def _normalize_alignment_symbol(symbol: str) -> str:
    """Lowercase a token and strip alignment punctuation (mt-asr normalization)."""
    return symbol.translate(ALIGNMENT_TRANSLATOR).lower()


@dataclass(frozen=True)
class _Word:
    symbol: str
    start: float
    end: float


@dataclass(frozen=True)
class _Turn:
    id: str
    source_id: str
    speaker: str
    gender: Optional[str]
    start: float
    end: float
    words: Tuple[_Word, ...]


@dataclass(frozen=True)
class _Chunk:
    id: str
    session_index: int
    session_id: str
    start: float
    end: float
    turns: Tuple[_Turn, ...]
    # Speaker -> utterance unfinished at the chunk boundary (mt-asr per_spk_flags).
    per_spk_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class _Example:
    chunk: _Chunk
    speakers: Tuple[str, ...]          # speakers kept for this chunk (<= max_speakers)


def normalize_manifest_paths(
    manifest_path: Union[str, Path, Sequence[Union[str, Path]]]
) -> List[str]:
    """Normalize a manifest argument into an ordered list of manifest paths.

    Accepts a single ``str``/``Path`` or a sequence (list/tuple) of them.
    Sequence order is preserved (cuts are concatenated in that order) and
    duplicates are kept as-is; glob wildcards are **not** expanded.
    ``ValueError`` is raised for an empty argument and ``TypeError`` for
    elements that are not path-like.
    """
    if isinstance(manifest_path, (str, Path)):
        raw = [manifest_path]
    elif isinstance(manifest_path, Sequence):
        raw = list(manifest_path)
    else:
        raise TypeError(
            "manifest_path must be a str, Path or a sequence of them, "
            f"got {type(manifest_path).__name__}"
        )
    if not raw:
        raise ValueError(
            "manifest_path sequence is empty; provide at least one manifest path"
        )
    paths: List[str] = []
    for item in raw:
        if not isinstance(item, (str, Path)):
            raise TypeError(
                "manifest_path elements must be str or Path, "
                f"got {type(item).__name__}"
            )
        path = str(item)
        if not path:
            raise ValueError("manifest_path contains an empty path string")
        paths.append(path)
    return paths


def _order_rows_by_onset(activity: torch.Tensor) -> torch.Tensor:
    """Reorder speaker rows by first active frame; silent rows go last."""
    S, T = activity.shape
    active = activity > 0
    onset = torch.where(active.any(dim=1), active.float().argmax(dim=1),
                        torch.full((S,), T, dtype=torch.long))
    order = torch.argsort(onset, stable=True)
    return activity[order]


class MT_AMI_Activity_Dataset(torch.utils.data.Dataset):
    """Multi-speaker activity dataset backed by AMI/NOTSOFAR Lhotse cut manifests.

    Session turns are grouped into conversation segments split at pauses
    longer than ``max_pause`` and divided into chunks of at most
    ``chunk_seconds`` following mt-asr-data-prep's boundary strategy (see
    ``_split_pause_group``). Each chunk is one example; its item is the
    turn-level activity of every speaker, shape ``(max_speakers, T)`` with
    ``T = round(chunk_seconds * fps)``. Chunks with more than ``max_speakers``
    speakers keep the ones with the most speech.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path, Sequence[Union[str, Path]]],
        audio_root: Optional[str] = None,
        fps: int = 100,
        chunk_seconds: float = 30.0,
        max_pause: float = 2.0,
        max_speakers: int = 4,
        random_offset: bool = False,
    ) -> None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive.")
        if max_pause < 0:
            raise ValueError("max_pause must be non-negative.")
        if max_speakers <= 0:
            raise ValueError("max_speakers must be positive.")

        self.manifest_paths = normalize_manifest_paths(manifest_path)
        self.audio_root = str(audio_root) if audio_root is not None else None
        self.fps = int(fps)
        self.max_chunk_duration = float(chunk_seconds)
        self.num_frames = int(round(chunk_seconds * fps))
        self.max_pause = float(max_pause)
        self.max_speakers = int(max_speakers)
        self.random_offset = bool(random_offset)

        self.sessions = []
        for path in self.manifest_paths:
            cuts = CutSet.from_file(path)
            if self.audio_root:
                cuts = cuts.with_recording_path_prefix(Path(self.audio_root))
            sessions = list(cuts)
            self.sessions.extend(sessions)
            self._warn_missing_alignments(path, sessions)
            print(f"loaded {len(sessions)} cuts from {path}")

        self.chunks: List[_Chunk] = []
        self.examples: List[_Example] = []
        self.stats: Dict[str, int] = defaultdict(int)
        self._prepare_index()

    # ---------------- manifest parsing (unchanged) ----------------
    @staticmethod
    def _warn_missing_alignments(path: str, sessions: Sequence) -> None:
        """Warn when a manifest lacks word-level alignments (needed by chunking)."""
        total = 0
        missing = 0
        for session in sessions:
            for supervision in getattr(session, "supervisions", ()):
                total += 1
                alignment = getattr(supervision, "alignment", None) or {}
                if hasattr(alignment, "get"):
                    words = alignment.get("word")
                else:
                    words = getattr(alignment, "word", None)
                if not words:
                    missing += 1
        if missing:
            print(
                f"warning: {missing}/{total} supervisions in {path} have no "
                "word alignment (boundary-crossing turns without alignments "
                "are dropped by the chunker)"
            )

    @staticmethod
    def _alignment_words(supervision) -> Tuple[_Word, ...]:
        alignment = supervision.alignment or {}
        items = alignment.get("word", [])
        words: List[_Word] = []
        for item in items:
            if hasattr(item, "symbol"):
                symbol = str(item.symbol).strip()
                start = float(item.start)
                duration = float(item.duration)
            else:
                symbol = str(item[0]).strip()
                start = float(item[1])
                duration = float(item[2])
            # mt-asr drops alignment tokens that are only punctuation.
            if not _normalize_alignment_symbol(symbol):
                continue
            words.append(
                _Word(symbol=symbol, start=start, end=start + max(duration, 0.0))
            )
        return tuple(words)

    @classmethod
    def _turn_from_supervision(cls, supervision) -> _Turn:
        start = float(supervision.start)
        end = start + float(supervision.duration)
        return _Turn(
            id=str(supervision.id),
            source_id=str(supervision.id),
            speaker=str(supervision.speaker),
            gender=getattr(supervision, "gender", None),
            start=start,
            end=end,
            words=cls._alignment_words(supervision),
        )

    # ---------------- mt-asr chunking (unchanged) ----------------
    @staticmethod
    def _words_in_window(
        turn: _Turn, window_start: float, window_end: float
    ) -> List[_Word]:
        """Select words fully inside the inclusive window [window_start, window_end]."""
        kept: List[_Word] = []
        for word in turn.words:
            if word.end > window_end:
                break
            if word.start >= window_start:
                kept.append(word)
        return kept

    @staticmethod
    def _renamed_member(turn: _Turn, index: int, position: int) -> _Turn:
        return replace(turn, id=f"{turn.id}-{index}-{position}")

    def _group_turns_by_pause(self, turns: Sequence[_Turn]) -> List[List[_Turn]]:
        """Split turns into conversation groups separated by pauses > max_pause."""
        groups: List[List[_Turn]] = []
        current: List[_Turn] = []
        running_end = float("-inf")
        for turn in sorted(turns, key=lambda item: (item.start, item.end, item.id)):
            if current and turn.start - running_end > self.max_pause:
                groups.append(current)
                current = []
                running_end = float("-inf")
            current.append(turn)
            running_end = max(running_end, turn.end)
        if current:
            groups.append(current)
        return groups

    def _split_pause_group(
        self, group: Sequence[_Turn]
    ) -> List[Tuple[List[Tuple[_Turn, int]], Dict[str, bool]]]:
        """Split one pause group into chunks of at most ``max_chunk_duration``.

        Port of mt-asr ``_split_cut_perseg`` operating on ``_Turn`` objects.
        Boundary-crossing turns are truncated at the last word inside the
        window; their remaining words continue in the next chunk (rewind or
        seeded tails), so no word is lost.
        """
        sups = sorted(group, key=lambda item: (item.start, item.end, item.id))
        max_len = self.max_chunk_duration
        group_span = max(turn.end for turn in sups) - sups[0].start
        if group_span < max_len:
            return [([(turn, index) for index, turn in enumerate(sups)], {})]

        results: List[Tuple[List[Tuple[_Turn, int]], Dict[str, bool]]] = []
        segment: List[Tuple[_Turn, int]] = [(self._renamed_member(sups[0], 0, 0), 0)]
        segment_flags: Dict[str, bool] = {}
        is_falling_back = False

        def window_end(start: float) -> float:
            return start + max_len - CHUNK_EPS

        def finalize(closer: Optional[Tuple[_Turn, int]]) -> Optional[int]:
            nonlocal segment, segment_flags, is_falling_back
            start_idx = segment[0][1]
            seg_start = segment[0][0].start
            w_end = window_end(seg_start)
            exceeding_tails: List[Tuple[int, List[_Word]]] = []
            for position, (turn, orig_idx) in enumerate(segment):
                segment_flags[turn.speaker] = False
                if turn.end - seg_start <= max_len:
                    continue
                kept = self._words_in_window(turn, seg_start, w_end)
                tail_words = [
                    word for word in turn.words if word.start >= seg_start and word.end > w_end
                ]
                exceeding_tails.append((orig_idx, tail_words))
                if kept:
                    segment_flags[turn.speaker] = True
                    segment[position] = (
                        replace(turn, end=kept[-1].end, words=tuple(kept)),
                        orig_idx,
                    )
                    self.stats["truncated_turns"] += 1
                else:
                    segment[position] = None
            members = [entry for entry in segment if entry is not None]
            if members:
                results.append(
                    (
                        members,
                        {
                            speaker: segment_flags[speaker]
                            for speaker in {entry[0].speaker for entry in members}
                        },
                    )
                )
            segment = []
            segment_flags = {}
            is_falling_back = False

            if not exceeding_tails:
                if closer is not None:
                    segment = [
                        (self._renamed_member(closer[0], closer[1], 0), closer[1])
                    ]
                return None
            min_exceeder = min(orig_idx for orig_idx, _ in exceeding_tails)
            if min_exceeder > start_idx and sups[min_exceeder].start > seg_start:
                is_falling_back = True
                return min_exceeder
            for orig_idx, tail_words in exceeding_tails:
                if tail_words and tail_words[0].start > seg_start:
                    original = sups[orig_idx]
                    segment.append(
                        (
                            replace(
                                original,
                                id=(
                                    f"{original.id}-{orig_idx}-tail-"
                                    f"{tail_words[0].start:.3f}"
                                ),
                                start=tail_words[0].start,
                                end=tail_words[-1].end,
                                words=tuple(tail_words),
                            ),
                            orig_idx,
                        )
                    )
                    self.stats["tail_continuations"] += 1
                else:
                    self.stats["dropped_boundary_turns"] += 1
            if closer is not None:
                segment.append(
                    (self._renamed_member(closer[0], closer[1], len(segment)), closer[1])
                )
            segment.sort(key=lambda entry: (entry[0].start, entry[0].end, entry[0].id))
            return None

        idx = 1
        while idx <= len(sups) or segment:
            sup = sups[idx] if idx < len(sups) else None
            if sup is not None and (
                not segment or sup.start - segment[0][0].start < max_len
            ):
                segment.append((self._renamed_member(sup, idx, len(segment)), idx))
                if is_falling_back and results:
                    previous_members, _ = results[-1]
                    for ovl_turn, ovl_idx in previous_members:
                        if not (ovl_turn.start <= sup.start < ovl_turn.end):
                            continue
                        if (
                            sup.source_id == ovl_turn.source_id
                            or ovl_turn.speaker == sup.speaker
                        ):
                            continue
                        original = sups[ovl_idx]
                        kept = self._words_in_window(
                            original, sup.start, window_end(sup.start)
                        )
                        if not kept:
                            continue
                        fragment = replace(
                            original,
                            id=f"{original.id}-{ovl_idx}-{len(segment)}_ovl",
                            start=kept[0].start,
                            end=kept[-1].end,
                            words=tuple(kept),
                        )
                        segment.append((fragment, ovl_idx))
                        self.stats["ovl_fragments"] += 1
                is_falling_back = False
                idx += 1
            else:
                if sup is None and not segment:
                    break
                closer = (sup, idx) if sup is not None else None
                rewind = finalize(closer)
                if rewind is not None:
                    idx = rewind
                else:
                    idx += 1
        return results

    def _make_chunks(
        self, session_index: int, session_id: str, turns: Sequence[_Turn]
    ) -> List[_Chunk]:
        chunks: List[_Chunk] = []
        for group in self._group_turns_by_pause(turns):
            for members, flags in self._split_pause_group(group):
                if not members:
                    continue
                chunk_turns = [turn for turn, _ in members]
                chunk_start = min(turn.start for turn in chunk_turns)
                chunk_end = max(turn.end for turn in chunk_turns)
                assert chunk_end - chunk_start <= self.max_chunk_duration
                start_sample = round(chunk_start * 16000)
                end_sample = round(chunk_end * 16000)
                chunks.append(
                    _Chunk(
                        id=f"{session_id}_{start_sample}-{end_sample}",
                        session_index=session_index,
                        session_id=session_id,
                        start=chunk_start,
                        end=chunk_end,
                        turns=tuple(
                            sorted(chunk_turns, key=lambda item: (item.start, item.end))
                        ),
                        per_spk_flags=flags,
                    )
                )
        return chunks

    # ---------------- index: one example per chunk ----------------
    @staticmethod
    def _rank_speakers_by_speech(turns: Sequence[_Turn]) -> List[str]:
        """Rank speakers by total speech duration in the chunk, then by ID."""
        total: Dict[str, float] = defaultdict(float)
        for turn in turns:
            total[turn.speaker] += turn.end - turn.start
        return sorted(total, key=lambda speaker: (-total[speaker], speaker))

    def _prepare_index(self) -> None:
        for session_index, session in enumerate(self.sessions):
            turns = tuple(
                self._turn_from_supervision(supervision)
                for supervision in session.supervisions
                if supervision.speaker is not None
            )
            session_id = str(getattr(session.recording, "id", session.id))
            self.chunks.extend(self._make_chunks(session_index, session_id, turns))

        seen_chunk_ids: Dict[str, int] = {}
        for chunk in self.chunks:
            previous_session = seen_chunk_ids.get(chunk.id)
            if previous_session is not None:
                raise ValueError(
                    f"Duplicate chunk id {chunk.id!r} produced by sessions "
                    f"{previous_session} and {chunk.session_index}; each "
                    "manifest must contribute distinct recording ids (the "
                    "same manifest cannot be passed twice)."
                )
            seen_chunk_ids[chunk.id] = chunk.session_index

        for chunk in self.chunks:
            ranked = self._rank_speakers_by_speech(chunk.turns)
            kept = ranked[: self.max_speakers]
            self.stats["dropped_speakers_cap"] += len(ranked) - len(kept)
            self.stats[f"chunks_with_{min(len(ranked), self.max_speakers)}_spk"] += 1
            self.examples.append(_Example(chunk=chunk, speakers=tuple(kept)))

        self.stats["sessions"] = len(self.sessions)
        self.stats["chunks"] = len(self.chunks)
        self.stats["examples"] = len(self.examples)

    # ---------------- activity ----------------
    def _activity(self, example: _Example, offset: float = 0.0) -> torch.Tensor:
        """Turn-level activity (as the original _stno), shape (max_speakers, T)."""
        chunk = example.chunk
        T = self.num_frames
        activity = torch.zeros(self.max_speakers, T)
        row = {speaker: i for i, speaker in enumerate(example.speakers)}
        for turn in chunk.turns:
            if turn.speaker not in row:
                continue
            start = int(round((turn.start - chunk.start + offset) * self.fps))
            end = int(round((turn.end - chunk.start + offset) * self.fps))
            start, end = max(0, start), min(T, end)
            if end > start:
                activity[row[turn.speaker], start:end] = 1.0
        return _order_rows_by_onset(activity)

    def __getitem__(self, index: int) -> torch.Tensor:
        example = self.examples[index]
        offset = 0.0
        if self.random_offset:
            slack = self.max_chunk_duration - (example.chunk.end - example.chunk.start)
            if slack > 0:
                offset = random.uniform(0.0, slack)
        return self._activity(example, offset)

    def __len__(self) -> int:
        return len(self.examples)


# ============================================================
# Simulated data: turn-taking conversation simulator
# ============================================================
class ConversationSimulator(torch.utils.data.Dataset):
    """Simulates multi-speaker activity, shape (max_speakers, T).

    Generative process (in seconds, then rasterised at ``fps``):
      * K in 1..max_speakers active speakers (1% fully silent chunks)
      * per-speaker profile: dominance (Dirichlet), median utterance length,
        backchannel rate, optional late join, optional listener-only speaker
      * floor-holder turn-taking with conversation-analysis-like transitions:
        short gaps (~0.2 s), ~25% overlapping starts, occasional long silences
      * utterances split into phrases with short intra-turn pauses
      * backchannels from listeners overlapping the floor holder
      * sometimes a shorter effective length (trailing silence), like real chunks
    Rows are ordered by first onset; unused speakers are zero rows.
    """

    def __init__(self, num_samples: int, fps: int = 100, chunk_seconds: float = 30.0,
                 max_speakers: int = 4, speaker_count_weights: Optional[Sequence[float]] = None,
                 min_segment: float = 0.1, seed: Optional[int] = None):
        super().__init__()
        self.num_samples = int(num_samples)
        self.fps = int(fps)
        self.chunk_seconds = float(chunk_seconds)
        self.num_frames = int(round(chunk_seconds * fps))
        self.max_speakers = int(max_speakers)
        self.min_segment = float(min_segment)
        self.seed = seed                       # None -> fresh randomness every item (training)
        if speaker_count_weights is None:
            if self.max_speakers == 4:
                speaker_count_weights = [0.15, 0.35, 0.30, 0.20]
            else:
                speaker_count_weights = [0.5] + [1.0] * (self.max_speakers - 1)
        w = np.asarray(speaker_count_weights, dtype=np.float64)
        assert len(w) == self.max_speakers, "speaker_count_weights must have max_speakers entries"
        self.speaker_count_weights = w / w.sum()

    # ---------------- generative process ----------------
    def _simulate(self, rng: np.random.Generator) -> List[Tuple[int, float, float]]:
        if rng.random() < 0.01:
            return []                                              # fully silent chunk

        # Effective length: real chunks are often shorter than the window
        eff = self.chunk_seconds if rng.random() < 0.7 else rng.uniform(5.0, self.chunk_seconds)
        K = int(rng.choice(np.arange(1, self.max_speakers + 1), p=self.speaker_count_weights))

        dominance = rng.dirichlet([1.5] * K)
        median_utt = np.exp(rng.normal(np.log(2.0), 0.4, K))       # ~1-4 s
        bc_rate = rng.uniform(0.0, 0.25, K)                        # backchannels / s while listening
        join = np.where(rng.random(K) < 0.15, rng.uniform(0.0, 0.7 * eff, K), 0.0)

        if K >= 2 and rng.random() < 0.1:                          # listener-only speaker
            k = int(rng.integers(K))
            dominance[k] = 0.0
            bc_rate[k] = rng.uniform(0.2, 0.5)
            dominance = dominance / dominance.sum()
        join[int(np.argmax(dominance))] = 0.0                      # someone talks from the start

        segments: List[Tuple[int, float, float]] = []
        floor_spans: List[Tuple[int, float, float]] = []
        t = rng.exponential(0.5)                                   # leading silence
        current = None

        while t < eff:
            available = [k for k in range(K) if join[k] <= t and dominance[k] > 0]
            if not available:
                pending = [join[k] for k in range(K) if dominance[k] > 0 and join[k] > t]
                if not pending:
                    break
                t = min(pending)
                continue

            # Next floor holder: mostly a different speaker than the current one
            if current is not None and len(available) > 1 and rng.random() > 0.15:
                candidates = [k for k in available if k != current]
            else:
                candidates = available
            p = dominance[candidates] / dominance[candidates].sum()
            speaker = int(rng.choice(candidates, p=p))

            duration = float(np.clip(rng.lognormal(np.log(median_utt[speaker]), 0.6), 0.2, 15.0))
            start, end = t, min(t + duration, eff)
            floor_spans.append((speaker, start, end))

            # Phrases separated by short intra-turn pauses
            pos = start
            while pos < end:
                phrase_end = min(pos + rng.lognormal(np.log(1.2), 0.5), end)
                if phrase_end - pos >= self.min_segment:
                    segments.append((speaker, pos, phrase_end))
                pos = phrase_end + rng.uniform(0.1, 0.4)

            # Transition to the next turn
            if speaker == current:
                gap = rng.uniform(0.3, 1.5)                        # same speaker continues after a pause
            else:
                r = rng.random()
                if r < 0.25:                                       # overlapping start / interruption
                    gap = -rng.uniform(0.0, min(1.0, 0.5 * (end - start)))
                elif r < 0.9:                                      # normal short gap
                    gap = max(0.0, rng.normal(0.2, 0.15))
                else:                                              # long silence
                    gap = rng.exponential(2.0)
            current = speaker
            t = end + gap

        # Backchannels from listeners while someone holds the floor
        for holder, start, end in floor_spans:
            for k in range(K):
                if k == holder or join[k] > start:
                    continue
                for _ in range(rng.poisson(bc_rate[k] * (end - start))):
                    bc_start = rng.uniform(start, end)
                    bc_end = min(bc_start + rng.uniform(0.2, 0.8), eff)
                    if bc_end - bc_start >= self.min_segment:
                        segments.append((k, bc_start, bc_end))

        return segments

    def _rasterize(self, segments: List[Tuple[int, float, float]]) -> torch.Tensor:
        T = self.num_frames
        activity = torch.zeros(self.max_speakers, T)
        for speaker, start, end in segments:
            s = max(0, int(round(start * self.fps)))
            e = min(T, int(round(end * self.fps)))
            if e > s:
                activity[speaker, s:e] = 1.0
        return _order_rows_by_onset(activity)

    def __getitem__(self, index: int) -> torch.Tensor:
        rng = np.random.default_rng(None if self.seed is None else self.seed + index)
        return self._rasterize(self._simulate(rng))

    def __len__(self) -> int:
        return self.num_samples


# ============================================================
# Data module
# ============================================================
def collate_activity(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)                               # (B, max_speakers, T)


class ActivityData(L.LightningDataModule):
    def __init__(self, batch_size: int, train_manifests: Sequence[str], val_manifests: Sequence[str],
                 train_source: str = "both", simu_samples: int = 100_000, fps: int = 100,
                 chunk_seconds: float = 30.0, max_speakers: int = 4, num_workers: int = 2):
        super().__init__()
        if train_source not in {"both", "real", "simu"}:
            raise ValueError(f"train_source must be 'both', 'real' or 'simu', got {train_source!r}")
        self.batch_size = batch_size
        self.num_workers = num_workers
        common = dict(fps=fps, chunk_seconds=chunk_seconds, max_speakers=max_speakers)

        train_sets = []
        if train_source in {"both", "real"}:
            real_train = MT_AMI_Activity_Dataset(train_manifests, random_offset=True, **common)
            print("real train stats:", dict(real_train.stats))
            train_sets.append(real_train)
        if train_source in {"both", "simu"}:
            train_sets.append(ConversationSimulator(simu_samples, **common))
        self.train_dataset = (train_sets[0] if len(train_sets) == 1
                              else torch.utils.data.ConcatDataset(train_sets))

        self.val_dataset = MT_AMI_Activity_Dataset(val_manifests, random_offset=False, **common)
        print("real val stats:", dict(self.val_dataset.stats))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           collate_fn=collate_activity)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           collate_fn=collate_activity)


# ============================================================
# Model: conv autoencoder with RVQ bottleneck
# ============================================================
class Encoder(nn.Module):
    """(B, S, T) -> (B, 256, T/32). T must be a multiple of 32 (padded by the codec)."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class Decoder(nn.Module):
    """(B, 256, T/32) -> logits (B, S, T). ConvTranspose(k=4, s=2, p=1) is an exact 2x upsample."""
    def __init__(self, out_channels: int):
        super().__init__()
        self.input_norm = nn.Sequential(nn.LayerNorm(256), nn.SiLU())
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            # Conv1D refinements -> logits (no Sigmoid: use BCE-with-logits)
            nn.Conv1d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.SiLU(),
            nn.Conv1d(16, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.input_norm(z.transpose(1, 2)).transpose(1, 2)
        return self.deconv_block(z)


class ActivityRVQCodec(nn.Module):
    STRIDE = 32                                            # 5 stride-2 stages

    def __init__(self, max_speakers: int = 4, latent_dim: int = 64,
                 n_q: int = 8, codebook_size: int = 1024):
        super().__init__()
        self.max_speakers = max_speakers
        self.encoder = Encoder(in_channels=max_speakers)
        self.pre_vq = nn.Conv1d(256, latent_dim, kernel_size=1)
        # Same RVQ configuration as soundstream_train_encodecloss.py
        self.quantizer = ResidualVQ(
            num_quantizers=n_q, dim=latent_dim, codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=100,
            dead_code_threshold_ratio=0.1,
            decay=0.99,
            quantize_dropout=True,
            quantize_dropout_cutoff_index=0,
        )
        self.post_vq = nn.Conv1d(latent_dim, 256, kernel_size=1)
        self.decoder = Decoder(out_channels=max_speakers)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        pad = (-x.shape[-1]) % self.STRIDE
        return F.pad(x, (0, pad)) if pad else x

    def _decode_latent(self, q: torch.Tensor, T: int) -> torch.Tensor:
        # q: (B, L, D) -> logits (B, S, T)
        return self.decoder(self.post_vq(q.transpose(1, 2)))[..., :T]

    def forward(self, x: torch.Tensor):
        T = x.shape[-1]
        z = self.pre_vq(self.encoder(self._pad(x))).transpose(1, 2)        # (B, L, D)
        quantized, indices, commit_loss = self.quantizer(z)
        logits = self._decode_latent(quantized, T)
        return logits, indices, commit_loss.sum()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Activity (B, S, T) -> codes (B, L, n_q). Call in eval mode."""
        z = self.pre_vq(self.encoder(self._pad(x))).transpose(1, 2)
        _, indices, _ = self.quantizer(z)
        return indices

    @torch.no_grad()
    def decode(self, indices: torch.Tensor, T: Optional[int] = None) -> torch.Tensor:
        """Codes (B, L, n_q) -> activity probabilities (B, S, T)."""
        T = T if T is not None else indices.shape[1] * self.STRIDE
        quantized = self.quantizer.get_output_from_indices(indices)
        return torch.sigmoid(self._decode_latent(quantized, T))


# ============================================================
# Lightning module
# ============================================================
class ActivityCodecTask(L.LightningModule):
    def __init__(self, codec: ActivityRVQCodec, rec_loss: str = "bce", lambda_commit: float = 1.0,
                 pos_weight: Optional[float] = None, lr: float = 1e-4, warmup_steps: int = 0):
        super().__init__()
        if rec_loss not in {"bce", "mse"}:
            raise ValueError(f"rec_loss must be 'bce' or 'mse', got {rec_loss!r}")
        self.save_hyperparameters(ignore=["codec"])
        self.codec = codec
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)), persistent=False)
        else:
            self.pos_weight = None

    def _rec_loss(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.hparams.rec_loss == "bce":
            return F.binary_cross_entropy_with_logits(logits, x, pos_weight=self.pos_weight)
        return F.mse_loss(torch.sigmoid(logits), x)

    @torch.no_grad()
    def _metrics(self, logits: torch.Tensor, x: torch.Tensor, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = logits > 0                                    # sigmoid > 0.5
        target = x > 0.5
        tp = (pred & target).sum().float()
        fp = (pred & ~target).sum().float()
        fn = (~pred & target).sum().float()
        codes = indices[..., 0]                              # first codebook (never dropped)
        usage = codes[codes >= 0].unique().numel() / self.codec.quantizer.codebook_size
        return {
            "acc": (pred == target).float().mean(),
            "f1": 2 * tp / (2 * tp + fp + fn + 1e-8),
            "use": torch.tensor(float(usage), device=logits.device),
        }

    def _step(self, batch: torch.Tensor):
        x = batch                                            # (B, S, T)
        logits, indices, commit = self.codec(x)
        rec = self._rec_loss(logits, x)
        loss = rec + self.hparams.lambda_commit * commit
        return loss, rec, commit, self._metrics(logits, x, indices)

    def training_step(self, batch, batch_idx):
        loss, rec, commit, m = self._step(batch)
        self.log_dict({"tl": loss, "tf1": m["f1"]},
                      on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict({"trec": rec, "tc": commit, "tacc": m["acc"], "tuse": m["use"]},
                      on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, rec, commit, m = self._step(batch)
        self.log_dict({"vl": loss, "vf1": m["f1"]},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict({"vrec": rec, "vc": commit, "vacc": m["acc"], "vuse": m["use"]},
                      on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, eps=1e-5)
        if self.hparams.warmup_steps <= 0:
            return optimizer
        total_steps = self.trainer.estimated_stepping_batches
        logging.info(f"Estimated stepping batches: {total_steps}")
        scheduler = get_linear_schedule_with_warmup(optimizer, self.hparams.warmup_steps, total_steps)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ============================================================
# Progress bar
# ============================================================
class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        # Định dạng này bỏ {rate_fmt} để không hiện it/s
        bar.bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        return bar


if __name__ == "__main__":
    # ---------------- config ----------------
    FPS = 100                  # frames per second (30 s -> 3000 frames)
    CHUNK_SECONDS = 30.0
    MAX_SPEAKERS = 4
    TRAIN_SOURCE = "both"      # "both" | "real" | "simu"
    SIMU_SAMPLES = 100_000
    REC_LOSS = "bce"           # "bce" | "mse"
    LAMBDA_COMMIT = 1.0
    LATENT_DIM = 64
    N_Q = 8
    CODEBOOK_SIZE = 1024
    BATCH_SIZE = 64
    LR = 1e-4
    WARMUP_STEPS = 1000        # 0 -> constant lr

    TRAIN_MANIFESTS = [
        '/kaggle/input/datasets/ngcthun/mt-ami/data/manifests/ami/ami-sdm_cutset_train.jsonl',
        '/kaggle/input/datasets/ngcthun/mt-notsofar/data/manifests/notsofar1/notsofar1_sdm_train_set_240825.1_train_cutset.jsonl',
        '/kaggle/input/datasets/ngcthun/mt-ami/data/manifests/ami/ami-sdm_cutset_dev.jsonl',
        '/kaggle/input/datasets/ngcthun/mt-notsofar/data/manifests/notsofar1/notsofar1_sdm_dev_set_240825.1_dev1_cutset.jsonl',
    ]
    VAL_MANIFESTS = [
        '/kaggle/input/datasets/ngcthun/mt-ami/data/manifests/ami/ami-sdm_cutset_test.jsonl',
        '/kaggle/input/datasets/ngcthun/mt-notsofar/data/manifests/notsofar1/notsofar1_sdm_eval_set_240629.1_eval_small_with_GT_cutset.jsonl/notsofar1_sdm_eval_set_240629.1_eval_small_with_GT_cutset.jsonl',
    ]

    L.seed_everything(42, workers=True)

    codec = ActivityRVQCodec(max_speakers=MAX_SPEAKERS, latent_dim=LATENT_DIM,
                             n_q=N_Q, codebook_size=CODEBOOK_SIZE)
    model = ActivityCodecTask(codec, rec_loss=REC_LOSS, lambda_commit=LAMBDA_COMMIT,
                              lr=LR, warmup_steps=WARMUP_STEPS)
    data = ActivityData(BATCH_SIZE, TRAIN_MANIFESTS, VAL_MANIFESTS,
                        train_source=TRAIN_SOURCE, simu_samples=SIMU_SAMPLES,
                        fps=FPS, chunk_seconds=CHUNK_SECONDS, max_speakers=MAX_SPEAKERS)

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    checkpoint_callback = ModelCheckpoint(dirpath='/kaggle/working/checkpoint2',
                                          save_top_k=10, monitor='vl',
                                          mode='min',
                                          filename='ActivityCodec-{epoch}-{vl:.10f}')
    logger = TensorBoardLogger("/kaggle/working/log", name="ActivityCodec", version='version_0')

    trainer = L.Trainer(accelerator="gpu",
                        callbacks=[checkpoint_callback, CustomProgressBar()],
                        enable_checkpointing=True,
                        max_epochs=5,
                        # precision='16-mixed',
                        logger=logger,
                        strategy='ddp_find_unused_parameters_true',
                        val_check_interval=0.1,
                        num_sanity_val_steps=0,       # don't let val data k-means-init the codebooks
                        devices=[0, 1])
    trainer.fit(model, data)
