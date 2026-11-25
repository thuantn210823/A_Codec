"""Multi-speaker speech-activity codec (conv AE + RVQ) -- difficulty-balanced training.

Same model/data as A_codec_train.py, plus:
  * in-validation frame-level DER (MS/FA/CS), broken down by difficulty bucket
    (easy / medium / hard) and by number of speakers, + optional n_q sweep
  * data balancing: difficulty-weighted sampling of real chunks, simulator
    `density` knob, activity-mixing augmentation (MixAugment)
  * loss balancing: boundary/overlap-weighted BCE (or MSE), optional focal term,
    speaker-count consistency loss
  * multi-stage curriculum (STAGES): warm-up -> densify -> real fine-tune,
    with per-stage quantize-dropout probability and LR scale

Each example: activity (max_speakers, T), T = chunk_seconds * fps (default 3000),
rows ordered by first onset, absent speakers = zero rows.
"""
import logging
import math
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
from pytorch_lightning.utilities import rank_zero_info
from lhotse import CutSet
from vector_quantize_pytorch import ResidualVQ


# ============================================================
# Difficulty & DER helpers
# ============================================================
BUCKET_NAMES = ("easy", "med", "hard")


def activity_difficulty(x: torch.Tensor, fps: int):
    """x: (B, S, T) binary -> (n_speakers, overlap_ratio, transitions_per_second), each (B,)."""
    x = (x > 0.5).float()
    n_spk = (x.sum(-1) > 0).sum(1)
    count = x.sum(1)                                            # (B, T) active speakers per frame
    speech = (count > 0).sum(-1).float()
    overlap = (count >= 2).sum(-1).float()
    overlap_ratio = overlap / speech.clamp(min=1.0)
    transitions = (x[..., 1:] != x[..., :-1]).sum((1, 2)).float()
    tps = transitions / (x.shape[-1] / fps)
    return n_spk, overlap_ratio, tps


def difficulty_bucket(x: torch.Tensor, fps: int) -> torch.Tensor:
    """0 = easy (<=2 spk, overlap < 5%), 2 = hard (>=3 spk and (overlap >= 15% or >= 1 transition/s)), 1 = medium."""
    n_spk, ovl, tps = activity_difficulty(x, fps)
    bucket = torch.ones_like(n_spk)
    bucket[(n_spk <= 2) & (ovl < 0.05)] = 0
    bucket[(n_spk >= 3) & ((ovl >= 0.15) | (tps >= 1.0))] = 2
    return bucket


def frame_der_stats(pred: torch.Tensor, ref: torch.Tensor):
    """Frame-level diarization errors with rows already aligned (no collar).

    pred, ref: (B, S, T) binary. Returns per-sample sums (ms, fa, cs, ref_speech), each (B,).
    """
    pred, ref = pred > 0.5, ref > 0.5
    n_ref = ref.sum(1).float()
    n_sys = pred.sum(1).float()
    n_cor = (pred & ref).sum(1).float()
    ms = (n_ref - n_sys).clamp(min=0).sum(-1)
    fa = (n_sys - n_ref).clamp(min=0).sum(-1)
    cs = (torch.minimum(n_ref, n_sys) - n_cor).sum(-1)
    return ms, fa, cs, n_ref.sum(-1)


def _order_rows_by_onset(activity: torch.Tensor) -> torch.Tensor:
    """Reorder speaker rows by first active frame; silent rows go last."""
    S, T = activity.shape
    active = activity > 0
    onset = torch.where(active.any(dim=1), active.float().argmax(dim=1),
                        torch.full((S,), T, dtype=torch.long))
    order = torch.argsort(onset, stable=True)
    return activity[order]


# ============================================================
# Curriculum stages
# ============================================================
def stage_at(stages: Sequence[dict], epoch: int):
    """Return (stage, stage_index, density) for a global epoch (last stage repeats)."""
    start = 0
    for i, stage in enumerate(stages):
        n = int(stage["epochs"])
        if epoch < start + n or i == len(stages) - 1:
            k = min(max(epoch - start, 0), n - 1)
            d0, d1 = stage.get("density", (0.5, 0.5))
            density = d0 if n <= 1 else d0 + (d1 - d0) * k / (n - 1)
            return stage, i, float(density)
        start += n
    raise ValueError("STAGES is empty")


# ============================================================
# Real data: AMI / NOTSOFAR cut manifests -> chunks (mt-asr chunking)
# ============================================================
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
    per_spk_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class _Example:
    chunk: _Chunk
    speakers: Tuple[str, ...]          # speakers kept for this chunk (<= max_speakers)


def normalize_manifest_paths(
    manifest_path: Union[str, Path, Sequence[Union[str, Path]]]
) -> List[str]:
    """Normalize a manifest argument into an ordered list of manifest paths."""
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


class MT_AMI_Activity_Dataset(torch.utils.data.Dataset):
    """Multi-speaker activity dataset backed by AMI/NOTSOFAR Lhotse cut manifests.

    One example per chunk (mt-asr chunking); item = turn-level activity of every
    kept speaker, shape (max_speakers, T). ``self.buckets`` holds each example's
    difficulty bucket (0 easy, 1 medium, 2 hard) for balanced sampling.
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
        self.buckets: List[int] = []
        self.stats: Dict[str, int] = defaultdict(int)
        self._prepare_index()

    # ---------------- manifest parsing (unchanged) ----------------
    @staticmethod
    def _warn_missing_alignments(path: str, sessions: Sequence) -> None:
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

        # Difficulty bucket of every example (from its offset-0 activity)
        for example in self.examples:
            bucket = int(difficulty_bucket(self._activity(example)[None], self.fps)[0])
            self.buckets.append(bucket)
            self.stats[f"bucket_{BUCKET_NAMES[bucket]}"] += 1

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
# Simulated data: turn-taking conversation simulator with density control
# ============================================================
class ConversationSimulator(torch.utils.data.Dataset):
    """Simulates multi-speaker activity (max_speakers, T).

    ``density`` in [0, 1] interpolates between an easy regime (few speakers,
    rare overlap, long turns) and a hard one (3-4 speakers, frequent overlap,
    many backchannels, short fast exchanges, parallel side conversations):

        speaker-count weights   [.30,.40,.20,.10] -> [.05,.15,.35,.45]   (max_speakers = 4)
        overlapping transition  0.15 -> 0.40
        max backchannel rate    0.15 -> 0.50 /s
        median utterance        ~2.5 s -> ~1.2 s
        parallel conversation   0.0 -> 0.3
    """

    EASY_COUNT_W4 = (0.30, 0.40, 0.20, 0.10)
    HARD_COUNT_W4 = (0.05, 0.15, 0.35, 0.45)

    def __init__(self, num_samples: int, fps: int = 100, chunk_seconds: float = 30.0,
                 max_speakers: int = 4, density: float = 0.5,
                 min_segment: float = 0.1, seed: Optional[int] = None):
        super().__init__()
        self.num_samples = int(num_samples)
        self.fps = int(fps)
        self.chunk_seconds = float(chunk_seconds)
        self.num_frames = int(round(chunk_seconds * fps))
        self.max_speakers = int(max_speakers)
        self.min_segment = float(min_segment)
        self.seed = seed                       # None -> fresh randomness every item (training)
        self.set_density(density)

    def set_density(self, density: float) -> None:
        d = float(np.clip(density, 0.0, 1.0))
        self.density = d
        lerp = lambda a, b: a + (b - a) * d
        S = self.max_speakers
        if S == 4:
            easy, hard = np.array(self.EASY_COUNT_W4), np.array(self.HARD_COUNT_W4)
        else:
            k = np.arange(1, S + 1, dtype=np.float64)
            easy, hard = (S + 1 - k), k
            easy, hard = easy / easy.sum(), hard / hard.sum()
        w = lerp(easy, hard)
        self.speaker_count_weights = w / w.sum()
        self.p_overlap = lerp(0.15, 0.40)
        self.bc_rate_max = lerp(0.15, 0.50)
        self.utt_median = math.exp(lerp(math.log(2.5), math.log(1.2)))
        self.p_parallel = lerp(0.0, 0.3)

    # ---------------- generative process ----------------
    def _turn_taking(self, rng, speakers, dominance, median_utt, join, eff):
        """Floor-holder turn-taking among ``speakers``. Returns (segments, floor_spans)."""
        segments, floor_spans = [], []
        talkers = [k for k in speakers if dominance[k] > 0]
        if not talkers:
            return segments, floor_spans
        t = rng.exponential(0.5)                                   # leading silence
        current = None
        while t < eff:
            available = [k for k in talkers if join[k] <= t]
            if not available:
                pending = [join[k] for k in talkers if join[k] > t]
                if not pending:
                    break
                t = min(pending)
                continue

            if current is not None and len(available) > 1 and rng.random() > 0.15:
                candidates = [k for k in available if k != current]
            else:
                candidates = available
            p = dominance[candidates] / dominance[candidates].sum()
            speaker = int(rng.choice(candidates, p=p))

            duration = float(np.clip(rng.lognormal(np.log(median_utt[speaker]), 0.6), 0.2, 15.0))
            start, end = t, min(t + duration, eff)
            floor_spans.append((speaker, start, end))

            pos = start                                            # phrases + intra-turn pauses
            while pos < end:
                phrase_end = min(pos + rng.lognormal(np.log(1.2), 0.5), end)
                if phrase_end - pos >= self.min_segment:
                    segments.append((speaker, pos, phrase_end))
                pos = phrase_end + rng.uniform(0.1, 0.4)

            if speaker == current:
                gap = rng.uniform(0.3, 1.5)
            else:
                r = rng.random()
                if r < self.p_overlap:                             # overlapping start / interruption
                    gap = -rng.uniform(0.0, min(1.0, 0.5 * (end - start)))
                elif r < self.p_overlap + 0.65 * (1 - self.p_overlap):
                    gap = max(0.0, rng.normal(0.2, 0.15))          # normal short gap
                else:
                    gap = rng.exponential(2.0 * (1.0 - 0.5 * self.density))   # long silence
            current = speaker
            t = end + gap
        return segments, floor_spans

    def _simulate(self, rng: np.random.Generator) -> List[Tuple[int, float, float]]:
        if rng.random() < 0.01:
            return []                                              # fully silent chunk

        eff = self.chunk_seconds if rng.random() < 0.7 else rng.uniform(5.0, self.chunk_seconds)
        K = int(rng.choice(np.arange(1, self.max_speakers + 1), p=self.speaker_count_weights))

        dominance = rng.dirichlet([1.5] * K)
        median_utt = np.exp(rng.normal(np.log(self.utt_median), 0.4, K))
        bc_rate = rng.uniform(0.0, self.bc_rate_max, K)
        join = np.where(rng.random(K) < 0.15, rng.uniform(0.0, 0.7 * eff, K), 0.0)

        if K >= 2 and rng.random() < 0.1:                          # listener-only speaker
            k = int(rng.integers(K))
            dominance[k] = 0.0
            bc_rate[k] = rng.uniform(0.2, max(0.5, self.bc_rate_max))
            dominance = dominance / dominance.sum()

        # Parallel side conversation: two subgroups take turns independently
        groups = [list(range(K))]
        if K >= 3 and rng.random() < self.p_parallel:
            perm = list(rng.permutation(K))
            split = int(rng.integers(1, K))
            g1, g2 = perm[:split], perm[split:]
            if any(dominance[k] > 0 for k in g1) and any(dominance[k] > 0 for k in g2):
                groups = [g1, g2]
        for g in groups:                                           # someone talks from the start
            talkers = [k for k in g if dominance[k] > 0]
            if talkers:
                join[max(talkers, key=lambda k: dominance[k])] = 0.0

        segments: List[Tuple[int, float, float]] = []
        floor_spans: List[Tuple[int, float, float]] = []
        for g in groups:
            seg, spans = self._turn_taking(rng, g, dominance, median_utt, join, eff)
            segments += seg
            floor_spans += spans

        for holder, start, end in floor_spans:                     # backchannels from listeners
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
# Activity-mixing augmentation
# ============================================================
class MixAugment(torch.utils.data.Dataset):
    """With prob ``p_mix``, stack the active rows of a random partner sample onto
    this one (only if the total number of speakers fits in ``max_speakers``)."""

    def __init__(self, dataset: torch.utils.data.Dataset, p_mix: float, max_speakers: int,
                 tries: int = 5):
        super().__init__()
        self.dataset = dataset
        self.p_mix = float(p_mix)
        self.max_speakers = int(max_speakers)
        self.tries = int(tries)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        x = self.dataset[index]
        if random.random() >= self.p_mix:
            return x
        rows_x = x[x.sum(-1) > 0]
        k1 = rows_x.shape[0]
        if k1 >= self.max_speakers:
            return x
        for _ in range(self.tries):
            y = self.dataset[random.randrange(len(self.dataset))]
            rows_y = y[y.sum(-1) > 0]
            k2 = rows_y.shape[0]
            if 0 < k2 <= self.max_speakers - k1:
                out = torch.zeros_like(x)
                out[:k1] = rows_x
                out[k1:k1 + k2] = rows_y
                return _order_rows_by_onset(out)
        return x


# ============================================================
# Data module (rebuilds the train loader every epoch for the curriculum)
# ============================================================
def collate_activity(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)                               # (B, max_speakers, T)


def _bucket_weights(buckets: Sequence[int], alpha: float) -> np.ndarray:
    """Per-example weights ∝ freq(bucket)^-alpha, normalised to sum 1."""
    b = np.asarray(buckets)
    counts = np.bincount(b, minlength=len(BUCKET_NAMES)).astype(np.float64)
    w = np.power(np.maximum(counts[b], 1.0), -float(alpha))
    return w / w.sum()


class ActivityData(L.LightningDataModule):
    def __init__(self, batch_size: int, train_manifests: Sequence[str], val_manifests: Sequence[str],
                 stages: Sequence[dict], simu_samples: int = 100_000, fps: int = 100,
                 chunk_seconds: float = 30.0, max_speakers: int = 4, num_workers: int = 2):
        super().__init__()
        for s in stages:
            if s["train_source"] not in {"both", "real", "simu"}:
                raise ValueError(f"train_source must be 'both', 'real' or 'simu', got {s['train_source']!r}")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stages = list(stages)
        self.simu_samples = int(simu_samples)
        self.common = dict(fps=fps, chunk_seconds=chunk_seconds, max_speakers=max_speakers)
        self.max_speakers = max_speakers

        self.real_train = None
        if any(s["train_source"] in {"both", "real"} for s in self.stages):
            self.real_train = MT_AMI_Activity_Dataset(train_manifests, random_offset=True, **self.common)
            print("real train stats:", dict(self.real_train.stats))
        self.val_dataset = MT_AMI_Activity_Dataset(val_manifests, random_offset=False, **self.common)
        print("real val stats:", dict(self.val_dataset.stats))

    def train_dataloader(self):
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        stage, stage_idx, density = stage_at(self.stages, epoch)
        source = stage["train_source"]

        parts, weights = [], []
        if source in {"both", "real"}:
            parts.append(self.real_train)
            weights.append(_bucket_weights(self.real_train.buckets, stage.get("alpha", 0.0)))
        if source in {"both", "simu"}:
            simu = ConversationSimulator(self.simu_samples, density=density, **self.common)
            parts.append(simu)
            weights.append(np.full(len(simu), 1.0 / len(simu)))

        # Mass of each part: stage['sim_share'] for the simulator, else its natural size proportion
        sizes = np.array([len(p) for p in parts], dtype=np.float64)
        shares = sizes / sizes.sum()
        if len(parts) == 2 and stage.get("sim_share") is not None:
            shares = np.array([1.0 - stage["sim_share"], stage["sim_share"]])
        sample_weights = np.concatenate([w * s for w, s in zip(weights, shares)])

        dataset = parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)
        if stage.get("p_mix", 0.0) > 0:
            dataset = MixAugment(dataset, stage["p_mix"], self.max_speakers)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(dataset), replacement=True)

        rank_zero_info(f"[epoch {epoch}] stage {stage_idx} '{stage.get('name', stage_idx)}': "
                       f"source={source} density={density:.2f} alpha={stage.get('alpha', 0.0)} "
                       f"p_mix={stage.get('p_mix', 0.0)} size={len(dataset)}")
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           sampler=sampler,
                                           num_workers=self.num_workers,
                                           collate_fn=collate_activity)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           collate_fn=collate_activity)


# ============================================================
# Model: conv autoencoder with RVQ bottleneck (configurable downsampling)
# ============================================================
def _channels(n_down: int) -> List[int]:
    # n_down=5 -> [16, 32, 64, 128, 256] (same as A_codec_train.py)
    return [16 * 2 ** i for i in range(n_down - 1)] + [256]


class Encoder(nn.Module):
    """(B, S, T) -> (B, 256, T / 2**n_down). T must be a multiple of 2**n_down."""
    def __init__(self, in_channels: int, n_down: int = 5):
        super().__init__()
        layers, c_in = [], in_channels
        for i, c_out in enumerate(_channels(n_down)):
            k, p = (5, 2) if i == 0 else (3, 1)
            layers += [nn.Conv1d(c_in, c_out, kernel_size=k, stride=2, padding=p), nn.SiLU()]
            c_in = c_out
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class Decoder(nn.Module):
    """(B, 256, T / 2**n_down) -> logits (B, S, T). ConvTranspose(k=4, s=2, p=1) = exact 2x."""
    def __init__(self, out_channels: int, n_down: int = 5):
        super().__init__()
        self.input_norm = nn.Sequential(nn.LayerNorm(256), nn.SiLU())
        outs = list(reversed(_channels(n_down)[:-1])) + [16]       # n_down=5 -> 128,64,32,16,16
        layers, c_in = [], 256
        for c_out in outs:
            layers += [nn.ConvTranspose1d(c_in, c_out, kernel_size=4, stride=2, padding=1), nn.SiLU()]
            c_in = c_out
        layers += [
            nn.Conv1d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.SiLU(),
            nn.Conv1d(16, out_channels, kernel_size=3, stride=1, padding=1),
        ]
        self.deconv_block = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.input_norm(z.transpose(1, 2)).transpose(1, 2)
        return self.deconv_block(z)


class ActivityRVQCodec(nn.Module):
    def __init__(self, max_speakers: int = 4, latent_dim: int = 64,
                 n_q: int = 8, codebook_size: int = 1024, n_down: int = 5):
        super().__init__()
        self.max_speakers = max_speakers
        self.stride = 2 ** n_down
        self.encoder = Encoder(in_channels=max_speakers, n_down=n_down)
        self.pre_vq = nn.Conv1d(256, latent_dim, kernel_size=1)
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
        self.decoder = Decoder(out_channels=max_speakers, n_down=n_down)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        pad = (-x.shape[-1]) % self.stride
        return F.pad(x, (0, pad)) if pad else x

    def _decode_latent(self, q: torch.Tensor, T: int) -> torch.Tensor:
        return self.decoder(self.post_vq(q.transpose(1, 2)))[..., :T]

    def forward(self, x: torch.Tensor):
        T = x.shape[-1]
        z = self.pre_vq(self.encoder(self._pad(x))).transpose(1, 2)        # (B, L, D)
        quantized, indices, commit_loss = self.quantizer(z)
        logits = self._decode_latent(quantized, T)
        return logits, indices, commit_loss.sum()

    def logits_from_indices(self, indices: torch.Tensor, T: int) -> torch.Tensor:
        """Codes (B, L, k<=n_q) -> logits (B, S, T); fewer codebooks = lower bit-rate."""
        return self._decode_latent(self.quantizer.get_output_from_indices(indices), T)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Activity (B, S, T) -> codes (B, L, n_q). Call in eval mode."""
        z = self.pre_vq(self.encoder(self._pad(x))).transpose(1, 2)
        _, indices, _ = self.quantizer(z)
        return indices

    @torch.no_grad()
    def decode(self, indices: torch.Tensor, T: Optional[int] = None) -> torch.Tensor:
        """Codes (B, L, n_q) -> activity probabilities (B, S, T)."""
        T = T if T is not None else indices.shape[1] * self.stride
        return torch.sigmoid(self.logits_from_indices(indices, T))


# ============================================================
# Losses
# ============================================================
def boundary_overlap_weights(x: torch.Tensor, beta_boundary: float, beta_overlap: float,
                             boundary_frames: int) -> torch.Tensor:
    """w = 1 + beta_b * near_boundary + beta_o * overlap, shape (B, S, T). Detached."""
    with torch.no_grad():
        change = (x[..., 1:] != x[..., :-1]).float()                    # (B, S, T-1)
        edge = torch.maximum(F.pad(change, (1, 0)), F.pad(change, (0, 1)))   # frames on both sides
        k = int(boundary_frames)
        boundary = F.max_pool1d(edge, kernel_size=2 * k + 1, stride=1, padding=k) if k > 0 else edge
        overlap = (x.sum(1, keepdim=True) >= 2).float().expand_as(x)
        return 1.0 + beta_boundary * boundary + beta_overlap * overlap


# ============================================================
# Lightning module
# ============================================================
class ActivityCodecTask(L.LightningModule):
    def __init__(self, codec: ActivityRVQCodec, stages: Sequence[dict], fps: int = 100,
                 rec_loss: str = "bce", lambda_commit: float = 1.0, pos_weight: Optional[float] = None,
                 beta_boundary: float = 2.0, beta_overlap: float = 1.0, boundary_frames: int = 3,
                 focal_gamma: float = 0.0, lambda_count: float = 0.1,
                 lr: float = 1e-4, warmup_steps: int = 0, eval_nq: Sequence[int] = (2, 4)):
        super().__init__()
        if rec_loss not in {"bce", "mse"}:
            raise ValueError(f"rec_loss must be 'bce' or 'mse', got {rec_loss!r}")
        self.save_hyperparameters(ignore=["codec"])
        self.codec = codec
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)), persistent=False)
        else:
            self.pos_weight = None
        n_q = len(self.codec.quantizer.layers)
        self.eval_nq = [k for k in eval_nq if 0 < k < n_q]
        self._nq_probe_ok = True
        # validation accumulator groups: total, easy, med, hard, spk1..spkS
        self._group_names = ["all"] + list(BUCKET_NAMES) + [f"spk{s}" for s in range(1, codec.max_speakers + 1)]

    # ---------------- helpers ----------------
    def _stage(self):
        return stage_at(self.hparams.stages, self.current_epoch)

    @staticmethod
    def _rng(*keys) -> random.Random:
        seed = 0
        for k in keys:
            seed = seed * 1_000_003 + int(k)
        return random.Random(seed)

    def _set_quantize_dropout(self, enabled: bool) -> None:
        if hasattr(self.codec.quantizer, "quantize_dropout"):
            self.codec.quantizer.quantize_dropout = bool(enabled)

    def _losses(self, logits: torch.Tensor, x: torch.Tensor, weighted: bool):
        hp = self.hparams
        probs = torch.sigmoid(logits)
        if hp.rec_loss == "bce":
            per_frame = F.binary_cross_entropy_with_logits(logits, x, pos_weight=self.pos_weight,
                                                           reduction="none")
        else:
            per_frame = (probs - x) ** 2
        if not weighted:
            return per_frame.mean(), torch.zeros((), device=x.device)

        if hp.focal_gamma > 0:
            p_t = probs * x + (1 - probs) * (1 - x)
            per_frame = per_frame * (1 - p_t).pow(hp.focal_gamma)
        w = boundary_overlap_weights(x, hp.beta_boundary, hp.beta_overlap, hp.boundary_frames)
        rec = (w * per_frame).sum() / w.sum()
        count = (probs.sum(1) - x.sum(1)).abs().mean()                # speaker-count consistency
        return rec, count

    @torch.no_grad()
    def _metrics(self, logits: torch.Tensor, x: torch.Tensor, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = logits > 0
        target = x > 0.5
        tp = (pred & target).sum().float()
        fp = (pred & ~target).sum().float()
        fn = (~pred & target).sum().float()
        codes = indices[..., 0]
        usage = codes[codes >= 0].unique().numel() / self.codec.quantizer.codebook_size
        return {
            "acc": (pred == target).float().mean(),
            "f1": 2 * tp / (2 * tp + fp + fn + 1e-8),
            "use": torch.tensor(float(usage), device=logits.device),
        }

    # ---------------- train ----------------
    def training_step(self, batch, batch_idx):
        x = batch
        hp = self.hparams
        stage, stage_idx, _ = self._stage()
        # Same decision on every DDP rank (seeded by epoch/batch)
        qdrop = self._rng(self.current_epoch, batch_idx).random() < stage.get("qdrop_prob", 1.0)
        self._set_quantize_dropout(qdrop)

        weighted = bool(stage.get("weighted_loss", False))
        logits, indices, commit = self.codec(x)
        rec, count = self._losses(logits, x, weighted)
        loss = rec + hp.lambda_commit * commit + (hp.lambda_count * count if weighted else 0.0)

        m = self._metrics(logits, x, indices)
        self.log_dict({"tl": loss, "tf1": m["f1"]},
                      on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict({"trec": rec, "tcount": count, "tc": commit, "tacc": m["acc"], "tuse": m["use"]},
                      on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log_dict({"stage": float(stage_idx), "qdrop": float(qdrop)},
                      on_step=True, on_epoch=False, logger=True)
        return loss

    # ---------------- validation (DER by difficulty) ----------------
    def on_validation_epoch_start(self):
        self._set_quantize_dropout(True)          # needed to decode from a subset of codebooks
        n_levels = len(self.eval_nq) + 1          # probes + full model (last)
        self._val_acc = torch.zeros(n_levels, len(self._group_names), 4, device=self.device)

    def _accumulate(self, level: int, pred: torch.Tensor, x: torch.Tensor,
                    bucket: torch.Tensor, n_spk: torch.Tensor) -> None:
        stats = torch.stack(frame_der_stats(pred, x), dim=1)            # (B, 4): ms, fa, cs, ref
        acc = self._val_acc[level]
        acc[0] += stats.sum(0)
        for b in range(len(BUCKET_NAMES)):
            acc[1 + b] += stats[bucket == b].sum(0)
        for s in range(1, self.codec.max_speakers + 1):
            acc[len(BUCKET_NAMES) + s] += stats[n_spk == s].sum(0)

    def validation_step(self, batch, batch_idx):
        x = batch
        hp = self.hparams
        logits, indices, commit = self.codec(x)
        rec, count = self._losses(logits, x, weighted=False)
        loss = rec + hp.lambda_commit * commit
        m = self._metrics(logits, x, indices)
        self.log_dict({"vl": loss, "vf1": m["f1"]},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict({"vrec": rec, "vc": commit, "vacc": m["acc"], "vuse": m["use"]},
                      on_step=False, on_epoch=True, logger=True, sync_dist=True)

        bucket = difficulty_bucket(x, hp.fps)
        n_spk, _, _ = activity_difficulty(x, hp.fps)
        self._accumulate(len(self.eval_nq), logits > 0, x, bucket, n_spk)

        if self._nq_probe_ok:
            for level, k in enumerate(self.eval_nq):
                try:
                    logits_k = self.codec.logits_from_indices(indices[..., :k], x.shape[-1])
                except Exception as err:          # API differences across vector_quantize_pytorch versions
                    rank_zero_info(f"n_q probe disabled: {err}")
                    self._nq_probe_ok = False
                    break
                self._accumulate(level, logits_k > 0, x, bucket, n_spk)
        return loss

    def on_validation_epoch_end(self):
        acc = self.trainer.strategy.reduce(self._val_acc, reduce_op="sum")
        full = acc[-1]

        def der(row):
            ms, fa, cs, ref = row.tolist()
            return None if ref <= 0 else (100 * (ms + fa + cs) / ref, 100 * ms / ref, 100 * fa / ref, 100 * cs / ref)

        total = der(full[0])
        if total is not None:
            d, ms, fa, cs = total
            self.log("vder", d, prog_bar=True, logger=True)
            self.log_dict({"vms": ms, "vfa": fa, "vcs": cs}, logger=True)
        for g, name in enumerate(self._group_names[1:], start=1):
            r = der(full[g])
            if r is not None:
                self.log(f"vder_{name}", r[0], logger=True)
        if self._nq_probe_ok:
            hard = 1 + BUCKET_NAMES.index("hard")
            for level, k in enumerate(self.eval_nq):
                for g, suffix in ((0, ""), (hard, "_hard")):
                    r = der(acc[level, g])
                    if r is not None:
                        self.log(f"vder{suffix}_nq{k}", r[0], logger=True)

    # ---------------- optimizer ----------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, eps=1e-5)
        warmup = int(self.hparams.warmup_steps)

        # Linear warmup, then constant x per-stage lr_scale (dataset size varies per
        # stage, so total steps aren't known up-front for a decay schedule).
        def lr_lambda(step: int) -> float:
            warm = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
            return warm * float(self._stage()[0].get("lr_scale", 1.0))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
    SIMU_SAMPLES = 100_000
    BATCH_SIZE = 64
    LR = 1e-4
    WARMUP_STEPS = 1000

    # Model (N_DOWN=4 / larger N_Q = more bit-rate; only if vder_hard_nq* shows a capacity limit)
    LATENT_DIM = 64
    N_Q = 8
    CODEBOOK_SIZE = 1024
    N_DOWN = 5                 # stride 2**N_DOWN = 32 -> ~3.1 latent frames/s at 100 fps

    # Loss
    REC_LOSS = "bce"           # "bce" | "mse"
    LAMBDA_COMMIT = 1.0
    BETA_BOUNDARY = 2.0        # extra weight on frames within +-BOUNDARY_FRAMES of an on/offset
    BETA_OVERLAP = 1.0         # extra weight on frames with >= 2 active speakers
    BOUNDARY_FRAMES = 3        # 30 ms at 100 fps
    FOCAL_GAMMA = 0.0          # 0 = off; try 1-2
    LAMBDA_COUNT = 0.1         # speaker-count consistency
    EVAL_NQ = (2, 4)           # extra n_q levels decoded in validation (full n_q always)

    # Curriculum. density: simulator (start, end) over the stage's epochs;
    # alpha: bucket-balancing exponent for real data (0 natural .. 1 fully balanced);
    # sim_share: fraction of samples from the simulator when source="both" (None = natural sizes).
    STAGES = [
        dict(name="warmup",  epochs=1, train_source="both", density=(0.3, 0.3), alpha=0.0,
             p_mix=0.0, weighted_loss=False, qdrop_prob=1.0, lr_scale=1.0, sim_share=None),
        dict(name="densify", epochs=3, train_source="both", density=(0.5, 1.0), alpha=0.5,
             p_mix=0.3, weighted_loss=True,  qdrop_prob=0.7, lr_scale=1.0, sim_share=None),
        dict(name="real_ft", epochs=1, train_source="real", density=(1.0, 1.0), alpha=0.5,
             p_mix=0.3, weighted_loss=True,  qdrop_prob=0.5, lr_scale=0.3, sim_share=None),
    ]
    # Optional warm start from a baseline A_codec_train.py checkpoint (same LATENT_DIM/N_Q/N_DOWN);
    # then you can drop the "warmup" stage.
    INIT_CKPT = None

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
                             n_q=N_Q, codebook_size=CODEBOOK_SIZE, n_down=N_DOWN)
    if INIT_CKPT:
        state = torch.load(INIT_CKPT, map_location="cpu", weights_only=False)["state_dict"]
        codec.load_state_dict({k[len("codec."):]: v for k, v in state.items() if k.startswith("codec.")})
        print(f"initialised codec from {INIT_CKPT}")

    model = ActivityCodecTask(codec, stages=STAGES, fps=FPS, rec_loss=REC_LOSS,
                              lambda_commit=LAMBDA_COMMIT, beta_boundary=BETA_BOUNDARY,
                              beta_overlap=BETA_OVERLAP, boundary_frames=BOUNDARY_FRAMES,
                              focal_gamma=FOCAL_GAMMA, lambda_count=LAMBDA_COUNT,
                              lr=LR, warmup_steps=WARMUP_STEPS, eval_nq=EVAL_NQ)
    data = ActivityData(BATCH_SIZE, TRAIN_MANIFESTS, VAL_MANIFESTS, stages=STAGES,
                        simu_samples=SIMU_SAMPLES, fps=FPS, chunk_seconds=CHUNK_SECONDS,
                        max_speakers=MAX_SPEAKERS)

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    checkpoint_callback = ModelCheckpoint(dirpath='/kaggle/working/checkpoint_balanced',
                                          save_top_k=10, monitor='vder',
                                          mode='min', save_last=True,
                                          filename='ActivityCodecBal-{epoch}-{vder:.4f}')
    logger = TensorBoardLogger("/kaggle/working/log", name="ActivityCodec_balanced", version='version_0')

    trainer = L.Trainer(accelerator="gpu",
                        callbacks=[checkpoint_callback, CustomProgressBar()],
                        enable_checkpointing=True,
                        max_epochs=sum(int(s["epochs"]) for s in STAGES),
                        reload_dataloaders_every_n_epochs=1,   # per-stage data settings
                        # precision='16-mixed',
                        logger=logger,
                        strategy='ddp_find_unused_parameters_true',
                        val_check_interval=0.1,
                        num_sanity_val_steps=0,       # don't let val data k-means-init the codebooks
                        devices=[0, 1])
    trainer.fit(model, data)
