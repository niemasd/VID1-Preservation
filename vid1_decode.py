#!/usr/bin/env python3
"""
Decode a Factor 5 VID1 .vid video to FFV1/Matroska or uncompressed Y4M.

This is deliberately a decoder front-end, not another VID1-to-M4V deliverable:

  1. It parses the VID1 container and proprietary picture headers.
  2. It builds a temporary FFmpeg-decodable MPEG-4 Part 2 adapter stream from the
     MPEG-4-derived macroblock payload.
  3. It invokes ffmpeg to perform the actual inverse quantisation, IDCT,
     motion compensation, B-frame reordering, and pixel reconstruction.
  4. It writes decoded pixels as lossless FFV1 (normally in Matroska) or as
     uncompressed YUV4MPEG2 (Y4M).

Only video is decoded; VID1 audio chunks are ignored.

S/GMC pictures are accepted in either standard MPEG-4 trajectory syntax or
the byte-aligned 16-bit form used by retail VID1 streams.  In the supplied
retail stream, the horizontal word stores a signed-magnitude code as
``raw >> 2`` and the vertical word stores one as ``raw >> 3``.  The low bit is
the sign (odd means negative) and the remaining bits are the magnitude.  Auto
mode recognises that exact low-bit layout and writes the corresponding MPEG-4
sprite trajectory.  The v4 folded/zig-zag interpretation and divisor-based
mappings remain available as explicit diagnostic compatibility options.

When auto mode cannot know the output sprite shape until the first S picture,
the resolved shape is backfilled to earlier pictures in the same VID1 sprite
state.  This keeps the MPEG-4 VOL stable from the first reference picture;
some decoder versions otherwise discard prediction references when a second
VOL appears mid-GOP and display white or block-concealed frames.

Extended VID1 luma/chroma quantisation matrices do not map one-for-one onto
MPEG-4 Part 2's intra/non-intra matrices.  For such pictures this program can
run two decoder passes, using the luma matrix for both MPEG-4 matrices in one
pass and the chroma matrix in the other, then combine Y from the first pass
with U/V from the second.  That is the closest component-wise mapping
available through a stock MPEG-4 decoder, but it remains an inferred mapping
because the public VID1 notes do not fully specify the extension.

Some retail streams set an extended-header field-syntax bit.  Their
macroblocks use MPEG-4's interlace-capable grammar even though the decoded
pictures are presented progressively, and their P pictures include a separate
zero preamble byte before the macroblock payload.  The adapter preserves that
grammar, removes the preamble, and marks the decoded output progressive.

Examples:

    python3 vid1_decode.py input.vid output.mkv
    python3 vid1_decode.py input.vid output.y4m --format y4m
    python3 vid1_decode.py input.vid output.mkv --progress always -v

The script requires an ffmpeg executable with the MPEG-4 Part 2 decoder and,
for FFV1 output, the FFV1 encoder.  It uses tqdm progress bars when tqdm is
installed and falls back to a built-in progress display otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Optional, Sequence, Union

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # tqdm is optional; a built-in fallback is provided.
    _tqdm = None


ByteBuffer = Union[bytes, memoryview]


PROGRAM = "vid1_decode.py"
VERSION = "5.2"


def tag(text: str) -> int:
    return int.from_bytes(text.encode("ascii"), "big")


TAG_VID1 = tag("VID1")
TAG_HEAD = tag("HEAD")
TAG_VIDH = tag("VIDH")
TAG_AUDH = tag("AUDH")
TAG_FRAM = tag("FRAM")
TAG_VIDD = tag("VIDD")
TAG_AUDD = tag("AUDD")
KNOWN_TAGS = (TAG_FRAM, TAG_VIDD, TAG_AUDD)

FRAME_NAMES = {0: "I", 1: "P", 2: "B", 3: "S"}

# MPEG-4 Part 2 zig-zag scan, expressed as row-major coefficient indices.
ZIGZAG = (
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
)

MPEG4_DEFAULT_INTRA = (
    8, 17, 18, 19, 21, 23, 25, 27,
    17, 18, 19, 21, 23, 25, 27, 28,
    20, 21, 22, 23, 24, 26, 28, 30,
    21, 22, 23, 24, 26, 28, 30, 32,
    22, 23, 24, 26, 28, 30, 32, 35,
    23, 24, 26, 28, 30, 32, 35, 38,
    25, 26, 28, 30, 32, 35, 38, 41,
    27, 28, 30, 32, 35, 38, 41, 45,
)

MPEG4_DEFAULT_INTER = (
    16, 17, 18, 19, 20, 21, 22, 23,
    17, 18, 19, 20, 21, 22, 23, 24,
    18, 19, 20, 21, 22, 23, 24, 25,
    19, 20, 21, 22, 23, 24, 26, 27,
    20, 21, 22, 23, 25, 26, 27, 28,
    21, 22, 23, 24, 26, 27, 28, 30,
    22, 23, 24, 26, 27, 28, 30, 31,
    23, 24, 25, 27, 28, 30, 31, 33,
)

# FFmpeg's MPEG-4 sprite trajectory VLC is a canonical code built from these
# symbol lengths.  The symbol itself is the number of following xbits (0..14).
SPRITE_TRAJECTORY_LENGTHS = (2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)


class VID1Error(Exception):
    """Input is malformed or uses a feature this decoder cannot safely map."""


class ExternalToolError(VID1Error):
    """ffmpeg failed while decoding or writing the output."""


class BitReaderMSB:
    """Most-significant-bit-first bit reader."""

    def __init__(self, data: ByteBuffer):
        self.data = data
        self.bitpos = 0

    @property
    def size_bits(self) -> int:
        return len(self.data) * 8

    @property
    def bits_left(self) -> int:
        return self.size_bits - self.bitpos

    @property
    def bytepos(self) -> int:
        if self.bitpos & 7:
            raise VID1Error("bit reader is not byte-aligned")
        return self.bitpos >> 3

    def read(self, count: int) -> int:
        if count < 0:
            raise ValueError("negative bit count")
        if self.bitpos + count > self.size_bits:
            raise VID1Error("unexpected end of VID1 picture header")
        value = 0
        for _ in range(count):
            byte = self.data[self.bitpos >> 3]
            value = (value << 1) | ((byte >> (7 - (self.bitpos & 7))) & 1)
            self.bitpos += 1
        return value

    def read_bit(self) -> int:
        return self.read(1)

    def align_byte(self) -> None:
        self.bitpos = (self.bitpos + 7) & ~7
        if self.bitpos > self.size_bits:
            raise VID1Error("picture header alignment runs past packet end")

    def bit_slice(self, start: int, end: int) -> tuple[int, ...]:
        if start < 0 or end < start or end > self.size_bits:
            raise ValueError("invalid bit slice")
        return tuple(
            (self.data[pos >> 3] >> (7 - (pos & 7))) & 1
            for pos in range(start, end)
        )


class BitReaderLSB:
    """Least-significant-bit-first reader used by VID1 variable audio headers."""

    def __init__(self, data: bytes):
        self.data = data
        self.bitpos = 0

    def read(self, count: int) -> int:
        if count < 0:
            raise ValueError("negative bit count")
        if self.bitpos + count > len(self.data) * 8:
            raise VID1Error("unexpected end of little-endian packet header")
        value = 0
        for shift in range(count):
            byte = self.data[self.bitpos >> 3]
            value |= ((byte >> (self.bitpos & 7)) & 1) << shift
            self.bitpos += 1
        return value


class BitWriter:
    """Most-significant-bit-first writer for the temporary MPEG-4 stream."""

    def __init__(self, out: BinaryIO):
        self.out = out
        self.current = 0
        self.count = 0
        self.bytes_written = 0

    def write_bit(self, bit: int) -> None:
        self.current = (self.current << 1) | (bit & 1)
        self.count += 1
        if self.count == 8:
            self.out.write(bytes((self.current,)))
            self.bytes_written += 1
            self.current = 0
            self.count = 0

    def write_bits(self, count: int, value: int) -> None:
        if count < 0:
            raise ValueError("negative bit count")
        if value < 0 or (count and value >= (1 << count)):
            raise ValueError(f"value {value} does not fit in {count} bits")
        for shift in range(count - 1, -1, -1):
            self.write_bit((value >> shift) & 1)

    def write_bit_sequence(self, bits: Iterable[int]) -> None:
        for bit in bits:
            self.write_bit(bit)

    def write_bytes_as_bits(self, data: ByteBuffer) -> None:
        if not data:
            return
        if self.count == 0:
            self.out.write(data)
            self.bytes_written += len(data)
            return

        # Shift a byte buffer into an already-partially-filled output byte.
        # Calling write_bits eight times per input byte made large VID1 files
        # need several minutes just to build the temporary decoder stream.
        count = self.count
        shift = 8 - count
        mask = (1 << count) - 1
        current = self.current
        output = bytearray(len(data))
        for index, byte in enumerate(data):
            output[index] = (current << shift) | (int(byte) >> count)
            current = int(byte) & mask
        self.out.write(output)
        self.bytes_written += len(output)
        self.current = current

    def align_zero(self) -> None:
        if self.count:
            self.current <<= 8 - self.count
            self.out.write(bytes((self.current,)))
            self.bytes_written += 1
            self.current = 0
            self.count = 0

    def start_code(self, suffix: int) -> None:
        if not 0 <= suffix <= 0xFF:
            raise ValueError("start-code suffix must fit in one byte")
        self.align_zero()
        self.out.write(b"\x00\x00\x01" + bytes((suffix,)))
        self.bytes_written += 4

    def close(self) -> None:
        self.align_zero()


@dataclass(frozen=True)
class VID1Info:
    big_endian: bool
    width: Optional[int]
    height: Optional[int]
    start_offset: int
    audio_codec: Optional[str]
    frame_count: Optional[int]
    header_rate: Optional[Fraction]


@dataclass(frozen=True)
class RawVideoChunk:
    index: int
    file_offset: int
    chunk_size: int
    body: bytes


@dataclass(frozen=True)
class SpriteConfig:
    warping_points: int
    accuracy: int


@dataclass(frozen=True)
class MatrixPair:
    intra: tuple[int, ...]
    inter: tuple[int, ...]


@dataclass(frozen=True)
class Picture:
    index: int
    chunk_offset: int
    frame_type: int
    rounding: int
    intra_dc_vlc_thr_idx: int
    quant: int
    fcode_forward: int
    fcode_backward: int
    timecode: int
    payload: ByteBuffer
    ignored16: int
    extended_info_present: bool
    sprite_config: Optional[SpriteConfig]
    trajectory_bits: tuple[int, ...]
    uses_extended_quant: bool
    luma_quant: Optional[MatrixPair]
    chroma_quant: Optional[MatrixPair]
    matrix_order_luma: Optional[str]
    matrix_order_chroma: Optional[str]
    display_index: int = -1
    gmc_source_format: Optional[str] = None
    gmc_raw_values: tuple[int, ...] = ()
    source_sprite_config: Optional[SpriteConfig] = None
    gmc_mapping: Optional[str] = None
    gmc_conversion_divisor: Optional[int] = None
    # The first trailing extended-header flag selects an MPEG-4-compatible
    # field/interlace macroblock grammar.  The decoded pictures themselves are
    # still presented progressively.  P pictures in this mode carry one
    # separate zero preamble byte before the macroblock payload.
    field_syntax: bool = False
    extension_flag_2: bool = False
    p_frame_preamble: Optional[int] = None

    @property
    def frame_name(self) -> str:
        return FRAME_NAMES.get(self.frame_type, f"?{self.frame_type}")


@dataclass
class ParseState:
    # VID1's 2-bit sprite field is not documented precisely enough to assume
    # that it is identical to MPEG-4's 6-bit num_sprite_warping_points field.
    # Keep the source-side state separate from the MPEG-4 VOL state that we
    # synthesize for FFmpeg.
    source_sprite: Optional[SpriteConfig] = None
    output_sprite: Optional[SpriteConfig] = None
    luma_matrix: Optional[tuple[int, ...]] = None
    chroma_matrix: Optional[tuple[int, ...]] = None
    gmc_format: Optional[str] = None
    fixed16_mapping: Optional[str] = None
    # Some VID1 streams use the MPEG-4 interlace-capable macroblock grammar
    # even though their pictures are displayed progressively.  The first of
    # the two trailing extended-header bits selects that grammar.  Keep the
    # second bit for diagnostics until its meaning is known.
    field_syntax: bool = False
    extension_flag_2: bool = False


@dataclass
class MPEG4TimeState:
    time_resolution: int
    current_time_base: int = 0
    last_time_base_for_b: int = 0


@dataclass(frozen=True)
class AdapterConfig:
    width: int
    height: int
    fps: Fraction
    has_b_frames: bool
    matrix_plane: str  # "luma" or "chroma"
    trim_payload_padding: bool


@dataclass(frozen=True)
class DecodeResult:
    raw_path: Path
    frame_count: int
    stderr: str


@dataclass(frozen=True)
class FFmpegRunResult:
    stderr: str
    frame_count: Optional[int]
    elapsed: float


@dataclass(frozen=True)
class SkipProbe:
    skip: int
    score: int
    parsed: int
    first_type: Optional[int]
    error: Optional[str]


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:d}:{remainder:02d}"


def format_size(byte_count: int) -> str:
    value = float(max(0, byte_count))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


class _NullProgress:
    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def update(self, amount: int = 1) -> None:
        return None

    def update_to(self, value: int) -> None:
        return None


class _TqdmProgress:
    def __init__(
        self,
        *,
        total: Optional[int],
        description: str,
        unit: str,
        unit_scale: bool,
    ) -> None:
        assert _tqdm is not None
        self._bar = _tqdm(
            total=total,
            desc=description,
            unit=unit,
            unit_scale=unit_scale,
            unit_divisor=1024,
            dynamic_ncols=True,
            leave=True,
            file=sys.stderr,
            mininterval=0.15,
            smoothing=0.15,
        )

    def __enter__(self) -> "_TqdmProgress":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._bar.close()

    def update(self, amount: int = 1) -> None:
        if amount > 0:
            self._bar.update(amount)

    def update_to(self, value: int) -> None:
        delta = value - int(self._bar.n)
        if delta > 0:
            self._bar.update(delta)


class _SimpleProgress:
    """Small stderr progress fallback used when tqdm is not installed."""

    def __init__(
        self,
        *,
        total: Optional[int],
        description: str,
        unit: str,
        unit_scale: bool,
    ) -> None:
        self.total = total
        self.description = description
        self.unit = unit
        self.unit_scale = unit_scale
        self.value = 0
        self.started = time.monotonic()
        self.last_print = 0.0
        self.last_percent = -1
        self.tty = sys.stderr.isatty()

    def __enter__(self) -> "_SimpleProgress":
        self._render(force=True)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.total is not None and exc_type is None:
            self.value = max(self.value, self.total)
        self._render(force=True, final=True)

    def update(self, amount: int = 1) -> None:
        self.update_to(self.value + amount)

    def update_to(self, value: int) -> None:
        self.value = max(self.value, value)
        self._render()

    def _quantity(self, value: int) -> str:
        if self.unit_scale and self.unit == "B":
            return format_size(value)
        return f"{value} {self.unit}"

    def _render(self, *, force: bool = False, final: bool = False) -> None:
        now = time.monotonic()
        percent: Optional[int] = None
        if self.total:
            percent = min(100, int(self.value * 100 / self.total))
        if not force:
            if self.tty:
                if now - self.last_print < 0.5:
                    return
            else:
                bucket = -1 if percent is None else percent // 10
                if bucket == self.last_percent and now - self.last_print < 5.0:
                    return
                self.last_percent = bucket
        elapsed = max(now - self.started, 1e-9)
        rate = self.value / elapsed
        if self.total is None:
            message = f"{self.description}: {self._quantity(self.value)}"
        else:
            message = (
                f"{self.description}: {percent:3d}% "
                f"({self._quantity(self.value)}/{self._quantity(self.total)})"
            )
        if self.unit_scale and self.unit == "B":
            message += f" [{format_size(int(rate))}/s]"
        else:
            message += f" [{rate:.1f} {self.unit}/s]"
        if self.tty and not final:
            print("\r" + message, end="", file=sys.stderr, flush=True)
        else:
            if self.tty:
                print("\r" + message, file=sys.stderr, flush=True)
            else:
                print(message, file=sys.stderr, flush=True)
        self.last_print = now


class Reporter:
    def __init__(
        self,
        verbose: int = 0,
        *,
        quiet: bool = False,
        progress_mode: str = "auto",
        ffmpeg_log: Optional[Path] = None,
    ) -> None:
        self.verbose = verbose
        self.quiet = quiet
        self.progress_mode = progress_mode
        self.ffmpeg_log = ffmpeg_log
        self.started = time.monotonic()
        self._warnings: list[str] = []

    @property
    def progress_enabled(self) -> bool:
        if self.quiet or self.progress_mode == "never":
            return False
        if self.progress_mode == "always":
            return True
        return sys.stderr.isatty()

    def status(self, message: str) -> None:
        if not self.quiet:
            print(f"[vid1] {message}", file=sys.stderr, flush=True)

    def info(self, message: str, level: int = 1) -> None:
        if not self.quiet and self.verbose >= level:
            print(f"[vid1] {message}", file=sys.stderr, flush=True)

    def warn(self, message: str) -> None:
        if message not in self._warnings:
            self._warnings.append(message)
            print(f"[vid1] warning: {message}", file=sys.stderr, flush=True)

    @contextlib.contextmanager
    def phase(self, message: str) -> Iterator[None]:
        self.status(message)
        started = time.monotonic()
        try:
            yield
        except Exception:
            self.status(f"{message} failed after {format_duration(time.monotonic() - started)}")
            raise
        else:
            self.info(
                f"{message} completed in {format_duration(time.monotonic() - started)}",
                level=1,
            )

    def progress(
        self,
        *,
        total: Optional[int],
        description: str,
        unit: str = "frame",
        unit_scale: bool = False,
    ) -> Union[_NullProgress, _TqdmProgress, _SimpleProgress]:
        if not self.progress_enabled:
            return _NullProgress()
        if _tqdm is not None:
            return _TqdmProgress(
                total=total,
                description=description,
                unit=unit,
                unit_scale=unit_scale,
            )
        return _SimpleProgress(
            total=total,
            description=description,
            unit=unit,
            unit_scale=unit_scale,
        )

    def track(
        self,
        iterable: Iterable[object],
        *,
        total: Optional[int],
        description: str,
        unit: str = "frame",
    ) -> Iterator[object]:
        with self.progress(total=total, description=description, unit=unit) as progress:
            for item in iterable:
                yield item
                progress.update(1)

    def record_ffmpeg_log(
        self,
        *,
        description: str,
        command: Sequence[str],
        stderr: str,
    ) -> None:
        if self.ffmpeg_log is None:
            return
        self.ffmpeg_log.parent.mkdir(parents=True, exist_ok=True)
        with self.ffmpeg_log.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n===== {description} =====\n")
            log.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
            log.write(stderr)
            if stderr and not stderr.endswith("\n"):
                log.write("\n")


# ---------------------------------------------------------------------------
# Generic binary/container parsing


def read_exact(stream: BinaryIO, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise EOFError
    return data


def read_u16(stream: BinaryIO, big_endian: bool) -> int:
    return int.from_bytes(read_exact(stream, 2), "big" if big_endian else "little")


def read_u32(stream: BinaryIO, big_endian: bool) -> int:
    return int.from_bytes(read_exact(stream, 4), "big" if big_endian else "little")


def on_disk_tag_bytes(value: int, big_endian: bool) -> bytes:
    encoded = value.to_bytes(4, "big")
    return encoded if big_endian else encoded[::-1]


def plausible_rate(numerator: int, denominator: int) -> Optional[Fraction]:
    if numerator <= 0 or denominator <= 0:
        return None
    rate = Fraction(numerator, denominator)
    if Fraction(1, 2) <= rate <= 240:
        return rate
    return None


def parse_vid1_file_header(stream: BinaryIO) -> VID1Info:
    magic_bytes = read_exact(stream, 4)
    magic_be = int.from_bytes(magic_bytes, "big")
    if magic_be == TAG_VID1:
        big_endian = True
    elif magic_bytes == b"1DIV":
        big_endian = False
    else:
        raise VID1Error(f"not a VID1 file: first four bytes are {magic_bytes!r}")

    try:
        head_offset = read_u32(stream, big_endian)
        stream.seek(head_offset)
        if read_u32(stream, big_endian) != TAG_HEAD:
            raise VID1Error("HEAD chunk not found at the header offset")
        head_size = read_u32(stream, big_endian)
    except EOFError as exc:
        raise VID1Error("truncated VID1 file header") from exc

    start_offset = head_offset + head_size
    width: Optional[int] = None
    height: Optional[int] = None
    frame_count: Optional[int] = None
    header_rate: Optional[Fraction] = None
    audio_codec: Optional[str] = None

    offset = head_offset + 12
    stream.seek(offset)
    try:
        chunk = read_u32(stream, big_endian)
    except EOFError as exc:
        raise VID1Error("truncated HEAD chunk") from exc

    if chunk == TAG_VIDH:
        try:
            vidh_size = read_u32(stream, big_endian)
        except EOFError as exc:
            raise VID1Error("truncated VIDH chunk") from exc
        if vidh_size < 16:
            raise VID1Error(f"invalid VIDH chunk size {vidh_size}")
        next_offset = offset + vidh_size
        try:
            stream.seek(offset + 8)
            stream.seek(4, io.SEEK_CUR)  # unknown/reserved
            width = read_u16(stream, big_endian)
            height = read_u16(stream, big_endian)
        except EOFError as exc:
            raise VID1Error("truncated VIDH dimensions") from exc
        if width <= 0 or height <= 0:
            raise VID1Error(f"invalid VIDH dimensions {width}x{height}")

        # Some files expose additional fields after width/height.  Their exact
        # status is not documented by the current demuxer, so accept only
        # strongly plausible values and keep 24 fps as the normal fallback.
        remaining = next_offset - stream.tell()
        if remaining >= 4:
            candidate_count = read_u32(stream, big_endian)
            remaining -= 4
            if 0 < candidate_count < (1 << 31):
                frame_count = candidate_count
        if remaining >= 4:
            _unknown_time = read_u32(stream, big_endian)
            remaining -= 4
        if remaining >= 6:
            rate_num = read_u32(stream, big_endian)
            rate_den = read_u16(stream, big_endian)
            header_rate = plausible_rate(rate_num, rate_den)
        offset = next_offset

    stream.seek(offset)
    try:
        chunk = read_u32(stream, big_endian)
    except EOFError:
        chunk = None
    if chunk == TAG_AUDH:
        try:
            stream.seek(offset + 12)
            codec = read_u32(stream, big_endian)
        except EOFError as exc:
            raise VID1Error("truncated AUDH chunk") from exc
        audio_codec = {
            tag("PC16"): "PC16",
            tag("XAPM"): "XAPM",
            tag("APCM"): "APCM",
            tag("VAUD"): "VAUD/Vorbis",
        }.get(codec, f"0x{codec:08x}")

    return VID1Info(
        big_endian=big_endian,
        width=width,
        height=height,
        start_offset=start_offset,
        audio_codec=audio_codec,
        frame_count=frame_count,
        header_rate=header_rate,
    )


def parse_variable_packet_header_at(stream: BinaryIO, offset: int) -> tuple[int, int]:
    stream.seek(offset)
    data = stream.read(4)
    if len(data) < 4:
        raise EOFError
    bits = BitReaderLSB(data)
    size_bits = bits.read(4)
    if size_bits > 30:
        raise VID1Error(f"unreasonable packet-header size_bits={size_bits} at 0x{offset:x}")
    size = bits.read(size_bits + 1)
    if size_bits == 0 and size == 0 and data[0] == 0x80:
        size = 1
    return (bits.bitpos + 7) // 8, size


def find_next_known_chunk(
    stream: BinaryIO,
    big_endian: bool,
    start: int,
    max_scan: int = 1 << 20,
) -> Optional[int]:
    stream.seek(start)
    data = stream.read(max_scan)
    best: Optional[int] = None
    for known in KNOWN_TAGS:
        location = data.find(on_disk_tag_bytes(known, big_endian))
        if location >= 0:
            absolute = start + location
            best = absolute if best is None else min(best, absolute)
    return best


def read_video_chunks(
    stream: BinaryIO,
    info: VID1Info,
    *,
    resync_scan: bool,
    max_chunk_size: int,
    reporter: Optional[Reporter] = None,
    input_size: Optional[int] = None,
) -> list[RawVideoChunk]:
    chunks: list[RawVideoChunk] = []
    big_endian = info.big_endian
    start_offset = info.start_offset
    stream.seek(start_offset)

    total_bytes: Optional[int] = None
    if input_size is not None:
        total_bytes = max(0, input_size - start_offset)
    progress_context = (
        reporter.progress(
            total=total_bytes,
            description="Scanning VID1 container",
            unit="B",
            unit_scale=True,
        )
        if reporter is not None
        else _NullProgress()
    )

    with progress_context as progress:
        while True:
            position = stream.tell()
            progress.update_to(max(0, position - start_offset))
            try:
                magic = read_u32(stream, big_endian)
            except EOFError:
                break

            if magic == TAG_FRAM:
                # The current librempeg demuxer skips the 28-byte FRAM
                # subheader and then reads the nested VIDD/AUDD tag.
                stream.seek(28, io.SEEK_CUR)
                position = stream.tell()
                try:
                    magic = read_u32(stream, big_endian)
                except EOFError:
                    break

            if magic == TAG_VIDD:
                try:
                    chunk_size = read_u32(stream, big_endian)
                except EOFError as exc:
                    raise VID1Error(f"truncated VIDD size at 0x{position:x}") from exc
                if chunk_size < 8 or chunk_size > max_chunk_size:
                    raise VID1Error(
                        f"invalid VIDD chunk size {chunk_size} at 0x{position:x}"
                    )
                try:
                    body = read_exact(stream, chunk_size - 8)
                except EOFError as exc:
                    raise VID1Error(f"truncated VIDD chunk at 0x{position:x}") from exc
                chunks.append(
                    RawVideoChunk(
                        index=len(chunks),
                        file_offset=position,
                        chunk_size=chunk_size,
                        body=body,
                    )
                )
                progress.update_to(max(0, stream.tell() - start_offset))
                continue

            if magic == TAG_AUDD:
                try:
                    chunk_size = read_u32(stream, big_endian)
                except EOFError as exc:
                    raise VID1Error(f"truncated AUDD size at 0x{position:x}") from exc
                if chunk_size < 8 or chunk_size > max_chunk_size:
                    raise VID1Error(
                        f"invalid AUDD chunk size {chunk_size} at 0x{position:x}"
                    )
                stream.seek(position + chunk_size)
                progress.update_to(max(0, stream.tell() - start_offset))
                continue

            if magic == 0:
                remainder = stream.read()
                if not remainder or all(value == 0 for value in remainder):
                    break
                stream.seek(position + 4)

            # Vorbis audio may appear as a bare variable-length packet between
            # regular chunks.  Skip it when the header is sane; otherwise
            # either fail or scan to the next known tag.
            try:
                header_length, packet_size = parse_variable_packet_header_at(
                    stream, position
                )
                if packet_size <= 0 or packet_size > max_chunk_size:
                    raise VID1Error("unreasonable bare audio packet size")
                stream.seek(position + header_length + packet_size)
            except Exception as exc:
                if not resync_scan:
                    raise VID1Error(
                        f"unknown chunk/tag 0x{magic:08x} at 0x{position:x}; "
                        "use --resync-scan for files with unusual bare audio packets"
                    ) from exc
                next_position = find_next_known_chunk(stream, big_endian, position + 1)
                if next_position is None:
                    break
                stream.seek(next_position)

        if total_bytes is not None:
            progress.update_to(total_bytes)

    if not chunks:
        raise VID1Error("no VIDD video chunks were found")
    return chunks


# ---------------------------------------------------------------------------
# VID1 picture-header parsing


def canonical_codes(lengths: Sequence[int]) -> dict[tuple[int, int], int]:
    """Return {(bit_length, code): symbol} for canonical Huffman lengths."""
    symbols = sorted((length, symbol) for symbol, length in enumerate(lengths) if length)
    result: dict[tuple[int, int], int] = {}
    code = 0
    previous_length = 0
    for length, symbol in symbols:
        code <<= length - previous_length
        result[(length, code)] = symbol
        code += 1
        previous_length = length
    return result


SPRITE_TRAJECTORY_CODES = canonical_codes(SPRITE_TRAJECTORY_LENGTHS)
MAX_SPRITE_CODE_LENGTH = max(SPRITE_TRAJECTORY_LENGTHS)


def read_sprite_length(bits: BitReaderMSB) -> int:
    code = 0
    for length in range(1, MAX_SPRITE_CODE_LENGTH + 1):
        code = (code << 1) | bits.read_bit()
        symbol = SPRITE_TRAJECTORY_CODES.get((length, code))
        if symbol is not None:
            return symbol
    raise VID1Error("invalid sprite trajectory VLC")


def read_sprite_trajectory(bits: BitReaderMSB, point_count: int) -> tuple[int, ...]:
    if not 0 <= point_count <= 3:
        raise VID1Error(f"invalid sprite warping point count {point_count}")
    start = bits.bitpos
    for point in range(point_count):
        x_length = read_sprite_length(bits)
        if x_length:
            bits.read(x_length)
        marker = bits.read_bit()
        if marker != 1:
            raise VID1Error(f"missing sprite x/y marker at warping point {point}")
        y_length = read_sprite_length(bits)
        if y_length:
            bits.read(y_length)
        marker = bits.read_bit()
        if marker != 1:
            raise VID1Error(f"missing sprite trajectory marker at warping point {point}")
    return bits.bit_slice(start, bits.bitpos)


def encode_sprite_length(symbol: int) -> tuple[int, ...]:
    if not 0 <= symbol < len(SPRITE_TRAJECTORY_LENGTHS):
        raise VID1Error(f"sprite trajectory component needs {symbol} bits; maximum is 14")
    for (length, code), candidate in SPRITE_TRAJECTORY_CODES.items():
        if candidate == symbol:
            return tuple((code >> shift) & 1 for shift in range(length - 1, -1, -1))
    raise AssertionError(symbol)


def encode_sprite_component(value: int) -> tuple[int, ...]:
    if value == 0:
        return encode_sprite_length(0)
    length = abs(value).bit_length()
    if length > 14:
        raise VID1Error(
            f"sprite trajectory component {value} needs {length} bits; maximum is 14"
        )
    # MPEG-4 get_xbits represents negatives by subtracting (2^length - 1)
    # from an unsigned code whose top bit is zero.  This is its inverse.
    encoded = value if value > 0 else value + ((1 << length) - 1)
    prefix = encode_sprite_length(length)
    suffix = tuple((encoded >> shift) & 1 for shift in range(length - 1, -1, -1))
    return prefix + suffix


def encode_sprite_trajectory(
    values: Sequence[int], point_count: int
) -> tuple[int, ...]:
    if len(values) != point_count * 2:
        raise VID1Error(
            f"sprite trajectory has {len(values)} components, expected {point_count * 2}"
        )
    result: list[int] = []
    for point in range(point_count):
        result.extend(encode_sprite_component(values[2 * point]))
        result.append(1)
        result.extend(encode_sprite_component(values[2 * point + 1]))
        result.append(1)
    return tuple(result)


def rounded_divide(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    if value >= 0:
        return (value + divisor // 2) // divisor
    return -((-value + divisor // 2) // divisor)


def read_fixed16_sprite_values(
    bits: BitReaderMSB,
    config: SpriteConfig,
) -> tuple[int, ...]:
    """Read the byte-aligned 16-bit GMC extension seen in retail VID1.

    Words are retained as signed Python integers for compatibility with the
    legacy divisor mappings; packed conversion reinterprets them as
    unsigned 16-bit words before unfolding.  This reader deliberately does not
    otherwise assign MPEG-4 semantics to the values.
    The public VID1 notes label the source fields only tentatively, and the
    supplied stream uses a two-component non-zero prefix followed by zero
    components.  Semantic conversion is handled separately below.
    """
    padding = (-bits.bitpos) & 7
    if padding and bits.read(padding) != 0:
        raise VID1Error("non-zero padding before fixed16 GMC data")

    component_count = config.warping_points * 2
    if bits.bits_left < component_count * 16:
        raise VID1Error(
            f"fixed16 GMC field needs {component_count * 2} bytes, "
            f"but only {bits.bits_left // 8} remain"
        )

    raw: list[int] = []
    for _ in range(component_count):
        value = bits.read(16)
        if value & 0x8000:
            value -= 0x10000
        raw.append(value)
    return tuple(raw)


def folded_signed_decode(code: int) -> int:
    """Decode legacy 0,-1,+1,-2,+2,... zig-zag folding.

    This was the v4 interpretation.  It is retained only for the explicit
    ``--fixed16-gmc-mode zigzag`` diagnostic mode.
    """
    if code < 0:
        raise ValueError("folded signed code cannot be negative")
    return -(code // 2 + 1) if code & 1 else code // 2


def signed_magnitude_decode(code: int) -> int:
    """Decode the VID1 packed sign/magnitude representation.

    The low bit is the sign and the remaining bits are the magnitude:
    ``2*m`` means ``+m`` and ``2*m+1`` means ``-m``.  Code 1 is therefore a
    redundant negative zero; the supplied retail stream never emits it.
    """
    if code < 0:
        raise ValueError("signed-magnitude code cannot be negative")
    magnitude = code >> 1
    return -magnitude if code & 1 else magnitude


def fixed16_packed_shifts(config: SpriteConfig) -> tuple[int, int]:
    """Return the observed horizontal/vertical packing shifts.

    The retail stream supplied with the bug report has sprite accuracy 3 and
    stores the horizontal sign/magnitude code in bits 15..2 and the vertical
    code in bits 15..3.  No public VID1 document currently defines this layout,
    so do not silently generalise it to other accuracy values.
    """
    if config.accuracy != 3:
        raise VID1Error(
            "the packed fixed16 mapping is only verified for sprite "
            f"accuracy 3, not {config.accuracy}"
        )
    return 2, 3


def fixed16_looks_like_packed(
    raw: Sequence[int],
    config: SpriteConfig,
) -> bool:
    if len(raw) != config.warping_points * 2 or not raw:
        return False
    try:
        x_shift, y_shift = fixed16_packed_shifts(config)
    except VID1Error:
        return False
    for component_index, signed_word in enumerate(raw):
        unsigned_word = signed_word & 0xFFFF
        shift = x_shift if component_index % 2 == 0 else y_shift
        if unsigned_word & ((1 << shift) - 1):
            return False
    return True


def decode_fixed16_packed_values(
    raw: Sequence[int],
    config: SpriteConfig,
    *,
    frame_index: int,
    signed_mode: str,
) -> tuple[int, ...]:
    """Decode the axis-shifted packed words into MPEG-4 trajectory values.

    ``signed_mode='vid1'`` uses the sign/magnitude mapping confirmed by the
    supplied stream.  ``signed_mode='zigzag'`` reproduces v4's off-by-one
    negative mapping for comparison.
    """
    if signed_mode not in ("vid1", "zigzag"):
        raise ValueError(f"unknown packed signed mode {signed_mode}")
    x_shift, y_shift = fixed16_packed_shifts(config)
    converted: list[int] = []
    decoder = signed_magnitude_decode if signed_mode == "vid1" else folded_signed_decode
    for component_index, signed_word in enumerate(raw):
        unsigned_word = signed_word & 0xFFFF
        shift = x_shift if component_index % 2 == 0 else y_shift
        discarded_mask = (1 << shift) - 1
        if unsigned_word & discarded_mask:
            axis = "x" if component_index % 2 == 0 else "y"
            raise VID1Error(
                f"frame {frame_index}: packed GMC {axis} word "
                f"0x{unsigned_word:04x} has non-zero low {shift} bit(s)"
            )
        converted.append(decoder(unsigned_word >> shift))
    return tuple(converted)


def convert_fixed16_sprite_trajectory(
    raw: Sequence[int],
    source_config: SpriteConfig,
    *,
    conversion_divisor: int,
    requested_mapping: str,
    state: ParseState,
    frame_index: int,
) -> tuple[tuple[int, ...], SpriteConfig, str]:
    """Map the proprietary byte-aligned 16-bit GMC representation.

    ``vid1`` is the mapping identified from the supplied retail stream:
    x is signed_magnitude(raw >> 2), y is signed_magnitude(raw >> 3), repeated
    for each configured point.  ``zigzag`` preserves v4's off-by-one negative
    interpretation for comparison.  ``translation`` and ``points`` retain the
    older divisor mappings for other VID1 variants.
    """
    if conversion_divisor <= 0:
        raise VID1Error("fixed16 GMC conversion divisor must be positive")
    if requested_mapping not in ("auto", "vid1", "zigzag", "translation", "points"):
        raise ValueError(f"unknown fixed16 GMC mapping {requested_mapping}")

    selected = requested_mapping
    if selected == "auto":
        if state.fixed16_mapping is None:
            if fixed16_looks_like_packed(raw, source_config):
                state.fixed16_mapping = "vid1"
            elif source_config.warping_points == 1:
                state.fixed16_mapping = "translation"
            elif (
                source_config.warping_points >= 2
                and len(raw) >= 4
                and all(value == 0 for value in raw[2:])
            ):
                state.fixed16_mapping = "translation"
            else:
                state.fixed16_mapping = "points"
        selected = state.fixed16_mapping

    if selected in ("vid1", "zigzag"):
        converted = decode_fixed16_packed_values(
            raw,
            source_config,
            frame_index=frame_index,
            signed_mode=selected,
        )
        output_config = source_config
        mapping_name = selected
    else:
        if selected == "translation":
            if len(raw) < 2:
                raise VID1Error(
                    f"frame {frame_index}: fixed16 translation needs an x/y pair"
                )
            if (
                requested_mapping == "auto"
                and len(raw) > 2
                and any(value != 0 for value in raw[2:])
            ):
                raise VID1Error(
                    f"frame {frame_index}: fixed16 GMC auto-detection selected the "
                    "one-point translation layout, but a later frame has non-zero "
                    "trailing components; retry with --fixed16-gmc-mode points"
                )
            source_values = tuple(raw[:2])
            output_config = SpriteConfig(
                warping_points=1,
                accuracy=source_config.accuracy,
            )
            mapping_name = "translation"
        else:
            source_values = tuple(raw)
            output_config = source_config
            mapping_name = "points"

        converted = tuple(
            rounded_divide(value, conversion_divisor) for value in source_values
        )

    trajectory = encode_sprite_trajectory(converted, output_config.warping_points)
    return trajectory, output_config, mapping_name


def parse_sprite_trajectory(
    bits: BitReaderMSB,
    config: SpriteConfig,
    *,
    requested_format: str,
    state: ParseState,
    frame_index: int,
    fixed16_divisor: int,
    fixed16_mapping: str,
) -> tuple[
    tuple[int, ...],
    str,
    tuple[int, ...],
    SpriteConfig,
    str,
]:
    selected = state.gmc_format if requested_format == "auto" else requested_format
    if selected is None:
        selected = "auto"

    errors: list[str] = []
    if selected in ("auto", "mpeg4"):
        start = bits.bitpos
        try:
            trajectory = read_sprite_trajectory(bits, config.warping_points)
            if requested_format == "auto":
                state.gmc_format = "mpeg4"
            state.output_sprite = config
            return trajectory, "mpeg4", (), config, "mpeg4"
        except VID1Error as exc:
            bits.bitpos = start
            errors.append(f"mpeg4: {exc}")
            if selected == "mpeg4":
                raise VID1Error(f"frame {frame_index}: {exc}") from exc

    if selected in ("auto", "fixed16"):
        start = bits.bitpos
        try:
            raw = read_fixed16_sprite_values(bits, config)
            trajectory, output_config, mapping_name = convert_fixed16_sprite_trajectory(
                raw,
                config,
                conversion_divisor=fixed16_divisor,
                requested_mapping=fixed16_mapping,
                state=state,
                frame_index=frame_index,
            )
            if requested_format == "auto":
                state.gmc_format = "fixed16"
            state.output_sprite = output_config
            return (
                trajectory,
                "fixed16",
                raw,
                output_config,
                mapping_name,
            )
        except VID1Error as exc:
            bits.bitpos = start
            errors.append(f"fixed16: {exc}")
            if selected == "fixed16":
                raise VID1Error(f"frame {frame_index}: {exc}") from exc

    raise VID1Error(
        f"frame {frame_index}: could not parse GMC trajectory "
        f"({'; '.join(errors)})"
    )


def matrix_neighbor_score(matrix: Sequence[int]) -> int:
    score = 0
    for y in range(8):
        for x in range(8):
            here = matrix[y * 8 + x]
            if x + 1 < 8:
                score += abs(here - matrix[y * 8 + x + 1])
            if y + 1 < 8:
                score += abs(here - matrix[(y + 1) * 8 + x])
    return score


def normalize_matrix(
    raw: Sequence[int],
    order: str,
    *,
    lenient: bool,
) -> tuple[tuple[int, ...], str]:
    if len(raw) != 64:
        raise VID1Error(f"quantisation matrix has {len(raw)} entries, expected 64")
    repaired = list(raw)
    if any(value == 0 for value in repaired):
        if not lenient:
            raise VID1Error("VID1 quantisation matrix contains a zero entry")
        repaired = [value if value else 1 for value in repaired]

    row_candidate = tuple(repaired)
    zigzag_candidate_list = [0] * 64
    for scan_index, coefficient_index in enumerate(ZIGZAG):
        zigzag_candidate_list[coefficient_index] = repaired[scan_index]
    zigzag_candidate = tuple(zigzag_candidate_list)

    if order == "row":
        return row_candidate, "row"
    if order == "zigzag":
        return zigzag_candidate, "zigzag"
    if order != "auto":
        raise ValueError(f"unknown matrix order {order}")

    row_score = matrix_neighbor_score(row_candidate)
    zigzag_score = matrix_neighbor_score(zigzag_candidate)
    if zigzag_score + max(16, row_score // 20) < row_score:
        return zigzag_candidate, "zigzag(auto)"
    return row_candidate, "row(auto)"


def parse_picture(
    packet: ByteBuffer,
    *,
    index: int,
    chunk_offset: int,
    state: ParseState,
    matrix_order: str,
    lenient: bool,
    gmc_format: str,
    fixed16_divisor: int,
    fixed16_mapping: str,
) -> Picture:
    if len(packet) < 8:
        raise VID1Error(f"frame {index}: packet is only {len(packet)} bytes")

    bits = BitReaderMSB(packet)
    ignored16 = bits.read(16)
    frame_type = bits.read(2)
    extended = bool(bits.read_bit())

    source_sprite = state.source_sprite
    output_sprite = state.output_sprite
    field_syntax = state.field_syntax
    extension_flag_2 = state.extension_flag_2
    uses_extended_quant = False
    luma_order: Optional[str] = None
    chroma_order: Optional[str] = None

    if extended:
        sprite_present = bool(bits.read_bit())
        if sprite_present:
            source_sprite = SpriteConfig(
                warping_points=bits.read(2),
                accuracy=bits.read(2),
            )
            state.source_sprite = source_sprite
            # For explicit modes, or after auto mode has identified the
            # stream representation, the output VOL shape is known immediately.
            # Before the first auto-detected S picture it remains unresolved and
            # is backfilled by resolve_sprite_vol_state() after the sequence has
            # been parsed.  Always clear the old value on a new source update so
            # a changed configuration cannot temporarily inherit stale VOL state.
            resolved_sprite: Optional[SpriteConfig] = None
            if gmc_format == "mpeg4" or state.gmc_format == "mpeg4":
                resolved_sprite = source_sprite
            elif gmc_format == "fixed16" or state.gmc_format == "fixed16":
                selected_mapping = (
                    fixed16_mapping
                    if fixed16_mapping != "auto"
                    else state.fixed16_mapping
                )
                if selected_mapping in ("points", "vid1", "zigzag"):
                    resolved_sprite = source_sprite
                elif (
                    selected_mapping == "translation"
                    and source_sprite.warping_points
                ):
                    resolved_sprite = SpriteConfig(1, source_sprite.accuracy)
            state.output_sprite = resolved_sprite
            output_sprite = resolved_sprite

        uses_extended_quant = bool(bits.read_bit())
        if uses_extended_quant:
            luma_present = bool(bits.read_bit())
            if luma_present:
                raw_luma = tuple(bits.read(8) for _ in range(64))
                state.luma_matrix, luma_order = normalize_matrix(
                    raw_luma, matrix_order, lenient=lenient
                )
            chroma_present = bool(bits.read_bit())
            if chroma_present:
                raw_chroma = tuple(bits.read(8) for _ in range(64))
                state.chroma_matrix, chroma_order = normalize_matrix(
                    raw_chroma, matrix_order, lenient=lenient
                )
        # These two stream-state bits were previously discarded.  Streams such
        # as THAW's GameCube movies set the first bit and then use the
        # interlace-capable MPEG-4 macroblock grammar (field-DCT/field-prediction
        # syntax) while still carrying progressive-looking pictures.  The
        # second bit remains unidentified, so retain it without assigning
        # semantics.
        field_syntax = bool(bits.read_bit())
        extension_flag_2 = bool(bits.read_bit())
        state.field_syntax = field_syntax
        state.extension_flag_2 = extension_flag_2

    rounding = bits.read_bit()
    intra_dc = bits.read(3)
    quant = bits.read(5)
    if quant == 0:
        raise VID1Error(f"frame {index}: quantiser 0 is invalid")

    fcode_forward = 1
    fcode_backward = 1
    if frame_type != 0:
        fcode_forward = bits.read(3)
        if fcode_forward == 0:
            raise VID1Error(f"frame {index}: forward fcode 0 is invalid")
    if frame_type == 2:
        fcode_backward = bits.read(3)
        if fcode_backward == 0:
            raise VID1Error(f"frame {index}: backward fcode 0 is invalid")

    timecode = bits.read(32)
    trajectory: tuple[int, ...] = ()
    gmc_source_format: Optional[str] = None
    gmc_raw_values: tuple[int, ...] = ()
    gmc_mapping: Optional[str] = None
    if frame_type == 3:
        if source_sprite is None:
            raise VID1Error(
                f"frame {index}: S/GMC frame has no active sprite configuration"
            )
        (
            trajectory,
            gmc_source_format,
            gmc_raw_values,
            output_sprite,
            gmc_mapping,
        ) = parse_sprite_trajectory(
            bits,
            source_sprite,
            requested_format=gmc_format,
            state=state,
            frame_index=index,
            fixed16_divisor=fixed16_divisor,
            fixed16_mapping=fixed16_mapping,
        )
    else:
        output_sprite = state.output_sprite

    bits.align_byte()
    payload_start = bits.bytepos
    p_frame_preamble: Optional[int] = None
    if frame_type == 1 and field_syntax:
        # In this VID1 syntax variant every P picture has one byte between the
        # byte-aligned proprietary picture header and the first MPEG-4
        # macroblock bit.  It is always zero in the retail stream and is not
        # part of the MPEG-4 payload; copying it made FFmpeg treat the frame as
        # corrupt and conceal most of the picture.
        if payload_start >= len(packet):
            raise VID1Error(
                f"frame {index}: field-syntax P picture has no preamble byte"
            )
        p_frame_preamble = int(packet[payload_start])
        payload_start += 1
        if p_frame_preamble != 0:
            raise VID1Error(
                f"frame {index}: field-syntax P preamble is "
                f"0x{p_frame_preamble:02x}, expected 0x00"
            )

    payload = packet[payload_start:]
    if not payload:
        raise VID1Error(f"frame {index}: no macroblock payload after picture header")

    luma_pair: Optional[MatrixPair] = None
    chroma_pair: Optional[MatrixPair] = None
    if uses_extended_quant:
        luma_pair = MatrixPair(
            intra=state.luma_matrix or MPEG4_DEFAULT_INTRA,
            inter=state.luma_matrix or MPEG4_DEFAULT_INTER,
        )
        chroma_pair = MatrixPair(
            intra=state.chroma_matrix or MPEG4_DEFAULT_INTRA,
            inter=state.chroma_matrix or MPEG4_DEFAULT_INTER,
        )

    return Picture(
        index=index,
        chunk_offset=chunk_offset,
        frame_type=frame_type,
        rounding=rounding,
        intra_dc_vlc_thr_idx=intra_dc,
        quant=quant,
        fcode_forward=fcode_forward,
        fcode_backward=fcode_backward,
        timecode=timecode,
        payload=payload,
        ignored16=ignored16,
        extended_info_present=extended,
        field_syntax=field_syntax,
        extension_flag_2=extension_flag_2,
        p_frame_preamble=p_frame_preamble,
        sprite_config=output_sprite,
        trajectory_bits=trajectory,
        uses_extended_quant=uses_extended_quant,
        luma_quant=luma_pair,
        chroma_quant=chroma_pair,
        matrix_order_luma=luma_order,
        matrix_order_chroma=chroma_order,
        gmc_source_format=gmc_source_format,
        gmc_raw_values=gmc_raw_values,
        source_sprite_config=source_sprite,
        gmc_mapping=gmc_mapping,
        gmc_conversion_divisor=(
            fixed16_divisor
            if gmc_source_format == "fixed16"
            and gmc_mapping in ("translation", "points")
            else None
        ),
    )


def resolve_sprite_vol_state(
    pictures: Sequence[Picture],
    *,
    reporter: Optional[Reporter] = None,
) -> list[Picture]:
    """Backfill an auto-resolved MPEG-4 sprite shape within each source-state run.

    VID1 declares its source sprite configuration before ordinary I/P pictures,
    but auto mode may need the first S trajectory to distinguish the proprietary
    fixed16 mappings.  Emitting those earlier pictures with ``sprite=None`` would
    write a no-sprite VOL followed by a GMC VOL at the first S picture.  Several
    MPEG-4 decoder versions treat that mid-GOP VOL replacement as a sequence
    reset and lose the reference picture, producing white/block-concealed output.

    Once all headers in a run have been parsed, the output shape is known.  Apply
    it to the preceding pictures so the adapter announces one stable VOL before
    the first reference picture.  A run with no S picture remains unresolved and
    needs no sprite-enabled VOL.
    """
    resolved_pictures = list(pictures)
    backfilled = 0
    start = 0

    while start < len(resolved_pictures):
        source_config = resolved_pictures[start].source_sprite_config
        end = start + 1
        while (
            end < len(resolved_pictures)
            and resolved_pictures[end].source_sprite_config == source_config
        ):
            end += 1

        output_configs = {
            picture.sprite_config
            for picture in resolved_pictures[start:end]
            if picture.sprite_config is not None
        }
        if len(output_configs) > 1:
            details = ", ".join(
                f"{config.warping_points} point(s), accuracy {config.accuracy}"
                for config in sorted(
                    output_configs,
                    key=lambda config: (config.warping_points, config.accuracy),
                )
            )
            first_frame = resolved_pictures[start].index
            last_frame = resolved_pictures[end - 1].index
            raise VID1Error(
                f"frames {first_frame}..{last_frame}: one VID1 sprite-state run "
                f"resolved to conflicting MPEG-4 VOL shapes ({details})"
            )

        if output_configs:
            output_config = next(iter(output_configs))
            for index in range(start, end):
                if resolved_pictures[index].sprite_config != output_config:
                    resolved_pictures[index] = replace(
                        resolved_pictures[index],
                        sprite_config=output_config,
                    )
                    backfilled += 1

        start = end

    if backfilled and reporter is not None:
        reporter.info(
            f"backfilled resolved GMC VOL state across {backfilled} earlier frame(s)",
            level=1,
        )
    return resolved_pictures


def parse_picture_sequence(
    chunks: Sequence[RawVideoChunk],
    *,
    header_skip: int,
    matrix_order: str,
    lenient: bool,
    gmc_format: str,
    fixed16_divisor: int,
    fixed16_mapping: str,
    limit: Optional[int] = None,
    reporter: Optional[Reporter] = None,
    progress_description: str = "Parsing VID1 pictures",
) -> list[Picture]:
    if header_skip < 0:
        raise VID1Error("VIDD header skip must be non-negative")
    state = ParseState()
    pictures: list[Picture] = []
    selected = chunks if limit is None else chunks[:limit]
    iterable: Iterable[RawVideoChunk] = selected
    if reporter is not None:
        iterable = reporter.track(
            selected,
            total=len(selected),
            description=progress_description,
            unit="frame",
        )
    for chunk in iterable:
        if len(chunk.body) <= header_skip:
            raise VID1Error(
                f"frame {chunk.index}: VIDD body is {len(chunk.body)} bytes, "
                f"not enough for a {header_skip}-byte subheader"
            )
        # Keep the macroblock payload as a zero-copy view of the chunk body.
        # Large game videos can otherwise briefly occupy roughly twice their
        # compressed video size while all pictures are retained for B-frame
        # timing and VOL-state reconstruction.
        packet = memoryview(chunk.body)[header_skip:]
        picture = parse_picture(
            packet,
            index=chunk.index,
            chunk_offset=chunk.file_offset,
            state=state,
            matrix_order=matrix_order,
            lenient=lenient,
            gmc_format=gmc_format,
            fixed16_divisor=fixed16_divisor,
            fixed16_mapping=fixed16_mapping,
        )
        pictures.append(picture)
    return resolve_sprite_vol_state(pictures, reporter=reporter)


def probe_header_skip(
    chunks: Sequence[RawVideoChunk],
    skip: int,
    *,
    matrix_order: str,
    probe_frames: int,
    gmc_format: str,
    fixed16_divisor: int,
    fixed16_mapping: str,
) -> SkipProbe:
    try:
        pictures = parse_picture_sequence(
            chunks,
            header_skip=skip,
            matrix_order=matrix_order,
            lenient=True,
            gmc_format=gmc_format,
            fixed16_divisor=fixed16_divisor,
            fixed16_mapping=fixed16_mapping,
            limit=probe_frames,
        )
    except Exception as exc:
        return SkipProbe(skip=skip, score=-100000, parsed=0, first_type=None, error=str(exc))

    score = 100 * len(pictures)
    if pictures:
        if pictures[0].frame_type == 0:
            score += 30
        elif pictures[0].frame_type in (1, 3):
            score += 5
        score += 4 * sum(pic.quant not in (1, 31) for pic in pictures)
        # The supplied retail stream uses 0x0001 as the 16-bit sync marker.
        # This is a much stronger discriminator than preferring a hard-coded
        # demuxer skip, because this parser intentionally retains that marker.
        score += 20 * sum(pic.ignored16 == 0x0001 for pic in pictures)
        score += 2 * sum(pic.ignored16 in (0, 0xFFFF) for pic in pictures)
        score -= 3 * sum(pic.ignored16 not in (0, 1, 0xFFFF) for pic in pictures)
        score += len({pic.timecode for pic in pictures})
        timecodes = [pic.timecode for pic in pictures]
        if all(b >= a for a, b in zip(timecodes, timecodes[1:])):
            score += 20
        if skip == 4:
            score += 5
    return SkipProbe(
        skip=skip,
        score=score,
        parsed=len(pictures),
        first_type=pictures[0].frame_type if pictures else None,
        error=None,
    )


def choose_header_skip(
    chunks: Sequence[RawVideoChunk],
    requested: str,
    *,
    matrix_order: str,
    probe_frames: int,
    gmc_format: str,
    fixed16_divisor: int,
    fixed16_mapping: str,
    reporter: Reporter,
) -> tuple[int, list[SkipProbe]]:
    if requested != "auto":
        try:
            value = int(requested, 0)
        except ValueError as exc:
            raise VID1Error("--vidd-header-skip must be 'auto' or an integer") from exc
        if value < 0:
            raise VID1Error("--vidd-header-skip cannot be negative")
        return value, []

    candidates = (6, 4, 2, 0, 8)
    probes = [
        probe_header_skip(
            chunks,
            candidate,
            matrix_order=matrix_order,
            probe_frames=probe_frames,
            gmc_format=gmc_format,
            fixed16_divisor=fixed16_divisor,
            fixed16_mapping=fixed16_mapping,
        )
        for candidate in candidates
    ]
    valid = [probe for probe in probes if probe.error is None]
    if not valid:
        details = "; ".join(f"skip {probe.skip}: {probe.error}" for probe in probes)
        raise VID1Error(f"could not identify the VIDD subheader length ({details})")
    best = max(valid, key=lambda probe: (probe.score, probe.skip == 4, -abs(probe.skip - 4)))
    reporter.info(
        "VIDD skip probes: "
        + ", ".join(
            f"{probe.skip}={'error' if probe.error else probe.score}"
            for probe in probes
        ),
        level=2,
    )
    return best.skip, probes


# ---------------------------------------------------------------------------
# Timing


def parse_fraction(text: str) -> Fraction:
    cleaned = text.strip().replace(":", "/")
    try:
        value = Fraction(cleaned)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(f"invalid rational value: {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("rate must be positive")
    return value


def choose_fps(requested: str, info: VID1Info, reporter: Reporter) -> Fraction:
    if requested != "auto":
        rate = parse_fraction(requested)
    elif info.header_rate is not None and Fraction(1, 1) <= info.header_rate <= Fraction(240, 1):
        rate = info.header_rate
        reporter.info(
            f"using VIDH frame rate {rate.numerator}/{rate.denominator}",
            level=2,
        )
    else:
        rate = Fraction(24, 1)
        reporter.info("using 24 fps fallback because VIDH has no plausible frame rate")

    rate = rate.limit_denominator(100000)
    if rate.numerator > 65535:
        raise VID1Error(
            f"frame-rate numerator {rate.numerator} exceeds MPEG-4's 16-bit "
            "time_increment_resolution; pass an equivalent lower-resolution rate"
        )
    return rate


def unwrap_timecodes(values: Sequence[int]) -> list[int]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum - minimum <= 0x80000000:
        return list(values)
    # A cluster around 0xffffffff followed by a cluster around zero is most
    # likely a 32-bit wrap.  Shift the low cluster forward one epoch.
    return [value + (1 << 32) if value < 0x80000000 else value for value in values]


def timecode_order_is_plausible(pictures: Sequence[Picture], values: Sequence[int]) -> bool:
    if len(set(values)) != len(values):
        return False
    references: list[int] = []
    for picture, value in zip(pictures, values):
        if picture.frame_type != 2:
            if references and value <= references[-1]:
                return False
            references.append(value)
        else:
            if len(references) < 2:
                return False
            if not references[-2] < value < references[-1]:
                return False
    return True


def assign_timecode_indices(pictures: Sequence[Picture]) -> Optional[list[int]]:
    values = unwrap_timecodes([picture.timecode for picture in pictures])
    if not timecode_order_is_plausible(pictures, values):
        return None
    ordered = sorted(values)
    rank = {value: index for index, value in enumerate(ordered)}
    return [rank[value] for value in values]


def assign_gop_indices(pictures: Sequence[Picture]) -> list[int]:
    if not pictures:
        return []
    if pictures[0].frame_type == 2:
        raise VID1Error("stream begins with a B frame; cannot synthesize reference timing")

    result = [-1] * len(pictures)
    first_reference = next(
        (index for index, picture in enumerate(pictures) if picture.frame_type != 2),
        None,
    )
    if first_reference is None:
        raise VID1Error("stream contains only B frames")
    if first_reference != 0:
        raise VID1Error("B frames precede the first reference frame")

    result[0] = 0
    previous_reference_display = 0
    index = 1
    while index < len(pictures):
        if pictures[index].frame_type == 2:
            raise VID1Error(
                f"frame {index}: B frame appears before its future reference in coded order"
            )

        reference_index = index
        b_start = reference_index + 1
        b_end = b_start
        while b_end < len(pictures) and pictures[b_end].frame_type == 2:
            b_end += 1
        b_count = b_end - b_start
        current_reference_display = previous_reference_display + b_count + 1
        result[reference_index] = current_reference_display
        for offset, b_index in enumerate(range(b_start, b_end), start=1):
            result[b_index] = previous_reference_display + offset
        previous_reference_display = current_reference_display
        index = b_end

    if any(value < 0 for value in result):
        raise AssertionError("internal timing assignment failure")
    return result


def assign_display_indices(
    pictures: Sequence[Picture],
    mode: str,
    reporter: Reporter,
) -> tuple[list[Picture], str]:
    selected_mode = mode
    indices: Optional[list[int]] = None
    if mode in ("auto", "timecode"):
        indices = assign_timecode_indices(pictures)
        if indices is not None:
            selected_mode = "timecode"
        elif mode == "timecode":
            raise VID1Error(
                "VID1 timecodes are not a plausible MPEG-4 display order; "
                "use --timing gop"
            )
    if indices is None:
        indices = assign_gop_indices(pictures)
        selected_mode = "gop"
        if mode == "auto":
            reporter.warn(
                "VID1 timecodes were not usable as display timestamps; "
                "synthesizing timing from I/P/S and B-frame coded order"
            )

    return [replace(picture, display_index=index) for picture, index in zip(pictures, indices)], selected_mode


# ---------------------------------------------------------------------------
# MPEG-4 Part 2 adapter


def time_increment_bits(time_resolution: int) -> int:
    if not 1 <= time_resolution <= 65535:
        raise VID1Error("MPEG-4 time_increment_resolution must be 1..65535")
    return max(1, (time_resolution - 1).bit_length())


def selected_quant_pair(picture: Picture, plane: str) -> Optional[MatrixPair]:
    if not picture.uses_extended_quant:
        return None
    if plane == "luma":
        return picture.luma_quant
    if plane == "chroma":
        return picture.chroma_quant
    raise ValueError(f"unknown matrix plane {plane}")


def write_quant_matrix(writer: BitWriter, matrix: Sequence[int]) -> None:
    if len(matrix) != 64:
        raise ValueError("quantisation matrix must have 64 entries")
    for coefficient_index in ZIGZAG:
        value = int(matrix[coefficient_index])
        if not 1 <= value <= 255:
            raise VID1Error(f"quantisation matrix entry {value} is outside 1..255")
        writer.write_bits(8, value)


def write_visual_object_prefix(writer: BitWriter) -> None:
    # Visual Object Sequence, Advanced Simple Profile level 5.
    writer.start_code(0xB0)
    writer.write_bits(8, 0xF5)

    # Visual Object: no identifier, type video, no video_signal_type.
    writer.start_code(0xB5)
    writer.write_bits(1, 0)
    writer.write_bits(4, 1)
    writer.write_bits(1, 0)
    writer.align_zero()

    # Video Object start, object id zero.
    writer.start_code(0x00)


def write_vol(
    writer: BitWriter,
    config: AdapterConfig,
    *,
    quant_pair: Optional[MatrixPair],
    sprite: Optional[SpriteConfig],
    field_syntax: bool,
) -> None:
    if not 1 <= config.width <= 8191 or not 1 <= config.height <= 8191:
        raise VID1Error(
            f"MPEG-4 VOL dimensions are outside 1..8191: {config.width}x{config.height}"
        )

    time_resolution = config.fps.numerator
    tick = config.fps.denominator
    increment_bits = time_increment_bits(time_resolution)
    if tick >= (1 << increment_bits):
        raise VID1Error(
            f"frame-rate denominator {tick} does not fit the MPEG-4 fixed increment field"
        )

    writer.start_code(0x20)  # Video Object Layer, layer zero.
    writer.write_bits(1, 0)       # random_accessible_vol
    writer.write_bits(8, 0x11)    # Advanced Simple video_object_type_indication
    writer.write_bits(1, 1)       # is_object_layer_identifier
    writer.write_bits(4, 5)       # video_object_layer_verid
    writer.write_bits(3, 1)       # video_object_layer_priority
    writer.write_bits(4, 1)       # square pixels

    # Explicit VOL control parameters prevent decoders from forcing low_delay
    # for an unidentified ASP stream.  Chroma format 1 is 4:2:0.
    writer.write_bits(1, 1)       # vol_control_parameters
    writer.write_bits(2, 1)       # chroma_format = 4:2:0
    writer.write_bits(1, 0 if config.has_b_frames else 1)
    writer.write_bits(1, 0)       # vbv_parameters absent

    writer.write_bits(2, 0)       # rectangular shape
    writer.write_bits(1, 1)       # marker
    writer.write_bits(16, time_resolution)
    writer.write_bits(1, 1)       # marker
    writer.write_bits(1, 1)       # fixed_vop_rate
    writer.write_bits(increment_bits, tick)
    writer.write_bits(1, 1)       # marker before width
    writer.write_bits(13, config.width)
    writer.write_bits(1, 1)       # marker before height
    writer.write_bits(13, config.height)
    writer.write_bits(1, 1)       # marker after height
    # VID1's field-syntax variant carries the two extra per-VOP field flags and
    # uses the corresponding MPEG-4 macroblock grammar.  Setting the VOL flag
    # is an adapter requirement; decoded output is still tagged progressive.
    writer.write_bits(1, int(field_syntax))
    writer.write_bits(1, 1)       # obmc_disable = true

    if sprite is None:
        writer.write_bits(2, 0)   # no sprite, verid != 1
    else:
        writer.write_bits(2, 2)   # GMC sprite
        writer.write_bits(6, sprite.warping_points)
        writer.write_bits(2, sprite.accuracy)
        writer.write_bits(1, 0)   # sprite_brightness_change

    writer.write_bits(1, 0)       # not_8_bit = false
    if quant_pair is None:
        writer.write_bits(1, 0)   # H.263 quantisation
    else:
        writer.write_bits(1, 1)   # MPEG quantisation matrices
        writer.write_bits(1, 1)   # custom intra matrix present
        write_quant_matrix(writer, quant_pair.intra)
        writer.write_bits(1, 1)   # custom non-intra matrix present
        write_quant_matrix(writer, quant_pair.inter)

    writer.write_bits(1, 0)       # quarter_sample = false
    writer.write_bits(1, 1)       # complexity_estimation_disable
    # Keep MPEG-4 resync/end-of-VOP recognition enabled.  VID1 retains the
    # standard MCBPC sync/stuffing handling, and disabling it makes FFmpeg
    # misinterpret valid packet-end padding as another macroblock in some B
    # pictures.
    writer.write_bits(1, 0)       # resync_marker_disable = false
    writer.write_bits(1, 0)       # data_partitioned
    writer.write_bits(1, 0)       # newpred_enable
    writer.write_bits(1, 0)       # reduced_resolution_vop_enable
    writer.write_bits(1, 0)       # scalability
    writer.align_zero()


def write_vop_header(
    writer: BitWriter,
    picture: Picture,
    config: AdapterConfig,
    state: MPEG4TimeState,
) -> None:
    time_value = picture.display_index * config.fps.denominator
    seconds = time_value // state.time_resolution
    increment = time_value % state.time_resolution

    if picture.frame_type == 2:
        modulo = seconds - state.last_time_base_for_b
    else:
        modulo = seconds - state.current_time_base
        state.last_time_base_for_b = state.current_time_base
        state.current_time_base = seconds
    if modulo < 0:
        raise VID1Error(
            f"frame {picture.index}: negative MPEG-4 modulo_time_base; "
            "try --timing gop"
        )

    writer.start_code(0xB6)
    writer.write_bits(2, picture.frame_type)
    for _ in range(modulo):
        writer.write_bit(1)
    writer.write_bit(0)
    writer.write_bit(1)  # marker before time increment
    writer.write_bits(time_increment_bits(state.time_resolution), increment)
    writer.write_bit(1)  # marker before vop_coded
    writer.write_bit(1)  # vop_coded

    if picture.frame_type in (1, 3):
        writer.write_bit(picture.rounding)
    writer.write_bits(3, picture.intra_dc_vlc_thr_idx)

    if picture.field_syntax:
        # The source pictures do not expose meaningful field-order controls.
        # Neutral values make FFmpeg consume the VID1 field-syntax macroblock
        # grammar without changing the displayed line order.
        writer.write_bit(0)  # top_field_first
        writer.write_bit(0)  # alternate_vertical_scan_flag

    if picture.frame_type == 3:
        writer.write_bit_sequence(picture.trajectory_bits)

    writer.write_bits(5, picture.quant)
    if picture.frame_type != 0:
        writer.write_bits(3, picture.fcode_forward)
    if picture.frame_type == 2:
        writer.write_bits(3, picture.fcode_backward)


def packet_alignment_zero_bits(payload: ByteBuffer) -> int:
    """Return up to seven trailing zero bits used for byte alignment.

    VID1 VIDD chunks are byte-sized and the public picture description says
    the custom header is byte-aligned before the MPEG-4-derived macroblocks.
    A bit writer therefore commonly leaves 0..7 zero fill bits at packet end.
    Those fill bits cannot be copied verbatim after a replacement VOP header:
    the replacement header usually has a different bit length, so the old fill
    plus the new fill can become a complete bogus byte.  MPEG-4's own end
    stuffing terminates in one bits, making the following zero run a useful
    boundary signal.  Never remove more than one byte-alignment field.
    """
    if not payload:
        return 0
    last = int(payload[-1])
    count = 0
    while count < 7 and ((last >> count) & 1) == 0:
        count += 1
    return count


def write_picture_payload(
    writer: BitWriter,
    payload: ByteBuffer,
    *,
    trim_packet_padding: bool,
) -> None:
    trim = packet_alignment_zero_bits(payload) if trim_packet_padding else 0
    bit_count = len(payload) * 8 - trim
    full_bytes, remaining_bits = divmod(bit_count, 8)
    if full_bytes:
        writer.write_bytes_as_bits(payload[:full_bytes])
    if remaining_bits:
        writer.write_bits(remaining_bits, int(payload[full_bytes]) >> (8 - remaining_bits))


def write_adapted_stream(
    output: Path,
    pictures: Sequence[Picture],
    config: AdapterConfig,
    *,
    reporter: Optional[Reporter] = None,
    progress_description: str = "Writing MPEG-4 adapter",
) -> None:
    time_state = MPEG4TimeState(time_resolution=config.fps.numerator)
    previous_vol_key: object = object()

    with output.open("wb") as stream:
        writer = BitWriter(stream)
        write_visual_object_prefix(writer)

        iterable: Iterable[Picture] = pictures
        if reporter is not None:
            iterable = reporter.track(
                pictures,
                total=len(pictures),
                description=progress_description,
                unit="frame",
            )
        for picture in iterable:
            quant_pair = selected_quant_pair(picture, config.matrix_plane)
            # Sprite/GMC parameters are VOL state, not per-VOP state.  Once the
            # VID1 header has supplied them, keep them active for ordinary I/P/B
            # pictures as well as S pictures.  Sprite state and field-syntax
            # state are both VOL-level adapter settings; S-VOP macroblocks alone
            # consume the GMC trajectory, while field syntax changes the common
            # macroblock grammar for every picture type.
            sprite = picture.sprite_config
            vol_key = (quant_pair, sprite, picture.field_syntax)
            if vol_key != previous_vol_key:
                write_vol(
                    writer,
                    config,
                    quant_pair=quant_pair,
                    sprite=sprite,
                    field_syntax=picture.field_syntax,
                )
                previous_vol_key = vol_key

            write_vop_header(writer, picture, config, time_state)
            write_picture_payload(
                writer,
                picture.payload,
                trim_packet_padding=config.trim_payload_padding,
            )
            writer.align_zero()

        writer.close()


def pictures_need_dual_matrix_pass(pictures: Sequence[Picture]) -> bool:
    for picture in pictures:
        if picture.uses_extended_quant and picture.luma_quant != picture.chroma_quant:
            return True
    return False


# ---------------------------------------------------------------------------
# ffmpeg execution and decoded-pixel output


def executable_or_error(name: str) -> str:
    candidate = shutil.which(name)
    if candidate is None:
        raise VID1Error(f"ffmpeg executable not found: {name!r}")
    return candidate


def run_ffmpeg(
    command: Sequence[str],
    reporter: Reporter,
    *,
    description: str,
    expected_frames: Optional[int] = None,
) -> FFmpegRunResult:
    """Run one ffmpeg command while consuming ``-progress`` output live."""
    if not command:
        raise ValueError("empty ffmpeg command")

    # Every command built by this script has exactly one output and its path is
    # the final argument.  Inject ffmpeg's machine-readable progress protocol
    # immediately before that output path, leaving stdout free of media data.
    full_command = list(command[:-1]) + [
        "-progress",
        "pipe:1",
        "-nostats",
        command[-1],
    ]
    reporter.info(
        "+ " + " ".join(shlex.quote(part) for part in full_command),
        level=2,
    )
    reporter.status(description)
    started = time.monotonic()
    last_frame: Optional[int] = None

    with tempfile.TemporaryFile(
        mode="w+t",
        encoding="utf-8",
        errors="replace",
    ) as error_stream:
        process = subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=error_stream,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            with reporter.progress(
                total=expected_frames,
                description=description,
                unit="frame",
            ) as progress:
                for raw_line in process.stdout:
                    line = raw_line.strip()
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    if key == "frame":
                        try:
                            frame = int(value.strip())
                        except ValueError:
                            continue
                        if frame >= 0:
                            last_frame = frame
                            progress.update_to(frame)
                return_code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise

        error_stream.seek(0)
        stderr = error_stream.read()

    elapsed = time.monotonic() - started
    reporter.record_ffmpeg_log(
        description=description,
        command=full_command,
        stderr=stderr,
    )
    if return_code != 0:
        tail = "\n".join(stderr.rstrip().splitlines()[-40:])
        raise ExternalToolError(
            f"{description} failed with exit status {return_code}"
            + (f":\n{tail}" if tail else "")
        )

    if reporter.verbose >= 3 and stderr.strip():
        print(stderr.rstrip(), file=sys.stderr)
    frame_text = f", {last_frame} frames" if last_frame is not None else ""
    reporter.status(
        f"{description} completed in {format_duration(elapsed)}{frame_text}"
    )
    return FFmpegRunResult(
        stderr=stderr,
        frame_count=last_frame,
        elapsed=elapsed,
    )


def warn_on_ffmpeg_damage(stderr: str, reporter: Reporter) -> None:
    markers = ("damaged", "invalid data", "corrupt", "conceal", "error while")
    diagnostic_lines = [
        line
        for line in stderr.splitlines()
        if any(marker in line.lower() for marker in markers)
    ]
    if not diagnostic_lines:
        return
    reporter.warn(
        f"ffmpeg emitted {len(diagnostic_lines)} possible bitstream-damage "
        "diagnostic line(s); use -vvv or --ffmpeg-log PATH for details"
    )
    if reporter.verbose >= 2:
        for line in diagnostic_lines[:10]:
            reporter.info(f"ffmpeg: {line}", level=2)
        if len(diagnostic_lines) > 10:
            reporter.info(
                f"ffmpeg: ... {len(diagnostic_lines) - 10} more matching line(s)",
                level=2,
            )


def yuv420_frame_size(width: int, height: int) -> int:
    chroma_width = (width + 1) // 2
    chroma_height = (height + 1) // 2
    return width * height + 2 * chroma_width * chroma_height


def count_raw_frames(path: Path, width: int, height: int) -> int:
    frame_size = yuv420_frame_size(width, height)
    total = path.stat().st_size
    if total % frame_size:
        raise ExternalToolError(
            f"decoded raw output size {total} is not a multiple of frame size {frame_size}"
        )
    return total // frame_size


def m4v_decode_prefix(
    ffmpeg: str,
    adapted: Path,
    *,
    idct: str,
    decode_threads: int,
    force_progressive: bool,
) -> list[str]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-threads",
        str(decode_threads),
        "-f",
        "m4v",
        "-c:v",
        "mpeg4",
        # VIDD chunks are padded to a container boundary.  FFmpeg's MPEG-4
        # no_padding workaround accepts this packet-end junk after all
        # macroblocks have already been decoded.
        "-bug",
        "no_padding",
        "-idct",
        idct,
        "-i",
        str(adapted),
        "-map",
        "0:v:0",
        "-an",
    ]
    if force_progressive:
        # The adapter may enable MPEG-4's interlace-capable syntax solely so
        # FFmpeg consumes VID1's extra macroblock fields.  setfield changes
        # frame metadata only; it does not deinterlace or alter pixel values.
        command += ["-vf", "setfield=prog"]
    command += [
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "yuv420p",
    ]
    return command


def decode_m4v_to_raw(
    ffmpeg: str,
    adapted: Path,
    raw_output: Path,
    *,
    width: int,
    height: int,
    idct: str,
    decode_threads: int,
    force_progressive: bool,
    expected_frames: int,
    reporter: Reporter,
    description: str = "Decoding MPEG-4 adapter",
) -> DecodeResult:
    command = m4v_decode_prefix(
        ffmpeg,
        adapted,
        idct=idct,
        decode_threads=decode_threads,
        force_progressive=force_progressive,
    ) + [
        "-f",
        "rawvideo",
        "-y",
        str(raw_output),
    ]
    result = run_ffmpeg(
        command,
        reporter,
        description=description,
        expected_frames=expected_frames,
    )
    warn_on_ffmpeg_damage(result.stderr, reporter)
    frame_count = count_raw_frames(raw_output, width, height)
    if frame_count == 0:
        raise ExternalToolError("ffmpeg produced zero decoded frames")
    if result.frame_count is not None and result.frame_count != frame_count:
        reporter.warn(
            f"ffmpeg progress reported {result.frame_count} frames, but raw output "
            f"contains {frame_count} complete frames"
        )
    return DecodeResult(
        raw_path=raw_output,
        frame_count=frame_count,
        stderr=result.stderr,
    )


def decode_m4v_to_output(
    ffmpeg: str,
    adapted: Path,
    output: Path,
    *,
    output_format: str,
    width: int,
    height: int,
    fps: Fraction,
    idct: str,
    decode_threads: int,
    force_progressive: bool,
    expected_frames: int,
    reporter: Reporter,
) -> tuple[int, str]:
    command = m4v_decode_prefix(
        ffmpeg,
        adapted,
        idct=idct,
        decode_threads=decode_threads,
        force_progressive=force_progressive,
    )
    if output_format == "ffv1":
        description = "Decoding and encoding lossless FFV1"
        command += [
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-g",
            "1",
            "-f",
            "matroska",
            "-y",
            str(output),
        ]
    elif output_format == "y4m":
        description = "Decoding and writing uncompressed Y4M"
        command += [
            "-c:v",
            "rawvideo",
            "-f",
            "yuv4mpegpipe",
            "-y",
            str(output),
        ]
    elif output_format == "raw":
        description = "Decoding and writing raw yuv420p"
        command += [
            "-f",
            "rawvideo",
            "-y",
            str(output),
        ]
    else:
        raise AssertionError(f"unhandled output format {output_format}")

    result = run_ffmpeg(
        command,
        reporter,
        description=description,
        expected_frames=expected_frames,
    )
    warn_on_ffmpeg_damage(result.stderr, reporter)
    if not output.is_file() or output.stat().st_size == 0:
        raise ExternalToolError(f"ffmpeg did not create a non-empty output: {output}")

    if output_format == "raw":
        frame_count = count_raw_frames(output, width, height)
    elif result.frame_count is not None:
        frame_count = result.frame_count
    else:
        # ``-progress`` normally always emits frame=.  Retain a usable result
        # for unusual ffmpeg builds, but make the unverifiable fallback clear.
        reporter.warn(
            "ffmpeg did not report a final frame count; using the VID1 packet count"
        )
        frame_count = expected_frames
    if frame_count == 0:
        raise ExternalToolError("ffmpeg produced zero decoded frames")
    return frame_count, result.stderr


def merge_component_passes(
    luma_raw: Path,
    chroma_raw: Path,
    output: Path,
    *,
    width: int,
    height: int,
    luma_frames: int,
    chroma_frames: int,
    reporter: Optional[Reporter] = None,
) -> int:
    if luma_frames != chroma_frames:
        raise ExternalToolError(
            f"luma and chroma decoder passes produced different frame counts "
            f"({luma_frames} versus {chroma_frames})"
        )

    y_size = width * height
    chroma_size = ((width + 1) // 2) * ((height + 1) // 2)
    progress_context = (
        reporter.progress(
            total=luma_frames,
            description="Combining luma/chroma decoder passes",
            unit="frame",
        )
        if reporter is not None
        else _NullProgress()
    )
    with (
        luma_raw.open("rb") as luma,
        chroma_raw.open("rb") as chroma,
        output.open("wb") as merged,
        progress_context as progress,
    ):
        for frame in range(luma_frames):
            y = luma.read(y_size)
            if len(y) != y_size:
                raise ExternalToolError(f"short luma frame {frame}")
            luma.seek(2 * chroma_size, io.SEEK_CUR)

            chroma.seek(y_size, io.SEEK_CUR)
            uv = chroma.read(2 * chroma_size)
            if len(uv) != 2 * chroma_size:
                raise ExternalToolError(f"short chroma frame {frame}")
            merged.write(y)
            merged.write(uv)
            progress.update(1)
    return luma_frames


def write_y4m(
    raw_input: Path,
    output: Path,
    *,
    width: int,
    height: int,
    fps: Fraction,
    frame_count: int,
    reporter: Optional[Reporter] = None,
) -> None:
    frame_size = yuv420_frame_size(width, height)
    header = (
        f"YUV4MPEG2 W{width} H{height} F{fps.numerator}:{fps.denominator} "
        "Ip A1:1 C420mpeg2 XYSCSS=420MPEG2\n"
    ).encode("ascii")
    progress_context = (
        reporter.progress(
            total=frame_count,
            description="Writing Y4M",
            unit="frame",
        )
        if reporter is not None
        else _NullProgress()
    )
    with (
        raw_input.open("rb") as source,
        output.open("wb") as destination,
        progress_context as progress,
    ):
        destination.write(header)
        for frame in range(frame_count):
            data = source.read(frame_size)
            if len(data) != frame_size:
                raise ExternalToolError(f"short raw frame {frame} while writing Y4M")
            destination.write(b"FRAME\n")
            destination.write(data)
            progress.update(1)
        if source.read(1):
            raise ExternalToolError("raw input contains data after the expected final frame")


def encode_ffv1(
    ffmpeg: str,
    raw_input: Path,
    output: Path,
    *,
    width: int,
    height: int,
    fps: Fraction,
    overwrite: bool,
    expected_frames: int,
    reporter: Reporter,
) -> int:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pixel_format",
        "yuv420p",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps.numerator}/{fps.denominator}",
        "-i",
        str(raw_input),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "matroska",
        "-y" if overwrite else "-n",
        str(output),
    ]
    result = run_ffmpeg(
        command,
        reporter,
        description="Encoding lossless FFV1",
        expected_frames=expected_frames,
    )
    if result.frame_count is None:
        reporter.warn(
            "ffmpeg did not report a final FFV1 frame count; using the raw frame count"
        )
        return expected_frames
    return result.frame_count


def copy_raw_output(
    raw_input: Path,
    output: Path,
    *,
    reporter: Optional[Reporter] = None,
) -> None:
    total = raw_input.stat().st_size
    progress_context = (
        reporter.progress(
            total=total,
            description="Copying raw yuv420p",
            unit="B",
            unit_scale=True,
        )
        if reporter is not None
        else _NullProgress()
    )
    with (
        raw_input.open("rb") as source,
        output.open("wb") as destination,
        progress_context as progress,
    ):
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            destination.write(block)
            progress.update(len(block))


# ---------------------------------------------------------------------------
# Diagnostics and top-level conversion


def picture_summary(pictures: Sequence[Picture]) -> dict[str, object]:
    counts = {name: 0 for name in ("I", "P", "B", "S")}
    extended_quant = 0
    custom_luma = 0
    custom_chroma = 0
    gmc_formats: dict[str, int] = {}
    gmc_mappings: dict[str, int] = {}
    fixed16_zero_tail = 0
    fixed16_nonzero_tail = 0
    field_syntax_frames = 0
    field_syntax_preambles = 0
    extension_flag_2_frames = 0
    preamble_values: set[int] = set()
    for picture in pictures:
        counts[picture.frame_name] = counts.get(picture.frame_name, 0) + 1
        extended_quant += int(picture.uses_extended_quant)
        custom_luma += int(picture.matrix_order_luma is not None)
        custom_chroma += int(picture.matrix_order_chroma is not None)
        field_syntax_frames += int(picture.field_syntax)
        extension_flag_2_frames += int(picture.extension_flag_2)
        if picture.p_frame_preamble is not None:
            field_syntax_preambles += 1
            preamble_values.add(picture.p_frame_preamble)
        if picture.gmc_source_format is not None:
            gmc_formats[picture.gmc_source_format] = (
                gmc_formats.get(picture.gmc_source_format, 0) + 1
            )
        if picture.gmc_mapping is not None:
            gmc_mappings[picture.gmc_mapping] = (
                gmc_mappings.get(picture.gmc_mapping, 0) + 1
            )
        if picture.gmc_source_format == "fixed16" and len(picture.gmc_raw_values) > 2:
            if all(value == 0 for value in picture.gmc_raw_values[2:]):
                fixed16_zero_tail += 1
            else:
                fixed16_nonzero_tail += 1
    return {
        "frames": len(pictures),
        "frame_types": counts,
        "extended_quant_frames": extended_quant,
        "luma_matrix_updates": custom_luma,
        "chroma_matrix_updates": custom_chroma,
        "gmc_formats": gmc_formats,
        "gmc_mappings": gmc_mappings,
        "fixed16_zero_tail_frames": fixed16_zero_tail,
        "fixed16_nonzero_tail_frames": fixed16_nonzero_tail,
        "field_syntax_frames": field_syntax_frames,
        "field_syntax_p_frames": field_syntax_preambles,
        "field_syntax_preamble_values": sorted(preamble_values),
        "extension_flag_2_frames": extension_flag_2_frames,
        "first_timecode": pictures[0].timecode if pictures else None,
        "last_timecode": pictures[-1].timecode if pictures else None,
    }


def inspect_report(
    *,
    input_path: Path,
    info: VID1Info,
    header_skip: int,
    fps: Fraction,
    timing_mode: str,
    fixed16_gmc_divisor: int,
    fixed16_gmc_mode: str,
    pictures: Sequence[Picture],
    probes: Sequence[SkipProbe],
) -> dict[str, object]:
    return {
        "input": str(input_path),
        "endianness": "big" if info.big_endian else "little",
        "width": info.width,
        "height": info.height,
        "start_offset": info.start_offset,
        "audio_codec": info.audio_codec,
        "header_frame_count": info.frame_count,
        "header_rate": (
            f"{info.header_rate.numerator}/{info.header_rate.denominator}"
            if info.header_rate is not None
            else None
        ),
        "selected_fps": f"{fps.numerator}/{fps.denominator}",
        "timing_mode": timing_mode,
        "vidd_header_skip": header_skip,
        "fixed16_gmc_divisor": fixed16_gmc_divisor,
        "fixed16_gmc_mode": fixed16_gmc_mode,
        "skip_probes": [
            {
                "skip": probe.skip,
                "score": probe.score,
                "parsed": probe.parsed,
                "first_type": (
                    FRAME_NAMES.get(probe.first_type) if probe.first_type is not None else None
                ),
                "error": probe.error,
            }
            for probe in probes
        ],
        **picture_summary(pictures),
    }


def determine_output_format(requested: str, output: Path) -> str:
    if requested != "auto":
        return requested
    suffix = output.suffix.lower()
    if suffix == ".y4m":
        return "y4m"
    if suffix == ".yuv":
        return "raw"
    return "ffv1"


def ensure_output_available(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise VID1Error(f"output already exists: {path}; pass --overwrite to replace it")
        if not path.is_file():
            raise VID1Error(f"output path is not a regular file: {path}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def choose_fixed16_gmc_divisor(requested: str) -> int:
    if requested == "auto":
        # Used only by the legacy translation/points mappings.  The verified
        # packed VID1/zigzag mappings are exact and do not use a divisor.
        return 16
    try:
        divisor = int(requested, 0)
    except ValueError as exc:
        raise VID1Error(
            "--fixed16-gmc-divisor must be 'auto' or a positive integer"
        ) from exc
    if divisor <= 0:
        raise VID1Error("--fixed16-gmc-divisor must be positive")
    return divisor


def convert(args: argparse.Namespace) -> None:
    ffmpeg_log_path = Path(args.ffmpeg_log) if args.ffmpeg_log else None
    if ffmpeg_log_path is not None:
        ffmpeg_log_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_log_path.write_text("", encoding="utf-8")

    reporter = Reporter(
        args.verbose,
        quiet=args.quiet,
        progress_mode=args.progress,
        ffmpeg_log=ffmpeg_log_path,
    )
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output is not None else None
    if not input_path.is_file():
        raise VID1Error(f"input file not found: {input_path}")

    input_size = input_path.stat().st_size
    reporter.status(f"Input: {input_path} ({format_size(input_size)})")
    fixed16_divisor = choose_fixed16_gmc_divisor(args.fixed16_gmc_divisor)

    with input_path.open("rb") as stream:
        info = parse_vid1_file_header(stream)
        chunks = read_video_chunks(
            stream,
            info,
            resync_scan=args.resync_scan,
            max_chunk_size=args.max_chunk_size,
            reporter=reporter,
            input_size=input_size,
        )
    reporter.status(f"Found {len(chunks)} VIDD video packet(s)")

    width = args.width if args.width is not None else info.width
    height = args.height if args.height is not None else info.height
    if width is None or height is None:
        raise VID1Error("width/height were not found in VIDH; pass --width and --height")
    if width <= 0 or height <= 0:
        raise VID1Error(f"invalid dimensions {width}x{height}")

    fps = choose_fps(args.fps, info, reporter)
    header_skip, probes = choose_header_skip(
        chunks,
        args.vidd_header_skip,
        matrix_order=args.matrix_order,
        probe_frames=args.probe_frames,
        gmc_format=args.gmc_format,
        fixed16_divisor=fixed16_divisor,
        fixed16_mapping=args.fixed16_gmc_mode,
        reporter=reporter,
    )
    reporter.status(f"VIDD picture-header offset: {header_skip} byte(s)")
    pictures = parse_picture_sequence(
        chunks,
        header_skip=header_skip,
        matrix_order=args.matrix_order,
        lenient=args.lenient,
        gmc_format=args.gmc_format,
        fixed16_divisor=fixed16_divisor,
        fixed16_mapping=args.fixed16_gmc_mode,
        reporter=reporter,
    )
    pictures, timing_mode = assign_display_indices(pictures, args.timing, reporter)

    if info.frame_count is not None and info.frame_count != len(pictures):
        reporter.warn(
            f"VIDH frame count says {info.frame_count}, but {len(pictures)} VIDD packets were parsed"
        )

    summary = picture_summary(pictures)
    reporter.status(
        f"Video: {width}x{height}, {fps.numerator}/{fps.denominator} fps, "
        f"{summary['frames']} frames, types={summary['frame_types']}, "
        f"timing={timing_mode}"
    )
    field_syntax_count = int(summary["field_syntax_frames"])
    if field_syntax_count:
        reporter.status(
            f"VID1 field-syntax mode: {field_syntax_count} frame(s); "
            f"removed {summary['field_syntax_p_frames']} zero P-frame preamble(s)"
        )
    if int(summary["extension_flag_2_frames"]):
        reporter.warn(
            f"the unidentified second VID1 extension flag is set for "
            f"{summary['extension_flag_2_frames']} frame(s); its semantics are not mapped"
        )

    fixed16_count = int(summary["gmc_formats"].get("fixed16", 0))
    if fixed16_count:
        mappings = summary["gmc_mappings"]
        vid1_count = int(mappings.get("vid1", 0))
        zigzag_count = int(mappings.get("zigzag", 0))
        legacy_count = fixed16_count - vid1_count - zigzag_count
        if vid1_count:
            reporter.status(
                f"Packed VID1 GMC: {vid1_count} S frame(s), "
                "x=signmag(raw>>2), y=signmag(raw>>3), "
                f"mapping={mappings}"
            )
        if zigzag_count:
            reporter.warn(
                f"Legacy v4 zigzag GMC is active for {zigzag_count} S frame(s); "
                "negative components are one half-pel larger than VID1 sign/magnitude"
            )
        if legacy_count:
            mode_note = (
                "legacy automatic scale"
                if args.fixed16_gmc_divisor == "auto"
                else "user scale override"
            )
            reporter.status(
                f"Legacy fixed16 GMC: {legacy_count} S frame(s), "
                f"divisor={fixed16_divisor} ({mode_note}), mapping={mappings}"
            )
        if int(summary["fixed16_zero_tail_frames"]):
            reporter.info(
                f"fixed16 zero-tail layout: {summary['fixed16_zero_tail_frames']} "
                "frame(s)",
                level=1,
            )

    if args.inspect:
        report = inspect_report(
            input_path=input_path,
            info=replace(info, width=width, height=height),
            header_skip=header_skip,
            fps=fps,
            timing_mode=timing_mode,
            fixed16_gmc_divisor=fixed16_divisor,
            fixed16_gmc_mode=args.fixed16_gmc_mode,
            pictures=pictures,
            probes=probes,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if output_path is None:
        raise VID1Error("an output path is required unless --inspect is used")
    if input_path.resolve() == output_path.resolve():
        raise VID1Error("input and output paths must be different")
    ensure_output_available(output_path, args.overwrite)
    output_format = determine_output_format(args.format, output_path)
    ffmpeg = executable_or_error(args.ffmpeg)
    reporter.status(
        f"Output: {output_path} ({output_format}; decode threads={args.decode_threads or 'auto'})"
    )

    has_b_frames = any(picture.frame_type == 2 for picture in pictures)
    dual_pass = pictures_need_dual_matrix_pass(pictures)
    if dual_pass:
        reporter.warn(
            "component-specific extended quantisation matrices require two MPEG-4 "
            "decode passes and Y/U/V plane recombination"
        )

    keep_directory: Optional[Path] = None
    temporary_parent = Path(args.temp_dir) if args.temp_dir else None
    if temporary_parent is not None:
        temporary_parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(
            prefix="vid1-decode-",
            dir=str(temporary_parent) if temporary_parent else None,
        ) as temporary_name:
            temporary = Path(temporary_name)
            reporter.info(f"temporary directory: {temporary}", level=1)
            if args.keep_temp:
                keep_directory = Path(str(temporary) + "-kept")

            base_config = AdapterConfig(
                width=width,
                height=height,
                fps=fps,
                has_b_frames=has_b_frames,
                matrix_plane="luma",
                trim_payload_padding=not args.keep_payload_padding,
            )
            luma_m4v = temporary / "adapted-luma.m4v"
            write_adapted_stream(
                luma_m4v,
                pictures,
                base_config,
                reporter=reporter,
                progress_description="Writing luma MPEG-4 adapter",
            )
            reporter.info(
                f"adapter size: {format_size(luma_m4v.stat().st_size)}",
                level=1,
            )

            if not dual_pass:
                decoded_frames, _decoder_stderr = decode_m4v_to_output(
                    ffmpeg,
                    luma_m4v,
                    output_path,
                    output_format=output_format,
                    width=width,
                    height=height,
                    fps=fps,
                    idct=args.idct,
                    decode_threads=args.decode_threads,
                    force_progressive=bool(field_syntax_count),
                    expected_frames=len(pictures),
                    reporter=reporter,
                )
            else:
                luma_yuv = temporary / "decoded-luma.yuv"
                luma_result = decode_m4v_to_raw(
                    ffmpeg,
                    luma_m4v,
                    luma_yuv,
                    width=width,
                    height=height,
                    idct=args.idct,
                    decode_threads=args.decode_threads,
                    force_progressive=bool(field_syntax_count),
                    expected_frames=len(pictures),
                    reporter=reporter,
                    description="Decoding luma-matrix pass",
                )

                chroma_config = replace(base_config, matrix_plane="chroma")
                chroma_m4v = temporary / "adapted-chroma.m4v"
                chroma_yuv = temporary / "decoded-chroma.yuv"
                merged_yuv = temporary / "decoded-merged.yuv"
                write_adapted_stream(
                    chroma_m4v,
                    pictures,
                    chroma_config,
                    reporter=reporter,
                    progress_description="Writing chroma MPEG-4 adapter",
                )
                chroma_result = decode_m4v_to_raw(
                    ffmpeg,
                    chroma_m4v,
                    chroma_yuv,
                    width=width,
                    height=height,
                    idct=args.idct,
                    decode_threads=args.decode_threads,
                    force_progressive=bool(field_syntax_count),
                    expected_frames=len(pictures),
                    reporter=reporter,
                    description="Decoding chroma-matrix pass",
                )
                decoded_frames = merge_component_passes(
                    luma_result.raw_path,
                    chroma_result.raw_path,
                    merged_yuv,
                    width=width,
                    height=height,
                    luma_frames=luma_result.frame_count,
                    chroma_frames=chroma_result.frame_count,
                    reporter=reporter,
                )

                if output_format == "y4m":
                    reporter.status("Packaging merged pixels as Y4M")
                    write_y4m(
                        merged_yuv,
                        output_path,
                        width=width,
                        height=height,
                        fps=fps,
                        frame_count=decoded_frames,
                        reporter=reporter,
                    )
                elif output_format == "raw":
                    reporter.status("Copying merged raw pixels")
                    copy_raw_output(merged_yuv, output_path, reporter=reporter)
                elif output_format == "ffv1":
                    encoded_frames = encode_ffv1(
                        ffmpeg,
                        merged_yuv,
                        output_path,
                        width=width,
                        height=height,
                        fps=fps,
                        overwrite=True,
                        expected_frames=decoded_frames,
                        reporter=reporter,
                    )
                    if encoded_frames != decoded_frames:
                        reporter.warn(
                            f"FFV1 encoder reported {encoded_frames} frames for "
                            f"{decoded_frames} merged frames"
                        )
                else:
                    raise AssertionError(f"unhandled output format {output_format}")

            if decoded_frames != len(pictures):
                message = (
                    f"ffmpeg returned {decoded_frames} decoded frames for "
                    f"{len(pictures)} VID1 pictures"
                )
                if args.strict_frame_count:
                    raise ExternalToolError(message)
                reporter.warn(message)

            if args.dump_adapted:
                destination = Path(args.dump_adapted)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(luma_m4v, destination)
                reporter.status(f"Copied diagnostic adapter to {destination}")

            if keep_directory is not None:
                reporter.status(f"Copying temporary files to {keep_directory}")
                shutil.copytree(temporary, keep_directory)
                reporter.status(f"Kept temporary files in {keep_directory}")

    except Exception:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise

    output_size = output_path.stat().st_size
    reporter.status(
        f"Done: {decoded_frames} frames -> {output_path} "
        f"({format_size(output_size)}) in {format_duration(time.monotonic() - reporter.started)}"
    )


# ---------------------------------------------------------------------------
# Self-tests


def run_self_tests() -> None:
    # Canonical sprite VLC round trip.
    for symbol in range(15):
        sequence = encode_sprite_length(symbol)
        packed = bytearray((len(sequence) + 7) // 8)
        for index, bit in enumerate(sequence):
            packed[index >> 3] |= bit << (7 - (index & 7))
        reader = BitReaderMSB(bytes(packed))
        assert read_sprite_length(reader) == symbol
        assert reader.bitpos == len(sequence)

    # Bit writer/reader symmetry.
    buffer = io.BytesIO()
    writer = BitWriter(buffer)
    writer.write_bits(3, 0b101)
    writer.write_bits(9, 0x12F)
    writer.close()
    reader = BitReaderMSB(buffer.getvalue())
    assert reader.read(3) == 0b101
    assert reader.read(9) == 0x12F

    # Matrix auto-order keeps an already smooth row-major matrix row-major.
    smooth = tuple(8 + x + y for y in range(8) for x in range(8))
    normalized, selected = normalize_matrix(smooth, "auto", lenient=False)
    assert normalized == smooth
    assert selected.startswith("row")

    # Timecode and GOP ordering for I P B B P B.
    mock = [
        Picture(
            index=i,
            chunk_offset=0,
            frame_type=t,
            rounding=0,
            intra_dc_vlc_thr_idx=0,
            quant=2,
            fcode_forward=1,
            fcode_backward=1,
            timecode=tc,
            payload=b"x",
            ignored16=0,
            extended_info_present=False,
            field_syntax=False,
            extension_flag_2=False,
            p_frame_preamble=None,
            sprite_config=None,
            trajectory_bits=(),
            uses_extended_quant=False,
            luma_quant=None,
            chroma_quant=None,
            matrix_order_luma=None,
            matrix_order_chroma=None,
        )
        for i, (t, tc) in enumerate(((0, 0), (1, 3), (2, 1), (2, 2), (1, 5), (2, 4)))
    ]
    assert assign_timecode_indices(mock) == [0, 3, 1, 2, 5, 4]
    assert assign_gop_indices(mock) == [0, 3, 1, 2, 5, 4]

    # The FFmpeg raw frame-size formula supports odd dimensions too.
    assert yuv420_frame_size(4, 4) == 24
    assert yuv420_frame_size(3, 3) == 17

    # Packet alignment trimming removes only the final 0..7 fill bits.
    assert packet_alignment_zero_bits(b"\x81") == 0
    assert packet_alignment_zero_bits(b"\x82") == 1
    assert packet_alignment_zero_bits(b"\x80") == 7
    trimmed = io.BytesIO()
    writer = BitWriter(trimmed)
    write_picture_payload(writer, b"\x80", trim_packet_padding=True)
    writer.close()
    assert trimmed.getvalue() == b"\x80"

    # Extended VID1 headers propagate the field-syntax flag, and P pictures in
    # that mode discard their separate zero preamble before exposing MPEG-4
    # macroblock bits.
    extended_packet = io.BytesIO()
    writer = BitWriter(extended_packet)
    writer.write_bits(16, 1)  # VID1 sync
    writer.write_bits(2, 0)   # I
    writer.write_bit(1)       # extended header
    writer.write_bit(0)       # sprite update absent
    writer.write_bit(0)       # extended quant absent
    writer.write_bit(1)       # field syntax
    writer.write_bit(0)       # second extension flag
    writer.write_bit(0)       # rounding
    writer.write_bits(3, 0)   # intra DC threshold
    writer.write_bits(5, 2)   # quantiser
    writer.write_bits(32, 0)  # timecode
    writer.align_zero()
    writer.write_bits(8, 0x80)
    writer.close()
    field_state = ParseState()
    extended_picture = parse_picture(
        extended_packet.getvalue(),
        index=0,
        chunk_offset=0,
        state=field_state,
        matrix_order="auto",
        lenient=False,
        gmc_format="auto",
        fixed16_divisor=16,
        fixed16_mapping="auto",
    )
    assert extended_picture.field_syntax
    assert field_state.field_syntax
    assert bytes(extended_picture.payload) == b"\x80"

    p_packet = io.BytesIO()
    writer = BitWriter(p_packet)
    writer.write_bits(16, 1)  # VID1 sync
    writer.write_bits(2, 1)   # P
    writer.write_bit(0)       # no extended header; inherit field syntax
    writer.write_bit(0)       # rounding
    writer.write_bits(3, 0)   # intra DC threshold
    writer.write_bits(5, 2)   # quantiser
    writer.write_bits(3, 1)   # forward fcode
    writer.write_bits(32, 1)  # timecode
    writer.align_zero()
    writer.write_bits(8, 0x00)  # VID1 P-picture preamble
    writer.write_bits(8, 0x80)  # macroblock payload
    writer.close()
    p_picture = parse_picture(
        p_packet.getvalue(),
        index=1,
        chunk_offset=0,
        state=field_state,
        matrix_order="auto",
        lenient=False,
        gmc_format="auto",
        fixed16_divisor=16,
        fixed16_mapping="auto",
    )
    assert p_picture.p_frame_preamble == 0
    assert bytes(p_picture.payload) == b"\x80"

    # Auto mode may resolve the output sprite shape only at the first S picture.
    # Backfill it to earlier pictures in the same source-state run so the adapter
    # does not replace a no-sprite VOL in the middle of a predictive GOP.
    unresolved_sprite_picture = replace(
        extended_picture,
        index=0,
        source_sprite_config=SpriteConfig(2, 3),
        sprite_config=None,
    )
    resolved_sprite_picture = replace(
        extended_picture,
        index=1,
        frame_type=3,
        source_sprite_config=SpriteConfig(2, 3),
        sprite_config=SpriteConfig(2, 3),
    )
    sprite_resolved = resolve_sprite_vol_state(
        (unresolved_sprite_picture, resolved_sprite_picture)
    )
    assert sprite_resolved[0].sprite_config == SpriteConfig(2, 3)
    assert sprite_resolved[1].sprite_config == SpriteConfig(2, 3)

    # A legacy fixed16 translation can resolve to a one-point MPEG-4 sprite even
    # when VID1 advertised more source points; backfill the converted shape.
    translated_sprite_picture = replace(
        resolved_sprite_picture,
        sprite_config=SpriteConfig(1, 3),
    )
    translation_resolved = resolve_sprite_vol_state(
        (unresolved_sprite_picture, translated_sprite_picture)
    )
    assert all(
        picture.sprite_config == SpriteConfig(1, 3)
        for picture in translation_resolved
    )

    # Retail packed GMC words use sign/magnitude codes with axis-specific shifts.
    assert choose_fixed16_gmc_divisor("auto") == 16
    assert signed_magnitude_decode(0) == 0
    assert signed_magnitude_decode(1) == 0  # redundant negative zero
    assert signed_magnitude_decode(2) == 1
    assert signed_magnitude_decode(3) == -1
    assert signed_magnitude_decode(5) == -2
    assert folded_signed_decode(5) == -3  # v4 compatibility mode
    sprite = SpriteConfig(2, 3)
    assert decode_fixed16_packed_values(
        (192, 256, 0, 0), sprite, frame_index=4, signed_mode="vid1"
    ) == (24, 16, 0, 0)
    assert decode_fixed16_packed_values(
        (52, 424, 0, 0), sprite, frame_index=808, signed_mode="vid1"
    ) == (-6, -26, 0, 0)
    assert decode_fixed16_packed_values(
        (8, 40, 0, 0), sprite, frame_index=1462, signed_mode="vid1"
    ) == (1, -2, 0, 0)
    trajectory = encode_sprite_trajectory((24, 16, 0, 0), 2)
    packed = bytearray((len(trajectory) + 7) // 8)
    for bit_index, bit in enumerate(trajectory):
        packed[bit_index >> 3] |= bit << (7 - (bit_index & 7))
    trajectory_reader = BitReaderMSB(bytes(packed))
    assert read_sprite_trajectory(trajectory_reader, 2) == trajectory

    print("all self-tests passed")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", help="input Factor 5 VID1 .vid file")
    parser.add_argument("output", nargs="?", help="output .mkv/.y4m/.yuv file")
    parser.add_argument(
        "--format",
        choices=("auto", "ffv1", "y4m", "raw"),
        default="auto",
        help="output representation; auto uses y4m for .y4m, raw for .yuv, otherwise FFV1",
    )
    parser.add_argument(
        "--fps",
        default="auto",
        help="output/display frame rate as integer or rational, or auto (plausible VIDH rate, otherwise 24)",
    )
    parser.add_argument("--width", type=int, help="override width from VIDH")
    parser.add_argument("--height", type=int, help="override height from VIDH")
    parser.add_argument(
        "--vidd-header-skip",
        default="auto",
        help="bytes between VIDD size and VID1 picture header; auto tests 6,4,2,0,8",
    )
    parser.add_argument(
        "--probe-frames",
        type=int,
        default=16,
        help="number of frames used to auto-detect the VIDD subheader length",
    )
    parser.add_argument(
        "--gmc-format",
        choices=("auto", "mpeg4", "fixed16"),
        default="auto",
        help=(
            "S/GMC trajectory representation: standard MPEG-4 VLCs, or the "
            "byte-aligned packed 16-bit form found in retail VID1 files"
        ),
    )
    parser.add_argument(
        "--fixed16-gmc-divisor",
        default="auto",
        help=(
            "divisor used only by legacy translation/points mappings; "
            "auto uses 16. The packed VID1/zigzag mappings are exact and ignore it"
        ),
    )
    parser.add_argument(
        "--fixed16-gmc-mode",
        choices=("auto", "vid1", "zigzag", "translation", "points"),
        default="auto",
        help=(
            "how proprietary 16-bit GMC words map to MPEG-4: auto detects "
            "the retail packed sign/magnitude layout; vid1 forces the corrected "
            "x=signmag(raw>>2), y=signmag(raw>>3) mapping; zigzag reproduces v4; "
            "translation/points retain legacy divisor mappings"
        ),
    )
    parser.add_argument(
        "--matrix-order",
        choices=("auto", "row", "zigzag"),
        default="auto",
        help="storage order of VID1's 64-byte luma/chroma matrices",
    )
    parser.add_argument(
        "--timing",
        choices=("auto", "timecode", "gop"),
        default="auto",
        help="derive MPEG-4 display timestamps from VID1 timecodes or coded GOP order",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable name or path",
    )
    parser.add_argument(
        "--decode-threads",
        type=int,
        default=0,
        help="FFmpeg MPEG-4 decoder thread count; 0 lets FFmpeg choose automatically",
    )
    parser.add_argument(
        "--idct",
        default="auto",
        help=(
            "FFmpeg MPEG-4 inverse-DCT implementation (for example auto, simple, "
            "or xvid); auto normally selects FFmpeg's generic implementation"
        ),
    )
    parser.add_argument(
        "--keep-payload-padding",
        action="store_true",
        help=(
            "do not remove up to seven trailing zero alignment bits from each "
            "VID1 macroblock payload (diagnostic compatibility option)"
        ),
    )
    parser.add_argument(
        "--temp-dir",
        help="parent directory for temporary adapted streams and raw video",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="copy temporary streams beside the temporary directory before cleanup",
    )
    parser.add_argument(
        "--dump-adapted",
        help="copy the internal luma-adapted MPEG-4 stream to this path for diagnostics",
    )
    parser.add_argument(
        "--resync-scan",
        action="store_true",
        help="scan for the next known chunk after an unrecognised/bare audio packet",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=512 * 1024 * 1024,
        help="safety limit for a single VID1 chunk",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="repair zero quant-matrix entries to 1 instead of rejecting the file",
    )
    parser.add_argument(
        "--strict-frame-count",
        action="store_true",
        help="fail if ffmpeg emits a different number of frames than VIDD packets",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="parse and print JSON metadata/frame statistics without invoking ffmpeg",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="progress display policy; auto shows bars when standard error is a terminal",
    )
    parser.add_argument(
        "--ffmpeg-log",
        help="write complete ffmpeg command lines and diagnostics to this text file",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress status messages and progress bars (warnings/errors still print)",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase diagnostics; -vv shows commands and -vvv prints ffmpeg warnings")
    parser.add_argument("--self-test", action="store_true", help="run internal tests and exit")
    parser.add_argument("--version", action="version", version=f"{PROGRAM} {VERSION}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_tests()
        return 0
    if args.input is None:
        parser.error("input is required unless --self-test is used")
    if args.output is None and not args.inspect:
        parser.error("output is required unless --inspect is used")
    if args.probe_frames <= 0:
        parser.error("--probe-frames must be positive")
    if args.max_chunk_size <= 0:
        parser.error("--max-chunk-size must be positive")
    if args.decode_threads < 0:
        parser.error("--decode-threads cannot be negative")
    if args.quiet and args.verbose:
        parser.error("--quiet cannot be combined with --verbose")
    try:
        choose_fixed16_gmc_divisor(args.fixed16_gmc_divisor)
    except VID1Error as exc:
        parser.error(str(exc))

    try:
        convert(args)
    except (VID1Error, OSError, subprocess.SubprocessError) as exc:
        print(f"{PROGRAM}: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
