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
the byte-aligned signed-16-bit fixed-point form used by retail VID1 streams.

Extended VID1 luma/chroma quantisation matrices do not map one-for-one onto
MPEG-4 Part 2's intra/non-intra matrices.  For such pictures this program can
run two decoder passes, using the luma matrix for both MPEG-4 matrices in one
pass and the chroma matrix in the other, then combine Y from the first pass
with U/V from the second.  That is the closest component-wise mapping
available through a stock MPEG-4 decoder, but it remains an inferred mapping
because the public VID1 notes do not fully specify the extension.

Examples:

    python3 vid1_decode.py input.vid output.mkv
    python3 vid1_decode.py input.vid output.y4m --format y4m
    python3 vid1_decode.py input.vid output.mkv --fps 30000/1001 -v

The script has no third-party Python dependencies.  It requires an ffmpeg
executable with the MPEG-4 Part 2 decoder and, for FFV1 output, the FFV1 encoder.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, Iterable, Optional, Sequence, Union


ByteBuffer = Union[bytes, memoryview]


PROGRAM = "vid1_decode.py"
VERSION = "2.0"


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

    @property
    def frame_name(self) -> str:
        return FRAME_NAMES.get(self.frame_type, f"?{self.frame_type}")


@dataclass
class ParseState:
    sprite: Optional[SpriteConfig] = None
    luma_matrix: Optional[tuple[int, ...]] = None
    chroma_matrix: Optional[tuple[int, ...]] = None
    gmc_format: Optional[str] = None


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
class SkipProbe:
    skip: int
    score: int
    parsed: int
    first_type: Optional[int]
    error: Optional[str]


class Reporter:
    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self._warnings: list[str] = []

    def info(self, message: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(message, file=sys.stderr)

    def warn(self, message: str) -> None:
        if message not in self._warnings:
            self._warnings.append(message)
            print(f"warning: {message}", file=sys.stderr)


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
) -> list[RawVideoChunk]:
    chunks: list[RawVideoChunk] = []
    big_endian = info.big_endian
    stream.seek(info.start_offset)

    while True:
        position = stream.tell()
        try:
            magic = read_u32(stream, big_endian)
        except EOFError:
            break

        if magic == TAG_FRAM:
            # The current librempeg demuxer skips the 28-byte FRAM subheader
            # and then reads the nested VIDD/AUDD tag.
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
                raise VID1Error(f"invalid VIDD chunk size {chunk_size} at 0x{position:x}")
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
            continue

        if magic == TAG_AUDD:
            try:
                chunk_size = read_u32(stream, big_endian)
            except EOFError as exc:
                raise VID1Error(f"truncated AUDD size at 0x{position:x}") from exc
            if chunk_size < 8 or chunk_size > max_chunk_size:
                raise VID1Error(f"invalid AUDD chunk size {chunk_size} at 0x{position:x}")
            stream.seek(position + chunk_size)
            continue

        if magic == 0:
            remainder = stream.read()
            if not remainder or all(value == 0 for value in remainder):
                break
            stream.seek(position + 4)

        # Vorbis audio may appear as a bare variable-length packet between
        # regular chunks.  Skip it when the header is sane; otherwise either
        # fail or scan to the next known tag.
        try:
            header_length, packet_size = parse_variable_packet_header_at(stream, position)
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


def read_fixed16_sprite_trajectory(
    bits: BitReaderMSB, config: SpriteConfig
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    # The retail sample supplied with the bug report stores the GMC values as
    # byte-aligned, big-endian signed 16-bit fixed-point components rather
    # than MPEG-4's variable-length trajectory syntax.  There are x/y values
    # for each configured warping point.  The fixed-point scale is the same
    # 2 << accuracy factor used by MPEG-4's GMC equations.
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

    scale = 2 << config.accuracy
    converted = tuple(rounded_divide(value, scale) for value in raw)
    return encode_sprite_trajectory(converted, config.warping_points), tuple(raw)


def parse_sprite_trajectory(
    bits: BitReaderMSB,
    config: SpriteConfig,
    *,
    requested_format: str,
    state: ParseState,
    frame_index: int,
) -> tuple[tuple[int, ...], str, tuple[int, ...]]:
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
            return trajectory, "mpeg4", ()
        except VID1Error as exc:
            bits.bitpos = start
            errors.append(f"mpeg4: {exc}")
            if selected == "mpeg4":
                raise VID1Error(f"frame {frame_index}: {exc}") from exc

    if selected in ("auto", "fixed16"):
        start = bits.bitpos
        try:
            trajectory, raw = read_fixed16_sprite_trajectory(bits, config)
            if requested_format == "auto":
                state.gmc_format = "fixed16"
            return trajectory, "fixed16", raw
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
) -> Picture:
    if len(packet) < 8:
        raise VID1Error(f"frame {index}: packet is only {len(packet)} bytes")

    bits = BitReaderMSB(packet)
    ignored16 = bits.read(16)
    frame_type = bits.read(2)
    extended = bool(bits.read_bit())

    active_sprite = state.sprite
    uses_extended_quant = False
    luma_order: Optional[str] = None
    chroma_order: Optional[str] = None

    if extended:
        sprite_present = bool(bits.read_bit())
        if sprite_present:
            active_sprite = SpriteConfig(
                warping_points=bits.read(2),
                accuracy=bits.read(2),
            )
            state.sprite = active_sprite

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
        bits.read_bit()  # ignored
        bits.read_bit()  # ignored

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
    if frame_type == 3:
        if active_sprite is None:
            raise VID1Error(
                f"frame {index}: S/GMC frame has no active sprite configuration"
            )
        trajectory, gmc_source_format, gmc_raw_values = parse_sprite_trajectory(
            bits,
            active_sprite,
            requested_format=gmc_format,
            state=state,
            frame_index=index,
        )

    bits.align_byte()
    payload = packet[bits.bytepos :]
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
        sprite_config=active_sprite,
        trajectory_bits=trajectory,
        uses_extended_quant=uses_extended_quant,
        luma_quant=luma_pair,
        chroma_quant=chroma_pair,
        matrix_order_luma=luma_order,
        matrix_order_chroma=chroma_order,
        gmc_source_format=gmc_source_format,
        gmc_raw_values=gmc_raw_values,
    )


def parse_picture_sequence(
    chunks: Sequence[RawVideoChunk],
    *,
    header_skip: int,
    matrix_order: str,
    lenient: bool,
    gmc_format: str,
    limit: Optional[int] = None,
) -> list[Picture]:
    if header_skip < 0:
        raise VID1Error("VIDD header skip must be non-negative")
    state = ParseState()
    pictures: list[Picture] = []
    selected = chunks if limit is None else chunks[:limit]
    for chunk in selected:
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
        )
        pictures.append(picture)
    return pictures


def probe_header_skip(
    chunks: Sequence[RawVideoChunk],
    skip: int,
    *,
    matrix_order: str,
    probe_frames: int,
    gmc_format: str,
) -> SkipProbe:
    try:
        pictures = parse_picture_sequence(
            chunks,
            header_skip=skip,
            matrix_order=matrix_order,
            lenient=True,
            gmc_format=gmc_format,
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
    writer.write_bits(1, 0)       # interlaced = false
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
) -> None:
    time_state = MPEG4TimeState(time_resolution=config.fps.numerator)
    previous_vol_key: object = object()

    with output.open("wb") as stream:
        writer = BitWriter(stream)
        write_visual_object_prefix(writer)

        for picture in pictures:
            quant_pair = selected_quant_pair(picture, config.matrix_plane)
            # Sprite/GMC parameters are VOL state, not per-VOP state.  Once the
            # VID1 header has supplied them, keep them active for ordinary I/P/B
            # pictures as well as S pictures.  The ordinary macroblock grammar
            # is unchanged; only S-VOP macroblocks consume mcsel/trajectory data.
            sprite = picture.sprite_config
            vol_key = (quant_pair, sprite)
            if vol_key != previous_vol_key:
                write_vol(writer, config, quant_pair=quant_pair, sprite=sprite)
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


def run_command(command: Sequence[str], reporter: Reporter, *, description: str) -> str:
    reporter.info("+ " + " ".join(command), level=2)
    process = subprocess.run(
        list(command),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        check=False,
    )
    stderr = process.stderr or ""
    if process.returncode != 0:
        tail = "\n".join(stderr.rstrip().splitlines()[-40:])
        raise ExternalToolError(
            f"{description} failed with exit status {process.returncode}"
            + (f":\n{tail}" if tail else "")
        )
    if reporter.verbose >= 3 and stderr.strip():
        print(stderr.rstrip(), file=sys.stderr)
    return stderr


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


def decode_m4v_to_raw(
    ffmpeg: str,
    adapted: Path,
    raw_output: Path,
    *,
    width: int,
    height: int,
    idct: str,
    reporter: Reporter,
) -> DecodeResult:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-threads",
        "1",
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
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        "-y",
        str(raw_output),
    ]
    stderr = run_command(command, reporter, description="ffmpeg MPEG-4 pixel decode")
    warning_text = stderr.lower()
    if any(
        marker in warning_text
        for marker in ("damaged", "invalid data", "corrupt", "conceal", "error while")
    ):
        reporter.warn(
            "ffmpeg reported possible MPEG-4 bitstream damage while decoding; "
            "rerun with -vvv to see its full diagnostics"
        )
    frame_count = count_raw_frames(raw_output, width, height)
    if frame_count == 0:
        raise ExternalToolError("ffmpeg produced zero decoded frames")
    return DecodeResult(raw_path=raw_output, frame_count=frame_count, stderr=stderr)


def merge_component_passes(
    luma_raw: Path,
    chroma_raw: Path,
    output: Path,
    *,
    width: int,
    height: int,
    luma_frames: int,
    chroma_frames: int,
) -> int:
    if luma_frames != chroma_frames:
        raise ExternalToolError(
            f"luma and chroma decoder passes produced different frame counts "
            f"({luma_frames} versus {chroma_frames})"
        )

    y_size = width * height
    chroma_size = ((width + 1) // 2) * ((height + 1) // 2)
    with luma_raw.open("rb") as luma, chroma_raw.open("rb") as chroma, output.open("wb") as merged:
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
    return luma_frames


def write_y4m(
    raw_input: Path,
    output: Path,
    *,
    width: int,
    height: int,
    fps: Fraction,
    frame_count: int,
) -> None:
    frame_size = yuv420_frame_size(width, height)
    header = (
        f"YUV4MPEG2 W{width} H{height} F{fps.numerator}:{fps.denominator} "
        "Ip A1:1 C420mpeg2 XYSCSS=420MPEG2\n"
    ).encode("ascii")
    with raw_input.open("rb") as source, output.open("wb") as destination:
        destination.write(header)
        for frame in range(frame_count):
            data = source.read(frame_size)
            if len(data) != frame_size:
                raise ExternalToolError(f"short raw frame {frame} while writing Y4M")
            destination.write(b"FRAME\n")
            destination.write(data)
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
    reporter: Reporter,
) -> None:
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
        "-y" if overwrite else "-n",
        str(output),
    ]
    run_command(command, reporter, description="ffmpeg FFV1 encode")


def copy_raw_output(raw_input: Path, output: Path) -> None:
    shutil.copyfile(raw_input, output)


# ---------------------------------------------------------------------------
# Diagnostics and top-level conversion


def picture_summary(pictures: Sequence[Picture]) -> dict[str, object]:
    counts = {name: 0 for name in ("I", "P", "B", "S")}
    extended_quant = 0
    custom_luma = 0
    custom_chroma = 0
    gmc_formats: dict[str, int] = {}
    for picture in pictures:
        counts[picture.frame_name] = counts.get(picture.frame_name, 0) + 1
        extended_quant += int(picture.uses_extended_quant)
        custom_luma += int(picture.matrix_order_luma is not None)
        custom_chroma += int(picture.matrix_order_chroma is not None)
        if picture.gmc_source_format is not None:
            gmc_formats[picture.gmc_source_format] = (
                gmc_formats.get(picture.gmc_source_format, 0) + 1
            )
    return {
        "frames": len(pictures),
        "frame_types": counts,
        "extended_quant_frames": extended_quant,
        "luma_matrix_updates": custom_luma,
        "chroma_matrix_updates": custom_chroma,
        "gmc_formats": gmc_formats,
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


def convert(args: argparse.Namespace) -> None:
    reporter = Reporter(args.verbose)
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output is not None else None
    if not input_path.is_file():
        raise VID1Error(f"input file not found: {input_path}")

    with input_path.open("rb") as stream:
        info = parse_vid1_file_header(stream)
        chunks = read_video_chunks(
            stream,
            info,
            resync_scan=args.resync_scan,
            max_chunk_size=args.max_chunk_size,
        )

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
        reporter=reporter,
    )
    reporter.info(f"parsing {len(chunks)} video packets with VIDD header skip {header_skip}")
    pictures = parse_picture_sequence(
        chunks,
        header_skip=header_skip,
        matrix_order=args.matrix_order,
        lenient=args.lenient,
        gmc_format=args.gmc_format,
    )
    pictures, timing_mode = assign_display_indices(pictures, args.timing, reporter)

    if info.frame_count is not None and info.frame_count != len(pictures):
        reporter.warn(
            f"VIDH frame count says {info.frame_count}, but {len(pictures)} VIDD packets were parsed"
        )

    summary = picture_summary(pictures)
    reporter.info(
        f"VID1 {width}x{height}, {fps.numerator}/{fps.denominator} fps, "
        f"frames={summary['frames']} types={summary['frame_types']} "
        f"timing={timing_mode}"
    )

    if args.inspect:
        report = inspect_report(
            input_path=input_path,
            info=replace(info, width=width, height=height),
            header_skip=header_skip,
            fps=fps,
            timing_mode=timing_mode,
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
            luma_yuv = temporary / "decoded-luma.yuv"
            write_adapted_stream(luma_m4v, pictures, base_config)
            luma_result = decode_m4v_to_raw(
                ffmpeg,
                luma_m4v,
                luma_yuv,
                width=width,
                height=height,
                idct=args.idct,
                reporter=reporter,
            )

            decoded_raw = luma_result.raw_path
            decoded_frames = luma_result.frame_count
            if dual_pass:
                chroma_config = replace(base_config, matrix_plane="chroma")
                chroma_m4v = temporary / "adapted-chroma.m4v"
                chroma_yuv = temporary / "decoded-chroma.yuv"
                merged_yuv = temporary / "decoded-merged.yuv"
                write_adapted_stream(chroma_m4v, pictures, chroma_config)
                chroma_result = decode_m4v_to_raw(
                    ffmpeg,
                    chroma_m4v,
                    chroma_yuv,
                    width=width,
                    height=height,
                    idct=args.idct,
                    reporter=reporter,
                )
                decoded_frames = merge_component_passes(
                    luma_result.raw_path,
                    chroma_result.raw_path,
                    merged_yuv,
                    width=width,
                    height=height,
                    luma_frames=luma_result.frame_count,
                    chroma_frames=chroma_result.frame_count,
                )
                decoded_raw = merged_yuv

            if decoded_frames != len(pictures):
                message = (
                    f"ffmpeg returned {decoded_frames} decoded frames for "
                    f"{len(pictures)} VID1 pictures"
                )
                if args.strict_frame_count:
                    raise ExternalToolError(message)
                reporter.warn(message)

            if output_format == "y4m":
                write_y4m(
                    decoded_raw,
                    output_path,
                    width=width,
                    height=height,
                    fps=fps,
                    frame_count=decoded_frames,
                )
            elif output_format == "raw":
                copy_raw_output(decoded_raw, output_path)
            elif output_format == "ffv1":
                encode_ffv1(
                    ffmpeg,
                    decoded_raw,
                    output_path,
                    width=width,
                    height=height,
                    fps=fps,
                    overwrite=True,
                    reporter=reporter,
                )
            else:
                raise AssertionError(f"unhandled output format {output_format}")

            if args.dump_adapted:
                destination = Path(args.dump_adapted)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(luma_m4v, destination)
                reporter.info(f"copied diagnostic adapted stream to {destination}")

            if keep_directory is not None:
                shutil.copytree(temporary, keep_directory)
                reporter.info(f"kept temporary files in {keep_directory}")

    except Exception:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise

    reporter.info(
        f"wrote {decoded_frames} decoded frames to {output_path} "
        f"({output_format}, yuv420p)"
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
        Picture(i, 0, t, 0, 0, 2, 1, 1, tc, b"x", 0, False, None, (), False, None, None, None, None)
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

    # Signed fixed16 GMC conversion and MPEG-4 VLC round trip.
    assert rounded_divide(192, 16) == 12
    assert rounded_divide(-192, 16) == -12
    trajectory = encode_sprite_trajectory((12, 16, 0, 0), 2)
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
            "byte-aligned signed-16-bit fixed-point form found in retail VID1 files"
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
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase diagnostics")
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

    try:
        convert(args)
    except (VID1Error, OSError, subprocess.SubprocessError) as exc:
        print(f"{PROGRAM}: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
