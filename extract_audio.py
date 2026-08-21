#!/usr/bin/env python3
"""Extract GameCube DSP-ADPCM audio from SC-family archives as FLAC files.

Usage:

    python3 extract_audio.py fro03.scg

The script uses NiemaFS to expose an archive's logical resources.  It extracts
both audio layouts observed in the GameCube SCG files:

* ``.shdr`` + ``.samp`` sample banks, whose entries are written as individual
  mono FLAC files.
* Long, interleaved DSP streams stored in the archive's unindexed ``bulk``
  area.  Their offsets, sizes, rates, and channel counts are read from either
  the flat or grouped streaming-table layout in the matching ``.shdr``
  metadata.

Stream names are recovered from ``RPNS`` resources.  When an IGC script binds a
movie to a named audio event, the event is appended to the filename.  For
example, stream 22 in the supplied archive becomes:

    streamed/ATTRFMV1_Global_AttractLoop.flac

RPNS class-4 movie references include a third numeric selector.  Only
selector 255 has been verified to address the archive's type-0x0F movie-stream
table.  Other selector values are retained in ``manifest.json`` but are not
mapped to local stream records or used to rename outputs.  In particular, a
selector such as 253 is not assumed to encode record type 0x0D.

Decoded signed 16-bit PCM is piped directly to FFmpeg and encoded losslessly
with FFmpeg's highest FLAC compression level plus exact Rice-parameter search.
Bitexact muxing and zero metadata-header padding avoid per-file encoder tags
and FFmpeg's normal reserved padding.  No intermediate WAV files or lossy
re-encoding are used.

The input suffix selects the NiemaFS class automatically.  Both the established
SCG/SCW/SCX spellings and the transposed SGC/SGW/SGX spellings are accepted:

    .scg / .sgc -> ScgFS or SgcFS
    .scw / .sgw -> ScwFS or SgwFS
    .scx / .sgx -> ScxFS or SgxFS

The DSP layouts implemented here are verified against GameCube SCG samples.
For PC/Xbox containers, the script still dispatches to the correct filesystem
class and will extract any compatible ``.shdr``/``.samp`` DSP data it finds,
but it refuses malformed or incompatible audio rather than guessing a codec.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from struct import unpack_from
from tqdm import tqdm
from typing import Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Constants and exceptions
# ---------------------------------------------------------------------------

DSP_RECORD_MIN_SIZE = 0x30
DSP_RECORD_OBSERVED_SIZE = 0x38
STREAM_RECORD_MIN_SIZE = 0x18
STREAM_RECORD_OBSERVED_SIZE = 0x18
STREAM_BLOCK_SIZE = 0x8000
STREAM_BLOCK_HEADER_SIZE = 0x100
STREAM_GROUP_COUNT_OFFSET = 0x3C
STREAM_GROUP_COUNT_COPY_OFFSET = 0x40
STREAM_GROUP_OFFSETS_OFFSET = 0x44
STREAM_GROUP_DATA_OFFSET = 0x48
STREAM_GROUP_DATA_SIZE_OFFSET = 0x4C
STREAM_GROUP_END_OFFSET = 0x50
# RPNS class 4 names movie-related resources.  In the supplied ``fro03``
# archive, selector 255 directly addresses the type-0x0F movie-stream table.
# No meaning has been verified for other selector values, so they must not be
# converted into record types merely from their low nibble.
RPNS_MOVIE_STREAM_CLASS = 4
RPNS_VERIFIED_LOCAL_SELECTOR = 255
RPNS_STANDARD_STREAM_RECORD_TYPE = 0x0F
MAX_REASONABLE_SAMPLE_ENTRIES = 1_000_000
MAX_REASONABLE_STREAM_ENTRIES = 100_000
MIN_REASONABLE_SAMPLE_RATE = 4_000
MAX_REASONABLE_SAMPLE_RATE = 384_000
FLAC_COMPRESSION_LEVEL = 12
FLAC_EXACT_RICE_PARAMETERS = True
FLAC_BITEXACT = True
FLAC_METADATA_HEADER_PADDING = 0
COMMON_SAMPLE_RATES = {
    8_000,
    11_025,
    12_000,
    16_000,
    22_050,
    24_000,
    32_000,
    44_100,
    48_000,
    64_000,
    88_200,
    96_000,
    176_400,
    192_000,
}
RESOURCE_ID_RE = re.compile(r"(?:^|_)id([0-9a-f]+)(?=\.)", re.IGNORECASE)
RPNS_MOVIE_REFERENCE_RE = re.compile(
    rb"(?<![0-9])4\s*,\s*(\d+)\s*,\s*(\d+)(?:\x00|$)"
)
RPNS_NAME_RE = re.compile(rb"#([A-Za-z0-9_]+)(?:\x00|$)")
BIND_AUDIO_RE = re.compile(
    rb"_BindAudio\s+['\"]([^'\"\x00\r\n]+)['\"]",
    re.IGNORECASE,
)
SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


class AudioExtractionError(RuntimeError):
    """Base class for an archive-audio extraction failure."""


class ShdrFormatError(AudioExtractionError):
    """Raised when a .shdr resource does not match a supported layout."""


class DspDecodeError(AudioExtractionError):
    """Raised when DSP-ADPCM data is truncated or structurally invalid."""


class StreamTableError(AudioExtractionError):
    """Raised when long-stream metadata cannot be interpreted safely."""


class FlacEncodeError(AudioExtractionError):
    """Raised when FFmpeg cannot encode decoded PCM as FLAC."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DspEntry:
    """One low-level DSP-ADPCM stream description from a .shdr sample table."""

    index: int
    start_address: int
    loop_address: int
    end_address: int
    current_address: int
    coefficients: tuple[int, ...]
    trailer_words: tuple[int, ...]


@dataclass(frozen=True)
class ParsedSampleShdr:
    """Relevant information recovered from a .shdr sample-bank table."""

    byte_order: str
    endian: str
    declared_samp_size: int
    entry_count: int
    entry_table_offset: int
    entry_stride: int
    entries: tuple[DspEntry, ...]
    rate_values: tuple[int, ...]
    rate_source: str


@dataclass(frozen=True)
class DecodedSample:
    """Decoded PCM and source geometry for one .samp DSP entry."""

    pcm: array
    sample_count: int
    encoded_byte_offset: int
    encoded_byte_size: int
    loop_start_sample: int | None


@dataclass(frozen=True)
class ResourcePair:
    """A .samp resource and its matching .shdr resource."""

    samp_path: Path
    samp_data: bytes
    shdr_path: Path
    shdr_data: bytes


@dataclass(frozen=True)
class StreamAudioEntry:
    """One long streamed-audio record from the .shdr streaming table."""

    index: int
    sample_rate: int
    unknown_04: int
    bulk_offset: int
    stored_size: int
    field_10: int
    block_count: int
    flags: int
    channels: int
    field_17: int
    raw_record: bytes
    group_index: int | None = None
    group_entry_index: int | None = None
    record_type: int | None = None


@dataclass(frozen=True)
class StreamGroup:
    """One subtable in the grouped streamed-audio metadata layout."""

    index: int
    relative_offset: int
    data_offset: int
    record_count: int
    record_type: int


@dataclass(frozen=True)
class ParsedStreamTable:
    """A validated table describing long audio stored in the archive bulk area."""

    source_path: Path
    byte_order: str
    endian: str
    section_offset: int
    section_size: int
    descriptor: int | None
    record_stride: int
    record_type: int | None
    entries: tuple[StreamAudioEntry, ...]
    score: int
    layout: str = "flat"
    groups: tuple[StreamGroup, ...] = ()
    group_offset_table_offset: int | None = None
    group_data_offset: int | None = None
    group_data_size: int | None = None
    group_end_offset: int | None = None


@dataclass(frozen=True)
class LoadedArchive:
    """NiemaFS output and the metadata needed by the audio extractors."""

    files: dict[Path, bytes]
    fs: object
    fs_class_name: str
    canonical_format: str
    archive_format: str | None
    archive_variant: str | None
    byte_order: str | None
    bulk_offset: int | None
    bulk_size: int | None


# ---------------------------------------------------------------------------
# NiemaFS loading and path helpers
# ---------------------------------------------------------------------------


_SUFFIX_TO_FORMAT_AND_CLASSES: dict[str, tuple[str, tuple[str, ...]]] = {
    ".scg": ("scg", ("ScgFS", "SgcFS")),
    ".sgc": ("scg", ("SgcFS", "ScgFS")),
    ".scw": ("scw", ("ScwFS", "SgwFS")),
    ".sgw": ("scw", ("SgwFS", "ScwFS")),
    ".scx": ("scx", ("ScxFS", "SgxFS")),
    ".sgx": ("scx", ("SgxFS", "ScxFS")),
}


def _resolve_filesystem_class(target_path: Path) -> tuple[type, str]:
    suffix = target_path.suffix.lower()
    try:
        canonical_format, class_names = _SUFFIX_TO_FORMAT_AND_CLASSES[suffix]
    except KeyError as error:
        accepted = ", ".join(sorted(_SUFFIX_TO_FORMAT_AND_CLASSES))
        raise AudioExtractionError(
            f"unsupported archive suffix {suffix or '<none>'!r}; expected one of: "
            f"{accepted}"
        ) from error

    import_errors: list[str] = []
    for module_name in ("niemafs", "niemafs.sc", "niemafs.scg"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            import_errors.append(f"{module_name}: {error}")
            continue

        for class_name in class_names:
            fs_class = getattr(module, class_name, None)
            if isinstance(fs_class, type):
                return fs_class, canonical_format

    wanted = " or ".join(class_names)
    details = "; ".join(import_errors) if import_errors else "classes not exported"
    raise AudioExtractionError(
        f"unable to import {wanted} from NiemaFS ({details}). Export the class "
        "from niemafs/__init__.py or keep sc.py inside the niemafs package."
    )


def load_archive(target_path: Path) -> LoadedArchive:
    """Read all NiemaFS-exposed files, including the SCG/SCW bulk tail."""

    fs_class, canonical_format = _resolve_filesystem_class(target_path)
    with target_path.open("rb") as target_file:
        kwargs: dict[str, object] = {
            "path": target_path,
            "file_obj": target_file,
        }
        if canonical_format in {"scg", "scw"}:
            kwargs["bulk_mode"] = "file"

        try:
            fs = fs_class(**kwargs)
        except TypeError as error:
            # A third-party alias may omit bulk_mode even though its base class
            # still exposes a stream archive.  Retry without it, then report the
            # original failure only if construction remains impossible.
            if "bulk_mode" not in kwargs:
                raise
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("bulk_mode")
            try:
                fs = fs_class(**retry_kwargs)
            except Exception:
                raise error

        files = {
            Path(path): bytes(data)
            for path, _timestamp, data in fs
            if data is not None
        }

        return LoadedArchive(
            files=files,
            fs=fs,
            fs_class_name=fs_class.__name__,
            canonical_format=canonical_format,
            archive_format=getattr(fs, "format_code", None),
            archive_variant=getattr(fs, "variant", None),
            byte_order=getattr(fs, "byte_order", None),
            bulk_offset=getattr(fs, "bulk_offset", None),
            bulk_size=getattr(fs, "bulk_size", None),
        )


def _logical_name(path: Path) -> str:
    """Return a resource name with an optional final .stored removed."""

    name = path.name
    if name.lower().endswith(".stored"):
        name = name[: -len(".stored")]
    return name


def _logical_extension(path: Path) -> str:
    """Return a normalized resource extension.

    FourCCs sometimes contain a trailing underscore after NiemaFS sanitizes a
    space or NUL, so ``.igc_`` is normalized to ``.igc``.
    """

    suffix = Path(_logical_name(path)).suffix.lower()
    return suffix.rstrip("_")


def _resource_id(path: Path) -> str | None:
    match = RESOURCE_ID_RE.search(_logical_name(path))
    return match.group(1).lower() if match else None


def _without_logical_extension(path: Path) -> str:
    name = _logical_name(path)
    suffix = Path(name).suffix
    return name[: -len(suffix)] if suffix else name


def _safe_component(value: str, fallback: str) -> str:
    value = value.strip(" \t\r\n\x00")
    value = SAFE_COMPONENT_RE.sub("_", value)
    value = value.strip("._")
    return value or fallback


def _path_order_key(path: Path) -> tuple[str, int, str]:
    match = re.match(r"(\d+)", path.name)
    numeric = int(match.group(1)) if match else 2**63 - 1
    return path.parent.as_posix(), numeric, path.name.lower()


# ---------------------------------------------------------------------------
# Primitive readers
# ---------------------------------------------------------------------------


def _read_u16(data: bytes, offset: int, endian: str, context: str) -> int:
    if offset < 0 or offset + 2 > len(data):
        raise ShdrFormatError(
            f"16-bit read at 0x{offset:X} is outside {context} "
            f"(size 0x{len(data):X})"
        )
    return unpack_from(endian + "H", data, offset)[0]


def _read_u32(data: bytes, offset: int, endian: str, context: str) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ShdrFormatError(
            f"32-bit read at 0x{offset:X} is outside {context} "
            f"(size 0x{len(data):X})"
        )
    return unpack_from(endian + "I", data, offset)[0]


# ---------------------------------------------------------------------------
# .shdr/.samp sample-bank parsing and decoding
# ---------------------------------------------------------------------------


def _dsp_address_to_sample(address: int) -> int:
    """Convert a Nintendo DSP nibble address into a decoded-sample position."""

    if address < 0:
        raise ValueError("DSP nibble addresses cannot be negative")
    frame, nibble_in_frame = divmod(address, 16)
    samples_in_partial_frame = max(0, min(14, nibble_in_frame - 2))
    return frame * 14 + samples_in_partial_frame


def _entry_geometry_is_valid(
    data: bytes,
    samp_size: int,
    table_offset: int,
    stride: int,
    count: int,
    endian: str,
) -> bool:
    if stride < DSP_RECORD_MIN_SIZE:
        return False
    if table_offset < 0 or table_offset + stride * count > len(data):
        return False

    pool_nibbles = samp_size * 2
    previous_start = -1
    checks = min(count, 32)
    for index in range(checks):
        offset = table_offset + index * stride
        start, _loop, end, current = unpack_from(endian + "4I", data, offset)
        if start < 2 or end < start or end >= pool_nibbles:
            return False
        if start < previous_start:
            return False
        if current not in (0, start):
            return False
        previous_start = start
    return True


def _candidate_entry_strides(
    data: bytes,
    table_offset: int,
    count: int,
    endian: str,
) -> list[int]:
    candidates: list[int] = [DSP_RECORD_OBSERVED_SIZE]

    possible_ends = [len(data)]
    if len(data) >= 0x64:
        declared_end = _read_u32(data, 0x60, endian, ".shdr")
        if table_offset < declared_end <= len(data):
            possible_ends.insert(0, declared_end)

    for end_offset in possible_ends:
        remaining = end_offset - table_offset
        if count and remaining > 0 and remaining % count == 0:
            stride = remaining // count
            if DSP_RECORD_MIN_SIZE <= stride <= 0x100:
                candidates.append(stride)

    return list(dict.fromkeys(candidates))


def _extract_rate_values(data: bytes, endian: str) -> tuple[list[int], str]:
    """Recover plausible sample rates from the structured .shdr sections."""

    if len(data) >= 0x48:
        try:
            section_offset = _read_u32(data, 0x44, endian, ".shdr")
            if 0 <= section_offset <= len(data) - 12:
                section_size = _read_u32(
                    data, section_offset + 4, endian, ".shdr rate section"
                )
                descriptor = _read_u32(
                    data, section_offset + 8, endian, ".shdr rate section"
                )
                stride = descriptor >> 16
                section_end = section_offset + 8 + section_size
                if (
                    4 <= stride <= 0x100
                    and section_size >= 4
                    and (section_size - 4) % stride == 0
                    and section_end <= len(data)
                ):
                    count = (section_size - 4) // stride
                    values = [
                        _read_u32(
                            data,
                            section_offset + 12 + index * stride,
                            endian,
                            ".shdr rate record",
                        )
                        for index in range(count)
                    ]
                    plausible = [
                        value
                        for value in values
                        if MIN_REASONABLE_SAMPLE_RATE
                        <= value
                        <= MAX_REASONABLE_SAMPLE_RATE
                    ]
                    if plausible and len(plausible) == len(values):
                        return plausible, "structured .shdr section"
        except ShdrFormatError:
            pass

    counts: Counter[int] = Counter()
    for offset in range(0, len(data) - 3, 4):
        value = unpack_from(endian + "I", data, offset)[0]
        if value in COMMON_SAMPLE_RATES:
            counts[value] += 1
    if counts:
        values: list[int] = []
        for value, occurrences in counts.items():
            values.extend([value] * occurrences)
        return values, "aligned conventional-rate scan"

    return [], "not found"


def _parse_sample_shdr_with_endian(
    shdr_data: bytes,
    samp_data: bytes,
    endian: str,
) -> ParsedSampleShdr:
    if len(shdr_data) < 0x20:
        raise ShdrFormatError(".shdr is too small for a sample table header")

    declared_samp_size = _read_u32(shdr_data, 0x14, endian, ".shdr")
    count = _read_u32(shdr_data, 0x18, endian, ".shdr")
    table_offset = _read_u32(shdr_data, 0x1C, endian, ".shdr")

    if not 0 < count <= MAX_REASONABLE_SAMPLE_ENTRIES:
        raise ShdrFormatError(f"implausible DSP entry count: {count}")
    if not 0 <= table_offset < len(shdr_data):
        raise ShdrFormatError(
            f"DSP entry table offset 0x{table_offset:X} is outside the .shdr"
        )

    stride = None
    for candidate in _candidate_entry_strides(
        shdr_data, table_offset, count, endian
    ):
        if _entry_geometry_is_valid(
            shdr_data,
            len(samp_data),
            table_offset,
            candidate,
            count,
            endian,
        ):
            stride = candidate
            break
    if stride is None:
        raise ShdrFormatError(
            "could not identify a valid DSP sample-table stride; the metadata "
            "may use another platform layout or may still be compressed"
        )

    entries: list[DspEntry] = []
    pool_nibbles = len(samp_data) * 2
    previous_start = -1
    for index in range(count):
        offset = table_offset + index * stride
        start, loop, end, current = unpack_from(endian + "4I", shdr_data, offset)
        coefficients = tuple(unpack_from(endian + "16h", shdr_data, offset + 0x10))
        trailer_size = max(0, min(stride - 0x30, 8))
        if trailer_size >= 2:
            word_count = trailer_size // 2
            trailer_words = tuple(
                unpack_from(
                    endian + f"{word_count}H", shdr_data, offset + 0x30
                )
            )
        else:
            trailer_words = ()

        if start < 2 or end < start or end >= pool_nibbles:
            raise ShdrFormatError(
                f"entry {index} has invalid nibble range 0x{start:X}..0x{end:X} "
                f"for a 0x{len(samp_data):X}-byte .samp"
            )
        if start < previous_start:
            raise ShdrFormatError(f"entry {index} starts before its predecessor")

        entries.append(
            DspEntry(
                index=index,
                start_address=start,
                loop_address=loop,
                end_address=end,
                current_address=current,
                coefficients=coefficients,
                trailer_words=trailer_words,
            )
        )
        previous_start = start

    rate_values, rate_source = _extract_rate_values(shdr_data, endian)
    return ParsedSampleShdr(
        byte_order="big" if endian == ">" else "little",
        endian=endian,
        declared_samp_size=declared_samp_size,
        entry_count=count,
        entry_table_offset=table_offset,
        entry_stride=stride,
        entries=tuple(entries),
        rate_values=tuple(rate_values),
        rate_source=rate_source,
    )


def parse_sample_shdr(
    shdr_data: bytes,
    samp_data: bytes,
    preferred_endian: str | None = None,
) -> ParsedSampleShdr:
    """Parse a supported .shdr sample table, trying both byte orders safely."""

    endian_order = [preferred_endian] if preferred_endian in {">", "<"} else []
    endian_order.extend(endian for endian in (">", "<") if endian not in endian_order)

    errors: list[str] = []
    for endian in endian_order:
        try:
            parsed = _parse_sample_shdr_with_endian(shdr_data, samp_data, endian)
            for entry in parsed.entries[: min(32, len(parsed.entries))]:
                frame_offset = (entry.start_address // 16) * 8
                if frame_offset + 8 > len(samp_data):
                    raise ShdrFormatError(
                        f"entry {entry.index} starts outside the .samp pool"
                    )
                predictor = samp_data[frame_offset] >> 4
                if predictor >= 8:
                    raise ShdrFormatError(
                        f"entry {entry.index} begins with invalid DSP predictor "
                        f"{predictor}"
                    )
            return parsed
        except ShdrFormatError as error:
            label = "big endian" if endian == ">" else "little endian"
            errors.append(f"{label}: {error}")

    raise ShdrFormatError("; ".join(errors))


def choose_sample_rates(
    parsed: ParsedSampleShdr,
    override: int | None,
) -> tuple[list[int], str]:
    """Choose one playback rate per low-level .samp entry."""

    if override is not None:
        if not MIN_REASONABLE_SAMPLE_RATE <= override <= MAX_REASONABLE_SAMPLE_RATE:
            raise ShdrFormatError(
                f"forced sample rate {override} is outside "
                f"{MIN_REASONABLE_SAMPLE_RATE}..{MAX_REASONABLE_SAMPLE_RATE}"
            )
        return [override] * parsed.entry_count, "command-line override"

    rates = list(parsed.rate_values)
    if not rates:
        raise ShdrFormatError(
            "could not locate a sample rate in the .shdr; use --sample-rate RATE"
        )

    unique_rates = set(rates)
    if len(unique_rates) == 1:
        return [rates[0]] * parsed.entry_count, parsed.rate_source

    if len(rates) == parsed.entry_count:
        return rates, parsed.rate_source + " (one rate per DSP entry)"

    counts = Counter(rates)
    summary = ", ".join(
        f"{rate} Hz x{count}" for rate, count in sorted(counts.items())
    )
    raise ShdrFormatError(
        "the .shdr contains multiple rates but no verified mapping to its "
        f"{parsed.entry_count} DSP entries ({summary}); use --sample-rate RATE"
    )


def _decode_dsp_frames(encoded: bytes, coefficients: Sequence[int]) -> array:
    """Decode complete Nintendo DSP-ADPCM frames into native-endian int16."""

    if len(coefficients) != 16:
        raise DspDecodeError(
            f"DSP decoding requires 16 coefficients, got {len(coefficients)}"
        )
    if len(encoded) % 8:
        raise DspDecodeError(
            f"DSP payload length 0x{len(encoded):X} is not a multiple of 8"
        )

    history_1 = 0
    history_2 = 0
    decoded = array("h")

    for frame_offset in range(0, len(encoded), 8):
        frame = encoded[frame_offset : frame_offset + 8]
        predictor_scale = frame[0]
        predictor = predictor_scale >> 4
        exponent = predictor_scale & 0x0F
        if predictor >= 8:
            raise DspDecodeError(
                f"invalid DSP predictor {predictor} at encoded offset "
                f"0x{frame_offset:X}"
            )

        coefficient_1 = coefficients[predictor * 2]
        coefficient_2 = coefficients[predictor * 2 + 1]
        scale = 1 << exponent

        for packed in frame[1:]:
            for nibble in (packed >> 4, packed & 0x0F):
                if nibble >= 8:
                    nibble -= 16

                value = (
                    ((nibble * scale) << 11)
                    + 1024
                    + coefficient_1 * history_1
                    + coefficient_2 * history_2
                ) >> 11
                value = max(-32768, min(32767, value))
                history_2, history_1 = history_1, value
                decoded.append(value)

    return decoded


def decode_dsp_entry(samp_data: bytes, entry: DspEntry) -> DecodedSample:
    """Decode one nibble-addressed .samp entry to native-endian PCM16."""

    start_sample_position = _dsp_address_to_sample(entry.start_address)
    end_sample_position = _dsp_address_to_sample(entry.end_address + 1)
    sample_count = end_sample_position - start_sample_position
    if sample_count <= 0:
        raise DspDecodeError(
            f"entry {entry.index} has a non-positive decoded length "
            f"({sample_count})"
        )

    start_frame = entry.start_address // 16
    end_frame = entry.end_address // 16
    first_sample_in_start_frame = max(
        0, min(14, entry.start_address % 16 - 2)
    )
    encoded_byte_offset = start_frame * 8
    encoded_byte_size = (end_frame - start_frame + 1) * 8
    encoded_end = encoded_byte_offset + encoded_byte_size
    if encoded_end > len(samp_data):
        raise DspDecodeError(
            f"entry {entry.index} needs bytes 0x{encoded_byte_offset:X}.."
            f"0x{encoded_end:X}, outside the .samp"
        )

    full_pcm = _decode_dsp_frames(
        samp_data[encoded_byte_offset:encoded_end], entry.coefficients
    )
    first = first_sample_in_start_frame
    last = first + sample_count
    if last > len(full_pcm):
        raise DspDecodeError(
            f"entry {entry.index} decoded only {len(full_pcm)} samples but "
            f"requires {last}"
        )
    pcm = array("h", full_pcm[first:last])

    loop_start_sample: int | None = None
    if entry.start_address <= entry.loop_address <= entry.end_address:
        loop_start_sample = (
            _dsp_address_to_sample(entry.loop_address) - start_sample_position
        )

    return DecodedSample(
        pcm=pcm,
        sample_count=sample_count,
        encoded_byte_offset=encoded_byte_offset,
        encoded_byte_size=encoded_byte_size,
        loop_start_sample=loop_start_sample,
    )


def pair_sample_resources(
    files: Mapping[Path, bytes],
) -> tuple[list[ResourcePair], list[str]]:
    """Match .samp resources to .shdr resources by ID, stem, or flat order."""

    headers_by_parent_and_id: defaultdict[tuple[Path, str], list[Path]] = (
        defaultdict(list)
    )
    headers_by_id: defaultdict[str, list[Path]] = defaultdict(list)
    headers_by_parent_and_stem: dict[tuple[Path, str], Path] = {}
    header_paths: list[Path] = []

    for path in files:
        if _logical_extension(path) != ".shdr":
            continue
        header_paths.append(path)
        resource_id = _resource_id(path)
        if resource_id is not None:
            headers_by_parent_and_id[(path.parent, resource_id)].append(path)
            headers_by_id[resource_id].append(path)
        headers_by_parent_and_stem[
            (path.parent, _without_logical_extension(path))
        ] = path

    pairs: list[ResourcePair] = []
    warnings: list[str] = []
    used_headers: set[Path] = set()
    samp_paths = sorted(
        (path for path in files if _logical_extension(path) == ".samp"),
        key=_path_order_key,
    )

    for samp_path in samp_paths:
        candidates: list[Path] = []
        resource_id = _resource_id(samp_path)
        if resource_id is not None:
            candidates = headers_by_parent_and_id.get(
                (samp_path.parent, resource_id), []
            )
            if not candidates:
                global_candidates = headers_by_id.get(resource_id, [])
                if len(global_candidates) == 1:
                    candidates = global_candidates

        if not candidates:
            same_stem = headers_by_parent_and_stem.get(
                (samp_path.parent, _without_logical_extension(samp_path))
            )
            if same_stem is not None:
                candidates = [same_stem]

        if not candidates:
            # SCX flat chunks do not necessarily carry resource IDs.  Pair with
            # the closest earlier unused header in the same directory.
            preceding = [
                path
                for path in header_paths
                if path.parent == samp_path.parent
                and _path_order_key(path) < _path_order_key(samp_path)
                and path not in used_headers
            ]
            if preceding:
                candidates = [max(preceding, key=_path_order_key)]

        if len(candidates) != 1:
            if not candidates:
                warnings.append(
                    f"no matching .shdr found for {samp_path.as_posix()}"
                )
            else:
                candidate_text = ", ".join(
                    path.as_posix() for path in candidates
                )
                warnings.append(
                    f"ambiguous .shdr match for {samp_path.as_posix()}: "
                    f"{candidate_text}"
                )
            continue

        shdr_path = candidates[0]
        used_headers.add(shdr_path)
        pairs.append(
            ResourcePair(
                samp_path=samp_path,
                samp_data=files[samp_path],
                shdr_path=shdr_path,
                shdr_data=files[shdr_path],
            )
        )

    return pairs, warnings


# ---------------------------------------------------------------------------
# Long streamed-audio metadata and decoding
# ---------------------------------------------------------------------------


def _candidate_section_offsets(data: bytes, endian: str) -> list[int]:
    offsets: list[int] = []
    if len(data) >= 0x48:
        offsets.append(unpack_from(endian + "I", data, 0x44)[0])

    # .shdr headers store section pointers in their first several dozen words.
    for offset in range(0, min(len(data), 0x80) - 3, 4):
        value = unpack_from(endian + "I", data, offset)[0]
        if 0 <= value <= len(data) - 12:
            offsets.append(value)

    return list(dict.fromkeys(offsets))


def _parse_stream_table_at(
    data: bytes,
    source_path: Path,
    section_offset: int,
    endian: str,
    bulk_size: int,
) -> ParsedStreamTable:
    if not 0 <= section_offset <= len(data) - 12:
        raise StreamTableError("section pointer is outside the .shdr")

    section_size = unpack_from(endian + "I", data, section_offset + 4)[0]
    descriptor = unpack_from(endian + "I", data, section_offset + 8)[0]
    stride = descriptor >> 16
    record_type = descriptor & 0xFFFF

    if not STREAM_RECORD_MIN_SIZE <= stride <= 0x100:
        raise StreamTableError(f"implausible stream-record stride 0x{stride:X}")
    if section_size < 4 or (section_size - 4) % stride:
        raise StreamTableError(
            f"section size 0x{section_size:X} does not fit stride 0x{stride:X}"
        )

    count = (section_size - 4) // stride
    if not 1 <= count <= MAX_REASONABLE_STREAM_ENTRIES:
        raise StreamTableError(f"implausible stream count: {count}")

    records_offset = section_offset + 12
    records_end = records_offset + count * stride
    if records_end > len(data):
        raise StreamTableError("stream table extends beyond the .shdr")

    entries: list[StreamAudioEntry] = []
    valid_rates = 0
    valid_channels = 0
    monotonic = True
    contiguous = 0
    previous_offset = -1
    previous_end = 0

    for index in range(count):
        offset = records_offset + index * stride
        raw_record = data[offset : offset + stride]
        (
            sample_rate,
            unknown_04,
            bulk_offset,
            stored_size,
        ) = unpack_from(endian + "4I", raw_record, 0)
        field_10, block_count, flags = unpack_from(
            endian + "3H", raw_record, 0x10
        )
        channels = raw_record[0x16]
        field_17 = raw_record[0x17]

        if MIN_REASONABLE_SAMPLE_RATE <= sample_rate <= MAX_REASONABLE_SAMPLE_RATE:
            valid_rates += 1
        if 1 <= channels <= 8:
            valid_channels += 1

        if bulk_offset < previous_offset:
            monotonic = False
        if index and bulk_offset == previous_end:
            contiguous += 1
        previous_offset = bulk_offset
        previous_end = bulk_offset + stored_size

        if stored_size and bulk_offset + stored_size > bulk_size:
            raise StreamTableError(
                f"entry {index} points to bulk range 0x{bulk_offset:X}.."
                f"0x{bulk_offset + stored_size:X}, beyond 0x{bulk_size:X}"
            )

        entries.append(
            StreamAudioEntry(
                index=index,
                sample_rate=sample_rate,
                unknown_04=unknown_04,
                bulk_offset=bulk_offset,
                stored_size=stored_size,
                field_10=field_10,
                block_count=block_count,
                flags=flags,
                channels=channels,
                field_17=field_17,
                raw_record=raw_record,
                record_type=record_type,
            )
        )

    if valid_rates < max(1, count - 1):
        raise StreamTableError(
            f"only {valid_rates}/{count} records contain plausible sample rates"
        )
    if valid_channels < max(1, count - 1):
        raise StreamTableError(
            f"only {valid_channels}/{count} records contain plausible channel counts"
        )
    if not monotonic:
        raise StreamTableError("bulk offsets are not monotonically increasing")
    if sum(entry.stored_size for entry in entries) == 0:
        raise StreamTableError("all stream records have zero stored size")

    score = count * 20 + valid_rates * 4 + valid_channels * 4 + contiguous * 3
    if stride == 0x18:
        score += 50
    if record_type == 0x0F:
        score += 25
    if entries and entries[0].bulk_offset == 0:
        score += 10
    if entries and entries[-1].bulk_offset + entries[-1].stored_size <= bulk_size:
        score += 10

    return ParsedStreamTable(
        source_path=source_path,
        byte_order="big" if endian == ">" else "little",
        endian=endian,
        section_offset=section_offset,
        section_size=section_size,
        descriptor=descriptor,
        record_stride=stride,
        record_type=record_type,
        entries=tuple(entries),
        score=score,
        layout="flat",
    )


def _is_zero_length_stream_placeholder(entry: StreamAudioEntry) -> bool:
    """Return whether a grouped-table record is an explicit empty slot."""

    return (
        entry.stored_size == 0
        and entry.sample_rate == 0
        and entry.channels == 0
        and entry.bulk_offset in {0, 0xFFFFFFFF}
    )


def _parse_grouped_stream_table(
    data: bytes,
    source_path: Path,
    endian: str,
    bulk_size: int,
) -> ParsedStreamTable:
    """Parse the grouped stream catalog used by some GameCube SCG archives.

    The .shdr header supplies a group count, an array of relative offsets, and
    one packed data area.  Every group begins with ``u16 count, u16 type`` and
    is followed by ``count`` standard 0x18-byte stream records.  Empty records
    are retained because RPNS references may deliberately point at them.
    """

    header_end = STREAM_GROUP_END_OFFSET + 4
    if len(data) < header_end:
        raise StreamTableError(".shdr is too small for grouped stream pointers")

    group_count = unpack_from(
        endian + "I", data, STREAM_GROUP_COUNT_OFFSET
    )[0]
    group_count_copy = unpack_from(
        endian + "I", data, STREAM_GROUP_COUNT_COPY_OFFSET
    )[0]
    offsets_offset = unpack_from(
        endian + "I", data, STREAM_GROUP_OFFSETS_OFFSET
    )[0]
    group_data_offset = unpack_from(
        endian + "I", data, STREAM_GROUP_DATA_OFFSET
    )[0]
    group_data_size = unpack_from(
        endian + "I", data, STREAM_GROUP_DATA_SIZE_OFFSET
    )[0]
    group_end_offset = unpack_from(
        endian + "I", data, STREAM_GROUP_END_OFFSET
    )[0]

    if not 1 <= group_count <= 0x1000:
        raise StreamTableError(f"implausible stream-group count: {group_count}")
    if group_count_copy != group_count:
        raise StreamTableError(
            "group-count header copies disagree: "
            f"{group_count} versus {group_count_copy}"
        )

    offsets_size = group_count * 4
    if not 0 <= offsets_offset <= len(data) - offsets_size:
        raise StreamTableError("group-offset array is outside the .shdr")
    if group_data_offset != offsets_offset + offsets_size:
        raise StreamTableError(
            "group data does not immediately follow its offset array"
        )
    if group_end_offset != group_data_offset + group_data_size:
        raise StreamTableError(
            "group data size and end pointer disagree"
        )
    if not group_data_offset <= group_end_offset <= len(data):
        raise StreamTableError("group data area is outside the .shdr")

    relative_offsets = list(
        unpack_from(endian + f"{group_count}I", data, offsets_offset)
    )
    if not relative_offsets or relative_offsets[0] != 0:
        raise StreamTableError("first stream-group relative offset is not zero")
    if any(
        next_offset <= offset
        for offset, next_offset in zip(relative_offsets, relative_offsets[1:])
    ):
        raise StreamTableError(
            "stream-group relative offsets are not strictly increasing"
        )
    if relative_offsets[-1] >= group_data_size:
        raise StreamTableError("last stream-group offset is outside group data")

    groups: list[StreamGroup] = []
    entries: list[StreamAudioEntry] = []
    valid_rates = 0
    valid_channels = 0
    contiguous = 0
    previous_offset: int | None = None
    previous_end: int | None = None
    global_index = 0

    for group_index, relative_offset in enumerate(relative_offsets):
        start = group_data_offset + relative_offset
        next_relative = (
            relative_offsets[group_index + 1]
            if group_index + 1 < group_count
            else group_data_size
        )
        limit = group_data_offset + next_relative
        if start + 4 > limit:
            raise StreamTableError(
                f"stream group {group_index} is smaller than its four-byte header"
            )

        record_count, record_type = unpack_from(endian + "2H", data, start)
        if record_count > MAX_REASONABLE_STREAM_ENTRIES:
            raise StreamTableError(
                f"stream group {group_index} has implausible record count "
                f"{record_count}"
            )
        required_size = 4 + record_count * STREAM_RECORD_OBSERVED_SIZE
        available_size = limit - start
        if required_size != available_size:
            raise StreamTableError(
                f"stream group {group_index} occupies 0x{available_size:X} bytes, "
                f"but its {record_count} records require 0x{required_size:X}"
            )

        groups.append(
            StreamGroup(
                index=group_index,
                relative_offset=relative_offset,
                data_offset=start,
                record_count=record_count,
                record_type=record_type,
            )
        )

        records_offset = start + 4
        for group_entry_index in range(record_count):
            offset = (
                records_offset
                + group_entry_index * STREAM_RECORD_OBSERVED_SIZE
            )
            raw_record = data[offset : offset + STREAM_RECORD_OBSERVED_SIZE]
            (
                sample_rate,
                unknown_04,
                bulk_offset,
                stored_size,
            ) = unpack_from(endian + "4I", raw_record, 0)
            field_10, block_count, flags = unpack_from(
                endian + "3H", raw_record, 0x10
            )
            channels = raw_record[0x16]
            field_17 = raw_record[0x17]

            entry = StreamAudioEntry(
                index=global_index,
                sample_rate=sample_rate,
                unknown_04=unknown_04,
                bulk_offset=bulk_offset,
                stored_size=stored_size,
                field_10=field_10,
                block_count=block_count,
                flags=flags,
                channels=channels,
                field_17=field_17,
                raw_record=raw_record,
                group_index=group_index,
                group_entry_index=group_entry_index,
                record_type=record_type,
            )
            entries.append(entry)
            global_index += 1

            if _is_zero_length_stream_placeholder(entry):
                continue
            if stored_size == 0:
                raise StreamTableError(
                    f"group {group_index} entry {group_entry_index} has zero "
                    "stored size but is not a recognized placeholder"
                )
            if not (
                MIN_REASONABLE_SAMPLE_RATE
                <= sample_rate
                <= MAX_REASONABLE_SAMPLE_RATE
            ):
                raise StreamTableError(
                    f"group {group_index} entry {group_entry_index} has "
                    f"implausible sample rate {sample_rate}"
                )
            if not 1 <= channels <= 8:
                raise StreamTableError(
                    f"group {group_index} entry {group_entry_index} has "
                    f"implausible channel count {channels}"
                )
            if bulk_offset + stored_size > bulk_size:
                raise StreamTableError(
                    f"group {group_index} entry {group_entry_index} points to "
                    f"bulk range 0x{bulk_offset:X}.."
                    f"0x{bulk_offset + stored_size:X}, beyond 0x{bulk_size:X}"
                )
            if previous_offset is not None and bulk_offset < previous_offset:
                raise StreamTableError(
                    "non-placeholder grouped stream offsets are not monotonic"
                )
            if previous_end is not None and bulk_offset == previous_end:
                contiguous += 1
            previous_offset = bulk_offset
            previous_end = bulk_offset + stored_size
            valid_rates += 1
            valid_channels += 1

    meaningful_entries = [
        entry for entry in entries if not _is_zero_length_stream_placeholder(entry)
    ]
    if not meaningful_entries:
        raise StreamTableError("grouped stream catalog has no stored audio")

    score = (
        len(meaningful_entries) * 20
        + valid_rates * 4
        + valid_channels * 4
        + contiguous * 3
        + min(group_count, 64) * 2
    )
    if any(group.record_type == RPNS_STANDARD_STREAM_RECORD_TYPE for group in groups):
        score += 10
    if meaningful_entries[0].bulk_offset == 0:
        score += 10
    if (
        meaningful_entries[-1].bulk_offset
        + meaningful_entries[-1].stored_size
        <= bulk_size
    ):
        score += 10

    return ParsedStreamTable(
        source_path=source_path,
        byte_order="big" if endian == ">" else "little",
        endian=endian,
        section_offset=offsets_offset,
        section_size=group_end_offset - offsets_offset,
        descriptor=None,
        record_stride=STREAM_RECORD_OBSERVED_SIZE,
        record_type=None,
        entries=tuple(entries),
        score=score,
        layout="grouped",
        groups=tuple(groups),
        group_offset_table_offset=offsets_offset,
        group_data_offset=group_data_offset,
        group_data_size=group_data_size,
        group_end_offset=group_end_offset,
    )


def find_stream_table(
    files: Mapping[Path, bytes],
    bulk_size: int,
    preferred_endian: str | None,
) -> ParsedStreamTable:
    """Find the highest-confidence long-stream table in any .shdr resource."""

    endian_order = [preferred_endian] if preferred_endian in {">", "<"} else []
    endian_order.extend(endian for endian in (">", "<") if endian not in endian_order)

    candidates: list[ParsedStreamTable] = []
    errors: list[str] = []
    for path, data in files.items():
        if _logical_extension(path) != ".shdr":
            continue
        for endian in endian_order:
            for section_offset in _candidate_section_offsets(data, endian):
                try:
                    table = _parse_stream_table_at(
                        data=data,
                        source_path=path,
                        section_offset=section_offset,
                        endian=endian,
                        bulk_size=bulk_size,
                    )
                except StreamTableError as error:
                    errors.append(
                        f"{path.as_posix()} @0x{section_offset:X}: {error}"
                    )
                    continue
                candidates.append(table)

            try:
                grouped_table = _parse_grouped_stream_table(
                    data=data,
                    source_path=path,
                    endian=endian,
                    bulk_size=bulk_size,
                )
            except StreamTableError as error:
                errors.append(
                    f"{path.as_posix()} grouped layout: {error}"
                )
            else:
                candidates.append(grouped_table)

    if not candidates:
        detail = errors[-1] if errors else "no .shdr resources were present"
        raise StreamTableError(f"no valid streamed-audio table found ({detail})")

    candidates.sort(
        key=lambda table: (table.score, table.layout == "flat"),
        reverse=True,
    )
    best = candidates[0]
    if len(candidates) > 1 and candidates[1].score == best.score:
        other = candidates[1]
        if (
            other.source_path != best.source_path
            or other.section_offset != best.section_offset
            or other.endian != best.endian
        ):
            raise StreamTableError(
                "multiple stream tables have equal confidence: "
                f"{best.source_path}@0x{best.section_offset:X} and "
                f"{other.source_path}@0x{other.section_offset:X}"
            )
    return best


def _trim_trailing_zero_dsp_frames(payload: bytes) -> tuple[bytes, int, int]:
    """Remove complete all-zero frames from the end of a DSP payload.

    SCG streamed-audio regions are block-aligned and may contain one or more
    zero-filled padding blocks after the last meaningful DSP frame.  Trimming
    only complete eight-byte zero frames reproduces the game's observed stream
    boundaries while retaining non-zero encoded silence and all earlier data.
    """

    aligned_size = len(payload) - (len(payload) % 8)
    discarded_partial_bytes = len(payload) - aligned_size
    end = aligned_size
    zero_frame = b"\x00" * 8
    while end >= 8 and payload[end - 8 : end] == zero_frame:
        end -= 8
    trimmed_frames = (aligned_size - end) // 8
    return payload[:end], trimmed_frames, discarded_partial_bytes


def decode_streamed_dsp(
    bulk_data: bytes,
    entry: StreamAudioEntry,
    coefficient_endian: str,
) -> tuple[array, list[int], list[int], list[int], int]:
    """Decode one interleaved long DSP stream to native-endian PCM16.

    Returns ``(interleaved_pcm, channel_sample_counts, trimmed_zero_frames,
    discarded_partial_bytes, padded_samples)``.
    """

    if entry.channels not in {1, 2}:
        raise DspDecodeError(
            f"stream {entry.index} uses unsupported channel count "
            f"{entry.channels}; only mono and stereo DSP streams are known"
        )
    if entry.stored_size <= 0:
        raise DspDecodeError(f"stream {entry.index} has no stored data")

    start = entry.bulk_offset
    end = start + entry.stored_size
    if start < 0 or end > len(bulk_data):
        raise DspDecodeError(
            f"stream {entry.index} needs bulk range 0x{start:X}..0x{end:X}, "
            f"outside 0x{len(bulk_data):X} bytes"
        )
    region = bulk_data[start:end]

    # A zero-filled record at index 0 is used as a placeholder in the sample.
    if not any(region):
        raise DspDecodeError(f"stream {entry.index} is entirely zero-filled")

    chunks = [
        region[offset : offset + STREAM_BLOCK_SIZE]
        for offset in range(0, len(region), STREAM_BLOCK_SIZE)
    ]
    if len(chunks) < entry.channels:
        raise DspDecodeError(
            f"stream {entry.index} has only {len(chunks)} physical block(s) "
            f"for {entry.channels} channels"
        )

    channels_pcm: list[array] = []
    channel_sample_counts: list[int] = []
    channel_trimmed_frames: list[int] = []
    channel_partial_bytes: list[int] = []

    for channel in range(entry.channels):
        channel_chunks = chunks[channel :: entry.channels]
        first_chunk = channel_chunks[0]
        if len(first_chunk) < STREAM_BLOCK_HEADER_SIZE:
            raise DspDecodeError(
                f"stream {entry.index} channel {channel} has a truncated "
                "coefficient/header block"
            )
        coefficients = tuple(
            unpack_from(coefficient_endian + "16h", first_chunk, 0)
        )

        payload = b"".join(
            chunk[STREAM_BLOCK_HEADER_SIZE:]
            for chunk in channel_chunks
            if len(chunk) > STREAM_BLOCK_HEADER_SIZE
        )
        payload, trimmed_frames, partial_bytes = _trim_trailing_zero_dsp_frames(
            payload
        )
        if not payload:
            raise DspDecodeError(
                f"stream {entry.index} channel {channel} contains no "
                "non-padding DSP frames"
            )

        pcm = _decode_dsp_frames(payload, coefficients)
        channels_pcm.append(pcm)
        channel_sample_counts.append(len(pcm))
        channel_trimmed_frames.append(trimmed_frames)
        channel_partial_bytes.append(partial_bytes)

    max_samples = max(channel_sample_counts)
    padded_samples = sum(max_samples - count for count in channel_sample_counts)

    if entry.channels == 1:
        interleaved = channels_pcm[0]
    else:
        interleaved = array("h", [0]) * (max_samples * entry.channels)
        for channel, pcm in enumerate(channels_pcm):
            if len(pcm) < max_samples:
                padded = array("h", pcm)
                padded.extend([0] * (max_samples - len(pcm)))
            else:
                padded = pcm
            interleaved[channel::entry.channels] = padded

    return (
        interleaved,
        channel_sample_counts,
        channel_trimmed_frames,
        channel_partial_bytes,
        padded_samples,
    )


# ---------------------------------------------------------------------------
# RPNS/IGC naming
# ---------------------------------------------------------------------------


def _nearby_rpns_name(
    data: bytes,
    reference_match: re.Match[bytes],
    window_start: int,
    window_end: int,
) -> str | None:
    """Find the nearest ``#NAME`` inside one RPNS reference window."""

    candidates: list[tuple[int, int, str]] = []
    for name_match in RPNS_NAME_RE.finditer(data, window_start, window_end):
        if name_match.end() <= reference_match.start():
            distance = reference_match.start() - name_match.end()
            direction_rank = 1
        elif name_match.start() >= reference_match.end():
            distance = name_match.start() - reference_match.end()
            direction_rank = 0
        else:
            distance = 0
            direction_rank = 0
        name = name_match.group(1).decode("ascii", "replace")
        candidates.append((distance, direction_rank, name))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def parse_rpns_named_references(
    files: Mapping[Path, bytes],
) -> tuple[list[dict[str, object]], list[str]]:
    """Recover named class-4 movie references from RPNS resources.

    Selector 255 is the only form currently verified to address a stream table
    in the same SCG archive.  Every other selector is preserved for diagnosis
    but deliberately left unresolved; treating its low nibble as a record type
    would turn an unverified coincidence into a false filename.
    """

    named_references: list[dict[str, object]] = []
    warnings: list[str] = []
    seen: set[tuple[str, int, int, str]] = set()

    for path, data in files.items():
        if _logical_extension(path) != ".rpns":
            continue

        matches = list(RPNS_MOVIE_REFERENCE_RE.finditer(data))
        for match_index, match in enumerate(matches):
            stream_index = int(match.group(1))
            selector = int(match.group(2))
            window_start = max(0, match.start() - 0x200)
            window_end = min(len(data), match.end() + 0x200)
            if match_index:
                window_start = max(window_start, matches[match_index - 1].end())
            if match_index + 1 < len(matches):
                window_end = min(window_end, matches[match_index + 1].start())

            name = _nearby_rpns_name(
                data,
                match,
                window_start=window_start,
                window_end=window_end,
            )
            if name is None:
                continue

            identity = (path.as_posix(), stream_index, selector, name.lower())
            if identity in seen:
                continue
            seen.add(identity)

            verified_local = selector == RPNS_VERIFIED_LOCAL_SELECTOR
            named_references.append(
                {
                    "source_path": path.as_posix(),
                    "rpns_offset": match.start(),
                    "rpns_offset_hex": f"0x{match.start():X}",
                    "stream_class": RPNS_MOVIE_STREAM_CLASS,
                    "stream_index": stream_index,
                    "selector": selector,
                    "selector_hex": f"0x{selector:X}",
                    "selector_interpretation": (
                        "verified local type-0x0F movie-stream table"
                        if verified_local
                        else "unverified; not mapped to this archive"
                    ),
                    "selected_record_type": (
                        RPNS_STANDARD_STREAM_RECORD_TYPE
                        if verified_local
                        else None
                    ),
                    "selected_record_type_hex": (
                        f"0x{RPNS_STANDARD_STREAM_RECORD_TYPE:X}"
                        if verified_local
                        else None
                    ),
                    "name": name,
                }
            )

    return named_references, warnings


def _entry_manifest_location(entry: StreamAudioEntry) -> dict[str, object]:
    return {
        "resolved_global_stream_index": entry.index,
        "resolved_group_index": entry.group_index,
        "resolved_group_entry_index": entry.group_entry_index,
        "resolved_record_type": entry.record_type,
        "resolved_record_type_hex": (
            f"0x{entry.record_type:X}" if entry.record_type is not None else None
        ),
        "resolved_stored_size": entry.stored_size,
        "resolved_bulk_offset": entry.bulk_offset,
        "resolved_bulk_offset_hex": f"0x{entry.bulk_offset:X}",
    }


def _verified_local_movie_candidates(
    table: ParsedStreamTable,
    stream_index: int,
) -> list[StreamAudioEntry]:
    """Find selector-255 movie-stream candidates at one zero-based index."""

    candidates: list[StreamAudioEntry] = []
    for entry in table.entries:
        if entry.record_type != RPNS_STANDARD_STREAM_RECORD_TYPE:
            continue
        reference_index = (
            entry.group_entry_index
            if entry.group_entry_index is not None
            else entry.index
        )
        if reference_index == stream_index:
            candidates.append(entry)
    return candidates


def resolve_rpns_named_references(
    table: ParsedStreamTable,
    named_references: Sequence[Mapping[str, object]],
) -> tuple[dict[int, str], list[dict[str, object]], list[str]]:
    """Resolve only the RPNS selector/index form verified by supplied data."""

    names_by_global_index: dict[int, str] = {}
    resolved: list[dict[str, object]] = []
    warnings: list[str] = []

    for reference in named_references:
        row = dict(reference)
        stream_index = int(row["stream_index"])
        selector = int(row["selector"])
        name = str(row["name"])

        if selector != RPNS_VERIFIED_LOCAL_SELECTOR:
            row.update(
                {
                    "resolved": False,
                    "resolved_to_stored_audio": False,
                    "status": "unresolved selector; not mapped to this archive",
                    "name_applied_to_output": False,
                }
            )
            warnings.append(
                f"RPNS name {name!r} uses selector {selector}; only selector "
                f"{RPNS_VERIFIED_LOCAL_SELECTOR} has a verified mapping to this "
                "archive's movie-stream table, so no local stream was renamed"
            )
            resolved.append(row)
            continue

        candidates = _verified_local_movie_candidates(table, stream_index)
        if not candidates:
            row.update(
                {
                    "resolved": False,
                    "resolved_to_stored_audio": False,
                    "status": "no matching type-0x0F stream record",
                    "name_applied_to_output": False,
                }
            )
            warnings.append(
                f"RPNS name {name!r} selector {selector} index {stream_index} "
                "did not match a type-0x0F stream record"
            )
            resolved.append(row)
            continue

        if len(candidates) > 1:
            row.update(
                {
                    "resolved": False,
                    "resolved_to_stored_audio": False,
                    "status": "ambiguous type-0x0F stream record",
                    "candidate_global_stream_indices": [
                        entry.index for entry in candidates
                    ],
                    "name_applied_to_output": False,
                }
            )
            warnings.append(
                f"RPNS name {name!r} selector {selector} index {stream_index} "
                "matches multiple type-0x0F stream records"
            )
            resolved.append(row)
            continue

        candidate = candidates[0]
        placeholder = _is_zero_length_stream_placeholder(candidate)
        row.update(_entry_manifest_location(candidate))
        row.update(
            {
                "resolution_method": "verified-selector-255-type-0x0F-index",
                "resolution_confidence": "direct",
                "resolved": True,
                "resolved_to_stored_audio": not placeholder,
                "status": (
                    "zero-length placeholder"
                    if placeholder
                    else "stored SCG audio"
                ),
            }
        )

        if placeholder:
            row["name_applied_to_output"] = False
            resolved.append(row)
            continue

        previous = names_by_global_index.get(candidate.index)
        if previous is not None and previous.lower() != name.lower():
            row["name_applied_to_output"] = False
            warnings.append(
                f"stream record {candidate.index} has conflicting RPNS names "
                f"{previous!r} and {name!r}; keeping {previous!r}"
            )
        else:
            names_by_global_index.setdefault(candidate.index, name)
            row["name_applied_to_output"] = True

        resolved.append(row)

    return names_by_global_index, resolved, warnings


def parse_igc_audio_bindings(
    files: Mapping[Path, bytes],
    stream_names: Mapping[int, str],
) -> tuple[dict[str, str], list[str]]:
    """Recover ``#movie -> _BindAudio 'cue'`` mappings from IGC resources."""

    bindings: dict[str, str] = {}
    warnings: list[str] = []
    known_names = sorted(set(stream_names.values()), key=len, reverse=True)
    if not known_names:
        return bindings, warnings

    for path, data in files.items():
        if _logical_extension(path) != ".igc":
            continue

        markers: list[tuple[int, int, str]] = []
        for name in known_names:
            pattern = re.compile(rb"#" + re.escape(name.encode("ascii")), re.I)
            for match in pattern.finditer(data):
                markers.append((match.start(), match.end(), name))
        markers.sort()

        for marker_index, (start, end, name) in enumerate(markers):
            stop = min(len(data), start + 0x400)
            if marker_index + 1 < len(markers):
                stop = min(stop, markers[marker_index + 1][0])
            cue_match = BIND_AUDIO_RE.search(data, end, stop)
            if cue_match is None:
                continue

            cue = cue_match.group(1).decode("ascii", "replace").strip()
            key = name
            previous = bindings.get(key)
            if previous is not None and previous != cue:
                warnings.append(
                    f"IGC movie {name!r} binds both {previous!r} and {cue!r}; "
                    f"keeping {previous!r}"
                )
                continue
            bindings.setdefault(key, cue)

    return bindings, warnings


def stream_output_name(
    entry: StreamAudioEntry,
    stream_names: Mapping[int, str],
    bindings: Mapping[str, str],
    used_names: set[str],
) -> tuple[str, str | None, str | None]:
    movie_name = stream_names.get(entry.index)
    cue_name = bindings.get(movie_name) if movie_name else None

    if movie_name:
        components = [_safe_component(movie_name, f"STREAM_{entry.index:04d}")]
        if cue_name:
            components.append(_safe_component(cue_name, "Audio"))
        stem = "_".join(components)
    elif entry.group_index is not None and entry.group_entry_index is not None:
        record_type = entry.record_type if entry.record_type is not None else 0
        stem = (
            f"STREAM_G{entry.group_index:02d}_"
            f"I{entry.group_entry_index:04d}_T{record_type:02X}"
        )
    else:
        stem = f"STREAM_{entry.index:04d}"

    candidate = stem + ".flac"
    if candidate.lower() in used_names:
        candidate = f"{stem}_stream{entry.index:04d}.flac"
    used_names.add(candidate.lower())
    return candidate, movie_name, cue_name


# ---------------------------------------------------------------------------
# FLAC writing and extraction orchestration
# ---------------------------------------------------------------------------


def resolve_ffmpeg(name: str) -> str:
    """Return a usable FFmpeg executable that includes the FLAC encoder."""

    candidate = shutil.which(name)
    if candidate is None:
        raise FlacEncodeError(f"ffmpeg executable not found: {name!r}")

    try:
        probe = subprocess.run(
            [
                candidate,
                "-hide_banner",
                "-loglevel",
                "error",
                "-h",
                "encoder=flac",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise FlacEncodeError(
            f"unable to run ffmpeg executable {candidate!r}: {error}"
        ) from error

    if "Encoder flac " not in probe.stdout:
        detail = probe.stdout.strip() or "the encoder probe produced no output"
        raise FlacEncodeError(
            f"ffmpeg executable {candidate!r} does not expose the FLAC encoder: "
            f"{detail}"
        )
    return candidate


def _pcm_bytes_little_endian(pcm: array) -> bytes:
    if pcm.typecode != "h":
        raise TypeError("PCM array must use signed 16-bit typecode 'h'")
    if sys.byteorder == "little":
        return pcm.tobytes()
    copy = array("h", pcm)
    copy.byteswap()
    return copy.tobytes()


def write_flac(
    path: Path,
    pcm: array,
    sample_rate: int,
    channels: int,
    ffmpeg: str,
) -> None:
    """Encode interleaved native-endian PCM16 at FFmpeg's maximum FLAC level."""

    if channels <= 0:
        raise ValueError("channels must be positive")
    if len(pcm) % channels:
        raise ValueError(
            f"interleaved PCM length {len(pcm)} is not divisible by {channels}"
        )
    if not MIN_REASONABLE_SAMPLE_RATE <= sample_rate <= MAX_REASONABLE_SAMPLE_RATE:
        raise ValueError(
            f"sample rate {sample_rate} is outside "
            f"{MIN_REASONABLE_SAMPLE_RATE}..{MAX_REASONABLE_SAMPLE_RATE}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".part")
    partial_path.unlink(missing_ok=True)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-map",
        "0:a:0",
        "-c:a",
        "flac",
        "-compression_level",
        str(FLAC_COMPRESSION_LEVEL),
        "-exact_rice_parameters",
        "1" if FLAC_EXACT_RICE_PARAMETERS else "0",
        "-sample_fmt",
        "s16",
        *(["-bitexact"] if FLAC_BITEXACT else []),
        "-metadata_header_padding",
        str(FLAC_METADATA_HEADER_PADDING),
        "-f",
        "flac",
        "-y",
        str(partial_path),
    ]

    try:
        result = subprocess.run(
            command,
            input=_pcm_bytes_little_endian(pcm),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        partial_path.unlink(missing_ok=True)
        raise FlacEncodeError(
            f"unable to run ffmpeg while writing {path}: {error}"
        ) from error

    if result.returncode != 0:
        partial_path.unlink(missing_ok=True)
        stderr = result.stderr.decode("utf-8", "replace").strip()
        tail = "\n".join(stderr.splitlines()[-40:])
        raise FlacEncodeError(
            f"ffmpeg failed while writing {path} with exit status "
            f"{result.returncode}" + (f":\n{tail}" if tail else "")
        )

    if not partial_path.is_file() or partial_path.stat().st_size == 0:
        partial_path.unlink(missing_ok=True)
        raise FlacEncodeError(f"ffmpeg did not create a non-empty FLAC: {path}")

    partial_path.replace(path)


def _preferred_endian(archive: LoadedArchive) -> str | None:
    if archive.byte_order == "big":
        return ">"
    if archive.byte_order == "little":
        return "<"
    if archive.canonical_format == "scg":
        return ">"
    if archive.canonical_format in {"scw", "scx"}:
        return "<"
    return None


def extract_sample_pair(
    pair: ResourcePair,
    output_root: Path,
    forced_sample_rate: int | None,
    preferred_endian: str | None,
    ffmpeg: str,
) -> tuple[dict[str, object], list[str]]:
    parsed = parse_sample_shdr(
        pair.shdr_data,
        pair.samp_data,
        preferred_endian=preferred_endian,
    )
    rates, rate_mapping_source = choose_sample_rates(parsed, forced_sample_rate)

    warnings: list[str] = []
    if parsed.declared_samp_size not in (0, len(pair.samp_data)):
        warnings.append(
            f"{pair.shdr_path.as_posix()} declares a .samp size of "
            f"0x{parsed.declared_samp_size:X}, but {pair.samp_path.as_posix()} "
            f"is 0x{len(pair.samp_data):X} bytes"
        )

    bank_directory = (
        output_root
        / pair.samp_path.parent
        / _without_logical_extension(pair.samp_path)
    )
    sample_manifest: list[dict[str, object]] = []

    for entry, sample_rate in tqdm(zip(parsed.entries, rates)):
        decoded = decode_dsp_entry(pair.samp_data, entry)
        output_name = (
            f"sample_{entry.index:04d}_start{entry.start_address:08x}.flac"
        )
        output_path = bank_directory / output_name
        write_flac(
            output_path,
            decoded.pcm,
            sample_rate,
            channels=1,
            ffmpeg=ffmpeg,
        )

        sample_manifest.append(
            {
                "index": entry.index,
                "output_path": output_path.relative_to(output_root).as_posix(),
                "sample_rate": sample_rate,
                "channels": 1,
                "sample_format": "signed 16-bit PCM",
                "decoded_samples": decoded.sample_count,
                "duration_seconds": round(decoded.sample_count / sample_rate, 9),
                "start_nibble_address": entry.start_address,
                "start_nibble_address_hex": f"0x{entry.start_address:X}",
                "loop_nibble_address": entry.loop_address,
                "loop_nibble_address_hex": f"0x{entry.loop_address:X}",
                "end_nibble_address_inclusive": entry.end_address,
                "end_nibble_address_inclusive_hex": f"0x{entry.end_address:X}",
                "current_nibble_address": entry.current_address,
                "current_nibble_address_hex": f"0x{entry.current_address:X}",
                "loop_start_sample": decoded.loop_start_sample,
                "encoded_byte_offset": decoded.encoded_byte_offset,
                "encoded_byte_offset_hex": f"0x{decoded.encoded_byte_offset:X}",
                "encoded_byte_size": decoded.encoded_byte_size,
                "coefficients": list(entry.coefficients),
                "trailer_words": list(entry.trailer_words),
            }
        )

    return (
        {
            "samp_path": pair.samp_path.as_posix(),
            "shdr_path": pair.shdr_path.as_posix(),
            "samp_size": len(pair.samp_data),
            "declared_samp_size": parsed.declared_samp_size,
            "shdr_byte_order": parsed.byte_order,
            "entry_count": parsed.entry_count,
            "entry_table_offset": parsed.entry_table_offset,
            "entry_table_offset_hex": f"0x{parsed.entry_table_offset:X}",
            "entry_stride": parsed.entry_stride,
            "entry_stride_hex": f"0x{parsed.entry_stride:X}",
            "rate_values_found": list(parsed.rate_values),
            "rate_table_source": parsed.rate_source,
            "rate_mapping_source": rate_mapping_source,
            "samples": sample_manifest,
        },
        warnings,
    )


def _find_bulk_file(
    archive: LoadedArchive,
) -> tuple[Path, bytes]:
    candidates: list[tuple[Path, bytes]] = []
    for path, data in archive.files.items():
        if path.parts and path.parts[0].lower() == "bulk":
            candidates.append((path, data))

    if not candidates:
        raise StreamTableError(
            "NiemaFS did not expose a bulk file; streamed audio cannot be read"
        )
    if len(candidates) > 1:
        candidates.sort(key=lambda item: len(item[1]), reverse=True)
    path, data = candidates[0]

    if archive.bulk_size not in (None, 0, len(data)):
        raise StreamTableError(
            f"NiemaFS reports bulk_size=0x{archive.bulk_size:X}, but "
            f"{path.as_posix()} contains 0x{len(data):X} bytes"
        )
    return path, data


def extract_streamed_audio(
    archive: LoadedArchive,
    output_root: Path,
    forced_sample_rate: int | None,
    ffmpeg: str,
) -> tuple[dict[str, object], list[str], list[dict[str, str]]]:
    bulk_path, bulk_data = _find_bulk_file(archive)
    preferred_endian = _preferred_endian(archive)
    table = find_stream_table(
        archive.files,
        bulk_size=len(bulk_data),
        preferred_endian=preferred_endian,
    )

    named_references, warnings = parse_rpns_named_references(archive.files)
    stream_names, resolved_references, reference_warnings = (
        resolve_rpns_named_references(table, named_references)
    )
    warnings.extend(reference_warnings)
    bindings, binding_warnings = parse_igc_audio_bindings(
        archive.files, stream_names
    )
    warnings.extend(binding_warnings)

    references_by_global_index: defaultdict[int, list[dict[str, object]]] = (
        defaultdict(list)
    )
    for reference in resolved_references:
        if reference.get("resolved"):
            references_by_global_index[
                int(reference["resolved_global_stream_index"])
            ].append(reference)

    # The supplied and validated GameCube layout stores DSP coefficients in
    # big-endian order.  For a related little-endian archive, use its metadata
    # byte order but let predictor validation reject incompatible codecs.
    coefficient_endian = (
        ">" if archive.canonical_format == "scg" else table.endian
    )

    streamed_root = output_root / "streamed"
    used_names: set[str] = set()
    stream_manifest: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    ignored_streams: list[dict[str, object]] = []

    for entry in tqdm(table.entries):
        if entry.stored_size == 0:
            ignored: dict[str, object] = {
                "stream_index": entry.index,
                "stream_group": entry.group_index,
                "stream_index_in_group": entry.group_entry_index,
                "stream_record_type": entry.record_type,
                "reason": "zero-length placeholder",
            }
            references = references_by_global_index.get(entry.index, [])
            if references:
                ignored["rpns_references"] = references
            ignored_streams.append(ignored)
            continue

        region = bulk_data[
            entry.bulk_offset : entry.bulk_offset + entry.stored_size
        ]
        if not any(region):
            ignored_streams.append(
                {
                    "stream_index": entry.index,
                    "stream_group": entry.group_index,
                    "stream_index_in_group": entry.group_entry_index,
                    "stream_record_type": entry.record_type,
                    "reason": "entirely zero-filled placeholder",
                    "stored_size": entry.stored_size,
                }
            )
            continue

        sample_rate = forced_sample_rate or entry.sample_rate
        if not MIN_REASONABLE_SAMPLE_RATE <= sample_rate <= MAX_REASONABLE_SAMPLE_RATE:
            failures.append(
                {
                    "stream_index": str(entry.index),
                    "error": f"implausible sample rate {sample_rate}",
                }
            )
            continue

        try:
            (
                pcm,
                channel_sample_counts,
                trimmed_zero_frames,
                discarded_partial_bytes,
                padded_samples,
            ) = decode_streamed_dsp(
                bulk_data=bulk_data,
                entry=entry,
                coefficient_endian=coefficient_endian,
            )
        except DspDecodeError as error:
            # Keep a malformed or unsupported record local to this stream
            # instead of discarding the other valid entries in the table.
            failures.append(
                {
                    "stream_index": str(entry.index),
                    "error": str(error),
                }
            )
            continue

        output_name, movie_name, cue_name = stream_output_name(
            entry, stream_names, bindings, used_names
        )
        output_path = streamed_root / output_name
        try:
            write_flac(
                output_path,
                pcm,
                sample_rate,
                entry.channels,
                ffmpeg=ffmpeg,
            )
        except FlacEncodeError as error:
            failures.append(
                {
                    "stream_index": str(entry.index),
                    "error": str(error),
                }
            )
            continue

        output_frames = len(pcm) // entry.channels

        stream_manifest.append(
            {
                "stream_index": entry.index,
                "stream_group": entry.group_index,
                "stream_index_in_group": entry.group_entry_index,
                "stream_record_type": entry.record_type,
                "movie_name": movie_name,
                "bound_audio_cue": cue_name,
                "output_path": output_path.relative_to(output_root).as_posix(),
                "sample_rate": sample_rate,
                "channels": entry.channels,
                "sample_format": "signed 16-bit PCM",
                "output_frames": output_frames,
                "duration_seconds": round(output_frames / sample_rate, 9),
                "channel_sample_counts_before_padding": channel_sample_counts,
                "padded_samples": padded_samples,
                "bulk_offset": entry.bulk_offset,
                "bulk_offset_hex": f"0x{entry.bulk_offset:X}",
                "archive_offset": (
                    archive.bulk_offset + entry.bulk_offset
                    if archive.bulk_offset is not None
                    else None
                ),
                "archive_offset_hex": (
                    f"0x{archive.bulk_offset + entry.bulk_offset:X}"
                    if archive.bulk_offset is not None
                    else None
                ),
                "stored_size": entry.stored_size,
                "stored_size_hex": f"0x{entry.stored_size:X}",
                "metadata_block_count": entry.block_count,
                "metadata_field_10": entry.field_10,
                "metadata_field_10_hex": f"0x{entry.field_10:X}",
                "metadata_flags": entry.flags,
                "metadata_flags_hex": f"0x{entry.flags:X}",
                "trimmed_trailing_zero_frames_per_channel": trimmed_zero_frames,
                "discarded_partial_bytes_per_channel": discarded_partial_bytes,
                "rpns_references": references_by_global_index.get(entry.index, []),
            }
        )

    result: dict[str, object] = {
        "bulk_path": bulk_path.as_posix(),
        "bulk_size": len(bulk_data),
        "stream_table_shdr_path": table.source_path.as_posix(),
        "stream_table_layout": table.layout,
        "stream_table_byte_order": table.byte_order,
        "stream_table_section_offset": table.section_offset,
        "stream_table_section_offset_hex": f"0x{table.section_offset:X}",
        "stream_table_section_size": table.section_size,
        "stream_table_record_stride": table.record_stride,
        "stream_table_record_type": table.record_type,
        "stream_record_count": len(table.entries),
        "stream_group_offset_table_offset": table.group_offset_table_offset,
        "stream_group_data_offset": table.group_data_offset,
        "stream_group_data_size": table.group_data_size,
        "stream_group_end_offset": table.group_end_offset,
        "stream_groups": [
            {
                "group_index": group.index,
                "relative_offset": group.relative_offset,
                "relative_offset_hex": f"0x{group.relative_offset:X}",
                "data_offset": group.data_offset,
                "data_offset_hex": f"0x{group.data_offset:X}",
                "record_count": group.record_count,
                "record_type": group.record_type,
                "record_type_hex": f"0x{group.record_type:X}",
            }
            for group in table.groups
        ],
        "stream_name_map_index_space": "global parsed stream index",
        "stream_name_map": {
            str(index): name for index, name in sorted(stream_names.items())
        },
        "rpns_named_movie_references": resolved_references,
        "igc_audio_bindings": dict(sorted(bindings.items())),
        "extracted_stream_count": len(stream_manifest),
        "ignored_streams": ignored_streams,
        "streams": stream_manifest,
    }
    return result, warnings, failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract short .samp sounds and long streamed DSP audio from an "
            "SCG/SCW/SCX-family archive as lossless FLAC files."
        )
    )
    parser.add_argument(
        "archive",
        type=Path,
        help="input .scg/.scw/.scx archive (transposed .sgc/.sgw/.sgx also accepted)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output directory (default: <input-stem>_audio beside the archive)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        help=(
            "force a uniform rate for both sample-bank and streamed audio; "
            "normally the rate is read from .shdr metadata"
        ),
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable name or path (must include the FLAC encoder)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    target_path: Path = args.archive
    if not target_path.is_file():
        print(f"error: input file does not exist: {target_path}", file=sys.stderr)
        return 2

    try:
        ffmpeg = resolve_ffmpeg(args.ffmpeg)
    except AudioExtractionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    output_root: Path = args.output or target_path.with_name(
        target_path.stem + "_audio"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        archive = load_archive(target_path)
    except Exception as error:
        print(f"error: unable to read {target_path}: {error}", file=sys.stderr)
        return 1

    warnings: list[str] = []
    failures: list[dict[str, str]] = []
    sample_bank_manifests: list[dict[str, object]] = []

    pairs, pair_warnings = pair_sample_resources(archive.files)
    warnings.extend(pair_warnings)
    preferred_endian = _preferred_endian(archive)

    for pair in pairs:
        print(
            f"Extracting sample bank {pair.samp_path.as_posix()} using "
            f"{pair.shdr_path.as_posix()} ..."
        )
        try:
            bank_manifest, bank_warnings = extract_sample_pair(
                pair=pair,
                output_root=output_root,
                forced_sample_rate=args.sample_rate,
                preferred_endian=preferred_endian,
                ffmpeg=ffmpeg,
            )
        except AudioExtractionError as error:
            failures.append(
                {
                    "kind": "sample_bank",
                    "source": pair.samp_path.as_posix(),
                    "error": str(error),
                }
            )
            print(
                f"warning: skipped sample bank {pair.samp_path.as_posix()}: "
                f"{error}",
                file=sys.stderr,
            )
            continue

        sample_bank_manifests.append(bank_manifest)
        warnings.extend(bank_warnings)
        print(f"  wrote {bank_manifest['entry_count']} mono FLAC files")

    streamed_manifest: dict[str, object] | None = None
    if archive.canonical_format in {"scg", "scw"}:
        print("Extracting long streamed audio from the archive bulk area ...")
        try:
            streamed_manifest, stream_warnings, stream_failures = (
                extract_streamed_audio(
                    archive=archive,
                    output_root=output_root,
                    forced_sample_rate=args.sample_rate,
                    ffmpeg=ffmpeg,
                )
            )
            warnings.extend(stream_warnings)
            failures.extend(
                {
                    "kind": "streamed_audio",
                    "source": f"stream {failure['stream_index']}",
                    "error": failure["error"],
                }
                for failure in stream_failures
            )
            group_count = len(streamed_manifest["stream_groups"])
            group_text = (
                f", {group_count} groups"
                if streamed_manifest["stream_table_layout"] == "grouped"
                else ""
            )
            print(
                f"  detected {streamed_manifest['stream_table_layout']} "
                f"stream table ({streamed_manifest['stream_record_count']} "
                f"records{group_text})"
            )
            print(
                f"  wrote {streamed_manifest['extracted_stream_count']} "
                "streamed FLAC files"
            )
        except AudioExtractionError as error:
            warnings.append(f"streamed audio was not extracted: {error}")
            print(
                f"warning: streamed audio was not extracted: {error}",
                file=sys.stderr,
            )

    short_audio_count = sum(
        int(bank["entry_count"]) for bank in sample_bank_manifests
    )
    streamed_audio_count = (
        int(streamed_manifest["extracted_stream_count"])
        if streamed_manifest is not None
        else 0
    )
    total_audio_count = short_audio_count + streamed_audio_count

    manifest = {
        "tool": "extract_audio.py",
        "source_archive": str(target_path),
        "filesystem_class": archive.fs_class_name,
        "archive_format": archive.archive_format,
        "archive_variant": archive.archive_variant,
        "archive_byte_order": archive.byte_order,
        "ffmpeg_executable": ffmpeg,
        "output_representation": "FLAC (lossless) from signed 16-bit PCM",
        "flac_compression_level": FLAC_COMPRESSION_LEVEL,
        "flac_exact_rice_parameters": FLAC_EXACT_RICE_PARAMETERS,
        "ffmpeg_bitexact": FLAC_BITEXACT,
        "flac_metadata_header_padding": FLAC_METADATA_HEADER_PADDING,
        "audio_file_count": total_audio_count,
        "sample_bank_audio_file_count": short_audio_count,
        "streamed_audio_file_count": streamed_audio_count,
        "sample_banks": sample_bank_manifests,
        "streamed_audio": streamed_manifest,
        "warnings": warnings,
        "failures": failures,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if total_audio_count == 0:
        print(f"No audio was extracted. Manifest: {manifest_path}", file=sys.stderr)
        return 1

    print(
        f"Extracted {total_audio_count} FLAC files "
        f"({short_audio_count} sample-bank, {streamed_audio_count} streamed)."
    )
    print(f"Output:   {output_root}")
    print(f"Manifest: {manifest_path}")
    if failures:
        print(
            f"Completed with {len(failures)} skipped item(s); see manifest.json.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
