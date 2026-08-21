# Factor 5 VID1 Preservation
On the Nintendo GameCube, some video files were stored in the [Factor 5 VID1](https://wiki.multimedia.cx/index.php/Factor_5_VID1) format, which constituted a `.vid` file in the [Factor 5 VID1 ***container*** format](https://wiki.multimedia.cx/index.php/Factor_5_VID1#Container_format) that contained a video stream in the [Factor 5 VID1 ***video codec***](https://wiki.multimedia.cx/index.php/Factor_5_VID1#Video_codec). The ***container*** format has already been [implemented into `librempeg`](https://github.com/librempeg/librempeg/blob/master/libavformat/vid1.c), meaning compiling `librempeg` (and thus `ffmpeg`) from the latest source enables support for it. The ***video codec***, however, has not been. The Factor 5 VID1 ***video codec*** is [very similar to MPEG-4 ASP](https://wiki.multimedia.cx/index.php/Factor_5_VID1#Video_codec). This repository documents my efforts in trying to preserve the contents of these VID1 files for archival purposes.

# TL;DR Summary
This README will document all of my exploration, but here is a brief summary of what I did in the end:

1. Use [`librempeg`](https://github.com/librempeg/librempeg)'s `ffmpeg` implementation (which supports the VID1 *container*) to decode all audio streams and re-encode them as lossless-compressed FLAC files
    * Most `.vid` files have exactly 1 audio stream, some have no audio, and some have multiple audio streams (e.g. different languages) 
2. Use [`vid1_decode.py`](vid1_decode.py) to decode the VID1 *video* stream and re-encode it as lossless-compressed FFV1 in an MKV container
3. For *The Lord of the Rings: The Return of the King* and *The Lord of the Rings: The Third Age*, use [`extract_sc.py`](extract_sc.py) to extract all audio from the `.scg` files, and manually match them to their corresponding `.vid` files
    * Some audio will be extracted with the same name as the corresponding `.vid` file (just with the `.flac` extension)
    * However, some are ambiguously named and are longer than their corresponding `.vid` file and need to be trimmed (e.g. using `ffmpeg`)
    * This matching takes a lot of manual investigative work
4. Use [MKVToolNix](https://mkvtoolnix.download/) to remux the video and audio streams of each `.vid` file into an MKV file
    * Not strictly necessary, but when remuxing with MKVToolNix, I also set the correct language for each video and audio stream

# Compiling the Latest `librempeg` from Source
The latest `librempeg` (and thus `ffmpeg`) source supports the VID1 ***container***. We won't be using `ffmpeg` to extract the video stream, but we *will* be using `ffmpeg` to extract the audio stream. Thus, we first need to compile `ffmpeg` from the latest source:

```bash
git clone https://github.com/librempeg/librempeg.git
cd librempeg
./configure --prefix="$PWD/build" --enable-agpl --disable-debug
make -j8
make install
```

There might be other dependencies that you need to install as well. You can Google your compile error messages as needed.

# Extracting Audio
Fortunately, [`librempeg`](https://github.com/librempeg/librempeg)'s `ffmpeg` implementation supports the VID1 *container*, and the audio in `.vid` files seems to be in standard formats supported by `ffmpeg`, so extracting audio from a `.vid` file is simple:

```bash
ffmpeg -i original.vid -vn -c:a flac -compression_level 12 audio.flac
```

Note that the above command is decoding the audio in the original file and is then *re-encoding* it (rather than copying it) using the FLAC codec. I decided to do this instead of *copying* the original audio stream because there is some variation in what audio codecs are used in different `.vid` files, and I wanted my final results to all be consistent. FLAC is a lossless-compressed audio codec, so the resulting files are larger than if I had directly copied the audio stream, but they're completely identical in content (i.e., no loss in quality).

To recursively run this on all `.vid` files nested within the current directory (outputting the different audio streams into different output files if the `.vid` file has multiple):

```bash
find . -type f -name '*.vid' -exec bash -c '
  tools="$HOME/VID1-Preservation/librempeg"
  ffmpeg="$tools/ffmpeg"
  ffprobe="$tools/ffprobe"

  for input in "$@"; do
    output_args=()
    audio_number=0

    while IFS= read -r stream_index; do
      ((audio_number += 1))

      output_args+=(
        -map "0:${stream_index}"
        -vn
        -c:a flac
        -compression_level 12
        "${input}.audio.${audio_number}.flac"
      )
    done < <(
      "$ffprobe" -v error \
        -select_streams a \
        -show_entries stream=index \
        -of csv=p=0 \
        "$input"
    )

    if ((audio_number > 0)); then
      "$ffmpeg" -i "$input" "${output_args[@]}"
    fi
  done
' _ {} +
```

## *The Lord of the Rings*
For *The Lord of the Rings: The Return of the King* and *The Lord of the Rings: The Third Age*, the `.vid` files do not contain any audio, but the videos *do* have audio when played in the game. The audio exists elsewhere on the disc, within proprietary `.scg` archives. The reason for this is that most cutscenes in these games mix FMVs (stored in `.vid` files) with in-engine rendering, so the audio of a given cutscene can be *longer* than the `.vid` file. Thus, to remux these videos with their correct audio, the audio segment corresponding to the `.vid` file may need to be cut out of the overall cutscene's audio (e.g. using `ffmpeg`).

First, use [`extract_sc.py`](extract_sc.py) to extract the contents of the corresponding `.scg` file:

```bash
python3 extract_sc.py archive.scg
```

To recursively run this on all `.scg` files nested within the current directory (where `OUT_DIR` is the desired output directory):

```bash
find . -type f -name '*.scg' -exec python3 extract_sc.py -o OUT_DIR/{} {} \;
```

Some audio (e.g. for cutscenes that are *just* `.vid` FMV) will be extracted with the same name as the corresponding `.vid` file (just with the `.flac` extension), but some are ambiguously named. Unfortunately, many of the audio streams in a `.scg` file are not identified in any meaningful way (at least not by the extraction script), and many of the audio streams are irrelevant (e.g. in-game sound effects).

As a result, matching a `.vid` file to its corresponding `.flac` file takes some manual work. My approach was to start with `.flac` files that had a name matching a `.vid` file (double-checking that the duration of the `.flac` file is less than or equal to the duration of the `.vid` file). Then, for all remaining `.vid` files, I went through the `.flac` files with a duration greater than or equal to it (or maybe slightly shorter) in search of one that begins with the audio that matches the video (using YouTube longplay videos of the game to find the cutscene and hear the correct audio). The relevant FLACs seem to all be in the `streamed` sub-folders.

If a `.flac` file matched the *start* of the `.vid` file but had audio *after the end* of the `.vid` file (e.g. a cutscene that transitions from `.vid` FMV to in-engine), I used the following command to extract the **beginning** of the `.flac` file:

```bash
ffmpeg -i audio.flac -map 0:a:0 -t "$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 video.mkv)" -c:a flac -compression_level 12 audio.trimmed.flac
```

If a a `.flac` file matched the *end* of the `.vid` fil but had audio *before the start* of the `.vid` file (e.g. a cutscene that transitions from in-engine to `.vid` FMV), I used the following command to extract the **end** of the `.flac` file:

```bash
d="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 video.mkv)" && ffmpeg -sseof "-$d" -i audio.flac -map 0:a:0 -t "$d" -c:a flac -compression_level 12 audio.trimmed.flac
```

However, the `.vid` file might not be *exactly* at the beginning or end of the `.flac` file, in which case the timing needs to be calibrated manually. Once the desired start and end times of the `.flac` file are found, the following command will extract that exact segment:

```bash
ffmpeg -i audio.flac -map 0:a:0 -ss START_TIME -to END_TIME -c:a flac -compression_level 12 audio.trimmed.flac
```

### YouTube Longplay Matches
For the `.vid` files that did *not* have an extracted `.flac` with a matching name, here are the YouTube longplay matches to know what audio should be playing with it:

#### *The Lord of the Rings: The Return of the King* (GKLE69)
I used this YouTube playlist:

https://www.youtube.com/playlist?list=PLGGFX-hnXvGpQXRrbGgdYpQSXLwsojndG

* **`crakfmv1.vid`:** https://youtu.be/RS9HYaSOp5Y?t=20
    * `Data/Game/cra/cra01.scg/streamed/STREAM_G02_I0002_T0F.flac`
* **`endfmv1.vid`:** https://youtu.be/RS9HYaSOp5Y?t=261
    * `Data/Game/cra/cra01.scg/streamed/STREAM_G02_I0000_T0F.flac`
* **`gatefmv1.vid`:** https://youtu.be/N0XYUlxVmWM?t=19
* **`gatefmv2.vid`:** https://youtu.be/N0XYUlxVmWM?t=656
* **`helmfmv1.vid`:** https://youtu.be/Uode2xTpBsU?t=0
* **`helmfmv2.vid`:** https://youtu.be/Uode2xTpBsU?t=382
* **`isenfmv1.vid`:** https://youtu.be/Uode2xTpBsU?t=715
* **`isenfmv2.vid`:** https://youtu.be/Uode2xTpBsU?t=1306
* **`mtwlfmv1.vid`:** https://youtu.be/BKy6IZ8HuCE?t=21
* **`mtydfmv2.vid`:** https://youtu.be/BKy6IZ8HuCE?t=1514
* **`osgifmv1.vid`:** https://youtu.be/5IjEfHZxcSM?t=21
* **`osgifmv2.vid`:** https://youtu.be/5IjEfHZxcSM?t=829
* **`pat1fmv1.vid`:** https://youtu.be/JtgPV8jRLLg?t=29
* **`pat1fmv2.vid`:** https://youtu.be/JtgPV8jRLLg?t=866
* **`pat2fmv1.vid`:** https://youtu.be/JtgPV8jRLLg?t=1079
* **`pelefmv1.vid`:** https://youtu.be/Fe0SuecJ7QQ?t=21
* **`shelfmv1.vid`:** https://youtu.be/Z_a3P4TS4lk?t=19

#### *The Lord of the Rings: The Third Age* (TODO)
TODO

# Translating VID1 Video into M4V (deprecated)
Because the VID1 *video codec* is very similar to MPEG-4 ASP, my original idea was to *translate* the VID1 video stream into MPEG-4 ASP as an M4V file using [`vid1_to_m4v.py`](vid1_to_m4v.py):

```bash
python3 vid1_to_m4v.py -i original.vid -o translated.m4v
```

This converts the VID1 video stream of `original.vid` into an MPEG-4 ASP video stream and save the result into `translated.m4v`. It **does *not*** re-encode the video stream: it directly translates it.

However, this was *extremely* fragile and only worked on a handful of videos from a handful of discs, so I didn't ultimately go this route in the final result. Instead, I attempted to properly *decode* the VID1 video streams (see below).

# Decoding VID1 Video
The [`vid1_decode.py`](vid1_decode.py) script decodes the VID1 *video* stream within a `.vid` VID1 *container* and re-encodes it into a lossless-compressed FFV1 codec in an MKV file:

```bash
python3 vid1_decode.py original.vid video.mkv
```

To recursively run it on all `.vid` files nested within the current directory (automatically reattempting failed runs with `--resync-scan`):

```bash
find . -type f -name '*.vid' -exec sh -c '"$HOME/VID1-Preservation/vid1_decode.py" "$1" "$1.video.mkv" || "$HOME/VID1-Preservation/vid1_decode.py" --resync-scan "$1" "$1.video.mkv"' sh {} \;
```

# Remuxing Video and Audio
The above

# Acknowledgements
* [Paul B Mahol's VID1 container demuxer code](https://github.com/librempeg/librempeg/blob/f3b1734d25ad2baf023af729cca7a0e30427d8a1/libavformat/vid1.c)
* [MultimediaWiki's VID1 video codec description](https://wiki.multimedia.cx/index.php/Factor_5_VID1#Video_codec)
* Much of the code to decode a VID1 video stream and to extract an SCG file was written by ChatGPT
    * All extracted video and audio files were manually verified by me before and after remuxing
