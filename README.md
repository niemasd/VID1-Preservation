# Factor 5 VID1 Archiving
On the Nintendo GameCube, some video files were stored in the [Factor 5 VID1](https://wiki.multimedia.cx/index.php/Factor_5_VID1) format, which constituted a `.vid` file in the [Factor 5 VID1 ***container*** format](https://wiki.multimedia.cx/index.php/Factor_5_VID1#Container_format) that contained a video stream in the [Factor 5 VID1 ***video codec***](https://wiki.multimedia.cx/index.php/Factor_5_VID1#Video_codec). The ***container*** format has already been [implemented into `librempeg`](https://github.com/librempeg/librempeg/blob/master/libavformat/vid1.c), meaning compiling `librempeg` (and thus `ffmpeg`) from the latest source enables support for it. The ***video codec***, however, has not been.

The Factor 5 VID1 ***video codec*** is [very similar to MPEG-4 ASP](https://wiki.multimedia.cx/index.php/Factor_5_VID1#Video_codec). Based on these similarities, [`vid1_to_m4v.py`](vid1_to_m4v.py) takes as input a VID1 `.vid` file, and it ***translates*** (no re-encoding!) the video stream into a standard MPEG-4 Visual Elementary Stream `.m4v` file. You can then compile `ffmpeg` from the latest source code to remux the translated video stream from the `.m4v` file and the original audio stream from the `.vid` file into a desired output container (e.g. `.mkv`).

This seems to work for a good number of VID1 files, but there are many files this approach is incompatible with. To truly archive these videos, we need a proper VID1 video codec decoder (rather than my translator) so we can decode VID1 video streams into a lossless format (e.g. [FFV1](https://en.wikipedia.org/wiki/FFV1)).

## Usage: VID1 Decoder
Basic usage of the [`vid1_decode.py`](vid1_decode.py) is as follows (and full usage details can be viewed using the `-h/--help` argument):

```bash
python3 vid1_decode.py original.vid decoded.mkv
```

This will decode the video stream of the input VID1 file `original.vid` and re-encode it to the lossless-compressed FFV1 video codec into the output file `decoded.mkv`.

## Usage: VID1 to M4V Translator
Basic usage of the [`vid1_to_m4v.py`](vid1_to_m4v.py) is as follows (and full usage details can be viewed using the `-h/--help` argument):

```bash
python3 vid1_to_m4v.py -i original.vid -o translated.m4v
```

This will convert the video stream of the input VID1 file `original.vid` into the output file `translated.m4v`. It will ***not*** copy the audio stream into the output file: just the translated video stream.

# Full Pipeline
The intended full pipeline to go from a GameCube VID1 file to a playable MKV file is as follows.

## Compile Latest `ffmpeg` from Source
The latest `librempeg` (and thus `ffmpeg`) source supports the VID1 ***container***. We won't be using `ffmpeg` to extract the video stream, but we *will* be using `ffmpeg` to extract the audio stream. Thus, we first need to compile `ffmpeg` from the latest source:

```bash
git clone https://github.com/librempeg/librempeg.git
cd librempeg
./configure --prefix="$PWD/build" --enable-agpl --disable-debug
make -j8
make install
```

There might be other dependencies that you need to install as well. You can Google your compile error messages as needed.

## Decode VID1 Video Stream to FFV1 MKV
We will first use the [`vid1_decode.py`](vid1_decode.py) script to decode the VID1 video stream and re-encode it into an FFV1 (lossless-compressed) MKV file:

```bash
python3 vid1_decode.py original.vid video.mkv
```

To recursively run it on all `.vid` files nested within the current directory (automatically reattempting failed runs with `--resync-scan`):

```bash
find . -type f -name '*.vid' -exec sh -c '"$HOME/VID1-Video-Translator/vid1_decode.py" "$1" "$1.video.mkv" || "$HOME/VID1-Video-Translator/vid1_decode.py" --resync-scan "$1" "$1.video.mkv"' sh {} \;
```

## Extract Audio Stream to FLAC
We will next use librempeg's `ffmpeg` to convert the VID1 file's audio stream to a lossless-compressed FLAC file:

```bash
ffmpeg -i original.vid -c:a flac audio.flac
```

This isn't strictly necessary (we can directly copy the original audio stream), but not all `.vid` files have audio streams: some (such as those in *The Lord of the Rings: The Return of the King*) only have a video stream, and the corresponding audio is elsewhere on the disc. By converting all audio to FLAC, we maintain consistency (every final file will have FFV1 video and FLAC audio), and we enable ourselves to fill in missing audio files manually on a disc-by-disc basis if needed.

To recursively run it on all `.vid` files nested within the current directory:

```bash
find . -type f -name '*.vid' -exec "$HOME/VID1-Video-Translator/librempeg/ffmpeg" "$1" -vn -c:a flac "$1.audio.flac" \;
```

## Remux Translated Video Stream and FLAC Audio Stream
Next, we will use the source-compiled `ffmpeg` to remux the translated video stream with the original audio stream.

```bash
./build/bin/ffmpeg -i video.mkv -i audio.flac -c copy -map 0:v:0 -map 1:a:0 remuxed.mkv
```

This does ***not*** re-encode the video or audio stream: it's directly remuxing them into an MKV container.

# Acknowledgements
* [Paul B Mahol's VID1 container demuxer code](https://github.com/librempeg/librempeg/blob/f3b1734d25ad2baf023af729cca7a0e30427d8a1/libavformat/vid1.c)
* [MultimediaWiki's VID1 video codec description](https://wiki.multimedia.cx/index.php/Factor_5_VID1#Video_codec)
