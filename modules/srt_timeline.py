from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import soundfile as sf


@dataclass
class TimelineClip:
    subtitle_index: int
    file_name: str
    file_path: str
    start_seconds: float
    end_seconds: float
    audio_duration_seconds: float
    track_index: int = 1
    track_group_index: int = 0


def _seconds_to_frames(seconds: float, fps: int) -> int:
    return max(0, int(round(float(seconds) * int(fps))))


def _seconds_to_ppro_ticks(seconds: float) -> int:
    return int(round(float(seconds) * 254016000000.0))


def _get_audio_file_info(audio_path: str) -> tuple[int, int, int]:
    info = sf.info(audio_path)
    if info.samplerate <= 0:
        raise ValueError(f"Invalid sample rate for audio file: {audio_path}")
    if info.channels <= 0:
        raise ValueError(f"Invalid channel count for audio file: {audio_path}")
    return int(info.frames), int(info.samplerate), int(info.channels)


def _assign_timeline_track_groups(clips: list[TimelineClip]) -> int:
    track_end_times: list[float] = []

    for clip in sorted(clips, key=lambda item: (item.start_seconds, item.subtitle_index, item.file_name)):
        assigned_track = None
        for idx, track_end in enumerate(track_end_times):
            if clip.start_seconds >= track_end:
                assigned_track = idx
                break

        if assigned_track is None:
            track_end_times.append(clip.end_seconds)
            clip.track_group_index = len(track_end_times) - 1
        else:
            track_end_times[assigned_track] = max(track_end_times[assigned_track], clip.end_seconds)
            clip.track_group_index = assigned_track

        clip.track_index = clip.track_group_index + 1

    return len(track_end_times)


def _manifest_to_clips(manifest: list[dict], job_dir: Path) -> list[TimelineClip]:
    clips: list[TimelineClip] = []
    for item in manifest:
        if item.get("status") != "ok":
            continue
        output_file = str(item.get("output_file", "") or "")
        if not output_file:
            continue
        file_path = job_dir / output_file
        if not file_path.exists() or not file_path.is_file():
            continue

        start_seconds = float(item.get("timeline_start_seconds", item.get("start_seconds", 0.0)))
        audio_duration_seconds = float(item.get("audio_duration_seconds", 0.0))
        if audio_duration_seconds <= 0:
            frames, sample_rate, _ = _get_audio_file_info(str(file_path))
            audio_duration_seconds = float(frames) / float(sample_rate)
        end_seconds = start_seconds + audio_duration_seconds

        clips.append(
            TimelineClip(
                subtitle_index=int(item.get("subtitle_index", len(clips) + 1)),
                file_name=Path(output_file).name,
                file_path=str(file_path),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                audio_duration_seconds=audio_duration_seconds,
            )
        )
    return clips


def build_premiere_xml(job_name: str, manifest: list[dict], job_dir: Path, xml_path: Path, fps: int = 30) -> Path | None:
    clips = _manifest_to_clips(manifest, job_dir)
    if not clips:
        return None

    timeline_fps = int(fps) if int(fps) > 0 else 30
    track_group_count = _assign_timeline_track_groups(clips)
    sequence_duration_frames = max(_seconds_to_frames(clip.end_seconds, timeline_fps) for clip in clips)

    sequence_sample_rate = 48000
    for clip in clips:
        _, sample_rate, _ = _get_audio_file_info(clip.file_path)
        sequence_sample_rate = sample_rate
        break

    tracks: list[list[TimelineClip]] = [[] for _ in range(track_group_count)]
    for clip in clips:
        tracks[clip.track_group_index].append(clip)

    audio_track_blocks: list[str] = []
    clip_item_counter = 1
    file_counter = 1

    for track_clips in tracks:
        clip_blocks: list[str] = []
        for clip in sorted(track_clips, key=lambda item: (item.start_seconds, item.subtitle_index, item.file_name)):
            start_frame = _seconds_to_frames(clip.start_seconds, timeline_fps)
            duration_frames = max(1, _seconds_to_frames(clip.audio_duration_seconds, timeline_fps))
            end_frame = start_frame + duration_frames
            frames, file_sample_rate, file_channels = _get_audio_file_info(clip.file_path)
            file_duration_frames = max(1, _seconds_to_frames(frames / file_sample_rate, timeline_fps))
            pathurl = Path(clip.file_path).resolve().as_uri()
            clip_id = clip_item_counter
            clip_item_counter += 1
            file_id = f"file-{file_counter}"
            file_counter += 1
            ppro_ticks_out = _seconds_to_ppro_ticks(clip.audio_duration_seconds)

            full_file_block = """
                        <file id=\"{file_id}\">
                            <name>{name}</name>
                            <pathurl>{pathurl}</pathurl>
                            <rate>
                                <timebase>{fps}</timebase>
                                <ntsc>FALSE</ntsc>
                            </rate>
                            <duration>{duration_frames}</duration>
                            <timecode>
                                <rate>
                                    <timebase>{fps}</timebase>
                                    <ntsc>FALSE</ntsc>
                                </rate>
                                <string>00:00:00:00</string>
                                <frame>0</frame>
                                <displayformat>NDF</displayformat>
                            </timecode>
                            <media>
                                <audio>
                                    <samplecharacteristics>
                                        <depth>16</depth>
                                        <samplerate>{samplerate}</samplerate>
                                    </samplecharacteristics>
                                    <channelcount>{channelcount}</channelcount>
                                </audio>
                            </media>
                        </file>""".format(
                file_id=file_id,
                name=escape(clip.file_name),
                pathurl=escape(pathurl),
                fps=timeline_fps,
                duration_frames=file_duration_frames,
                samplerate=file_sample_rate,
                channelcount=file_channels,
            )

            clip_blocks.append(
                """
                    <clipitem id=\"clipitem-{clip_id}\">
                        <name>{name}</name>
                        <enabled>TRUE</enabled>
                        <duration>{duration}</duration>
                        <rate>
                            <timebase>{fps}</timebase>
                            <ntsc>FALSE</ntsc>
                        </rate>
                        <start>{start}</start>
                        <end>{end}</end>
                        <in>0</in>
                        <out>{out}</out>
                        <pproTicksIn>0</pproTicksIn>
                        <pproTicksOut>{ppro_ticks_out}</pproTicksOut>
{file_block}
                        <sourcetrack>
                            <mediatype>audio</mediatype>
                            <trackindex>1</trackindex>
                        </sourcetrack>
                    </clipitem>""".format(
                    clip_id=clip_id,
                    name=escape(clip.file_name),
                    duration=duration_frames,
                    fps=timeline_fps,
                    start=start_frame,
                    end=end_frame,
                    out=duration_frames,
                    ppro_ticks_out=ppro_ticks_out,
                    file_block=full_file_block,
                )
            )

        audio_track_blocks.append(
            """
                <track>
{clips}
                    <enabled>TRUE</enabled>
                    <locked>FALSE</locked>
                    <outputchannelindex>1</outputchannelindex>
                </track>""".format(clips="\n".join(clip_blocks))
        )

    xml_text = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE xmeml>
<xmeml version=\"4\">
    <sequence id=\"sequence-1\">
        <uuid>{sequence_uuid}</uuid>
        <duration>{duration}</duration>
        <rate>
            <timebase>{fps}</timebase>
            <ntsc>FALSE</ntsc>
        </rate>
        <name>{sequence_name}</name>
        <media>
            <video>
                <track>
                    <enabled>TRUE</enabled>
                    <locked>FALSE</locked>
                </track>
            </video>
            <audio>
                <format>
                    <samplecharacteristics>
                        <depth>16</depth>
                        <samplerate>{sequence_sample_rate}</samplerate>
                    </samplecharacteristics>
                </format>
{audio_tracks}
            </audio>
        </media>
        <timecode>
            <rate>
                <timebase>{fps}</timebase>
                <ntsc>FALSE</ntsc>
            </rate>
            <string>00:00:00:00</string>
            <frame>0</frame>
            <displayformat>NDF</displayformat>
        </timecode>
    </sequence>
</xmeml>
""".format(
        sequence_uuid=f"premiere-{job_name}",
        sequence_name=escape(job_name),
        duration=max(1, sequence_duration_frames),
        fps=timeline_fps,
        sequence_sample_rate=sequence_sample_rate,
        audio_tracks="\n".join(audio_track_blocks),
    )

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(xml_text, encoding="utf-8")
    return xml_path
