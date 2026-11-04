"""Convert sixteenth-note SATB CSV files to MIDI and a two-staff MusicXML score.

Uses only Python's standard library. Run with --help for options.
"""

import argparse
import csv
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


VOICES = ("Soprano", "Alto", "Tenor", "Bass")
TICKS_PER_QUARTER = 480


def read_chorale(path):
    """Read either repository CSV format, preserving all rest rows."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        for line, row in enumerate(csv.reader(source), 1):
            if not row or all(not value.strip() for value in row):
                continue
            fields = [value.strip() for value in row]
            if not rows and (fields == ["note0", "note1", "note2", "note3"]
                             or (len(fields) == 2 and fields[1] in ("major", "minor"))):
                continue
            if len(fields) != 4:
                raise ValueError(f"{path}:{line}: expected four SATB pitches")
            try:
                pitches = tuple(int(value) for value in fields)
            except ValueError as error:
                raise ValueError(f"{path}:{line}: pitches must be integers") from error
            if any(pitch < 0 or pitch > 127 for pitch in pitches):
                raise ValueError(f"{path}:{line}: pitches must be 0 (rest) or 1–127")
            rows.append(pitches)
    if not rows:
        raise ValueError(f"{path}: no note rows found")
    return rows


def voice_runs(rows, voice):
    """Yield (pitch, start, length) in sixteenth-note units, including rests."""
    start = 0
    for end in range(1, len(rows) + 1):
        if end == len(rows) or rows[end][voice] != rows[start][voice]:
            yield rows[start][voice], start, end - start
            start = end


def variable_length(value):
    result = [value & 127]
    while value >> 7:
        value >>= 7
        result.insert(0, (value & 127) | 128)
    return bytes(result)


def midi_track(events, end_tick):
    data = bytearray()
    previous = 0
    for tick, message in sorted(events, key=lambda item: item[0]):
        data.extend(variable_length(tick - previous))
        data.extend(message)
        previous = tick
    data.extend(variable_length(end_tick - previous))
    data.extend(b"\xff\x2f\x00")
    return b"MTrk" + struct.pack(">I", len(data)) + data


def write_midi(rows, path, bpm=100, velocity=90, meter=(4, 4)):
    """Write type-1 MIDI: conductor track plus four independent voice tracks."""
    tempo = round(60_000_000 / bpm)
    end_tick = len(rows) * (TICKS_PER_QUARTER // 4)
    numerator, denominator = meter
    conductor = [(0, b"\xff\x51\x03" + tempo.to_bytes(3, "big")),
                 (0, bytes([255, 88, 4, numerator, denominator.bit_length() - 1, 24, 8]))]
    tracks = [midi_track(conductor, end_tick)]
    for voice, name in enumerate(VOICES):
        encoded = name.encode("ascii")
        events = [(0, b"\xff\x03" + variable_length(len(encoded)) + encoded),
                  (0, bytes([192 + voice, 0]))]  # General MIDI acoustic grand piano
        for pitch, start, length in voice_runs(rows, voice):
            if pitch:
                events.append((start * 120, bytes([144 + voice, pitch, velocity])))
                events.append(((start + length) * 120, bytes([128 + voice, pitch, 0])))
        tracks.append(midi_track(events, end_tick))
    Path(path).write_bytes(b"MThd" + struct.pack(">IHHH", 6, 1, 5, TICKS_PER_QUARTER)
                           + b"".join(tracks))


def element(parent, tag, text=None, **attributes):
    node = ET.SubElement(parent, tag, attributes)
    if text is not None:
        node.text = str(text)
    return node


# Duration, MusicXML type, dot count (units are sixteenth notes).
NOTE_LENGTHS = ((16, "whole", 0), (12, "half", 1), (8, "half", 0),
                (6, "quarter", 1), (4, "quarter", 0), (3, "eighth", 1),
                (2, "eighth", 0), (1, "16th", 0))
PITCH_NAMES = (("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
               ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0))


def write_musicxml(rows, path, title, bpm=100, meter=(4, 4)):
    """Write one grand-staff part with explicit SATB voices, stems, and ties."""
    score = ET.Element("score-partwise", version="4.0")
    element(element(score, "work"), "work-title", title)
    part_list = element(score, "part-list")
    element(element(part_list, "score-part", id="P1"), "part-name", "SATB")
    part = element(score, "part", id="P1")
    measure_length = meter[0] * 16 // meter[1]
    runs = [list(voice_runs(rows, voice)) for voice in range(4)]
    for number, start in enumerate(range(0, len(rows), measure_length), 1):
        length = min(measure_length, len(rows) - start)
        measure = element(part, "measure", number=str(number))
        if number == 1:
            attributes = element(measure, "attributes")
            element(attributes, "divisions", 4)
            time = element(attributes, "time")
            element(time, "beats", meter[0])
            element(time, "beat-type", meter[1])
            element(attributes, "staves", 2)
            for staff, sign, line in ((1, "G", 2), (2, "F", 4)):
                clef = element(attributes, "clef", number=str(staff))
                element(clef, "sign", sign)
                element(clef, "line", line)
            direction = element(measure, "direction", placement="above")
            metronome = element(element(direction, "direction-type"), "metronome")
            element(metronome, "beat-unit", "quarter")
            element(metronome, "per-minute", bpm)
            element(direction, "sound", tempo=str(bpm))
        for voice in range(4):
            if voice:
                element(element(measure, "backup"), "duration", length)
            for pitch, run_start, run_length in runs[voice]:
                position = max(start, run_start)
                stop = min(start + length, run_start + run_length)
                while position < stop:
                    duration, note_type, dots = next(
                        item for item in NOTE_LENGTHS if item[0] <= stop - position)
                    note = element(measure, "note")
                    if pitch == 0:
                        element(note, "rest")
                    else:
                        step, alter = PITCH_NAMES[pitch % 12]
                        pitch_node = element(note, "pitch")
                        element(pitch_node, "step", step)
                        if alter:
                            element(pitch_node, "alter", alter)
                        element(pitch_node, "octave", pitch // 12 - 1)
                    element(note, "duration", duration)
                    ties = []
                    if pitch:
                        if position > run_start:
                            ties.append("stop")
                        if position + duration < run_start + run_length:
                            ties.append("start")
                    for tie in ties:
                        element(note, "tie", type=tie)
                    element(note, "voice", voice + 1)
                    element(note, "type", note_type)
                    for _ in range(dots):
                        element(note, "dot")
                    if pitch:
                        element(note, "stem", "up" if voice % 2 == 0 else "down")
                    element(note, "staff", 1 if voice < 2 else 2)
                    if ties:
                        notations = element(note, "notations")
                        for tie in ties:
                            element(notations, "tied", type=tie)
                    position += duration
        if start + length == len(rows):
            element(element(measure, "barline", location="right"), "bar-style", "light-heavy")
    ET.indent(score)
    ET.ElementTree(score).write(path, encoding="utf-8", xml_declaration=True)


def parse_meter(value):
    try:
        numerator, denominator = map(int, value.split("/"))
        if not (1 <= numerator <= 255 and denominator in (1, 2, 4, 8, 16)):
            raise ValueError
        return numerator, denominator
    except ValueError as error:
        raise argparse.ArgumentTypeError("use a meter such as 4/4 or 3/4 (denominator 1, 2, 4, 8, 16)") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="SATB chorale CSV")
    parser.add_argument("-o", "--output", type=Path, help="MIDI path (default: input path with .mid suffix)")
    parser.add_argument("--bpm", type=float, default=100, help="quarter-note tempo (default: 100)")
    parser.add_argument("--velocity", type=int, default=90, help="note velocity, 1–127 (default: 90)")
    parser.add_argument("--time-signature", type=parse_meter, default=(4, 4), help="notation meter (default: 4/4)")
    parser.add_argument("--no-score", action="store_true", help="skip companion MusicXML score")
    args = parser.parse_args()
    if not 4 <= args.bpm <= 1000:
        parser.error("--bpm must be between 4 and 1000")
    if not 1 <= args.velocity <= 127:
        parser.error("--velocity must be between 1 and 127")
    output = args.output or args.input.with_suffix(".mid")
    score_path = output.with_suffix(".musicxml")
    if output.resolve() == args.input.resolve() or (not args.no_score and score_path.resolve() in (args.input.resolve(), output.resolve())):
        parser.error("input, MIDI, and MusicXML paths must be different")
    try:
        rows = read_chorale(args.input)
        write_midi(rows, output, args.bpm, args.velocity, args.time_signature)
        print(f"MIDI: {output}")
        if not args.no_score:
            write_musicxml(rows, score_path, args.input.stem, args.bpm, args.time_signature)
            print(f"Score: {score_path}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
