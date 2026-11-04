import csv
import time
import tinysoundfont
from catalog import *

# Favorites: 012, 015, 021, 102, 120, 122, 123, 117, 155, 208, 223, 256, 287, 309, 311, 337
# 267 has rests for testing

CHORALE_PATH = "data transposed chorales/120.csv"

BPM = 100
CHANNEL_VOLUME = 127  # [0-127] Main output loudness
EXPRESSION = 90 # [0-127] Structural fullness
STRIKE_VELOCITY = 90  # [0-127] Hammer strike hardness
BRIGHTNESS = 120  # [0-127] Shaves off high frequencies

REST_VAL = -1000

# Initialize the tinysoundfont synthesizer
synth = tinysoundfont.Synth()
sfid = synth.sfload("UprightPianoKW-20220221.sf2")

# Configure the 4 MIDI channels with your initial global volume knobs
for channel in range(4):
    synth.program_select(channel, sfid, 0, 0)
    synth.control_change(channel, 7, CHANNEL_VOLUME)

# Start the audio engine
synth.start()

sleep_int = 15 / BPM
current_notes = [None, None, None, None]

try:
    with open(CHORALE_PATH, mode="r") as score:
        reader = csv.reader(score)
        head = next(reader)


        for row in reader:
            raw_notes = [int(val) for val in row[:4]]

            if "transposed" in CHORALE_PATH:
                print(all_roman_numerals[match_chord([pitch_to_note(x) for x in raw_notes], "minor")])

            for i, midi_note in enumerate(raw_notes):
                # 1. Clear previous note
                if current_notes[i] is not None and (midi_note != current_notes[i] or midi_note <= 10):
                    synth.noteoff(i, current_notes[i])
                    current_notes[i] = None

                # 2. Trigger new note using your active timbre knobs
                if midi_note > 10 and midi_note != current_notes[i]:
                    synth.control_change(i, 11, EXPRESSION)
                    synth.control_change(i, 74, BRIGHTNESS)

                    synth.noteon(i, midi_note, STRIKE_VELOCITY)
                    current_notes[i] = midi_note

            time.sleep(sleep_int)

finally:
    for i, note in enumerate(current_notes):
        if note is not None:
            synth.noteoff(i, note)
