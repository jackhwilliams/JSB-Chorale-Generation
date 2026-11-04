# CSV to MIDI

`chorale_to_midi.py` runs on Python 3.9+ with no extra packages. From the project root:

```powershell
python code/chorale_to_midi.py "data/data original chorales/chorale_001.csv"
python code/chorale_to_midi.py "data/data transposed chorales/000.csv" -o code/output.mid --bpm 80
```

By default, the program writes `.mid` and `.musicxml` files next to the input. Use `-o` to choose the MIDI output location (its parent directory must exist). The companion score uses the same stem. Add `--no-score` for MIDI only. `--velocity 90` sets the note attack strength; `--time-signature 3/4` changes the score's meter and MIDI meter metadata. Tempo is in quarter notes per minute (default 100).

Both repository formats are supported: original CSVs start with `note0,note1,note2,note3`; transposed CSVs start with original-tonic/mode metadata such as `1,major`. Those metadata rows are skipped. The four data columns are soprano, alto, tenor, bass; each row lasts one sixteenth note. Values 1–127 are MIDI pitches (69 = A440), and 0 means rest. Consecutive equal pitches in each voice produce one sustained MIDI note; a rest or pitch change ends it. All voices retain their own channels and tracks, including when two voices sing the same pitch. Trailing rests retain the full piece duration.

The CSVs do not encode original meter, pickup lengths, enharmonic spelling, or phrase marks. The score defaults to 4/4 starting at a full bar, uses sharp spellings for black keys, and may need editorial adjustment. Transposed-file metadata describes the original key, so it is not applied as the output key signature. Repeated attacks on the same pitch cannot be recovered from this representation.
