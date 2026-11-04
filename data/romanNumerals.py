import csv
import os
from catalog import *

def cond_1(ind, chord_frame, note_frame, pitch_frame, tonality):
    # check if a labeled neighbors has the same notes

    # check previous
    if chord_frame[ind-1] is not None and nset(note_frame[ind-1]) == nset(note_frame[ind]):
        return chord_frame[ind-1]
    # check next
    if chord_frame[ind+1] is not None and nset(note_frame[ind+1]) == nset(note_frame[ind]):
        return chord_frame[ind+1]
    # else none
    return None

def cond_2(ind, chord_frame, note_frame, pitch_frame, tonality):
    # check if both neighbors are the same
    if chord_frame[ind-1] is not None and chord_frame[ind+1] is not None and chord_frame[ind-1] == chord_frame[ind+1]:
        return chord_frame[ind-1]
    return None

def cond_3(ind, chord_frame, note_frame, pitch_frame, tonality):
    # check if ind chord is a subset of previous chord

    # check that previous chord has label
    if chord_frame[ind-1] is None:
        return None

    # check that previous chord matches label
    if match_chord(nset(note_frame[ind-1]), tonality) != chord_frame[ind+1]:
        return None

    # check that ind chord is a subset of previous
    ind_notes = nset(note_frame[ind])
    prev_notes = nset(note_frame[ind-1])
    diff = ind_notes - prev_notes
    if len(diff) != 0:
        return None

    # check that either bass note is the same or ind chord has two notes
    diff_bass = (pitch_frame[ind][3] != pitch_frame[ind-1][3])
    few_notes = (len(nset(note_frame[ind])) < 2)
    if diff_bass and few_notes:
        return None

    return chord_frame[ind-1]

def cond_4(ind, chord_frame, note_frame, pitch_frame, tonality):
    # check if ind chord is a suspension

    # check that next chord has label
    if chord_frame[ind+1] is None:
        return None

    # check that next chord matches label
    if match_chord(nset(note_frame[ind+1]), tonality) != chord_frame[ind+1]:
        return None

    # check that ind chord differs from next by one note
    ind_notes = nset(note_frame[ind])
    next_notes = nset(note_frame[ind+1])
    diff = ind_notes - next_notes
    if len(diff) != 1:
        return None

    # iterate over voices on the differing note
    diff = diff.pop()
    voices = [i for i, x in enumerate(note_frame[ind]) if x == diff]
    for voice in voices:
        # check that diff note was held from previous
        if pitch_frame[ind-1][voice] != pitch_frame[ind][voice]:
            return None
        # check that diff note is 1 or 2 steps above next
        pitch_diff = pitch_frame[ind][voice] - pitch_frame[ind+1][voice]
        if pitch_diff == 0 or pitch_diff > 2:
            return None

    return chord_frame[ind+1]

def cond_5(ind, chord_frame, note_frame, pitch_frame, tonality):
    # check if ind chord is a passing/neighbor tone or anticipation

    # check that previous chord has label
    if chord_frame[ind-1] is None:
        return None

    # check that previous chord matched label
    if match_chord(nset(note_frame[ind-1]), tonality) != chord_frame[ind-1]:
        return None

    # check that ind chord differs from previous by one note
    ind_notes = nset(note_frame[ind])
    prev_notes = nset(note_frame[ind-1])
    diff = ind_notes - prev_notes
    if len(diff) != 1:
        return None

    diff = diff.pop()
    voices = [i for i, x in enumerate(note_frame[ind]) if x == diff]
    for voice in voices:
        # check that diff note is 1 or 2 steps from previous
        pitch_diff = pitch_frame[ind][voice] - pitch_frame[ind-1][voice]
        if pitch_diff == 0 or abs(pitch_diff) > 2:
            return None
        # check that diff note is 0, 1, or 2 steps from next
        pitch_diff = pitch_frame[ind][voice] - pitch_frame[ind+1][voice]
        if abs(pitch_diff) > 2:
            return None

    return chord_frame[ind-1]

def cond_6(ind, chord_frame, note_frame, pitch_frame, tonality):
    # check if previous chord is close enough

    # check that previous chord has label
    if chord_frame[ind-1] is None:
        return None

    # check that previous chord matched label
    if match_chord(nset(note_frame[ind-1]), tonality) != chord_frame[ind-1]:
        return None

    # check that ind chord differs from previous by one note
    ind_notes = nset(note_frame[ind])
    prev_notes = nset(note_frame[ind - 1])
    diff = ind_notes - prev_notes
    if len(diff) != 1:
        return None

    # check that bass notes are the same
    if pitch_frame[ind][3] != pitch_frame[ind-1][3]:
        return None

    return chord_frame[ind-1]

def cond_7(ind, chord_frame, note_frame, pitch_frame, tonality):
    # check if next chord is close enough

    # check that next chord has label
    if chord_frame[ind+1] is None:
        return None

    # check that next chord matched label
    if match_chord(nset(note_frame[ind+1]), tonality) != chord_frame[ind+1]:
        return None

    # check that ind chord differs from previous by one note
    ind_notes = nset(note_frame[ind])
    next_notes = nset(note_frame[ind+1])
    diff = ind_notes - next_notes
    if len(diff) != 1:
        return None

    # check that bass notes are the same
    if pitch_frame[ind][3] != pitch_frame[ind+1][3]:
        return None

    return chord_frame[ind+1]

def cond_8(ind, chord_frame, note_frame, pitch_frame, tonality):
    # if ind chord has two neighbors, try to pick the best one

    # check that both neighbors have labels
    if chord_frame[ind-1] is None or chord_frame[ind+1] is None:
        return None

    # check if either neighbor shares more notes in common
    ind_notes = nset(note_frame[ind])
    prev_notes = nset(note_frame[ind-1])
    next_notes = nset(note_frame[ind+1])
    diff1 = ind_notes - prev_notes
    diff2 = ind_notes - next_notes

    if len(diff1) < len(diff2):
        return chord_frame[ind-1]
    if len(diff2) < len(diff1):
        return chord_frame[ind+1]

    # check if either neighbor shares a bass while other doesn't
    ind_bass = pitch_frame[ind][3]
    prev_bass = pitch_frame[ind-1][3]
    next_bass = pitch_frame[ind+1][3]

    if ind_bass == prev_bass and ind_bass != next_bass:
        return chord_frame[ind-1]
    if ind_bass == next_bass and ind_bass != prev_bass:
        return chord_frame[ind+1]

    # check if either is originally labeled while other isn't
    prev_labeled = match_chord(nset(note_frame[ind-1]), tonality) == chord_frame[ind-1]
    next_labeled = match_chord(nset(note_frame[ind+1]), tonality) == chord_frame[ind+1]

    if prev_labeled and not next_labeled:
        return chord_frame[ind-1]
    if next_labeled and not prev_labeled:
        return chord_frame[ind+1]

    # check if either shares more held notes
    prev_held = sum([pitch_frame[ind-1][x] == pitch_frame[ind][x] for x in range(4)])
    next_held = sum([pitch_frame[ind+1][x] == pitch_frame[ind][x] for x in range(4)])

    if prev_held > next_held:
        return chord_frame[ind-1]
    if next_held > prev_held:
        return chord_frame[ind+1]

    return None

def cond_9(ind, chord_frame, note_frame, pitch_frame, tonality):
    # assume the correct label is the previous
    return chord_frame[ind-1]


def main(filename):

    pitch_frame, head = csv_to_frame("data transposed chorales/" + filename)
    original_key, tonality = head
    note_frame = transform_frame(pitch_frame, func=lambda x: [pitch_to_note(t) for t in x])
    chord_frame = transform_frame(note_frame, func=lambda x: match_chord(nset(x), tonality=tonality))

    # run length encoding of chords
    rle = [1]*len(chord_frame)

    # combine None chords with same pitches
    ind = 0
    while ind < len(chord_frame) - 1:
        if chord_frame[ind] is None and chord_frame[ind+1] is None and pitch_frame[ind] == pitch_frame[ind+1]:
            rle[ind] += 1
            rle.pop(ind+1)
            pitch_frame.pop(ind+1)
            note_frame.pop(ind+1)
            chord_frame.pop(ind+1)
        ind += 1

    # variables for labeling loop
    row_ind = 0
    cond_ind = 0
    last_change = len(chord_frame) - 1
    conditions = [cond_1, cond_2, cond_3, cond_4, cond_5, cond_6, cond_7, cond_8, cond_9]

    # pick label for all None chords based on heuristics
    while None in chord_frame:

        if row_ind == last_change:
            cond_ind  = min(cond_ind + 1, len(conditions) - 1)

        if chord_frame[row_ind] is None:
            chord_frame[row_ind] = conditions[cond_ind](row_ind, chord_frame, note_frame, pitch_frame, tonality)

            if chord_frame[row_ind] is not None:
                last_change = row_ind
                cond_ind = 0

        row_ind = (row_ind + 1) % len(chord_frame)

    current_length = len(chord_frame)
    reduced_length = 0
    while reduced_length < current_length:

        current_length = reduced_length

        # combine adjacent labels if they are the same
        i = 0
        prev_chord = None
        while i < len(chord_frame):
            if prev_chord == chord_frame[i]:
                rle[i-1] += rle[i]
                rle.pop(i)
                chord_frame.pop(i)
            else:
                prev_chord = chord_frame[i]
                i += 1

        # combine triads/sevenths under certain conditions
        i = 0
        while i < len(chord_frame):
            label = all_roman_numerals[chord_frame[i]]
            if "maj7" in label:

                curr_triad = label.split(" ")[0]

                if i-1 >= 0:
                    prev_triad = all_roman_numerals[chord_frame[i-1]].split(" ")[0]
                    if curr_triad == prev_triad:
                        rle[i-1] += rle[i]
                        rle.pop(i)
                        chord_frame.pop(i)
                        i -= 1

                if i+1 < len(chord_frame):
                    next_triad = all_roman_numerals[chord_frame[i+1]].split(" ")[0]
                    if curr_triad == next_triad:
                        rle[i+1] += rle[i]
                        rle.pop(i)
                        chord_frame.pop(i)

            elif "7" in label and i+1 < len(chord_frame):
                curr_triad = label.split(" ")[0]
                next_triad = all_roman_numerals[chord_frame[i+1]].split(" ")[0]
                if curr_triad == next_triad:
                    rle[i] += rle[i+1]
                    rle.pop(i+1)
                    chord_frame.pop(i+1)
            i += 1

        reduced_length = len(chord_frame)

    label_frame = transform_frame(chord_frame, func=lambda x: all_roman_numerals[x])

    with open("chorale chords/" + filename, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([tonality])
        writer.writerow(["chord_index", "duration"])

        for row in zip(chord_frame, rle):
            writer.writerow(row)


if __name__ == "__main__":
    chorales = os.listdir("data transposed chorales")
    for chorale in chorales:
        main(chorale)
