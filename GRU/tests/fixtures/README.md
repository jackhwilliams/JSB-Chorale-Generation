# Regression fixtures

`generation_reference.json` records generated events, draw traces, termination,
and generation settings for the selected checkpoints. These samples were checked
against the pre-reorganization behavior before being extracted. Host paths,
Python/PyTorch version strings, and device metadata are deliberately omitted.
Tests require the same events and stopping behavior, and compare floating-point
trace fields to five decimal places to allow small CPU implementation differences.

`asset_hashes.json` covers selected checkpoints, configs, frozen preprocessing,
and the generation reference. Checkpoint hashes use raw bytes; text hashes use
UTF-8 with normalized newlines so Git's checkout settings do not change them.
Update fixtures only for an intentional, reviewed behavior or asset change.
Tests and normal workflows do not need the excluded historical run directories.
