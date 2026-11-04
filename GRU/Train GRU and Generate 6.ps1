param([switch]$Preview)

# SETTINGS - edit these, save, then double-click the matching .cmd file.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$HiddenSize = 128
$NumLayers = 2
$DurationHiddenSizes = @(64, 32)
$InputDropout = 0.2
$RecurrentDropout = 0.2
$TrainingSeed = 8675309
$MaxEpochs = 100
$Patience = 12
$LearningRate = 0.001
$MajorSeeds = @(101, 102, 103)
$MinorSeeds = @(201, 202, 203)
$ChordTemperature = 1.0
$DurationTemperature = 1.0
$MaxEvents = 512

# IMPLEMENTATION - no edits needed below.
$ErrorActionPreference = 'Stop'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python not found: $Python"
}
if ($MajorSeeds.Count -ne 3 -or $MinorSeeds.Count -ne 3) {
    throw 'Supply exactly three major seeds and three minor seeds.'
}
if ((@($MajorSeeds + $MinorSeeds | Select-Object -Unique)).Count -ne 6) {
    throw 'Use six different generation seeds.'
}
$RunName = 'gru_' + $HiddenSize + 'x' + $NumLayers + '_' + [guid]::NewGuid().ToString('N').Substring(0, 10)
$RunDirectory = Join-Path $ProjectRoot ('GRU\runs\' + $RunName)
$SamplesDirectory = Join-Path $ProjectRoot ('GRU\generated\' + $RunName)
$Checkpoint = Join-Path $RunDirectory 'best.pt'

function Invoke-PythonStep {
    param([string[]]$Arguments)
    if ($Preview) { $Arguments += '--dry-run' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE. Remaining steps were not run."
    }
}

Push-Location -LiteralPath $ProjectRoot
try {
    if ($Preview) { Write-Host 'PREVIEW ONLY: no training, generation, or output files.' }
    Write-Host 'Step 1/7: train the GRU and select its best validation checkpoint.'
    $TrainingArguments = @(
        'GRU/train.py',
        '--model-config', 'GRU/configs/tonality.json',
        '--training-config', 'GRU/configs/training.json',
        '--mode', 'train',
        '--hidden-size', [string]$HiddenSize,
        '--num-layers', [string]$NumLayers,
        '--duration-hidden-sizes'
    ) + @($DurationHiddenSizes | ForEach-Object { [string]$_ }) + @(
        '--input-dropout', [string]$InputDropout,
        '--recurrent-dropout', [string]$RecurrentDropout,
        '--tonality-conditioning', 'one_hot_direct',
        '--seed', [string]$TrainingSeed,
        '--max-epochs', [string]$MaxEpochs,
        '--patience', [string]$Patience,
        '--learning-rate', [string]$LearningRate,
        '--output', $RunDirectory
    )
    Invoke-PythonStep -Arguments $TrainingArguments
    if (-not $Preview -and -not (Test-Path -LiteralPath $Checkpoint)) {
        throw "Training finished without the expected checkpoint: $Checkpoint"
    }
    $Step = 1
    foreach ($Mode in @('major', 'minor')) {
        $Seeds = if ($Mode -eq 'major') { $MajorSeeds } else { $MinorSeeds }
        foreach ($SampleSeed in $Seeds) {
            $Step++
            Write-Host "Step $Step/7: generate $Mode progression, seed $SampleSeed."
            $OutputFile = Join-Path $SamplesDirectory ($Mode + '_seed' + $SampleSeed + '.csv')
            Invoke-PythonStep -Arguments @(
                'GRU/generate.py', '--checkpoint', $Checkpoint,
                '--tonality', $Mode, '--seed', [string]$SampleSeed,
                '--chord-temperature', [string]$ChordTemperature,
                '--duration-temperature', [string]$DurationTemperature,
                '--max-events', [string]$MaxEvents, '--output', $OutputFile
            )
        }
    }
    Write-Host ''
    Write-Host 'Finished. Each launch uses a new folder; previous runs are preserved.'
    Write-Host "Best checkpoint: $Checkpoint"
    Write-Host "Six CSVs and matching JSON files: $SamplesDirectory"
}
finally {
    Pop-Location
}
