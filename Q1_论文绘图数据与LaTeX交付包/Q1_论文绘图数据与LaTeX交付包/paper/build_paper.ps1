$ErrorActionPreference = 'Stop'
$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceDir = Split-Path -Parent $paperDir
$pythonExe = (Get-Command python.exe -ErrorAction Stop).Source
$strawberryPerl = 'C:\Strawberry\perl\bin'
if ((Test-Path (Join-Path $strawberryPerl 'perl.exe')) -and ($env:Path -notlike "*$strawberryPerl*")) {
  $env:Path = "$strawberryPerl;$env:Path"
}
foreach ($localeVariable in @('LC_ALL', 'LC_CTYPE', 'LANG')) {
  Remove-Item "Env:$localeVariable" -ErrorAction SilentlyContinue
}

# The competition workspace manifest is the single source for Q1 constants.
$numbersPath = Join-Path $workspaceDir 'outputs\q1_final_local\00_manifest\q1_paper_numbers.json'
$numbers = Get-Content -Raw $numbersPath | ConvertFrom-Json
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$breakableTextModel = [regex]::Replace($numbers.final_models.text.value, '([/-])', '$1\allowbreak ')
$breakableAudioModel = [regex]::Replace($numbers.final_models.audio.value, '([/-])', '$1\allowbreak ')
$breakableVisionModel = [regex]::Replace($numbers.final_models.vision.value, '([/-])', '$1\allowbreak ')
$nearest = @($numbers.alignment_ablation | Where-Object { $_.method.value -eq 'nearest_center' })[0]
$macroLines = @(
  '% Generated from outputs/q1_final_local/00_manifest/q1_paper_numbers.json; do not edit manually.',
  "\newcommand{\QOneSamples}{$($numbers.sample_count.value)}",
  "\newcommand{\QOneBins}{$($numbers.aligned_length.value)}",
  "\newcommand{\QOneTextDim}{$($numbers.feature_dimensions.text.value)}",
  "\newcommand{\QOneAudioDim}{$($numbers.feature_dimensions.audio.value)}",
  "\newcommand{\QOneVisionDim}{$($numbers.feature_dimensions.vision.value)}",
  "\newcommand{\QOneTextRows}{$($numbers.native_row_counts.text.value)}",
  "\newcommand{\QOneAudioRows}{$($numbers.native_row_counts.audio.value)}",
  "\newcommand{\QOneVisionRows}{$($numbers.native_row_counts.vision.value)}",
  "\newcommand{\QOneTextCoverage}{$(([double]$numbers.modality_coverage.text.value).ToString('F4', $culture))}",
  "\newcommand{\QOneAudioCoverage}{$(([double]$numbers.modality_coverage.audio.value).ToString('F4', $culture))}",
  "\newcommand{\QOneVisionCoverage}{$(([double]$numbers.modality_coverage.vision.value).ToString('F4', $culture))}",
  "\newcommand{\QOneOfficialTextCount}{$($numbers.text_source_counts.official_transcript.value)}",
  "\newcommand{\QOneMediaAsrCount}{$($numbers.text_source_counts.media_asr.value)}",
  "\newcommand{\QOneSelectedTimestampFallback}{$($numbers.timestamp_fallback_count.value)}",
  "\newcommand{\QOneRawTimestampFallback}{$($numbers.raw_asr_timestamp_fallback_count.value)}",
  "\newcommand{\QOneNearestZeroOverlap}{$($nearest.nearest_center_zero_overlap_assignments.value)}",
  "\newcommand{\QOneTextModel}{$breakableTextModel}",
  "\newcommand{\QOneAudioModel}{$breakableAudioModel}",
  "\newcommand{\QOneVisionModel}{$breakableVisionModel}"
)
Set-Content (Join-Path $paperDir 'q1_numbers.tex') $macroLines -Encoding UTF8

& $pythonExe (Join-Path $paperDir 'scripts\generate_frozen_tables.py')
if ($LASTEXITCODE -ne 0) { throw "Frozen table generation failed with exit code $LASTEXITCODE" }

Push-Location $paperDir
try {
  $latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
  $perl = Get-Command perl -ErrorAction SilentlyContinue
  if ($latexmk -and $perl) {
    & $latexmk.Source -g -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
    if ($LASTEXITCODE -ne 0) { throw "latexmk failed with exit code $LASTEXITCODE" }
  }
  else {
    New-Item -ItemType Directory -Force -Path build | Out-Null
    & xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
    if ($LASTEXITCODE -ne 0) { throw "XeLaTeX first pass failed with exit code $LASTEXITCODE" }
    & bibtex build/main
    if ($LASTEXITCODE -ne 0) { throw "BibTeX failed with exit code $LASTEXITCODE" }
    foreach ($pass in 1..2) {
      & xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
      if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass $pass failed with exit code $LASTEXITCODE" }
    }
  }
  $log = Get-Content (Join-Path $paperDir 'build\main.log') -Raw
  $severe = @('Undefined control sequence', 'LaTeX Warning: Reference .* undefined', 'Citation .* undefined', 'Missing character')
  foreach ($pattern in $severe) {
    if ($log -match $pattern) { throw "Build log check failed: $pattern" }
  }
  Write-Output "PDF: $(Join-Path $paperDir 'build\main.pdf')"
  Write-Output 'No undefined control sequence/reference, undefined citation, or missing-character diagnostics found.'
  $overfull = Select-String -Path (Join-Path $paperDir 'build\main.log') -Pattern 'Overfull \\hbox'
  if ($overfull) { Write-Warning "Overfull hbox diagnostics: $($overfull.Count)" } else { Write-Output 'No overfull hbox diagnostics found.' }
}
finally { Pop-Location }
