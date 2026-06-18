# Upload a speech-model archive to GitHub Releases (tag: model)
#
# Handles both model families:
#   * IndicConformer per language  (default)
#   * English Conformer-CTC         (-English)
#
# Prerequisites:
#   1. gh CLI authenticated as josephzloty (repo owner):
#        gh auth login --hostname github.com --web --git-protocol ssh
#   2. Archive built:
#        python tools/export_indicconformer.py --lang te --pack     # Indic
#        python tools/export_indicconformer.py --english --pack     # English
#
# Usage:
#   .\tools\upload_ic_release.ps1                # Telugu (default)
#   .\tools\upload_ic_release.ps1 -Lang ta       # Tamil
#   .\tools\upload_ic_release.ps1 -English       # English Conformer-CTC

[CmdletBinding(DefaultParameterSetName = "Indic")]
param(
    [Parameter(ParameterSetName = "Indic")]
    [ValidateSet("as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur")]
    [string]$Lang = "te",

    [Parameter(ParameterSetName = "English", Mandatory = $true)]
    [switch]$English
)

$ErrorActionPreference = "Stop"
$Repo = "josephzloty/mimicplay-models"
$Tag = "model"

# Resolve the asset for the selected model family
if ($English) {
    $AssetName = "english-conformer-ctc-int8.tar.gz"
    $BuildHint = "python tools/export_indicconformer.py --english --pack"
}
else {
    $AssetName = "indicconformer-$Lang-int8.tar.gz"
    $BuildHint = "python tools/export_indicconformer.py --lang $Lang --pack"
}
$Archive = "dist/$AssetName"

if (-not (Test-Path $Archive)) {
    Write-Error "Archive not found: $Archive`nRun: $BuildHint"
}

# Verify gh can see the repo (must be logged in as josephzloty)
gh repo view $Repo 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error @"
gh cannot access $Repo.
Log in as josephzloty first:
  gh auth login --hostname github.com --web --git-protocol ssh
Then re-run this script.
"@
}

$releaseExists = gh release view $Tag --repo $Repo 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating release tag '$Tag'..."
    gh release create $Tag `
        --repo $Repo `
        --title "Speech model assets" `
        --notes "On-demand ASR models for Mimicplay (IndicConformer int8 + English Conformer-CTC). Not tied to app version."
}

Write-Host "Uploading $Archive ..."
gh release upload $Tag $Archive --repo $Repo --clobber

$url = "https://github.com/$Repo/releases/download/$Tag/$AssetName"
Write-Host ""
Write-Host "Done. Download URL:"
Write-Host "  $url"
