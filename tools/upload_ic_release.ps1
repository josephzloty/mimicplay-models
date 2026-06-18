# Upload IndicConformer model archive to GitHub Releases (tag: model)
#
# Prerequisites:
#   1. gh CLI authenticated as josephzloty (repo owner):
#        gh auth login --hostname github.com --web --git-protocol ssh
#   2. Archive built:
#        python scripts/export_indicconformer.py --lang te --pack
#
# Usage:
#   .\scripts\upload_ic_release.ps1
#   .\scripts\upload_ic_release.ps1 -Lang ta

param(
    [ValidateSet("as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur")]
    [string]$Lang = "te"
)

$ErrorActionPreference = "Stop"
$Repo = "josephzloty/mimicplay-models"
$Tag = "model"
$Archive = "dist/indicconformer-$Lang-int8.tar.gz"

if (-not (Test-Path $Archive)) {
    Write-Error "Archive not found: $Archive`nRun: python scripts/export_indicconformer.py --lang $Lang --pack"
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
        --notes "On-demand ASR models for Mimicplay (IndicConformer int8 + future assets). Not tied to app version."
}

Write-Host "Uploading $Archive ..."
gh release upload $Tag $Archive --repo $Repo --clobber

$url = "https://github.com/$Repo/releases/download/$Tag/indicconformer-$Lang-int8.tar.gz"
Write-Host ""
Write-Host "Done. Download URL:"
Write-Host "  $url"
