# mimicplay-models

On-device speech-recognition **model packs** and the **export/upload tooling** that builds them, for the [MimicPlay](https://github.com/josephzloty/mimicplay) speech-practice app.

This repo exists so the (private) app repo stays code-only: large model binaries are hosted here as GitHub Release assets and downloaded by the app on demand at first use, per language.

## What's hosted here

Models are **AI4Bharat IndicConformer** (NeMo CTC) exported to [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)-compatible ONNX, int8-quantized, with the required sherpa-onnx metadata stamped (`vocab_size`, `subsampling_factor`, `normalize_type`, `model_type`). A parallel **English NeMo Conformer-CTC** pack powers the app's dual-engine path for code-mixed (Tenglish/Hinglish) speech.

Each pack is a flat `.tar.gz` containing exactly two files:

```
model.int8.onnx     # int8-quantized ONNX, sherpa-onnx metadata stamped
tokens.txt          # BPE vocabulary in sherpa-onnx format
```

## Release layout

All packs are published as assets on a single release tagged **`model`**. The app builds download URLs as:

```
https://github.com/josephzloty/mimicplay-models/releases/download/model/<asset>
```

| Asset | Contents |
|-------|----------|
| `indicconformer-te-int8.tar.gz` | Telugu IndicConformer (hosted) |
| `indicconformer-<lang>-int8.tar.gz` | Other Indic languages (as exported) |
| `english-conformer-ctc-int8.tar.gz` | English Conformer-CTC (dual-engine) |

`<lang>` is the AI4Bharat language code: `as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur`.

> The app references these names in `ModelDownloadManager.kt` (`icUrl()` / `enCtcUrl()`). Keep asset names in sync with that file.

## Tooling

```
tools/
├── export_indicconformer.py   # NeMo → sherpa-onnx ONNX export + int8 quantize + metadata stamp
└── upload_ic_release.ps1      # Upload a packed archive to the `model` GitHub Release
```

### Export

```bash
pip install nemo_toolkit[asr] torch onnx onnxruntime onnxconverter-common pip-system-certs

python tools/export_indicconformer.py --lang te        # Telugu
python tools/export_indicconformer.py --all            # all Indic languages
python tools/export_indicconformer.py --verify te      # verify a built pack
python tools/export_indicconformer.py --lang te --pack # produce the .tar.gz
```

Output lands in `dist/indicconformer-<lang>/` (`model.int8.onnx` + `tokens.txt`), then gets packed to `indicconformer-<lang>-int8.tar.gz`.

### Upload

```powershell
# requires the GitHub CLI (gh) authenticated as the repo owner
.\tools\upload_ic_release.ps1            # Telugu (default)
.\tools\upload_ic_release.ps1 -Lang ta   # Tamil
```

## How the app consumes a pack

1. App ships with **no** model files (~30 MB base APK; sherpa-onnx native libs only).
2. On first use of a language, `ModelDownloadManager` downloads the matching pack from this repo's `model` release, extracts `model.int8.onnx` + `tokens.txt` into `filesDir/models/indicconformer/<lang>/`, and caches it.
3. `IndicConformerEngine` loads the pack via sherpa-onnx `OfflineRecognizer`.

## Attribution & licensing

Indic models are derived from **AI4Bharat IndicConformer**; the English model from a NeMo Conformer-CTC checkpoint. These derived packs are subject to the upstream model licenses and terms — review and comply with the original AI4Bharat / NVIDIA NeMo licenses before redistribution. This repo's **tooling** (export/upload scripts) is part of the MimicPlay project.