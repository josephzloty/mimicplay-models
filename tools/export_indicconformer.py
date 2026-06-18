#!/usr/bin/env python3
"""
Export AI4Bharat IndicConformer to sherpa-onnx compatible ONNX format.

Run this ONCE on your desktop (not on device). Outputs go to dist/ then upload
to GitHub Releases so the app can download them on-demand.

Prerequisites (install once):
    pip install nemo_toolkit[asr] torch onnx onnxruntime onnxconverter-common pip-system-certs

Windows SSL note:
    If you see "CERTIFICATE_VERIFY_FAILED" when contacting huggingface.co, install
    pip-system-certs (uses the Windows certificate store with requests/urllib3).

Usage:
    python scripts/export_indicconformer.py --lang te     # Telugu
    python scripts/export_indicconformer.py --lang ta     # Tamil
    python scripts/export_indicconformer.py --lang hi     # Hindi
    python scripts/export_indicconformer.py --all         # all three

Output structure per language (e.g. te):
    dist/indicconformer-te/
        model.onnx          # fp32 (intermediate, used to produce int8)
        model.int8.onnx     # int8 quantized — upload this to GitHub Releases
        tokens.txt          # vocabulary

After export:
    1. Verify with: python scripts/export_indicconformer.py --verify te
    2. Create a tar.gz: tar -czf indicconformer-te-int8.tar.gz -C dist/indicconformer-te model.int8.onnx tokens.txt
    3. Upload to GitHub Releases as: indicconformer-<lang>-int8.tar.gz
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# ── Model registry ──────────────────────────────────────────────────────────
# All 22 scheduled Indian languages supported by AI4Bharat IndicConformer.
# HuggingFace model IDs follow: ai4bharat/indicconformer_stt_{code}_hybrid_ctc_rnnt_large

LANGUAGE_NAMES = {
    "as":  "Assamese",
    "bn":  "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu":  "Gujarati",
    "hi":  "Hindi",
    "kn":  "Kannada",
    "kok": "Konkani",
    "ks":  "Kashmiri",
    "mai": "Maithili",
    "ml":  "Malayalam",
    "mni": "Manipuri",
    "mr":  "Marathi",
    "ne":  "Nepali",
    "or":  "Odia",
    "pa":  "Punjabi",
    "sa":  "Sanskrit",
    "sat": "Santali",
    "sd":  "Sindhi",
    "ta":  "Tamil",
    "te":  "Telugu",
    "ur":  "Urdu",
}

MODELS = {
    code: f"ai4bharat/indicconformer_stt_{code}_hybrid_ctc_rnnt_large"
    for code in LANGUAGE_NAMES
}

ALL_LANG_CODES = sorted(LANGUAGE_NAMES.keys())

# ── Helpers ──────────────────────────────────────────────────────────────────

def check_imports():
    missing = []
    for pkg in ["nemo", "torch", "onnx", "onnxruntime", "huggingface_hub"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("Install with: pip install nemo_toolkit[asr] torch onnx onnxruntime huggingface_hub")
        sys.exit(1)


def hf_login(token: str | None):
    """Authenticate with HuggingFace (required for gated AI4Bharat models)."""
    from huggingface_hub import login, HfApi
    import os

    # Priority: CLI arg → env var → cached login (from `huggingface-cli login`)
    resolved = token or os.environ.get("HF_TOKEN")
    if resolved:
        login(token=resolved, add_to_git_credential=False)
        print(f"  HuggingFace: logged in via token")
    else:
        # Check if already logged in from a prior `huggingface-cli login`
        try:
            api = HfApi()
            user = api.whoami()
            print(f"  HuggingFace: already logged in as {user['name']}")
        except Exception:
            print("[ERROR] HuggingFace authentication required.")
            print("  Option 1: huggingface-cli login  (interactive, saves token)")
            print("  Option 2: set HF_TOKEN env var")
            print("  Option 3: pass --hf-token <token>")
            print()
            print("  Then request model access at:")
            print("  https://huggingface.co/ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large")
            sys.exit(1)


def add_sherpa_onnx_metadata(onnx_path: Path, vocab_size: int, subsampling_factor: int):
    """Embed the ONNX metadata sherpa-onnx's native NeMo-CTC loader requires.

    sherpa-onnx does NOT just run a generic NeMo ONNX export — its C++ loader
    (OfflineNemoEncDecCtcModel) reads custom metadata_props out of the ONNX file
    to configure its internal feature extractor and decode buffers:
    vocab_size, subsampling_factor, normalize_type, model_type.
    Without these keys, sherpa-onnx's native loader can crash the whole process
    (not a catchable JVM exception) as soon as OfflineRecognizer(config) tries
    to read the model — matching "app closes right after IC model download".
    See: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/nemo/how-to-export.html
    """
    import onnx

    model = onnx.load(str(onnx_path))
    existing_keys = {m.key for m in model.metadata_props}
    meta_data = {
        "vocab_size": str(vocab_size),
        "normalize_type": "per_feature",
        "subsampling_factor": str(subsampling_factor),
        "model_type": "EncDecCTCModelBPE",
        "version": "1",
        "model_author": "nemo",
        "comment": "ai4bharat indicconformer (CTC branch) via scripts/export_indicconformer.py",
    }
    added = []
    for key, value in meta_data.items():
        if key in existing_keys:
            continue
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value
        added.append(key)
    onnx.save(model, str(onnx_path))
    if added:
        print(f"  Added sherpa-onnx metadata to {onnx_path.name}: {meta_data}")
    else:
        print(f"  sherpa-onnx metadata already present in {onnx_path.name}, skipping")


def export_language(lang: str, dist_dir: Path, opset: int = 14):
    import torch
    import onnx

    model_id = MODELS[lang]
    out_dir = dist_dir / f"indicconformer-{lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_fp32_path = out_dir / "model.onnx"
    onnx_int8_path = out_dir / "model.int8.onnx"
    tokens_path = out_dir / "tokens.txt"

    print(f"\n[{LANGUAGE_NAMES[lang]}] Loading NeMo model from HuggingFace: {model_id}")
    print("  (first download ~500 MB NeMo .nemo file — cached in ~/.cache/huggingface)")

    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download
    import tarfile
    import yaml
    import tempfile
    import os

    # Downloading via hf_hub_download bypasses NeMo's buggy from_pretrained HF integration
    # which sometimes confuses the snapshot folder with an extracted NeMo folder.
    nemo_filename = f"indicconformer_stt_{lang}_hybrid_rnnt_large.nemo"
    print(f"  Downloading/Locating {nemo_filename} via hf_hub_download...")
    nemo_path = hf_hub_download(repo_id=model_id, filename=nemo_filename)

    # NeMo 2.7.3 has an incompatibility with AI4Bharat's older config format.
    # The config uses `type: multilingual` which NeMo 2.7.3 tries to parse as a monolingual
    # tokenizer, causing a KeyError: 'dir'. We dynamically patch the config to `type: agg`.
    print("  Patching model_config.yaml for NeMo 2.x compatibility...")
    with tarfile.open(nemo_path, "r") as t:
        config_member = [m for m in t.getmembers() if "model_config.yaml" in m.name][0]
        config_dict = yaml.safe_load(t.extractfile(config_member))

    if config_dict.get("tokenizer", {}).get("type") == "multilingual" and "langs" in config_dict.get("tokenizer", {}):
        config_dict["tokenizer"]["type"] = "agg"

    # NeMo 2.7.x removed multisoftmax from decoder modules (RNNT + CTC branches).
    if "decoder" in config_dict:
        config_dict["decoder"].pop("multisoftmax", None)
    # Joint multilingual kwargs removed in NeMo 2.7 — strip for init; load weights with strict=False.
    if "joint" in config_dict:
        config_dict["joint"].pop("multilingual", None)
        config_dict["joint"].pop("language_keys", None)
    aux_ctc = config_dict.get("aux_ctc", {})
    if isinstance(aux_ctc.get("decoder"), dict):
        aux_ctc["decoder"].pop("multisoftmax", None)
        
    patched_config_fd, patched_config_path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(patched_config_fd, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)

    # IndicConformer is a *hybrid* CTC/RNNT model — must use EncDecHybridRNNTCTCBPEModel,
    # not EncDecCTCModelBPE. The hybrid model contains both a CTC and an RNNT decoder;
    # we export only the CTC branch (lighter, no beam search required on-device).
    try:
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            nemo_path,
            override_config_path=patched_config_path,
            strict=False,  # RNNT joint weights differ after config patch; CTC branch still loads
        )
    finally:
        os.remove(patched_config_path)
    model.eval()

    # Switch to CTC decoding branch for export
    model.cur_decoder = "ctc"
    print(f"  Loaded (hybrid CTC/RNNT). CTC vocab size: {model.ctc_decoder.num_classes_with_blank}")

    # ── Export tokens.txt ────────────────────────────────────────────────────
    if not tokens_path.exists():
        print(f"  Writing tokens.txt -> {tokens_path}")
        # Hybrid model: vocabulary is on the tokenizer, blank is separate
        tokenizer = model.tokenizer
        
        # If the tokenizer is an AggregateTokenizer, get the vocab for this specific language
        if hasattr(tokenizer, "tokenizers") and lang in tokenizer.tokenizers:
            vocab = tokenizer.tokenizers[lang].vocab
        else:
            vocab = tokenizer.vocab  # dict {token: id} for SentencePiece tokenizer
            
        if isinstance(vocab, dict):
            sorted_vocab = [t for t, _ in sorted(vocab.items(), key=lambda x: x[1])]
        else:
            sorted_vocab = list(vocab)  # list-style tokenizer
        blank_id = model.ctc_decoder.num_classes_with_blank - 1
        with open(tokens_path, "w", encoding="utf-8") as f:
            for i, token in enumerate(sorted_vocab):
                f.write(f"{token} {i}\n")
            # Ensure blank token is at the expected position
            if "<blk>" not in sorted_vocab and "<blank>" not in sorted_vocab:
                f.write(f"<blk> {blank_id}\n")
        print(f"  Tokens: {len(sorted_vocab)} entries (blank_id={blank_id})")
    else:
        print(f"  tokens.txt already exists, skipping")

    # ── Export fp32 ONNX ─────────────────────────────────────────────────────
    if not onnx_fp32_path.exists():
        print(f"  Exporting fp32 ONNX (opset {opset}) -> {onnx_fp32_path}")
        # NeMo's export() handles ONNX tracing and opset selection
        model.export(
            str(onnx_fp32_path),
            onnx_opset_version=opset,
            check_trace=False,       # skip trace check for speed; verify separately
        )
        print(f"  fp32 size: {onnx_fp32_path.stat().st_size / 1e6:.1f} MB")
    else:
        print(f"  model.onnx already exists, skipping fp32 export")

    # ── Add sherpa-onnx required metadata (REQUIRED — see add_sherpa_onnx_metadata) ──
    # subsampling_factor: try the live model config first; fall back to the patched
    # YAML dict; fall back to 4 (NeMo Conformer-CTC default; FastConformer uses 8).
    try:
        subsampling_factor = int(model.cfg.encoder.subsampling_factor)
    except Exception:
        subsampling_factor = int(config_dict.get("encoder", {}).get("subsampling_factor", 4))
    vocab_size = sum(1 for _ in open(tokens_path, encoding="utf-8"))
    print(f"  sherpa-onnx metadata: vocab_size={vocab_size}, subsampling_factor={subsampling_factor}")
    add_sherpa_onnx_metadata(onnx_fp32_path, vocab_size, subsampling_factor)

    # ── Quantize to int8 ─────────────────────────────────────────────────────
    if not onnx_int8_path.exists():
        print(f"  Quantizing to int8 -> {onnx_int8_path}")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        import shutil
        # Windows: shape-infer temp file beside the input ONNX can stay locked; use system temp.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_in = Path(tmpdir) / "model.onnx"
            tmp_out = Path(tmpdir) / "model.int8.onnx"
            shutil.copy2(onnx_fp32_path, tmp_in)
            quantize_dynamic(
                model_input=str(tmp_in),
                model_output=str(tmp_out),
                weight_type=QuantType.QInt8,
            )
            shutil.copy2(tmp_out, onnx_int8_path)
        int8_mb = onnx_int8_path.stat().st_size / 1e6
        fp32_mb = onnx_fp32_path.stat().st_size / 1e6
        print(f"  int8 size: {int8_mb:.1f} MB (was {fp32_mb:.1f} MB fp32, "
              f"{100*(fp32_mb-int8_mb)/fp32_mb:.0f}% reduction)")
    else:
        print(f"  model.int8.onnx already exists, skipping quantization")

    # quantize_dynamic does not reliably preserve custom metadata_props across
    # all onnxruntime versions — re-stamp the int8 file unconditionally so the
    # shipped artifact is guaranteed to carry the sherpa-onnx metadata.
    add_sherpa_onnx_metadata(onnx_int8_path, vocab_size, subsampling_factor)

    print(f"\n  [OK] {LANGUAGE_NAMES[lang]} complete -> {out_dir}/")
    print(f"      model.int8.onnx: {onnx_int8_path.stat().st_size / 1e6:.1f} MB")
    print(f"      tokens.txt:      {tokens_path.stat().st_size / 1024:.1f} KB")
    return out_dir


def verify_language(lang: str, dist_dir: Path, test_wav: str | None = None):
    """Sanity-check an exported model: metadata present, graph loads, a
    correctly-ranked forward pass runs without error.

    IMPORTANT: `audio_signal` is NOT raw PCM. NeMo's exported CTC graph expects
    already-extracted mel-spectrogram features, shape (batch, n_mels, time) —
    sherpa-onnx computes those features itself in native code before calling the
    model; it never feeds raw waveform samples in. A 2D (batch, time) raw-audio
    input (the old behaviour here) is the WRONG contract for this graph — that
    mismatch is expected and is not evidence of a broken export (see ISSUE-008).
    This function builds a correctly-shaped placeholder instead, so it actually
    matches what sherpa-onnx will send on-device. It does not reproduce a real
    transcript (no fbank extraction here) — for real ASR output, test through
    sherpa-onnx (on-device or `pip install sherpa-onnx` on desktop).
    """
    import onnxruntime as ort
    import numpy as np

    out_dir = dist_dir / f"indicconformer-{lang}"
    onnx_path = out_dir / "model.int8.onnx"
    tokens_path = out_dir / "tokens.txt"

    if not onnx_path.exists():
        print(f"[ERROR] {onnx_path} not found. Run export first.")
        sys.exit(1)

    print(f"\n[{LANGUAGE_NAMES[lang]}] Verifying {onnx_path}")

    # ── Metadata check (this is what sherpa-onnx actually reads at load time) ──
    import onnx as onnx_lib
    onnx_model = onnx_lib.load(str(onnx_path))
    meta = {m.key: m.value for m in onnx_model.metadata_props}
    required = ["vocab_size", "normalize_type", "subsampling_factor", "model_type"]
    missing = [k for k in required if k not in meta]
    if missing:
        print(f"  [FAIL] Missing sherpa-onnx metadata keys: {missing}")
        print(f"         This WILL crash sherpa-onnx's native loader on-device.")
        print(f"         Re-run export (metadata step is now automatic) and re-pack/upload.")
        sys.exit(1)
    print(f"  [OK] sherpa-onnx metadata present: {meta}")

    # Load vocabulary
    vocab = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocab[int(parts[-1])] = parts[0]
    if str(len(vocab)) != meta.get("vocab_size"):
        print(f"  [WARN] tokens.txt has {len(vocab)} entries but metadata "
              f"vocab_size={meta.get('vocab_size')} — these must match.")

    # Load session, inspect declared input shapes/rank
    session = ort.InferenceSession(str(onnx_path))
    inputs = session.get_inputs()
    print(f"  Model inputs: {[(i.name, i.shape) for i in inputs]}")

    audio_input = inputs[0]
    rank = len(audio_input.shape)
    if rank == 3:
        # (batch, n_mels, time) — feature dim is usually the only concrete axis
        feat_dim = next((d for d in audio_input.shape if isinstance(d, int)), 80)
        time_frames = 100
        audio_batch = np.zeros((1, feat_dim, time_frames), dtype=np.float32)
        length = np.array([time_frames], dtype=np.int64)
        print(f"  Using placeholder features: shape=(1, {feat_dim}, {time_frames}) "
              f"— this is a load/shape smoke test, not a real transcript.")
    else:
        # Older/different export convention — fall back to raw-PCM-style 2D input.
        print(f"  Input rank is {rank} (expected 3 for mel features) — using legacy 2D fallback.")
        if test_wav:
            import wave, array
            with wave.open(test_wav) as wf:
                assert wf.getframerate() == 16000, "WAV must be 16kHz"
                assert wf.getnchannels() == 1, "WAV must be mono"
                raw = wf.readframes(wf.getnframes())
                pcm = array.array("h", raw)
                audio = np.array(pcm, dtype=np.float32) / 32768.0
        else:
            audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        audio_batch = audio[np.newaxis, :]
        length = np.array([audio.shape[0]], dtype=np.int64)

    try:
        outputs = session.run(None, {inputs[0].name: audio_batch, inputs[1].name: length})
    except Exception as e:
        length_f = length.astype(np.float32)
        try:
            outputs = session.run(None, {inputs[0].name: audio_batch, inputs[1].name: length_f})
        except Exception as e2:
            print(f"  [FAIL] Forward pass crashed: {e2}")
            sys.exit(1)

    print(f"  Output shape: {outputs[0].shape}")
    print(f"  [OK] Model loads and runs without error (metadata + graph sane).")
    if test_wav and rank != 3:
        print(f"  (ran legacy 2D path against {test_wav} — see WARN above)")


def make_archive(lang: str, dist_dir: Path):
    """Package model.int8.onnx + tokens.txt into a tar.gz for GitHub Releases."""
    out_dir = dist_dir / f"indicconformer-{lang}"
    archive_name = f"indicconformer-{lang}-int8.tar.gz"
    archive_path = dist_dir / archive_name

    print(f"\n[{LANGUAGE_NAMES[lang]}] Packing -> {archive_path}")
    subprocess.run(
        ["tar", "-czf", str(archive_path), "-C", str(out_dir),
         "model.int8.onnx", "tokens.txt"],
        check=True
    )
    print(f"  Archive: {archive_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Upload to GitHub Releases as: {archive_name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export AI4Bharat IndicConformer → sherpa-onnx ONNX (int8)"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--lang", choices=ALL_LANG_CODES,
                       help="Export a single language (e.g. te, ta, hi, bn, kn, ...)")
    group.add_argument("--all", action="store_true",
                       help="Export all 22 languages")
    parser.add_argument("--verify", metavar="LANG", choices=ALL_LANG_CODES,
                        help="Verify an already-exported model (optional WAV via --wav)")
    parser.add_argument("--wav", metavar="PATH",
                        help="Test WAV (16kHz mono) for --verify; defaults to synthetic tone")
    parser.add_argument("--pack", action="store_true",
                        help="After export, create tar.gz archives for GitHub Releases upload")
    parser.add_argument("--dist", default="dist",
                        help="Output directory (default: dist/)")
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX opset version (default: 14)")
    parser.add_argument("--hf-token", metavar="TOKEN",
                        help="HuggingFace access token (or set HF_TOKEN env var). "
                             "Required for gated AI4Bharat models.")

    args = parser.parse_args()
    dist_dir = Path(args.dist)

    if args.verify:
        check_imports()
        verify_language(args.verify, dist_dir, args.wav)
        return

    if not args.lang and not args.all:
        parser.error("one of --lang, --all, or --verify is required")

    check_imports()
    hf_login(args.hf_token)

    langs = list(MODELS.keys()) if args.all else [args.lang]
    for lang in langs:
        export_language(lang, dist_dir, opset=args.opset)
        if args.pack:
            make_archive(lang, dist_dir)

    print("\n-- Next steps ---------------------------------------------------")
    for lang in langs:
        print(f"  1. Verify:  python scripts/export_indicconformer.py --verify {lang} "
              f"--wav test_fixtures/audio/{LANGUAGE_NAMES[lang].lower()}_hello.wav")
        print(f"  2. Pack:    python scripts/export_indicconformer.py --lang {lang} --pack")
        print(f"  3. Upload:  dist/indicconformer-{lang}-int8.tar.gz  -> GitHub Releases")
    print()


if __name__ == "__main__":
    main()
