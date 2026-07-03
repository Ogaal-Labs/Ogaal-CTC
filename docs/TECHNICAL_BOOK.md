# Ogaal CTC Technical Book

![Ogaal CTC Architecture](figures/ogaal_ctc_architecture.png)

## 1. Summary

`Ogaal CTC` is a Somali automatic speech recognition system released by Ogaal Labs.

It combines a Wav2Vec2/XLSR CTC acoustic model with the published `transcript_only` LM decode path for practical Somali transcription workflows.

## 2. Ogaal Labs Positioning

Ogaal Labs focuses on local datasets and practical AI tools for Somali and African communities. The purpose of this release is to package a usable Somali CTC transcription system that developers can run immediately.

`Ogaal CTC` was intentionally trained and packaged for Somali speech recognition. English was not part of the training objective for this release.

## 3. Training Data

The training pool used for the frozen package contains:

- train clips: `39604`
- train hours: `72.1`
- source mix: `balanced_gold=3424, hardcase_recorded_400=400, ddd_kenya_somali_68hrs_mozilla=35780`

A private Ogaal Labs collection pipeline contributed a core part of this effort through roughly `5,000` curated prompts recorded by `19` speakers across varied genders, accents, and speaking styles.

## 4. System Design

- model family: `Wav2Vec2 / XLSR-300M`
- architecture: `Wav2Vec2ForCTC`
- published decode path: `transcript_only`
- train samples: `39604`
- validation samples: `429`
- test samples: `428`
- freeze feature encoder: `true`
- bf16: `true`
- fp16: `false`
- warmup steps: `396`

The release configuration pairs the acoustic model with a bundled LM decoder so developers can use the same path that produced the public metrics.

## 5. Published Somali Results

Held-out validation with the published decode path:

- WER: `0.2179`

Held-out Somali test with the published decode path:

- WER: `0.2114`
- CER: `0.0997`

These fixed-system results are the primary public metrics for the release.

## 6. Product Workflow

The public code release includes:

- a file and folder inference CLI
- a local browser demo for microphone recording or audio upload
- a released decode path that matches the public metrics

- chunked inference for longer audio

The browser demo is not live streaming. The workflow is:

1. record
2. stop
3. transcribe

## 7. Public Release Wording

Use wording close to:

> A Somali automatic speech recognition system from Ogaal Labs, built around a Wav2Vec2/XLSR CTC acoustic model and released with its published LM decode path for practical Somali transcription workflows.

This phrasing keeps the release aligned with the way developers will actually use it.

## 8. Ogaal Labs

Ogaal Labs website:

- `https://ogaallabs.com/`
