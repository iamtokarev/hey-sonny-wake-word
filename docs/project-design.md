good# Project Design

This document defines the purpose, boundaries, and high-level direction of the
Hey Sonny wake-word project. Detailed implementation findings and operational
notes are maintained separately in
[hf-jobs-openwakeword-options.md](hf-jobs-openwakeword-options.md).

## Project Summary

The project will produce a personal wake-word model that recognizes the phrase
"Hey Sonny". The model will be based on the
openWakeWord training approach and exported in ONNX format for lightweight local
inference.

Training will run on Hugging Face Jobs rather than relying on a local GPU. Data,
intermediate artifacts, and released models will be stored outside the temporary
Job environment so that training runs can be inspected and reproduced.

## Goals and Success Criteria

- Produce a working "Hey Sonny" wake-word model for personal use with Reachy Mini.
- Detect the phrase reliably across realistic speakers, distances, rooms, and
  background conditions while keeping false activations acceptably low.
- Evaluate model quality with held-out audio and tests using the target device,
  not only synthetic training results.
- Make training runs reproducible through versioned code, configuration,
  environment definitions, and run metadata.
- Preserve valuable generated data and training outputs independently of the
  lifetime of a Hugging Face Job.
- Keep the workflow understandable through tested scripts and focused notebooks
  for data and model exploration.

## Scope

The project includes:

- Preparing positive "Hey Sonny" examples, confusing phrases, general negative
  speech, and environmental audio needed for training and evaluation.
- Building a repeatable pipeline for data preparation, training, evaluation,
  and ONNX export.
- Running pilot and full training workloads on Hugging Face Jobs.
- Persisting datasets, intermediate artifacts, run metadata, and final models
  in appropriate Hugging Face storage.
- Exploring audio, model outputs, and evaluation results through notebooks.

The initial project does not include:


- Commercial deployment or a public model release.
- Distributed or multi-GPU optimization.
- Additional deployment formats unless ONNX proves insufficient for the target
  runtime.

## Deliverables

- A version-controlled source repository containing the pipeline, configuration,
  tests, environment definition, and project documentation.
- Focused notebooks for inspecting audio, validating artifacts, and analyzing
  model thresholds and errors.
- A persistent collection of generated data, features, logs, and run manifests.
- A curated evaluation dataset that remains separate from training inputs.
- A validated ONNX model with its configuration, provenance, and evaluation
  results in a Hugging Face Model repository.
- Instructions for running the model and validating it with Reachy Mini.

## High-Level Workflow

1. Define representative training and held-out evaluation data for "Hey Sonny."
2. Generate or collect the required audio and validate its quality.
3. Run a small end-to-end pilot to verify the environment, pipeline, and
   persistence behavior.
4. Prepare the full feature set and train one or more candidate models.
5. Evaluate candidates on held-out audio, inspect errors, and choose an operating
   threshold.
6. Test the selected candidate with the Reachy Mini microphone in realistic
   conditions.
7. Publish the validated ONNX artifact and its supporting metadata to the private
   model repository.

## Technical Approach

- **Model and training foundation:** openWakeWord and its custom-model training
  workflow.
- **Implementation:** Python modules and command-line scripts as the executable
  source of truth.
- **Exploration:** Jupyter notebooks that call the same modules used by the
  scripts instead of duplicating training logic.
- **Environment:** a pinned Docker image so local checks, pilot runs, and full
  runs use a consistent software stack.
- **Compute:** GPU-backed Hugging Face Jobs, beginning with a small pilot before
  allocating resources to a full run.
- **Working storage:** a private Hugging Face Storage Bucket for mutable generated
  data, features, checkpoints, logs, and other intermediate artifacts.
- **Versioned artifacts:** private Hugging Face Dataset and Model repositories for
  curated data and validated model releases.
- **Source control:** Git, with a private remote repository during development.
- **Deployment format:** ONNX for local inference and integration.
- **Access policy:** private by default; public release requires a separate privacy
  and licensing review.

## Milestones

1. **Project foundation** — establish source control, repository structure,
   environment definition, and basic validation tooling.
2. **Pipeline pilot** — complete a small training run and confirm that its inputs,
   outputs, and metadata persist correctly.
3. **First candidate** — prepare the full dataset and train an initial model.
4. **Evaluation and refinement** — measure real performance, inspect failure
   cases, and refine data or training as needed.
5. **Reachy Mini validation** — select a threshold and verify behavior on the
   target microphone and device.
6. **Private release** — publish the validated ONNX model, evaluation summary,
   provenance, and operating instructions.
