A personal project to build a "Hey Sonny" wake-word model, based on the
openWakeWord training approach and exported to ONNX for local inference.

## What this is

An exploratory build, not a formal spec. The plan is to start in notebooks —
poking at data, trying augmentation, running small training experiments — and
only formalize things (scripts, config, Docker) once an approach actually
proves out. Deeper research on specific topics (data strategy, evaluation,
reproducibility, HF storage/execution, the openWakeWord baseline) lives in
`docs/research/` — treat those as reference to pull from, not requirements to
satisfy upfront.

## Approach

1. **Explore in notebooks.** Generate/collect sample audio, try training
   locally on small data, sanity-check the openWakeWord pipeline end to end.
2. **Iterate.** Adjust data (positive examples, negatives, augmentation) and
   training setup based on what the notebooks show.
3. **Formalize once it's worth it.** When an approach feels solid enough to
   scale up, turn the notebook logic into scripts/config worth rerunning.
4. **Move final training to Hugging Face Jobs.** Local exploration doesn't
   need GPU infra; the full/final training run does.
5. **Evaluate and pick a threshold** against held-out audio.
6. **Keep the final model** (+ eval results, provenance) in a private HF Model
   repo.

## Non-goals

- Public release or commercial use.
- Process/engineering rigor for its own sake — just enough to not lose work
  or have to redo it.
