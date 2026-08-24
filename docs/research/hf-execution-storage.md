# Hugging Face execution and storage

Research snapshot: 2026-08-24. Scope: the Hugging Face platform design needed to train and preserve an openWakeWord “Hey Sonny” model. Model internals, data selection, evaluation methodology, and deployment are covered by other research tracks.

## Recommended project architecture

Use a **Docker-based Hugging Face Job** for both the pilot and the full run. The openWakeWord training stack has enough native and GPU-sensitive dependencies that a pinned image is a more reliable contract than installing the environment at Job startup. Run the same image digest and entry point at every scale.

Separate storage by responsibility:

| Component | Role in this project |
| --- | --- |
| Git source repository | Training code, Dockerfile, lock files, configurations, tests, and notebooks. |
| Dataset repositories | Versioned, curated input or evaluation datasets that should be reproducible and documented. Mount them read-only. |
| Private Storage Bucket | Mutable run state: generated clips, cached features, checkpoints, logs, manifests, and candidate outputs. Mount it read-write. |
| Model repository | Versioned release of the validated ONNX model, configuration, metrics, checksums, and model card. |
| Job ephemeral disk | Hot working directory and cache for repeated or random reads during a run. It is not durable. |

This division matches the Hub’s intended storage model: repositories provide version history and collaboration features, whereas Buckets are non-versioned working storage that supports overwrite and deletion in place ([Storage Buckets](https://huggingface.co/docs/hub/storage-buckets)). Jobs can mount model and Dataset repositories read-only and Buckets read-write; mounted files are fetched lazily and cached on ephemeral disk ([Jobs volumes](https://huggingface.co/docs/hub/jobs-configuration#volumes), [large datasets on Jobs](https://huggingface.co/docs/hub/jobs-large-datasets#mount-a-dataset-model-or-bucket)).

Do not use the Bucket as the release registry. Each run should write under a unique prefix, for example `runs/<run-id>/`, and only a verified candidate should be copied or pushed into a versioned Dataset or Model repository.

## Run lifecycle

1. **Preflight.** Confirm authentication, positive compute balance, image availability, Bucket access, input visibility, GPU detection, dependency imports, and a small write/read-back test. Hugging Face requires login with permission to start and manage Jobs; Jobs are available to users or organizations with a positive credit balance ([authentication](https://huggingface.co/docs/hub/jobs-configuration#authentication), [pricing](https://huggingface.co/docs/hub/jobs-pricing)).
2. **Pilot.** Run the real pipeline on a small T4 flavor and a small sample. The pilot must use the production image, mounts, manifest format, and persistence path. Its purpose is to measure compatibility, storage behavior, runtime, and resource use—not model quality.
3. **Full run.** Use `l4x1` as the initial full-run candidate if the pilot confirms the need for its 400 GB ephemeral disk and additional GPU memory. Current listed prices are `t4-small` $0.40/hour, `t4-medium` $0.60/hour, and `l4x1` $0.80/hour, billed per minute while Starting or Running. Hardware and prices are mutable, so check `hf jobs hardware` immediately before submission ([current hardware and pricing](https://huggingface.co/docs/hub/jobs-pricing#pricing)).
4. **Verify persistence.** Validate counts, formats, checksums, and completion markers from outside the Job. `COMPLETED` only proves that the command exited successfully; it does not prove the expected artifacts were persisted ([persisting results](https://huggingface.co/docs/hub/jobs-manage#persist-your-results)).
5. **Promote.** Push curated data to a private Dataset repository and the accepted ONNX candidate to a private Model repository. Record the source commit, image digest, input revisions, configuration hash, Job ID, hardware, and Bucket run prefix in the release metadata.

Dataset and Model mounts are publication inputs, not writable release destinations. Repository promotion must use the Hub API or CLI and therefore needs an `HF_TOKEN` with the relevant repository write permission and repository-creation permission if the target does not yet exist. A Bucket mount is authorized when the Job is created, so the script does not need a token merely to write through that mount ([Jobs result persistence](https://huggingface.co/docs/hub/jobs-manage#persist-your-results)).

## Operational safeguards

- **Secrets:** pass `HF_TOKEN` as a Job secret, never as a normal environment variable, command argument, configuration value, or logged value. The bare CLI form `--secrets HF_TOKEN` resolves the locally authenticated token and encrypts it server-side ([environment variables and secrets](https://huggingface.co/docs/hub/jobs-configuration#environment-variables-and-secrets)).
- **Timeout and cost:** always set an explicit timeout. The default is 30 minutes. Billing stops when the timeout stops the Job; cancel irrelevant Jobs promptly. Hugging Face documents per-minute pricing but no project-specific spending cap, so the timeout is the primary per-run cost guard ([pricing recommendations](https://huggingface.co/docs/hub/jobs-pricing#recommendations)).
- **Observability:** give every Job a stable name/label, print the run ID and final metrics, and use the Job page or `hf jobs inspect`, `hf jobs logs`, and `hf jobs stats`. Logs are retained after termination and can preserve key results if publication fails ([managing Jobs](https://huggingface.co/docs/hub/jobs-manage)).
- **Durability:** write important stage outputs to the Bucket before the next expensive stage. Write to temporary names, validate, then create an explicit completion marker; do not equate file existence with completeness.
- **Recovery:** design for phase-level restart. The Job filesystem and its local cache disappear at termination, while Bucket outputs survive. A retry should reuse validated earlier-stage artifacts and create a new attempt record rather than overwrite the prior manifest.
- **Cache and throughput:** mounts are lazy and cache reads on ephemeral disk. For this audio/feature workload, copy repeatedly accessed inputs into a run-local cache when disk capacity permits, then persist only durable outputs. Whether direct mounted access is fast enough is a pilot measurement, not a documented guarantee.
- **CLI compatibility:** volume mounting currently requires `huggingface_hub >= 1.8.0`; pin the local submission tool as well as the container dependencies ([volume requirements](https://huggingface.co/docs/hub/jobs-configuration#volumes)).
- **Storage budget:** private Buckets, Dataset repositories, and Model repositories all count toward Hub storage policy. The current PRO allowance includes 1 TB of private storage before pay-as-you-go overage; confirm the account’s live billing page before retaining large generated corpora ([storage limits](https://huggingface.co/docs/hub/storage-limits)).

## Audit of the previous HF note

The main conclusions in `docs/hf-jobs-openwakeword-options.md` remain valid:

- Docker rather than the default UV environment for the serious training path.
- A cheap T4 compatibility/pilot run before an L4 full run.
- Positive credit balance rather than a mandatory paid subscription tier.
- A private Bucket for mutable artifacts and a private Model repository for the release.
- Read-only Dataset/Model mounts, read-write Bucket mounts, ephemeral Job filesystems, encrypted secrets, explicit timeouts, and independent output verification.
- `huggingface_hub >= 1.8.0` for volume support.

Items to treat as recommendations rather than confirmed facts:

- The existing runtime and dollar estimates are planning guesses; only a pilot can establish this pipeline’s duration and total cost.
- `l4x1` is a sensible first full-run flavor, not a proven requirement. The pilot may show that `t4-medium` is adequate or that CPU, RAM, or disk—not GPU memory—is the constraint.
- The proposed phase timeouts are starting limits, not platform guarantees.
- Using local ephemeral storage as the hot training filesystem is technically motivated, but the best caching/copy strategy depends on measured Bucket-mount throughput and access patterns.
- Mid-training resume behavior belongs to the openWakeWord implementation audit. Hugging Face preserves Bucket checkpoints, but it cannot make the training code resume from them automatically.

The earlier hardware subset and prices are still current as of this snapshot, but the platform now lists additional flavors. A static table should not be treated as authoritative; use the live CLI or Jobs hardware API before each paid run. The observation about uncertain outbound-Internet guarantees remains unresolved by the public documentation; essential dependencies and large inputs should therefore be available through the pinned image or Hub storage.

## Unknowns that require the pilot

- Whether the pinned image starts successfully on both T4 and L4, with CUDA visible to every required runtime.
- Cold-start time for image pull, lazy mounts, and initial dependency/data access.
- Read throughput and cache growth when openWakeWord reads many audio files and large feature arrays from mounts.
- Peak ephemeral-disk, RAM, GPU-memory, and network usage for the intended pilot scale.
- Which GPU flavor provides the best cost/runtime trade-off for the full run.
- Whether interruption leaves files that look complete and which phase boundaries need stronger atomicity.
- Whether direct Internet access to any non-Hub data source is reliable enough; the preferred design is not to depend on it during the paid run.
- Actual end-to-end runtime and therefore the correct timeout and cost envelope.

These questions do not require more desk research. They should become explicit observations in one small end-to-end pilot before committing to the full dataset and training run.

## Primary sources

- [Jobs overview and quickstart](https://huggingface.co/docs/hub/jobs)
- [Jobs configuration: Docker, authentication, secrets, volumes, and hardware](https://huggingface.co/docs/hub/jobs-configuration)
- [Jobs pricing and billing](https://huggingface.co/docs/hub/jobs-pricing)
- [Managing Jobs, logs, failure inspection, and persistence](https://huggingface.co/docs/hub/jobs-manage)
- [Processing large datasets and mount caching](https://huggingface.co/docs/hub/jobs-large-datasets)
- [Storage Buckets and repository comparison](https://huggingface.co/docs/hub/storage-buckets)
- [Hub storage limits](https://huggingface.co/docs/hub/storage-limits)
