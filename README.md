# TriadCare

TriadCare is a multi-agent clinical framework for the follow-up of patients with
cognitive decline. A coordinator agent (DoctorAgent) answers a clinician request. It
consults three specialist agents. Each specialist reads one source of patient data:

- **EHR Agent** reads the electronic health record through a FHIR R4 server.
- **CareAgent** reads the patient and caregiver check-ins in the Patient Knowledge Ledger.
- **Radiologist Agent** requests MRI hippocampal volume measurements from an inference
  service.

A governance layer verifies each citation in an answer against the sources that the agents
retrieved. A stage-level provenance trace records each source from retrieval to the final
answer.

This repository contains the code and the scenario sources of the article "TriadCare:
Tracing Evidence Loss and Verifying Citations Across Agent Handoffs in Home Monitoring of
Cognitive Decline" (Applied Sciences, MDPI).

DoctorAgent request path. Each specialist reads one data source. The governance layer
checks the citations before the answer goes to the clinician.

```text
clinician request
      |
      v
+-------------+   consult   +-------------------+     +--------------------+
| DoctorAgent |------------>| EHR Agent         |---->| HAPI FHIR server   |
|             |------------>| CareAgent         |---->| PostgreSQL ledger  |
|             |------------>| Radiologist Agent |---->| MRI inference      |
+------+------+             +-------------------+     +--------------------+
       |
       v
citation verification (governance layer) --> answer + provenance trace
```

## Repository contents

The repository holds source files only. You generate all data files with the steps in
[Step 5](#step-5-generate-the-scenario-data).

| Path | Contents |
|---|---|
| `src/agent/` | DoctorAgent, specialist agents, citation verification |
| `src/fhir/` | FHIR tools, FHIR fixture loader |
| `src/ledger/` | Patient Knowledge Ledger, check-in recaps, search |
| `src/imaging/` | Imaging catalog, MRI gateway client, MRI warm-up script |
| `src/llm/` | OpenRouter client |
| `src/prompt/` | System prompts of the agents |
| `src/evaluation/` | Experiment runner, judge, scoring, reports, comparisons, sensitivity |
| `services/mri-inference/` | MRI gateway: token check, study allowlist, hippodeep run, agent-facing result |
| `docker/` | Docker Compose file for PostgreSQL, HAPI FHIR, and the MRI service |
| `scenarios/cases/` | The 18 authored scenario cases (one YAML file for each case) |
| `scenarios/manifest.yaml` | List of the cases and their split (development or test) |
| `scenarios/build.py` | Builder of the scenario data and the reference release |
| `scenarios/synthea_background.py` | Generator of the synthetic background EHR history |
| `scenarios/fetch_dlbs.py` | Download script for the MRI series |
| `eval/reference_units/held_out.bank.json` | Judge question bank and quality rubric |

All patient data is synthetic. The MRI series come from a public dataset of healthy
adults (see [Step 3](#step-3-download-the-mri-series)).

### Scenario cases

One case file holds all the authored facts of one synthetic patient:

- the patient, the caregiver, and the clinical record (conditions, medications,
  observations, notes);
- the imaging studies and the DLBS series of each study;
- the check-in transcripts between CareAgent and the patient or the caregiver;
- the reference units: the facts that an answer must keep, with their sources;
- the clinician tasks, in order, with the units that each task requires.

`manifest.yaml` puts 8 cases in the development split and 10 cases in the test split.
The article reports the test split only (10 scenarios, 76 tasks).

## Requirements

- Python 3.12 or 3.13 and [Poetry](https://python-poetry.org/) 2
- Java 17 or later (for Synthea)
- Docker with Docker Compose
- An [OpenRouter](https://openrouter.ai/) API key (agents and judge)
- An API key for an OpenAI-compatible embeddings endpoint (CareAgent search)
- About 12 GB of free disk space (Docker images, Synthea, MRI series)

## Setup

Do the steps in this order. Run each command from the repository root.

### Step 1. Install the Python dependencies

```bash
poetry install
```

### Step 2. Make the configuration file

```bash
cp .env.example .env
```

Fill in the empty values in `.env`:

- `OPENROUTER_API_KEY`: an [OpenRouter API key](https://openrouter.ai/settings/keys).
  The agents, the check-in recaps, the search rerank, and the judge send all model
  requests through OpenRouter with this key. Add credit to the OpenRouter account first.
- `EMBEDDING_API_KEY`: a key for the embeddings endpoint in `EMBEDDING_BASE_URL`. The
  default endpoint is the OpenAI API (`text-embedding-3-small`), so use an
  [OpenAI API key](https://platform.openai.com/api-keys).
- `POSTGRES_PASSWORD`: a password for the local PostgreSQL container.
- `DATABASE_URL`: `postgresql://triadcare:<POSTGRES_PASSWORD>@127.0.0.1:5434/triadcare`.
- `IMAGING_BEARER_TOKEN`: a random secret that the agents and the MRI service share, for
  example the output of `openssl rand -hex 32`.

### Step 3. Download the MRI series

The MRI service reads wave-1 T1-weighted series of the Dallas Lifespan Brain Study
(DLBS), OpenNeuro dataset [ds004856](https://openneuro.org/datasets/ds004856/versions/1.3.0)
version 1.3.0, CC0 license. This command downloads 11 series (about 72 MB) to
`data/dlbs/`:

```bash
poetry run python scenarios/fetch_dlbs.py
```

Each synthetic patient uses the series of one DLBS subject with a similar sex and age.
DLBS is a healthy-aging cohort, so the images do not show the dementia that the synthetic
records describe.

### Step 4. Download Synthea

[Synthea](https://github.com/synthetichealth/synthea) (Apache-2.0) generates the older
background history of each synthetic EHR. The article data uses the Synthea build of
commit `bd8a52b`. Download that build:

```bash
mkdir -p data/synthea
curl -L -o data/synthea/synthea.jar \
    https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar
```

Then check its digest:

```bash
shasum -a 256 data/synthea/synthea.jar
```

The digest must be
`20098ec70c24c715ea874b19140983634711cc4986196105b5b0a9ca72a4b644`. The
`master-branch-latest` file changes when Synthea publishes a new build. If the digest is
different, build the jar from source at commit `bd8a52b`:

```bash
git clone https://github.com/synthetichealth/synthea data/synthea/src
git -C data/synthea/src checkout bd8a52b
(cd data/synthea/src && ./gradlew uberJar)
cp data/synthea/src/build/libs/synthea-with-dependencies.jar data/synthea/synthea.jar
```

A jar that you build has a different digest. Step 5.3 checks the generated data instead.

### Step 5. Generate the scenario data

Data flow from the sources to the generated files:

```text
scenarios/cases/*.yaml ---+----------------------------+
                          |                            |
data/synthea/synthea.jar -+-> synthea_background.py    |
                                    |                  |
                                    v                  v
                          fixtures/background/ ---> build.py <--- held_out.bank.json
                                                       ^
data/dlbs/ (MRI series, digests only) -----------------+
                                                       |
        +-------------------+--------------------+-----+--------------------+
        v                   v                    v                          v
  fixtures/fhir/      generated/           fixtures/              eval/reference_units/
  (synthetic EHR)     conversations/       imaging-workflow.json  releases/triadcare/
                      (check-ins)          (study catalog)        + CURRENT.json
```

1. Generate the background EHR history. Synthea runs with a fixed seed for each case, so
   each run gives the same output. The step takes about 4 minutes. If `java` on your
   `PATH` is older than version 17, set `JAVA` to the path of a Java 17 executable.

   ```bash
   poetry run python scenarios/synthea_background.py
   ```

   The script writes `scenarios/fixtures/background/<case>.json`.

2. Build the FHIR fixtures, the check-in conversations, the imaging catalog, and the
   reference release. The check-in conversations are scripted in the case files, so this
   step makes no model calls. The build gives the same files as the article release:

   ```bash
   poetry run python scenarios/build.py
   ```

   The script writes these files:

   | Path | Contents |
   |---|---|
   | `scenarios/fixtures/fhir/{dev,held_out}/<case>.json` | Synthetic FHIR R4 records: the authored record plus the background history |
   | `scenarios/generated/conversations/{dev,held_out}/<case>.json` | Check-in sessions for the Patient Knowledge Ledger |
   | `scenarios/fixtures/imaging-workflow.json` | Patient-to-study bindings with the SHA-256 digest of each MRI series |
   | `eval/reference_units/releases/triadcare/` | Reference units, task plans, unit origins, question bank, and the manifest of digests |
   | `eval/reference_units/CURRENT.json` | Pointer to the active release and the digest of its manifest |

3. Check the result. `eval/reference_units/CURRENT.json` must contain this
   `manifest_sha256`:

   ```text
   2bd15b648388c7bf66741fa03d22a619f6791043ce8af23c89c8f5a697fe293e
   ```

   The manifest records the SHA-256 digest of each generated file. Thus the same manifest
   digest shows that each generated file is the same as in the article release.

To check one case without a full build, run
`poetry run python scenarios/build.py --check scenarios/cases/<case>.yaml`.

### Step 6. Start the services

Start PostgreSQL (with pgvector), HAPI FHIR, and the MRI service:

```bash
docker compose --env-file .env -f docker/compose.yml up -d --build --wait
```

`--wait` returns when PostgreSQL and the MRI service are healthy. HAPI FHIR needs about
one minute more to start. Wait until it answers:

```bash
until curl -sf http://127.0.0.1:8090/fhir/metadata >/dev/null; do sleep 5; done
```

The services listen on `127.0.0.1`: PostgreSQL on port 5434, HAPI FHIR on port 8090, and
the MRI service on port 8101. Stop other services that use these ports first. The Compose
file pins HAPI FHIR to the image digest of the article runs.

On the first start, `docker/postgres-init.sql` creates the pgvector extension. The script
runs only when the data volume is empty. If you reuse an old volume, run
`CREATE EXTENSION IF NOT EXISTS vector;` in the `triadcare` database yourself.

The MRI image build downloads [hippodeep](https://github.com/bthyreau/hippodeep_pytorch)
(code and trained weights) at commit `310305f` and CPU-only PyTorch. The first build needs
Docker BuildKit (the default in current Docker versions) and network access.

### Step 7. Warm up the MRI service

Analyze the MRI studies of the test split one time after each start of the MRI service.
The service keeps each result in memory, so no evaluation run waits for an analysis:

```bash
poetry run python -m src.imaging.prewarm
```

You do not load the EHR or the check-ins manually. For each scenario, the runner loads
the FHIR fixtures into HAPI FHIR and imports the check-ins into a new ledger schema.

## Reproduce the article results

The article uses these frozen settings:

| Item | Value |
|---|---|
| Reference release | `triadcare`, test split (10 scenarios, 76 tasks; the runner drops tasks W1 and W2) |
| Conditions | `full`, `full_ungoverned`, `single_agent` |
| Agent and recap model | `openai/gpt-6-luna`, medium reasoning effort, 16,384 maximum output tokens, 64-request budget |
| Judge | `anthropic/claude-haiku-5.5` on OpenRouter, low reasoning effort, 3 votes |
| Runs | 3 runs for each condition |

The three conditions:

- `full`: DoctorAgent with the three specialists and the governance layer.
- `full_ungoverned`: the same agents without the citation repair round and without the
  attached sources.
- `single_agent`: one agent that uses all the tools.

The runner refuses to start when a release file, conversation file, FHIR fixture, or the
imaging catalog does not match the digests in the release manifest. The test split also
requires `--confirm-test`.

1. Run each condition three times (nine runs):

   ```bash
   for condition in full full_ungoverned single_agent; do
     for run in 1 2 3; do
       poetry run python -m src.evaluation.run --split test --confirm-test \
           --condition "$condition" --models openai/gpt-6-luna \
           --out "output/evaluation/$condition-r$run"
     done
   done
   ```

   The output directory names must have the form `<condition>-r<n>`. The compare and
   sensitivity commands find the runs by these names. To run one scenario only, add
   `--scenarios <scenario_id>`.

2. Score and report each run directory:

   ```bash
   for dir in output/evaluation/*-r[123]; do
     JUDGE_TRANSPORT=openrouter JUDGE_MODEL=anthropic/claude-haiku-5.5 \
         JUDGE_REASONING_EFFORT=low \
         poetry run python -m src.evaluation.score "$dir" --votes 3
     poetry run python -m src.evaluation.report "$dir"
   done
   ```

   `report` reads every `scores.json` under the directory that you give it. To pool the
   three runs of one condition, put them in one directory and report that directory.

   The scorer saves each accepted judge answer in the run directory. A second score of
   the same run uses the saved answers. Use `--cached-only` to score with saved answers
   only, without judge requests.

3. Compare the conditions. The 95% intervals come from a bootstrap over scenarios:

   ```bash
   poetry run python -m src.evaluation.compare output/evaluation full single_agent
   poetry run python -m src.evaluation.compare output/evaluation full full_ungoverned
   poetry run python -m src.evaluation.compare output/evaluation full_ungoverned single_agent
   ```

   The third comparison is optional.

4. Run the sensitivity analyses (judge vote rules, vote ties, truncation of the attached
   sources, matched-unit and rubric transitions):

   ```bash
   poetry run python -m src.evaluation.sensitivity output/evaluation \
       eval/reference_units/releases/triadcare/test
   ```

Model outputs are not deterministic. A new run gives different answers. The scores can
differ from the article values within the reported intervals.

## Scoring

The scorer uses two types of measures:

- **Deterministic checks.** Code checks the citations, the source identifiers, and the
  provenance trace of each answer.
- **Model judge.** The judge checks each reference unit of the release against the answer.
  It also scores answer quality against the sources that the answer cites. It does not
  check the rest of the patient record. Each unit question has three votes. The majority
  decides the status of the unit.

All denominators come from the release. Tasks that did not run, or that the judge did not
score, stay in the denominator.

## Provenance of this release

The article runs used internal commit `1cbe97bf` with uncommitted changes. This
repository is that working tree without the generated data, the run outputs, the
clinician web application, the patient chat API, the PET and DTI services (no scenario
study uses them), the development tests, and experiment scripts that the article does not
use.

The code of the evaluated pipeline is not changed, with these exceptions:

- The judge supports only the OpenRouter transport, which the article uses.
- `build.py` makes the imaging catalog from the cases, without an earlier catalog file,
  and writes the release timestamp of the article release.
- `synthea_background.py` reads the Java path from `JAVA`.
- The MRI service has only the analysis route and the health route. The article runs
  used the same analysis code with a vendored copy of hippodeep that is byte-identical to
  the upstream commit. The repository does not contain hippodeep; the image build
  downloads it. The new image gives the same volumes and confidence as the article runs.

We checked this repository against the article data and runs:

- Steps 5.1 and 5.2 give files that are byte-identical to the article data.
- `report`, `compare`, and `sensitivity` give the same tables as the article runs.
- `score --cached-only` gives the same score files.

## Development

```bash
poetry run ruff check
```

## Authors

Minh-Kien Ha, Myungeun Lee, Duy-Phuong Dao, and Hyung-Jeong Yang (corresponding author),
Chonnam National University, Republic of Korea.

## License

MIT. See `LICENSE`. hippodeep (Thyreau et al., Medical Image Analysis, 2018,
doi:10.1016/j.media.2017.11.004), Synthea (Apache-2.0), and the DLBS data (CC0) are not
part of this repository. Their own terms apply.
