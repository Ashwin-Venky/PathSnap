# Team handoff

Prepared before the first project commit. These are proposed review and maintenance responsibilities. All implementation and runs were prepared with AI assistance. No teammate review, test, modification or authorship is asserted. Record actual review or changes in each PR. The full project is runnable locally. The first public commit contains Ashwin’s files only.

| Owner | Responsibility | Exact file list | Dependencies | Branch and suggested commit | Check and explain |
|---|---|---|---|---|---|
| bvarun-blip | Data, EDA and approximate labels | 20 paths in the **bvarun-blip** section below | Original ZIP and research environment | `data/alignment` · Add trip audit, candidate geometry and reference alignment | `test_geometry.py`. Compare archive fingerprints, occurrence order, training recall and 50 m alignment coverage. Explain bidirectional ties and the 153,064 s gap. |
| Tejas6499 | Classifier comparison, TPE and export | 105 paths in the **Tejas6499** section below | Data handoff, prepared arrays and research environment | `models/grouped-tuning` · Add grouped classifier experiments and Bayesian tuning | `test_models.py`. Inspect five disjoint folds, candidate distribution, trial 17, binary versus conditional loss and the joblib fingerprint. |
| ashrithahegde10 | Directed decoder and route evaluation | 9 paths in the **ashrithahegde10** section below | Data handoff, fitted model and metrics module | `routes/viterbi` · Add directed route reconstruction and decoder ablation | `test_decoder.py`. Trace shortest-path arc IDs and impossible transitions. Explain why accuracy on trip 00000077 fell while aggregate continuity improved. |
| Ashwin-Venky | API, UI, integration and report | 42 paths in the **Ashwin-Venky** section below | Full local review snapshot for integration; merged branches for a fresh public clone | `main` · Add PathSnap serving interface, report and team handoff | `test_api.py`. Match ten HTTP request IDs, inspect label isolation, inspect screenshots and reproduce cold-process fingerprints. Complete physical acceptance with the team. |

## Transfer and integration order

Ashwin bootstraps main. Each teammate accepts their own GitHub invitation, uses their own account and Git identity, and branches from main. Merge data/alignment first, models/grouped-tuning second, and routes/viterbi third. Ashwin owns shared report/figure generation and deployment configuration. Varun owns the single shared algorithm settings file. Tejas owns metrics and the model artifact. Ashritha owns decoder settings.

From the complete local project run `python scripts/handoff.py bundle`. Privately transfer **.cache/team-review.zip** within this team. The archive contains only the exact reviewed project paths in config/handoff.json. It excludes .git, environments, raw organisers’ files, working labels, and evaluator references. Do not publish or share this bundle with other teams. Obtain original inputs through the organisers.

On each teammate’s machine, clone the verified repository into a **new** folder. Copy team-review.zip into the parent folder of that clone. The commands below unpack this known same-team archive only into the fresh clone. Use a different clone if you already have edits. The full overlay lets everyone run integration checks before every branch has merged. Stage only the assigned manifest paths. Run commands from a shell with an activated Python 3.13 research environment. Git Bash works on Windows. `gh api user --jq .login` should show your own username. Inspect `git var GIT_AUTHOR_IDENT` before committing.

## Commands for each owner

### bvarun-blip

```sh
git clone https://github.com/Ashwin-Venky/PathSnap.git PathSnap-review
cd PathSnap-review
git switch -c data/alignment
python -m zipfile -e ../team-review.zip .
python -m pip install -r config/research.txt
gh api user --jq .login
git var GIT_AUTHOR_IDENT
python -m unittest discover -s tests -p test_geometry.py -v
python scripts/handoff.py stage bvarun-blip
git diff --cached --stat
git diff --cached -- . ":(exclude)*.npz" ":(exclude)*.joblib" ":(exclude)*.sqlite3"
python scripts/handoff.py check bvarun-blip
git commit -m "Add trip audit, candidate geometry and reference alignment"
git push -u origin data/alignment
gh pr create --base main --head data/alignment --web
```

Compare archive fingerprints, occurrence order, training recall and 50 m alignment coverage. Explain bidirectional ties and the 153,064 s gap.

Exact paths owned by bvarun-blip:

```text
config/settings.json
experiments/alignment.csv
experiments/candidate_pilot.csv
experiments/feature_identity.json
experiments/fit_distribution.json
experiments/input_identity.json
experiments/inventory.csv
experiments/manifest.csv
experiments/quality.csv
figures/gap.png
figures/student_test_eda.png
figures/train_eda.png
scripts/audit.py
scripts/prepare.py
scripts/quality.py
scripts/unpack.py
src/alignment.py
src/data.py
src/geometry.py
tests/test_geometry.py
```

### Tejas6499

```sh
git clone https://github.com/Ashwin-Venky/PathSnap.git PathSnap-review
cd PathSnap-review
git switch -c models/grouped-tuning
python -m zipfile -e ../team-review.zip .
python -m pip install -r config/research.txt
gh api user --jq .login
git var GIT_AUTHOR_IDENT
python -m unittest discover -s tests -p test_models.py -v
python scripts/handoff.py stage Tejas6499
git diff --cached --stat
git diff --cached -- . ":(exclude)*.npz" ":(exclude)*.joblib" ":(exclude)*.sqlite3"
python scripts/handoff.py check Tejas6499
git commit -m "Add grouped classifier experiments and Bayesian tuning"
git push -u origin models/grouped-tuning
gh pr create --base main --head models/grouped-tuning --web
```

Inspect five disjoint folds, candidate distribution, trial 17, binary versus conditional loss and the joblib fingerprint.

Exact paths owned by Tejas6499:

```text
artifacts/model.joblib
artifacts/model.json
experiments/folds.json
experiments/models.csv
experiments/runs/DecisionTree/configuration.json
experiments/runs/DecisionTree/cv0.csv
experiments/runs/DecisionTree/cv0.npz
experiments/runs/DecisionTree/cv1.csv
experiments/runs/DecisionTree/cv1.npz
experiments/runs/DecisionTree/cv2.csv
experiments/runs/DecisionTree/cv2.npz
experiments/runs/DecisionTree/cv3.csv
experiments/runs/DecisionTree/cv3.npz
experiments/runs/DecisionTree/cv4.csv
experiments/runs/DecisionTree/cv4.npz
experiments/runs/DecisionTree/student_test.csv
experiments/runs/DecisionTree/student_test.npz
experiments/runs/DecisionTree/train_refit.csv
experiments/runs/DecisionTree/train_refit.npz
experiments/runs/GaussianNB/configuration.json
experiments/runs/GaussianNB/cv0.csv
experiments/runs/GaussianNB/cv0.npz
experiments/runs/GaussianNB/cv1.csv
experiments/runs/GaussianNB/cv1.npz
experiments/runs/GaussianNB/cv2.csv
experiments/runs/GaussianNB/cv2.npz
experiments/runs/GaussianNB/cv3.csv
experiments/runs/GaussianNB/cv3.npz
experiments/runs/GaussianNB/cv4.csv
experiments/runs/GaussianNB/cv4.npz
experiments/runs/GaussianNB/student_test.csv
experiments/runs/GaussianNB/student_test.npz
experiments/runs/GaussianNB/train_refit.csv
experiments/runs/GaussianNB/train_refit.npz
experiments/runs/HistBoost/configuration.json
experiments/runs/HistBoost/cv0.csv
experiments/runs/HistBoost/cv0.npz
experiments/runs/HistBoost/cv1.csv
experiments/runs/HistBoost/cv1.npz
experiments/runs/HistBoost/cv2.csv
experiments/runs/HistBoost/cv2.npz
experiments/runs/HistBoost/cv3.csv
experiments/runs/HistBoost/cv3.npz
experiments/runs/HistBoost/cv4.csv
experiments/runs/HistBoost/cv4.npz
experiments/runs/HistBoost/student_test.csv
experiments/runs/HistBoost/student_test.npz
experiments/runs/HistBoost/train_refit.csv
experiments/runs/HistBoost/train_refit.npz
experiments/runs/Logistic/configuration.json
experiments/runs/Logistic/cv0.csv
experiments/runs/Logistic/cv0.npz
experiments/runs/Logistic/cv1.csv
experiments/runs/Logistic/cv1.npz
experiments/runs/Logistic/cv2.csv
experiments/runs/Logistic/cv2.npz
experiments/runs/Logistic/cv3.csv
experiments/runs/Logistic/cv3.npz
experiments/runs/Logistic/cv4.csv
experiments/runs/Logistic/cv4.npz
experiments/runs/Logistic/student_test.csv
experiments/runs/Logistic/student_test.npz
experiments/runs/Logistic/train_refit.csv
experiments/runs/Logistic/train_refit.npz
experiments/runs/RandomForest/configuration.json
experiments/runs/RandomForest/cv0.csv
experiments/runs/RandomForest/cv0.npz
experiments/runs/RandomForest/cv1.csv
experiments/runs/RandomForest/cv1.npz
experiments/runs/RandomForest/cv2.csv
experiments/runs/RandomForest/cv2.npz
experiments/runs/RandomForest/cv3.csv
experiments/runs/RandomForest/cv3.npz
experiments/runs/RandomForest/cv4.csv
experiments/runs/RandomForest/cv4.npz
experiments/runs/RandomForest/student_test.csv
experiments/runs/RandomForest/student_test.npz
experiments/runs/RandomForest/train_refit.csv
experiments/runs/RandomForest/train_refit.npz
experiments/runs/Tuned/configuration.json
experiments/runs/Tuned/cv0.csv
experiments/runs/Tuned/cv0.npz
experiments/runs/Tuned/cv1.csv
experiments/runs/Tuned/cv1.npz
experiments/runs/Tuned/cv2.csv
experiments/runs/Tuned/cv2.npz
experiments/runs/Tuned/cv3.csv
experiments/runs/Tuned/cv3.npz
experiments/runs/Tuned/cv4.csv
experiments/runs/Tuned/cv4.npz
experiments/runs/Tuned/student_test.csv
experiments/runs/Tuned/student_test.npz
experiments/runs/Tuned/train_refit.csv
experiments/runs/Tuned/train_refit.npz
experiments/summary.csv
experiments/training.log
experiments/trials.csv
experiments/tuned.csv
experiments/tuning.sqlite3
figures/model_comparison.png
figures/tuning.png
scripts/experiment.py
src/metrics.py
src/models.py
tests/test_models.py
```

### ashrithahegde10

```sh
git clone https://github.com/Ashwin-Venky/PathSnap.git PathSnap-review
cd PathSnap-review
git switch -c routes/viterbi
python -m zipfile -e ../team-review.zip .
python -m pip install -r config/research.txt
gh api user --jq .login
git var GIT_AUTHOR_IDENT
python -m unittest discover -s tests -p test_decoder.py -v
python scripts/handoff.py stage ashrithahegde10
git diff --cached --stat
git diff --cached -- . ":(exclude)*.npz" ":(exclude)*.joblib" ":(exclude)*.sqlite3"
python scripts/handoff.py check ashrithahegde10
git commit -m "Add directed route reconstruction and decoder ablation"
git push -u origin routes/viterbi
gh pr create --base main --head routes/viterbi --web
```

Trace shortest-path arc IDs and impossible transitions. Explain why accuracy on trip 00000077 fell while aggregate continuity improved.

Exact paths owned by ashrithahegde10:

```text
artifacts/decoder.json
experiments/ablation.csv
experiments/first_phase.json
experiments/profiles.csv
experiments/route_summary.csv
scripts/ablation.py
scripts/slice.py
src/decoder.py
tests/test_decoder.py
```

### Ashwin-Venky

Ashwin’s first branch is main in the newly initialised repository. These are the bootstrap commands from the complete local workspace. For later changes, create `integration/final-review` from the merged main branch.

```sh
git branch --show-current
git remote -v
gh api user --jq .login
git var GIT_AUTHOR_IDENT
python -m unittest discover -s tests -v
python scripts/handoff.py stage Ashwin-Venky
git diff --cached --stat
git diff --cached -- README.md src/api.py ui.py scripts/report.py
python scripts/handoff.py check Ashwin-Venky
git commit -m "Add PathSnap serving interface, report and team handoff"
git push -u origin main
```

Match ten HTTP request IDs, inspect label isolation, inspect screenshots and reproduce cold-process fingerprints. Complete physical acceptance with the team.

Exact paths owned by Ashwin-Venky:

```text
.gitattributes
.gitignore
HANDOFF.md
README.md
config/development.json
config/handoff.json
config/research.txt
config/server.txt
config/ui.txt
experiments/http.csv
experiments/reload.json
experiments/runtime.json
experiments/server_health.json
experiments/server_requests.jsonl
experiments/sources.json
experiments/terminal_requests.json
experiments/tests.log
experiments/ui_runs.json
experiments/video_review.json
figures/server_terminal.png
figures/student_test_map.png
figures/train_map.png
figures/ui.png
reports/PathSnap_Technical_Report.pdf
scripts/browser_check.py
scripts/environment.py
scripts/figures.py
scripts/handoff.py
scripts/integration.py
scripts/provenance.py
scripts/reload_check.py
scripts/report.py
scripts/serving_manifest.py
src/api.py
src/maps.py
src/predict.py
src/private_score.py
src/vendor/Leaflet-LICENSE.txt
src/vendor/leaflet.css
src/vendor/leaflet.js
tests/test_api.py
ui.py
```

## Ashwin’s review and merge

Use a separate clean review clone. Keep the original workspace intact while teammates have uncommitted files. For each PR number in the order above, inspect the changed paths against config/handoff.json, read the author’s actual review notes, and check their tests. Run the complete project tests and the HTTP/UI checks after all three merges.

```sh
git fetch origin
gh pr list --base main
gh pr view PR_NUMBER --json author,headRefName,files
gh pr diff PR_NUMBER
gh pr checkout PR_NUMBER
python -m unittest discover -s tests -v
gh pr merge PR_NUMBER --merge
git switch main
git pull --ff-only
```

Early branches may need the review bundle for modules from later branches. After the third merge, verify a fresh clone with no source overlay, inspect the PDF, verify public access, and run the physical two-device acceptance checklist in README.
