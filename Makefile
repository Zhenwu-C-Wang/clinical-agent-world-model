PYTHON ?= python

.PHONY: verify smoke reports check-reports release-check

verify:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -c "import subprocess, sys; subprocess.run([sys.executable, '-c', 'import clinical_world_model; import clinical_world_model.planner; print(clinical_world_model.__file__)'], cwd='/tmp', check=True)"
	$(PYTHON) scripts/generate_trajectories.py --count 40 --seed 42 --output /tmp/clinical_world_trajectories.jsonl
	$(PYTHON) scripts/train_world_model.py --input /tmp/clinical_world_trajectories.jsonl --seed 42 --output /tmp/clinical_world_model_eval.md
	$(PYTHON) scripts/run_planner.py --training-data /tmp/clinical_world_trajectories.jsonl --count 20 --seed 42 --horizon 3 --output /tmp/clinical_world_planner.md

reports:
	$(PYTHON) scripts/run_baselines.py --count 1000 --seed 42 --output reports/baseline_results.md
	$(PYTHON) scripts/train_world_model.py --input data/synthetic_trajectories.jsonl --seed 42 --output reports/world_model_eval.md
	$(PYTHON) scripts/run_planner.py --training-data data/synthetic_trajectories.jsonl --count 1000 --seed 42 --horizon 3 --output reports/planner_results.md
	$(PYTHON) scripts/run_stress_eval.py --training-data data/synthetic_trajectories.jsonl --count 500 --seed 99 --horizon 3 --output reports/stress_results.md

check-reports:
	$(PYTHON) scripts/check_reports_fresh.py

release-check: smoke verify check-reports
