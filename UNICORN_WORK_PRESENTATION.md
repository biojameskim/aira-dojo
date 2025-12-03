# UNICORN Branch Work Summary
**Presentation for Lab Meeting**
*Work Period: October 21 - November 19, 2025*

---

## Executive Summary

Over the past ~4 weeks, I've extended the AIRA-Dojo framework with two novel MCTS-based solvers, established a complete experimental infrastructure on the Cornell Unicorn cluster, and conducted extensive benchmarking experiments. The work comprises:

- **12,261 insertions, 93 deletions** across 111 files
- **2 new solver algorithms** with full implementations
- **Complete cluster integration** with comprehensive documentation
- **Baseline experiments** on MLEBench tasks with results analysis

---

## 1. Novel Solver Implementations

### 1.1 SA-UCB: Semantic Aware Upper Confidence Bound

**What it is:** An MCTS variant that incorporates LLM-based semantic scoring into node selection.

**Mathematical Innovation:**
```
SA-UCB(v) = Q_v/N_v + C·√(ln(N_parent)/N_v) + λ·P_v + γ·U_v
            └────────────────────┬────────────────────┘   └─────┬─────┘
                    Standard UCB                        Semantic Bonus
```

**Key Components:**
- **Promise Scorer (P)**: LLM evaluates how promising a solution approach appears (0-1 score)
- **Novelty Scorer (U)**: LLM evaluates solution diversity compared to siblings (0-1 score)
- Hyperparameters λ (promise weight) and γ (novelty weight) control semantic influence

**Implementation Details:**
- Core solver: `src/dojo/solvers/mcts/sa_ucb.py` (538 lines)
- Extended node class tracking promise/novelty scores and reasoning
- Comprehensive logging of all scorer outputs to `scorer_logs/`
- Full configuration system with dataclasses

**Files Added/Modified:**
- `src/dojo/solvers/mcts/sa_ucb.py` (new, 538 lines)
- `src/dojo/configs/solver/mlebench/sa_ucb.yaml` (new)
- `SA_UCB_DOCS.md` (new, 258 lines of documentation)

### 1.2 P-MCTS: Policy-Guided Monte Carlo Tree Search

**What it is:** MCTS enhanced with learned or heuristic policies to guide exploration.

**Key Features:**
- Policy network/heuristic guides node expansion and selection
- Balances exploration with policy priors
- Designed for MLEBench machine learning competition tasks

**Implementation Details:**
- Core solver: `src/dojo/solvers/p_mcts/p_mcts.py` (868 lines)
- New operator set for policy-guided actions
- Configuration dataclass system

**Files Added/Modified:**
- `src/dojo/solvers/p_mcts/p_mcts.py` (new, 868 lines)
- `src/dojo/config_dataclasses/solver/p_mcts.py` (new)
- `src/dojo/configs/solver/mlebench/p_mcts.yaml` (new)

---

## 2. LLM-Based Operators and Agents

### 2.1 New Operators

**Judge Operator:**
- Evaluates solution quality and provides feedback
- Implementation: `src/dojo/core/solvers/operators/judge.py` (146 lines)
- Config: `src/dojo/configs/solver/operators/mlebench/aira_operators/judge.yaml`

**Promise Scorer:**
- Semantic evaluation of solution promise
- Considers: method appropriateness, theoretical foundation, evaluation strategy
- Config: `src/dojo/configs/solver/operators/mlebench/aira_operators/promise_scorer.yaml` (83 lines)
- Uses Gemini 2.5-flash-lite for efficient scoring

**Novelty Scorer:**
- Evaluates solution diversity and uniqueness
- Compares approaches across sibling nodes
- Config: `src/dojo/configs/solver/operators/mlebench/aira_operators/novelty_scorer.yaml` (87 lines)
- Uses Gemini 2.5-flash-lite for efficient scoring

### 2.2 LLM Backend Updates

- Switched operators to use `gemini-2.5-flash-lite` for cost efficiency
- Added Gemini API configurations across multiple operator configs
- Updated all AIRA operators (draft, improve, debug, crossover)

---

## 3. UNICORN Cluster Integration

### 3.1 Complete Setup Documentation

**Created:** `UNICORN_SETUP.md` (186 lines)

**Covers:**
1. **Initial Setup**: Apptainer installation, environment configuration
2. **Python Environment**: PyTorch CUDA compatibility for RTX 6000 Ada GPUs
3. **Environment Variables**: Logging, data directories, Apptainer settings
4. **Apptainer Sandbox**: Custom bootstrap script for cluster limitations
5. **SLURM Configuration**: Job submission, resource allocation
6. **Task-Specific Configs**: MLEBench integration

**Key Technical Solutions:**
- Apptainer non-root execution workarounds
- Filesystem layout adaptations for Unicorn
- CUDA 12.8 compatibility for RTX 6000 Ada
- Shared storage configuration (`/share/j_sun/`)

### 3.2 Infrastructure Code Changes

**Modified Sandbox Script:**
- `src/dojo/core/interpreters/jupyter/sand` (+146 lines)
- Added non-root Apptainer compatibility
- Custom filesystem mount handling

**SLURM Integration:**
- `src/dojo/config_dataclasses/launcher/slurm.py` (+7 lines)
- Updated runtime limits to 10 hours for long experiments

**Environment Configuration:**
- `.env_default` (+19 lines, -14 deletions)
- Updated paths for Unicorn cluster
- Added Google API key configuration

---

## 4. Experimental Framework and Results

### 4.1 Baseline Experiments

**Created:** `run_three_tasks_local.py` (279 lines)
- Automated local testing script for quick iteration
- Runs 3 MLEBench tasks: configured for rapid prototyping

**Experiment Configurations:**
- `src/dojo/configs/_exp/run_three_tasks.yaml` (new)
- `src/dojo/configs/_exp/run_mlebench_aira_pmcts.yaml` (new)
- `src/dojo/configs/_exp/run_mlebench_aira_sa_ucb_gdm.yaml` (new)
- `src/dojo/configs/_exp/run_mlebench_aira_mcts_gdm.yaml` (new)

**Task Splits:**
- `src/dojo/tasks/mlebench/splits/three_tasks.txt` (new)
- `src/dojo/tasks/mlebench/splits/lite_test.txt` (new, 6 tasks)

### 4.2 Experimental Results

**AIRA MCTS Baselines:**
- Statistics for 3 lite tasks (`0817774`)
- Vanilla MCTS comparison baseline (`2b9d099`)

**P-MCTS Results:**
- Grading report for APTOS2019 blindness detection (seeds 1-3)
- File: `src/dojo/analysis_utils/tree_stats/aptos2019_grading_report.json`
- Contains detailed performance metrics

**Documentation:**
- Added grading reports (`258012c`, `3281e4f`)
- Comparative analysis between solvers

---

## 5. Infrastructure Improvements

### 5.1 Time Management

**Commit:** `79ca5ba` - "Add better time management (ends correctly at specified time)"
- Ensures experiments terminate properly at specified runtime limits
- Critical for SLURM job management on shared cluster

### 5.2 Logging Enhancements

**Commit:** `1d8cbab` - "update logging"
- Improved logging for MCTS tree searches
- Better tracking of LLM operator calls

**Meta-Error Summary:**
- `src/dojo/analysis_utils/meta_error_summary.py` (+3 lines)
- Enhanced error aggregation and analysis

### 5.3 Visualization Fixes

**Commit:** `05a4acf` - "Update error where tree visualization file wasn't being created"
- Fixed bug preventing tree visualization exports
- Updated: `src/dojo/core/solvers/utils/tree_export.py` (+11 lines, -11 deletions)
- Updated: `src/dojo/core/solvers/utils/search_exporter.py` (+20 lines, -20 deletions)

### 5.4 Experiment Artifact Management

**Commit:** `3551b4e` - "Increase agent runtime limit to 10 hours and clean up experiment artifacts"
- Extended runtime for complex MLEBench tasks
- Improved cleanup of temporary experiment files

---

## 6. Configuration System Enhancements

### 6.1 MLEBench Task Configurations

**New experiment configs:**
- `aide_greedy_gdm.yaml` (21 lines) - AIDE baseline with greedy search
- `aira_evo_gdm.yaml` (49 lines) - AIRA evolutionary approach
- `aira_greedy_gdm.yaml` (16 lines) - AIRA greedy baseline
- `aira_mcts_gdm.yaml` (45 lines) - AIRA MCTS with Gemini
- `aira_sa_ucb_gdm.yaml` (13 lines) - SA-UCB configuration

### 6.2 Promise/Novelty Client Configuration

**Added:** `src/dojo/configs/solver/client/promise_novelty.yaml` (5 lines)
- Centralized configuration for semantic scoring LLMs
- Enables easy switching between LLM backends

### 6.3 Benchmark Configuration

**Added:** `src/dojo/configs/benchmark/mlebench/three_tasks.yaml` (7 lines)
- Quick test suite for development
- Faster iteration during solver development

---

## 7. Timeline and Development Phases

### Phase 1: Foundation (Oct 21-22)
- UNICORN cluster setup and documentation
- Gemini API integration
- Environment configuration

### Phase 2: AIRA MCTS Baselines (Oct 25-26)
- Baseline experiments on 3 lite tasks
- `run_three_tasks_local.py` script development
- Initial grading reports

### Phase 3: Policy-Guided MCTS (Oct 26)
- P-MCTS solver implementation (868 lines in one day!)
- Operator set configuration
- Documentation

### Phase 4: Experimental Iteration (Oct 27-28)
- Vanilla MCTS baseline runs
- Time management improvements
- Logging enhancements
- P-MCTS APTOS2019 experiments

### Phase 5: SA-UCB Development (Nov 7-19)
- SA-UCB solver implementation
- Promise and novelty scorer operators
- Comprehensive documentation (SA_UCB_DOCS.md)
- Runtime limit extensions
- Visualization bug fixes
- LLM backend optimization (switch to flash-lite)

---

## 8. Key Metrics and Statistics

### Code Contribution
- **Total changes:** 12,261 additions, 93 deletions
- **Files modified:** 111 files
- **New major implementations:** 2 solvers (1,406 lines combined)
- **New operators:** 3 (judge, promise_scorer, novelty_scorer)
- **Documentation:** 444 lines (UNICORN_SETUP.md + SA_UCB_DOCS.md)

### Major New Files
1. `src/dojo/solvers/p_mcts/p_mcts.py` - 868 lines
2. `src/dojo/solvers/mcts/sa_ucb.py` - 538 lines
3. `run_three_tasks_local.py` - 279 lines
4. `SA_UCB_DOCS.md` - 258 lines
5. `UNICORN_SETUP.md` - 186 lines
6. `src/dojo/core/solvers/operators/judge.py` - 146 lines

### Experiments Run
- AIRA MCTS baselines (3 lite tasks)
- Vanilla MCTS comparison
- P-MCTS on APTOS2019 (seeds 1-3)
- SA-UCB development experiments

---

## 9. Next Steps and Future Work

### Short-term
1. **Complete SA-UCB evaluation** on full MLEBench benchmark suite
2. **Hyperparameter tuning** for λ (promise weight) and γ (novelty weight)
3. **Comparative analysis** between vanilla MCTS, P-MCTS, and SA-UCB

### Medium-term
1. **Multi-task learning** experiments across diverse MLEBench tasks
2. **Ablation studies** on semantic scoring components
3. **Operator optimization** for promise/novelty scoring efficiency

### Long-term
1. **Publication preparation** for SA-UCB algorithm
2. **Integration with other benchmarks** beyond MLEBench
3. **Meta-learning** approaches for λ and γ adaptation

---

## 10. Technical Challenges Solved

### Cluster Integration
- Non-root Apptainer execution in shared HPC environment
- CUDA 12.8 compatibility for RTX 6000 Ada GPUs
- Filesystem access across shared storage (`/share/`)

### LLM Integration
- Efficient semantic scoring with Gemini flash-lite
- Prompt engineering for promise/novelty evaluation
- Comprehensive logging of LLM reasoning

### Algorithmic Innovation
- Balancing semantic scores with traditional UCB exploration
- Preventing semantic scores from dominating selection
- Efficient sibling comparison for novelty scoring

### Software Engineering
- Modular operator system for LLM agents
- Flexible configuration with Hydra dataclasses
- Comprehensive visualization and logging

---

## Appendix: Commit Timeline

| Date | Commits | Focus |
|------|---------|-------|
| Oct 21 | 5 | UNICORN setup, documentation, configs |
| Oct 22 | 1 | README updates |
| Oct 25-26 | 10 | AIRA baselines, P-MCTS implementation |
| Oct 27-28 | 4 | Vanilla MCTS, time management, grading |
| Nov 7 | 1 | Runtime extensions |
| Nov 10 | 1 | Visualization fixes |
| Nov 19 | 2 | SA-UCB implementation, LLM optimization |

**Total:** 26 commits spanning ~4 weeks

---

## Questions for Discussion

1. **SA-UCB hyperparameters**: What's the optimal balance between semantic (λ, γ) and exploration (C) weights?
2. **Evaluation metrics**: Should we add task-specific metrics beyond MLEBench scores?
3. **Operator design**: Are there other semantic signals worth incorporating?
4. **Computational budget**: Trade-offs between LLM calls for scoring vs. additional MCTS rollouts?
5. **Generalization**: How well do these solvers transfer across different task types?

---

## Acknowledgments

- Cornell Unicorn cluster for computational resources
- Akanksha Sarkar's unicorn fork for initial cluster setup reference
- AIRA-Dojo framework by Facebook Research
- MLEBench benchmark by OpenAI
