# Changelog

## 0.9.9

**Human auditing and corrections stage**

The orchestrator main system tests were found to be DRAMATICALLY overcoupled, and had to be redone to emphasize testing of contract compliance and integration. The integration tests in abstract optimizer wrapper needed minor tweaks. 

- Refactored opchestrator_main_system.py dramatically, then audited and approved it
- Removed some black box contract violations from test_abstract_optimizer_wrapper.py, then approved it. 

## 0.9.8

**Human Auditing and Corrections Stage**

Fixed inconsitancy between documentation, propogating through distributed metrics subsystem test, involving typing; Used claude for patching with auditing. Generally, fixed the issues claude left behind. 


- test_step_management_subsystem.py: Now passes audit. Primarily we added tests for tensor cases, and made error types a bit more specific and clear. 
- test_distributed_management_subsystem.py: Like mentioned. Also narrowing allowed error types and better checking of error message contents. Fixed major test bug: We were not testing we were raising under the correct context. 
- Added scalar tensor tests into test_state_management_subsystems to keep to contract.
- Added into test_gradient_accumulation_step_subsystem.py specifications on the right way to store state for reporting. Identified and fixed major test issue - test for if gradient accumulation averages did not, in fact, test it averages. This has been fixed.
- Finished auditing test_reporting_subsystem.py. Spec is modified to aggregate even wrapper lists when possible; it reports STATISTICS after all. Tests modified to do so as well.
- Finished auditing test_optimizer_mocking_mixin.py. No major issues detected, though there will likely be bugs we bring up later. One minor issue was detected on how one of the tests was implemented regarding initialization order. Fixed. 

## 0.9.7

**Integration Tests with Distributed Synchronization and Test Organization**

- Implemented `test_abstract_optimizer_wrapper.py`: End-to-end integration tests with real subclassing
- Added distributed synchronization tests using `torch.multiprocessing` with CPU-only gloo backend
- Tests verify metric consensus across ranks and correct aggregation (mean for replicated, sum for sharded)
- Reorganized tests into `abstract_base_class/` and `concrete_implementations/` directories
- Updated `CLAUDE.md` with comment style guide (role-based names, complete sentences, no numbered enumeration)

## 0.9.6

**Test Suites for Remaining Subsystems**

- Implemented `test_gradient_accumulation_step_subsystem.py`: Tests counter management, gradient averaging, stepping mechanics
- Implemented `test_reporting_subsystem.py`: Tests statistics generation, aggregation, filtering
- Implemented `test_optimizer_mocking_mixin.py`: Tests attribute forwarding and initialization phases
- Implemented `test_orchestrator_main_system.py`: Integration tests for subsystem coordination via dependency injection

## 0.9.5

**Test Suites for State Management and Distributed Metrics, Architectural Fix**

- Implemented `test_state_management_subsystem.py` and `test_distributed_metrics_subsystem.py`
- Discovered aggregation logic was incorrectly placed in StateManagementSubsystem
- Refactored documentation to move aggregation to ReportingSubsystem where it belongs
- Updated `StateManagementSubsystem.get_state()` to return raw lists for optimizer params
- Updated `base_object_api.md` to maintain contract abstraction while reflecting architecture change

## 0.9.4

**Unfolded base_object_api into base_object_implementation**

- Decomposed AbstractOptimizerWrapper into 7 subsystems with implementation contracts in `base_object_implementation.md`:
  - StateManagementSubsystem: Handles all state storage and serialization
  - DistributedMetricsManagementSubsystem: Manages metric binding and distributed synchronization
  - GradientAccumulationStepSubsystem: Controls gradient accumulation mechanics and optimizer stepping
  - ReportingSubsystem: Generates statistics and vital statistics reports
  - OptimizerMockingMixin: Provides transparent optimizer duck-typing via attribute forwarding
  - OrchestratorMainSystem: Main facade coordinating subsystems via dependency injection
  - AbstractOptimizerWrapper: User-facing class with auto-construction

## 0.9.3

**Factory Function Test Coverage**

- Added comprehensive factory tests for all 9 factory functions across 5 optimizer wrappers:
  - `make_sbc_with_polynomial_schedule` and `make_sbc_with_polynomial_schedule_conventional_lr` (SBC)
  - `make_gnts_with_cosine_annealing_schedule` and `make_gnts_with_cosine_annealing_schedule_conventional_lr` (GNTS)
  - `make_gnr_with_cosine_annealing_schedule` and `make_gnr_with_cosine_annealing_schedule_conventional_lr` (GNR)
  - `make_gns_with_cosine_annealing_schedule` and `make_gns_default` (GNS)
  - `make_mht_with_warmup_schedule` (MHT)
- Tests validate return types (OptimizerWrapper, SynchronousSchedule)
- Tests verify schedule values match ScheduleAnything builtin formulas at key training steps
- Tests confirm wrapper configuration parameters (max_batch_draws, distributed_mode)
- All factory tests follow black box methodology with no inline imports

## 0.9.2

**Test Infrastructure Rewrite (Black Box Methodology)**

- Completely rewrote all 5 optimizer wrapper test suites (SBC, GNTS, GNR, MHT, GNS)
- Migrated from Mock-based testing to real PyTorch optimizers
- Integrated ScheduleAnything for proper schedule target binding
- Tests now validate documented contracts only, never implementation details
- Removed duplicate functionality following DRY/SOLID principles:
  - Statistics testing delegated to base class tests
  - Gradient accumulation testing delegated to base class tests
  - Child tests focus on optimizer-specific behavior only
- Added minimal statistics smoke tests to verify API exists
- Fixed all import paths and terminology (optimizer/base_optimizer, never "wrapper")

## 0.9.1

**Base Class API and Contract Updates**

- Added `distributed_mode` parameter to base class for distributed training support (replicated/sharded modes)
- Updated `step()` signature to accept `*args, **kwargs` for algorithms requiring additional inputs (e.g., MHT metric)
- Fixed terminology throughout: "Implementation Pattern" → "Subclassing Pattern" (contracts document subclassing, not implementation)
- Added internal utilities documentation for shared gradient norm computation
- Added `mode` parameter to OptimizerWrapperGNR: "global" (default) or "independent" scaling modes
  - Global mode: computes norm across all parameters, scales uniformly
  - Independent mode: scales each parameter to target norm separately
- Updated base class tests for `distributed_mode` parameter validation
- Updated base class tests for `step()` signature flexibility with `*args, **kwargs`

## 0.9.0

- BREAKING CHANGE: Rebuilding using Documentation Driven Development and ScheduleAnything
- Complete rebuild of all documentation
- Onto library rebuild

## 0.8.6

- Bugfix: Optimizer wrapper did not actually behave like wrapped optimizer. Fixed.
- A wide variety of changes to the readme. 

## 0.8.5

- Bugfix: SBC should always start with a learning rate of 1.0 for the scheduling system to work properly. Obsoleted manual setter. 

## 0.8.4

State update. You can now use .state_dict and .load_state_dict to resume training seamlessly.
- Minor other documentation changes. 

## 0.8.3

- Example is installed.
- A few more documentation changes. 

## 0.8.2

- statistic "last_step_num_draws" added.

## 0.8.1

- Addition of the get_batch_curved_schedule scheduler utility function for general batch seek performance, rather than restriction to only the quadratic curve. No api violations. 
- Minor formatting and typing issues fixed. 
- Moderate changes and streamlining of the readme.

## 0.8.0

Change made that breaks existing norm scheduling design, and as such we advance subversion. Still in beta, so this is okay, and is demonstratably superior from writing the examples.

- New utils get_direct_cosine_annealing_with_warmup, get_norm_threshold_cosine_annealing_with_warmup, get_quadratic_batch_schedule designed to provide direct solutions for practitioners for the schedules needed
- Usage is to provide necessary info and provide the wrapped optimizer, and it attaches and provides reasonable default schedules. 
- Users can define their own custom lambda schedules if desired still.


## 0.7.3

- The cleaners now cache the mean last grad norm before clearing the grads, making it much easier for users to identify what they were. See mean_last_grad_norms.
- Readme clarifies this is a component and the scope more quickly
- The cleaners now return "last_grad_norm" from statistics, indicating the mean last grad norm.

## 0.7.2 

Continous deployment YAML pushes. System should now publish when a pull request is accepted with the release tag.

## 0.7.2

- Modified ruff, black. 
- Included working CI/CD including workflow modifications
- -Added format_code.py root script to format files automatically. 

## 0.7.1 

- Included CD
- Added CHANGELOG.md 

## 0.7.0 - 2025-11-16

Initial deployment of the research as a deployable production library.

- Various Gradient Cleaners, including the expected flagship Gradient Norm Threshold Scheduler version.
- Unit tests for all
- Documentation and examples structure
- Research files structure.
- Standard repository features.
- CI preliminary work.

Note that as the effect has been fairly robustly tested, this version is mostly done, 