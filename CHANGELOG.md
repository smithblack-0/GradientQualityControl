# Changelog

## 0.7.4

- Added the NormWarmupAutoScheduler object and associated tests, which is considerably more intuitive to use as it automatically calibrates to the warmup target. Updated readme and various other documents to match.

## 0.7.3

- The cleaners now cache the mean last grad norm before clearing the grads, making it much easier for users to identify what they were. See mean_last_grad_norms.
- Readme clarifies this is a component and the scope more quickly
- The cleaners now return "last_grad_norm" from statistics, indicating the mean last grad norm.
- Added 'target_initial_norm_threshold' argument to the GNTS and GNS controller types. This should not affect existign code, but lets one easily define what a torch controller means when they say 'choose a schedule value of 1.0'

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