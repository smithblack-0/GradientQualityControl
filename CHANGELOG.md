# Changelog

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