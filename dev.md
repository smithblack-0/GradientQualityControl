# Things I need before release

* Installable Package: pip install gradient-quality-control works
* License: MIT probably.
* Documentation Site: Document how to use it, what it does, a bit of theory behind it
* Research Site: Document why it might be working, what is currently know, what anomalies have been observed. It currently could either be automatically tuning a set of hyperparameters at once, or actually lowering the noise flor and raising convergence.
* API Site: Probably still read the docs. Automatically generated and pulled according to file structure.
* Reproducible Benchmarks: We are making outlandish claims. We need ready-to-go colabs with pinned costs for reproduction. These need to be straightforward enough to be manually audited.
* Collaboration Guidelines: I want to publish, anyone want to help? Can someone help run some of the experiments in hyperparamter-seek mode? Resources appreciated. 
* CI: Tests are shown passing. 
* Versioning/Stability process: If it is not in experimental, api changes will never invalidate existing code except if underlying libraries change. 