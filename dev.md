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

---

# Development Practices

## Document-Driven Development (DDD) Workflow

Document-Driven Development is a contract-based development methodology where documentation defines formal specifications, and implementation must conform to those contracts. The workflow spawns bounded contracts (monads) that can be filled in immediately, later, or by other team members.

**Core Principle:**
Write the contract (documentation) → Write tests validating the contract → Implement to pass the tests

**The Fork-Based Workflow:**

When implementing a documented contract, you encounter dependencies:

```
Document Contract A
    ↓
Implement Tests for A (black box)
    ↓
Start Implementing A
    ↓
Realize: "A needs component B with properties X, Y, Z"
    ↓
    ├─→ Document Contract B (spawn new bounded contract)
    │       ↓
    │   Implement Tests for B (black box to B's contract)
    │       ↓
    │   Implement B to pass tests
    │       ↓
    │   Inject B into A ←──┘
    │
    ├─→ OR: Stub B, continue with A, fill B later
    │
    └─→ OR: Assign B to another team member
```

Each fork creates a **bounded contract** - a complete specification with:
- Documented public interface
- Black box tests validating the contract
- Implementation conforming to tests
- Clear injection points

**Example:**

1. Document `AbstractOptimizerWrapper` contract
2. Write tests validating the contract (this file)
3. Start implementing `AbstractOptimizerWrapper`
4. Realize: "I need a state management system with serialization"
5. **Fork**: Document `StateManager` contract separately
6. Write tests for `StateManager`
7. Implement `StateManager`
8. Inject `StateManager` into `AbstractOptimizerWrapper`
9. Continue implementing `AbstractOptimizerWrapper`

**Benefits:**
- Contracts can be filled asynchronously
- Team members can work on different forks independently
- Each contract is completely decoupled
- Implementation can be refactored freely within contract bounds
- Integration is guaranteed if both sides honor contracts

## Black Box Testing for Public Contracts

All **publicly documented contracts** MUST be tested using strict black box methodology.

**What you CAN test:**
1. **Public methods** - All documented public methods and their contracted behaviors
2. **Injected dependencies** - Correct usage of injected objects (e.g., optimizer.step() called, gradients modified)
3. **Observable state** - Properties, return values, and state accessible through public API
4. **Behavioral conditions** - All conditions under which documented behaviors occur
5. **Subclass contracts** - Protected methods tested ONLY through their documented contracts

**What you CANNOT test:**
1. **Implementation details** - How something works internally
2. **Undocumented behavior** - Anything not in the specification
3. **Private state** - Internal storage, fields, or mechanisms
4. **Non-contracted behavior** - If you want to test it, add it to the contract first

**Rationale:**
Testing the contract, not the implementation, ensures complete decoupling. Code can be refactored freely as long as the contract is preserved. This is what enables the fork-based DDD workflow - each bounded contract can evolve independently. 