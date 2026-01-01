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

## Document-Driven Development (DDD)

Document-Driven Development is a contract-based methodology where **documentation is the design medium**. Instead of designing in code and documenting afterward, you design by writing Natural Language Formal Specifications, then implement code to conform to those specifications.

### Core Workflow

```
Documentation (Design) → Tests (Validation) → Implementation (Execution)
```

**The key inversion:** Documentation defines what must be true, tests verify those properties hold, and implementation is constrained to satisfy both. Code cannot deviate from the contract without failing tests.

### Why DDD Works

**1. Documentation is Cheap to Refactor**
- Change a function signature in docs: 30 seconds
- Change a function signature in code: refactor all call sites, update tests, fix breaks
- Design iteration happens in the cheap phase (documentation), not expensive phase (code)

**2. Architecture Emerges from Constraints**
- Traditional: Guess architecture upfront → discover it doesn't fit → expensive refactoring
- DDD: Document high-level contracts → notice patterns ("these all need X") → abstract naturally
- Progressive refinement: Each iteration narrows solution space, converging on correct architecture
- By implementation phase, architecture has already been pressure-tested against requirements

**3. Implementation Becomes Mechanical**
- All dependencies identified during planning
- All interfaces defined before coding
- Implementation task: "Fill in this contract to pass these tests"
- No architectural decisions, no scope creep, clear success criteria

**4. True Parallel Development**
- Bounded contracts can be filled by different people/teams simultaneously
- No coordination needed beyond honoring contracts
- Integration guaranteed if both sides satisfy their contracts

**5. Refactoring Freedom**
- Black box tests validate contracts, not implementation
- Can completely rewrite internals as long as contract preserved
- No fear of breaking hidden assumptions

### The Documentation Phase

Documentation is where **forks happen**. When documenting contract A, you realize "A needs something with properties X, Y, Z". You have three options:

**Option 1: Fork immediately**
1. Document contract B with public API providing X, Y, Z
2. Write tests for B
3. Implement B
4. Return to implementing A, injecting B

**Option 2: Stub for later**
1. Document contract B with vague behavior "does X, Y, Z somehow"
2. Must include public API (method signatures, properties)
3. Continue with A, filling in B later (or assign to someone else)

**Option 3: All-at-once**
1. Document both contracts A and B together
2. Define clear injection points
3. Proceed to testing phase

**Critical rule:** If you discover a missing dependency during **implementation**, you failed at planning. Forks should happen during documentation or test writing, never during implementation.

**Why forks happen during documentation:**
- You're designing the public interface
- To write tests, you need to know what to inject
- Dependencies become obvious when specifying behavior
- Cheap to add new contracts at this stage

### The Testing Phase

Tests validate contracts using **strict black box methodology**. Tests can also spawn forks when you realize "to test A's behavior, I need to inject B with these properties".

**Black Box Testing Rules:**

What you CAN test:
- Public methods and their documented behaviors
- Observable state (properties, return values)
- Injected dependencies usage (e.g., optimizer.step() called)
- Emergent behavior from following the contract
- Protected methods ONLY through documented subclass contracts

What you CANNOT test:
- Implementation details (how something works internally)
- Undocumented behavior
- Private state or mechanisms
- Specific functions doing specific things (test the contract, not the implementation)

**Example of black box vs white box:**
- ❌ White box: "test_take_optimizer_step_averages_gradients" - tests specific function implementation
- ✓ Black box: "test_using_optimizer_results_in_gradient_averaging" - tests emergent behavior from contract

**Why black box testing:**
- Complete decoupling: implementation can change freely
- Tests remain valid across refactors
- Forces clear contracts (if you can't test it black box, contract is unclear)
- Enables true parallel development

### The Implementation Phase

If documentation and testing were done correctly, implementation should be **mechanical**:

1. Read the contract
2. Look at the test suite
3. Write code that makes tests pass
4. Refactor internally as needed (tests protect you)

**Red flags during implementation:**
- Discovering new dependencies → You missed them during documentation
- Unclear what to implement → Contract is underspecified
- Tests don't cover behavior → Tests incomplete
- Need to change contract → Architecture mismatch (refactor docs, update tests, then implement)

**Green flags:**
- Clear what to build
- Tests give immediate feedback
- Can refactor freely
- Just filling in bounded contracts

### Roles in DDD

**Designer Role:**
- Write contracts (documentation)
- Design architecture through progressive refinement
- Identify and document dependencies
- Make architectural trade-offs explicit
- Requires domain expertise and architectural vision

**Auditor Role:**
- Review contracts for completeness and consistency
- Verify tests properly validate contracts (black box compliance)
- Identify ambiguities or underspecified behavior
- Check that implementation honors contracts
- LLMs can sometimes serve as auditors (spotting inconsistencies, checking coverage)

**Implementer Role:**
- Implement test suites from contracts (mechanical transcription)
- Implement code to pass tests (bounded problem solving)
- Identify contract ambiguities during implementation (escalate to designer)
- Refactor within contract bounds
- LLMs often make good implementers (excel at "implement to spec")

**Why this separation works:**
- Designing contracts requires expertise and vision (human-centric)
- Auditing can be partially automated (LLMs good at consistency checking)
- Implementation from clear specs is mechanical (LLMs/juniors effective)
- Each role has clear success criteria and boundaries

### When to Use DDD

**Good fit:**
- Stable or well-understood requirements
- Projects needing parallel development
- Code that will be maintained long-term
- When architecture quality matters
- Working with LLMs or distributed teams

**Poor fit:**
- Rapidly changing requirements (contracts invalidate quickly)
- Throwaway prototypes
- Exploratory research code
- Very small projects (overhead not worth it)

### Practical Notes

**Documentation as code:**
- Documentation is held to formal specification standards
- If it's not documented, it doesn't exist
- Inconsistencies in docs are bugs
- Version control docs with same rigor as code

**Progressive refinement:**
- Start with high-level contracts
- Refactor docs as patterns emerge
- "These three features all need X" → Abstract X into shared contract
- Narrow solution space iteratively

**Integration:**
- If both sides honor contracts, integration just works
- No "how do I call this?" questions
- No mismatched assumptions
- Contract is the integration specification 