# Development Practices

## Document-Driven Development (DDD)

This project uses documentation driven development.

### Big Picture

Traditional software development faces a fundamental tension: designing architecture requires working at high abstraction levels with broad context, while implementing code requires working at low abstraction levels with specific details. Attempting both simultaneously produces cognitive overload and poor results. Teams typically resolve this by designing in code, discovering architectural problems during implementation when changes are expensive, then either accumulating technical debt using shortcuts or doing costly refactors. In parallel with this, good code requires documentation, which is usually done after-the-fact by a separate team and is often out of date.

Document-Driven Development inverts this priority. Correct documentation is more important than code, and code is compiled to the documentation. We can observe that will well-written documentation one can generate the objects themselves - this, when paired with modern LLM technology or a legion of code monkeys allows the documentation to be the primary source of truth which is progressively unfolded from the big picture into implementation details. The code is implemented according to a specific workflow that forms from api requirements documentation, then tests, then code, then integration tests.The methodology begins by a design phase that, at a big picture level, isolates what we are doing and why. This is then progressively unrolled under the methodology into smaller and smaller subcomponents. We note that initial viability testing is part of this process; DDD should be used only after initial viability is assured by whatever exploratory tests are needed, and will then take prototype code into refactored versions with no technical debt.

This enables architecture to emerge from constraints rather than guesswork. Traditional development guesses architecture upfront then refactors when guesses prove wrong. DDD progressively refines the big picture as patterns emerge - when "these three features all need X" you abstract X into a shared contract. Cycles of test-design->additional dependencies and their API generate additional needed architecture work that is passed back to design roles.  Really good project specifications can eliminate most refactors because dependencies surface during the cheap design phase rather than expensive implementation phase, and well-written documentation can depend heavily on LLM support for refactoring.

### Workflow

DDD requires a specific workflow to realize its benefits. A good workflow should allow working at one abstraction level at a time, naturally identify when sub-abstractions need fleshing out further, cleanly split up roles between designers, auditors, and implementers, and naturally slice out levels of abstraction to produce clean and easily refactored code.

The workflow operates in a cycle that spawns forks at progressively finer abstraction levels:

```
API Requirement → Documentation → Tests → Implementation → Integration Tests
                                    ↓ (spawns dependency API)
                                    API Requirement → Documentation → Tests → Implementation → Integration Tests
                                                                        ↓ (spawns sub-dependency)
                                                                        API Requirement → Documentation → Tests → ...
```

Starting from a **design foundation** that establishes the project's invariants, requirements, utilities, and directions, you progressively unroll detail:

1. While writing tests at one abstraction level, you realize "I need to inject something doing X, Y, Z"
2. This **spawns an API requirement** for that dependency
3. The API requirement gets **documented as a contract**
4. You can now **fork**: fill the contract immediately, stub it for later, or assign to someone else
5. Continue testing with the dependency interface defined
6. Implement code to pass tests
7. Write integration tests to verify pieces work together

**Spawning Documentation.** When a lower-level dependency emerges, you **contract it out** rather than implementing it inline. Forks should happen organtically during test writing or sometimes design as you realize that you need a dependency doing X. X then becomes the base of an api specification that can be handed back to the design team to be fleshed out. Critically, **Black box testing** validates contracts by testing what's observable (public methods, state, injected dependency usage, emergent behavior) rather than how it works internally. This leaves an abstract design space that just needs to be filled in downstream, and can largely be automated.

Throughout this cycle, the **auditor role backpropagates changes** through the documentation tree. When detailed work reveals new dependencies, auditing must occur to consider whether it should be elevated into a big-picture orthogonal dependency to be used in many places or a dependency of the particular class. This can occur as part of design work or as a separate step. Keeping the living specification synchronized with detailed work is critical, and the work is implemented to the specification. In an ideal world, it would be possible to hook up the system to an LLM, tell it to make the project, and it would rebuild the entire thing from scratch.

### Black Box Testing

Testing is extremely important. One of the benefits of this workflow is it makes black box testing natural as when tests are written the underlying implementation is not yet known, an advantage of the outside-in development pattern. However, this requires actually following a black box testing strategy.

The underlying philosophy: tests validate that the contract is honored, not that specific implementation approaches are used. You are held to the documented specification - if it's in the specification, it must be testable; if it's not in the specification, you cannot bind tests to it.

**What you CAN test:**
- Public methods with their documented input/output behaviors
- Observable state accessible through public properties or methods
- How injected dependencies are called (e.g., verify optimizer.step() was called, gradients were modified)
- Emergent behavior from following the contract (e.g., "using optimizer produces averaged gradients")
- Documented invariants and constraints
- Error conditions and exceptions specified in the contract

**What you CANNOT test:**
- Implementation details (which algorithm is used, internal data structures, private methods)
- Undocumented behavior or side effects
- Private state or internal variables
- Specific function implementations (binding tests to `_take_optimizer_step` doing X is white box)
- Performance characteristics unless documented in the contract
- Anything not specified in the public documentation

**Consequences of violating black box testing:**
- Tests break during refactoring even when contract is preserved
- Implementation becomes coupled to tests, losing refactoring freedom
- Cannot safely delegate implementation to LLMs or juniors
- Technical debt accumulates as "cannot change this, tests will break"
- The entire benefit of contract-based development is lost

Testers identify needed APIs by thinking through from a public perspective how the system must interact with its injected dependencies. This identifies what contracts need to be designed.

### Pros, Cons, Use cases

DDD is project-oriented and produces substantial documentation. It's not lightweight - expect to write significant documentation mass. The methodology works best when you can leverage its strengths and avoid contexts where its properties become liabilities.

**Excellent fit:**
- Building tools and libraries with stable, well-defined interfaces
- Projects requiring parallel development across distributed teams or with LLM assistance
- Code that will be maintained long-term where refactoring freedom matters
- Complex systems requiring multiple abstraction layers
- When architecture quality matters more than speed-to-first-prototype
- After initial viability is proven and you're refactoring prototype to production

**Poor fit:**
- Rapidly changing requirements where contracts invalidate quickly
- Throwaway prototypes or exploratory research code
- Very small single-file projects where overhead isn't justified
- Projects that don't naturally decompose into abstraction layers
- When you're still discovering what to build

**Why this fits GradientQualityControl:** We're building a tool library with stable optimizer interfaces, complex multi-layer abstractions (base wrapper, concrete implementations, scheduling integration), need LLM assistance for implementation, require long-term maintainability, and have proven viability through initial experiments. The workflow naturally handles the fork-heavy architecture where each optimizer type spawns its own contract while sharing the base abstraction.

### Roles

The methodology separates concerns across four roles with different experience requirements and automation potential. The system is designed to allow mistakes in everything except the auditor role, and to allow automation in anything except the auditor role.

**Implementer** (lowest experience): Implements to specification of design and tests. Does integration tests as well. Fills bounded contracts mechanically. Can be automated with LLMs.

**Tester** (moderate experience): Thinks through abstract dependencies and implements unit tests in a decoupled manner. Identifies new APIs to be designed to a given spec by figuring out from a public perspective how the system must interact with its injected dependencies. This is where API requirements emerge. Requires understanding of the public contract and black box testing principles.

**Designer** (high experience): Does documentation and establishes injection contracts as needed. Gathers requirements, forms big picture story, makes architectural trade-offs explicit, progressively refines documentation as patterns emerge. Requires domain expertise and architectural vision.

**Auditor** (highest experience, MOST IMPORTANT): Ensures consistency and crosschecks compliance between documentation and tests, or tests and code. Nothing gets committed without auditor's approval. Backpropagates changes through documentation tree, elevates discovered dependencies to big picture when appropriate, resolves conflicts. This is the most critical role - the entire system is designed to catch mistakes here. Cannot be fully automated - requires human judgment and experience.

Experience hierarchy: Implementer → Tester → Designer → Auditor

### Feedback and Change Management

As work progresses, issues and needed changes emerge. When implementation or testing reveals problems, they are propagated up the documentation chain, putting the documentation out of sync. These needed changes accumulate in tickets rather than being fixed ad-hoc.

When an issue is discovered:
1. The issue is documented (what's wrong, what needs to change)
2. Needed changes are elevated to the appropriate level (design, contract, test specification)
3. Changes are bound together as a ticket to be completed
4. The auditor reviews proposed changes for consistency with the big picture
5. Once approved, changes are implemented across documentation, tests, and code as a unit
6. Nothing is committed until the auditor confirms documentation, tests, and code are synchronized

This prevents documentation drift and ensures the living specification stays synchronized with implementation reality. Issues become opportunities to refine the contracts rather than accumulating as technical debt.