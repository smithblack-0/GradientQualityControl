# Development Practices

## Document-Driven Development (DDD)

This project uses documentation driven development.

### Big Picture

Traditional software development faces a fundamental tension: designing architecture requires working at high abstraction levels with broad context, while implementing code requires working at low abstraction levels with specific details. Attempting both simultaneously produces cognitive overload and poor results. Teams typically resolve this by designing in code, discovering architectural problems during implementation when changes are expensive, then either accumulating technical debt using shortcuts or doing costly refactors. In parallel with this, good code requires documentation, which is usually done after-the-fact by a separate team and is often out of date.

Document-Driven Development inverts this priority. It is a workflow that **progressively uncovers a specification which is the exact documentation.** We can observe that will well-written documentation one can generate the objects themselves - this, when paired with modern LLM technology or a legion of code monkeys allows the documentation to be the primary source of truth which is progressively unfolded from the big picture into implementation details. The code is implemented according to a specific workflow that forms from api requirements documentation, then tests, then code, then integration tests.The methodology begins by a design phase that, at a big picture level, isolates what we are doing and why. This is then progressively unrolled under the methodology into smaller and smaller subcomponents. We note that initial viability testing is part of this process; DDD should be used only after initial viability is assured by whatever exploratory tests are needed, and will then take prototype code into refactored versions with no technical debt.

This enables architecture to emerge from constraints rather than guesswork. Traditional development guesses architecture upfront then refactors when guesses prove wrong. DDD progressively refines the big picture as patterns emerge - when "these three features all need X" you abstract X into a shared contract. Cycles of test-design->additional dependencies and their API generate additional needed architecture work that is passed back to design roles.  Really good project specifications can eliminate most refactors because dependencies surface during the cheap design phase rather than expensive implementation phase, and well-written documentation can depend heavily on LLM support for refactoring.

Critically, Merge Requests must NEVER be accepted unless documentation, tests, AND code are resynchronized; **The documentation or tests being out of date is LITERALLY a bug**

### Constraints and Emergence

DDD emerged from optimizing for a specific set of constraints present in modern software development with LLM assistance:

**Given constraints:**
- **Implementation is cheap but dumb** - LLMs and junior developers can implement to specification mechanically, but require clear bounded contracts. However, issues with contracts are very expensive.
- **Testing is smarter but still dumb** - LLMs can write tests from specifications, but need human auditing for judgment calls about what APIs are needed. Human review of first-pass API identification and tests is almost always required
- **Design is required, slow, and manual** - Quality design requires human thought and domain expertise, cannot be fully automated.
- **Auditing is needed with human feedback** - Any LLM artifact requires human review. However, reviewing documentation to verify completeness and consistency is much faster than reviewing code implementation

DDD optimizes for these constraints by concentrating expensive human effort (documentation and auditing) where it provides maximum value, while automating cheap mechanical work (implementation and testing). Every role usually does some level of auditing, but we take advantage of a key insight: since documentation is required anyway, if you make it the primary reviewable artifact, then compile code to it, you can make implementation automatable.

Strengthening this to Natural Language Formal Specification (NLFS) contracts that are continously maintained as one source of truth with clean api breaks then vastly accelerates auditing and allows dispatch of testing and implementation tasks, with guidance, to juniors or LLMs. Auditing then concentrates on ensuring the documentation is consistant, and artifacts are produced to specification.

### Workflow

DDD requires a specific workflow to realize its benefits. A good workflow should allow working at one abstraction level at a time, naturally identify when sub-abstractions need fleshing out further, cleanly split up roles between designers, auditors, and implementers, and naturally slice out levels of abstraction to produce clean and easily refactored code.

The workflow operates in a cycle that spawns forks at progressively finer abstraction levels:

```
API Requirement → Documentation → Tests → Implementation → Factories→ Integration
                                    ↓ (spawns dependency API)
                                    API Requirement → Documentation → Tests → Implementation → Factories->Integration
                                                                        ↓ (spawns sub-dependency)
                                                                        API Requirement → Documentation → Tests → ...
```

Starting from a **design foundation** that establishes the project's invariants, requirements, utilities, and directions, you progressively unroll detail:

1. While writing tests at one abstraction level, you realize "I need to inject something doing X, Y, Z"
2. This **spawns an API requirement** for that dependency, usually an object or data structure.
3. The API requirement gets **documented as a contract** that collects dependencies while designing the test.
4. You can now **fork**: fill the contract immediately, stub it for later, or assign to someone else.
5. Continue testing with the dependency interface defined.
6. Implement code to pass tests.
7. Write factories to build the objects once all sub-dependencies are done (blocking)
7. Write integration tests to verify pieces work together.

**Spawning Documentation.** When a lower-level dependency emerges, you **contract it out** rather than implementing it inline. Forks should happen organtically during test writing or sometimes design as you realize that you need a dependency doing X. X then becomes the base of an api specification that can be handed back to the design team to be fleshed out. Critically, **Black box testing** validates contracts by testing what's observable (public methods, state, injected dependency usage, emergent behavior) rather than how it works internally. This leaves an abstract design space that just needs to be filled in downstream, and can largely be automated.

Throughout this cycle, when detailed work reveals new dependencies, auditing must occur to consider whether it should be elevated into a big-picture orthogonal dependency to be used in many places or a dependency of the particular class. This can occur as part of design work or as a separate step. Keeping the living specification synchronized with detailed work is critical, and the work is implemented to the specification. In an ideal world, it would be possible to hook up the system to an LLM, tell it to make the project, and it would rebuild the entire thing from scratch. 

### Black Box Testing

Testing is extremely important. One of the benefits of this workflow is it makes black box testing natural as when tests are written the underlying implementation is not yet known, an advantage of the outside-in development pattern. However, this requires actually following a black box testing strategy.

The underlying philosophy: tests validate that the contract is honored, not that specific implementation approaches are used. You are held to the documented specification - if it's in the specification, it must be testable; if it's not in the specification, you cannot bind tests to it.

**What you CAN test:**
- Public methods with their documented input/output behaviors
- Observable state accessible through public properties or methods
- How injected dependencies are called but ONLY from the public facing contract. You verify .step was called, not that it stepped in subfunction X.
- Emergent behavior from following the contract (e.g., "using optimizer produces averaged gradients")
- Documented invariants and constraints
- Error conditions and exceptions specified in the contract

**What you CANNOT test:**
- Implementation details (which algorithm is used, internal data structures, private methods)
- Undocumented behavior or side effects
- Private state or internal variables
- Testing behavior that needs to be tested outside of contracted access route or implementation.
- Anything not specified in the public documentation

If you need to test something, but cannot because of this, either the documentation needs tweaks or/and new apis need to be spawned abstracting that behavior away. Documentation changes would then be routed back to design, as would be the finished collection of api changes.

**Consequences of violating black box testing:**
- Tests break during refactoring even when contract is preserved
- Implementation becomes coupled to tests, losing refactoring freedom
- Cannot safely delegate implementation to LLMs or juniors
- Technical debt accumulates as "cannot change this, tests will break"
- The entire benefit of contract-based development is lost

Testers identify needed APIs by thinking through from a public perspective how the system must interact with its injected dependencies. This identifies what contracts need to be designed. Everything else becomes a black box that can be handed off to the implementation.

### Pros, Cons, Use cases

DDD is project-oriented and produces substantial documentation. It's not lightweight - expect to write a formal specification level of documentation while doing the project. At the same time, it is not waterflow. It is designed to start at the big picture, to avoid making horrible design mistakes, then get progressively better. It also is designed explictly for environments where handing contracts off for implementation is cheap.

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

We're building a tool library with stable optimizer interfaces, complex multi-layer abstractions (base wrapper, concrete implementations, scheduling integration), need LLM assistance for implementation, require long-term maintainability, and have proven viability through initial experiments. The workflow naturally handles the fork-heavy architecture where each optimizer type spawns its own contract while sharing the base abstraction.

### Roles

The methodology separates concerns across four roles with different experience requirements and automation potential. The system is designed to allow mistakes in everything except the Director role, and to allow automation in anything except the Director role.

**Implementer** (lowest experience): Implements to specification of design and tests. Does integration tests as well. Fills bounded contracts mechanically. Can be automated with LLMs. They will also form the factories used to construct the objects once the tests are done. Should escalate back to testing if behavior is impossible.

**Tester** (moderate experience): Thinks through abstract dependencies and implements unit tests in a decoupled manner. Identifies new APIs to be designed to a given spec by figuring out from a public perspective how the system must interact with its injected dependencies. This is where API requirements emerge and abstraction is defined in terms of injection and dependencies. Requires understanding of the public contract, black box testing principles, and how to identify different levels of abstractions. Often performs minor documentation changes too, but should escalate to design if major changes are needed. Can audit implementation.

**Designer** (high experience): Does documentation and establishes injection contracts as needed. Gathers requirements, forms big picture story, makes architectural trade-offs explicit, progressively refines documentation as patterns emerge. Requires domain expertise and architectural vision. Revises documentation and establishes what refactoring now needs to be executed. They implement API contracts into documentation for it, and pass it to testing. They also escalate major issues, along with a resolution plan, to the Director. Being able to produce quality documentation which begins holistically and progressively unfolds is **the** essential skill for all designers. Can audit tests.

**Director** (highest experience): Directors are the 'PM' of DDD. They attempt to maximize consistency between documents, tests, and code across the entire system. Nothing gets committed without the relevant Director's approval. Directors are always looking for and surface any detected inconsistencies between the parts of the system, whether documents, tests, or code. This means with enough audit rounds most bugs are found and fixed and overall code quality increases. They have authority to review any documentation (user guides, API docs, related systems) to catch system-level mistakes. Within their domain, they **are** the person responsible for the final quality on this commit. Cannot be fully automated - requires human judgment and experience. Can audit design.

Experience hierarchy: Implementer → Tester → Designer → Director