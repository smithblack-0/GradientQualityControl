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

Black box testing validates contracts by testing only what's observable through the public interface. You test public methods and their documented behaviors, observable state like properties and return values, how injected dependencies are used, and emergent behavior that results from following the contract. You cannot test implementation details, undocumented behavior, private state, or bind tests to specific internal functions.

The difference matters. Testing "take_optimizer_step averages gradients" is white box - it tests a specific function's implementation. Testing "using optimizer results in gradient averaging" is black box - it tests emergent behavior from following the contract. The first breaks when you refactor internals. The second remains valid as long as the contract is honored.

This creates complete decoupling. Implementation can change freely - different algorithms, data structures, optimizations - and tests remain valid. The contract is the specification, tests validate the contract, and implementation is free to evolve within those bounds. This is what enables LLMs and junior developers to safely implement to spec.

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

The methodology cleanly separates concerns across three distinct roles that can be filled by different people or even different types of workers (humans vs LLMs).

**Designer** writes contracts and designs architecture. They gather requirements, form the big picture story, identify dependencies, make architectural trade-offs explicit, and progressively refine documentation as patterns emerge. This requires domain expertise and architectural vision - understanding what needs to be built and why. Designers work at high abstraction levels.

**Auditor** maintains consistency across the living specification. They backpropagate changes through the documentation tree, elevating discovered dependencies to the big picture when appropriate, resolving conflicts like multiple logging frameworks, reviewing contracts for completeness, verifying tests follow black box methodology, and keeping the big picture synchronized with detailed work. LLMs can sometimes assist with auditing by spotting inconsistencies and checking coverage.

**Implementer** fills bounded contracts. They implement test suites from contracts (mechanical transcription of requirements), implement code to pass tests (bounded problem solving), identify contract ambiguities and escalate to designer, and refactor within contract bounds. LLMs excel as implementers since they're good at "implement to spec" but struggle with "design the spec". Junior developers also work well as implementers with clear contracts.

This separation works because each role operates at its natural abstraction level with clear success criteria and boundaries. Designers need creativity and vision, auditors need consistency checking, implementers need mechanical execution. The workflow ensures clean handoffs between roles.