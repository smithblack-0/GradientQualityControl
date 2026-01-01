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

- CLAUDE FILL IN DETAILS

### Pros, Cons, Use cases, and why this fits


- CLAUDE FILL IN DETAILS.