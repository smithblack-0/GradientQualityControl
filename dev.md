# Development Practices

## Document-Driven Development (DDD)

This project uses documentation driven development.

### Big Picture

Traditional software development faces a fundamental tension: designing architecture requires working at high abstraction levels with broad context, while implementing code requires working at low abstraction levels with specific details. Attempting both simultaneously produces cognitive overload and poor results. Teams typically resolve this by designing in code, discovering architectural problems during implementation when changes are expensive, then either accumulating technical debt using shortcuts or doing costly refactors. In parallel with this, good code requires documentation, which is usually done after-the-fact by a separate team and is often out of date.

Document-Driven Development inverts this priority. Correct documentation is more important than code, and code is compiled to the documentation. We can observe that will well-written documentation one can generate the objects themselves - this, when paired with modern LLM technology or a legion of code monkeys allows the documentation to be the primary source of truth which is progressively unfolded from the big picture into implementation details. The code is implemented according to a specific workflow that forms from api requirements documentation, then tests, then code, then integration tests.The methodology begins by a design phase that, at a big picture level, isolates what we are doing and why. This is then progressively unrolled under the methodology into smaller and smaller subcomponents. We note that initial viability testing is part of this process; DDD should be used only after initial viability is assured by whatever exploratory tests are needed, and will then take prototype code into refactored versions with no technical debt.

This enables architecture to emerge from constraints rather than guesswork. Traditional development guesses architecture upfront then refactors when guesses prove wrong. DDD progressively refines the big picture as patterns emerge - when "these three features all need X" you abstract X into a shared contract. Cycles of test-design->additional dependencies and their API generate additional needed architecture work that is passed back to design roles.  Really good project specifications can eliminate most refactors because dependencies surface during the cheap design phase rather than expensive implementation phase, and well-written documentation can depend heavily on LLM support for refactoring.

### Workflow

DDD requires a specific workflow to realize its benefits. A good workflow should allow working at one abstraction level at a time, naturally identify when sub-abstractions need fleshing out further, cleanly split up roles between designers, auditors, and implementers, and naturally slice out levels of abstraction to produce clean and easily refactored code.

The workflow operates in a cycle:

```
API Requirement → Documentation → Tests → Implementation → Integration Tests
                      ↓
                (spawns new API requirement for dependencies)
                      ↓
                  Fork to document new contract
```

Starting from a **design foundation** that establishes the project's invariants, requirements, utilities, and directions, you progressively unroll detail:

1. While writing tests at one abstraction level, you realize "I need to inject something doing X, Y, Z"
2. This **spawns an API requirement** for that dependency
3. The API requirement gets **documented as a contract**
4. You can now **fork**: fill the contract immediately, stub it for later, or assign to someone else
5. Continue testing with the dependency interface defined
6. Implement code to pass tests
7. Write integration tests to verify pieces work together

**Forks slice off bounded work.** When a lower-level dependency emerges, you **contract it out** rather than implementing it inline. The critical rule: **discovering dependencies during implementation means you failed at planning** - forks should happen during design or test writing.

**Black box testing** validates contracts by testing what's observable (public methods, state, injected dependency usage, emergent behavior) rather than how it works internally. Tests must remain valid when implementation changes completely.

Throughout this cycle, the **auditor role backpropagates changes** through the documentation tree. When detailed work reveals new dependencies, auditing elevates them to the big picture and resolves conflicts, keeping the living specification synchronized with detailed work.

### Mechanics

The fundamental principle enabling DDD is never working at more than one level of abstraction simultaneously. You isolate related abstractions at the same level and document them together, pushing lower-level concerns into future bounded contracts. During documentation you identify what abstraction level you're working at, recognize dependencies at lower levels, contract them out by defining interfaces with stubbed behavior, stay at your level, and fill in lower-level contracts later or assign to others. During testing this becomes creating interfaces for needed dependencies, testing against those interfaces through dependency injection and black box testing, stubbing dependencies for now, and passing them back to the documentation team to fill in later.

This works because each object and document contains things at the same abstraction level, lower-level details become contracted stubs, the hierarchy starts at the public interface and progressively deepens, you're always solving a bounded problem at one abstraction level, and there's no cognitive overload from mixing abstraction levels. This fails when insisting on single objects instead of multiple abstraction layers, not contracting out sub-roles as separate abstractions, trying to work at multiple abstraction levels simultaneously, or mixing high-level architecture with low-level implementation details.

The methodology creates three distinct roles. The designer writes contracts (documentation), designs architecture through progressive refinement, identifies and documents dependencies, makes architectural trade-offs explicit, and requires domain expertise and architectural vision. The auditor maintains documentation consistency by backpropagating changes through the documentation tree - when detailed work reveals new dependencies like a logging role, auditing elevates that to a primary dependency in the big picture and resolves conflicts such as multiple logging frameworks. The auditor reviews contracts for completeness and consistency, verifies tests properly validate contracts with black box compliance, identifies ambiguities or underspecified behavior, checks that implementation honors contracts, and keeps the big picture synchronized with detailed work as a living specification. LLMs can sometimes serve as auditors for spotting inconsistencies and checking coverage. The implementer implements test suites from contracts (mechanical transcription), implements code to pass tests (bounded problem solving), identifies contract ambiguities during implementation and escalates to designer, refactors within contract bounds, and LLMs often make good implementers since they excel at "implement to spec" but struggle with "design the spec". This separation works because designing contracts requires expertise and vision (human-centric), auditing maintains consistency across abstraction levels through backpropagation, and implementation from clear specs is mechanical (LLMs and juniors effective).

DDD is project-oriented and produces substantial documentation mass - it's not lightweight. It's an excellent fit for building tools and libraries with clear contracts and stable interfaces, stable or well-understood requirements, projects needing parallel development with distributed teams or LLM assistance, code maintained long-term, when architecture quality matters more than speed-to-first-prototype, and complex systems requiring multiple abstraction layers. It's a poor fit for rapidly changing requirements where contracts invalidate quickly and documentation effort is wasted, throwaway prototypes or exploratory code, research code where you're discovering what to build, very small single-file projects where overhead isn't justified, and projects that don't naturally decompose into multiple abstraction layers.

Practical notes: documentation is held to formal specification standards where if it's not documented it doesn't exist and inconsistencies in docs are bugs, requiring version control of docs with same rigor as code. Progressive refinement starts with high-level contracts, refactors docs as patterns emerge (when "these three features all need X" you abstract X into shared contract), and narrows the solution space iteratively. Integration just works if both sides honor contracts with no "how do I call this?" questions, no mismatched assumptions, and the contract serving as the integration specification.

### Benefits

The deeper theory of why DDD works rests on several fundamental insights. Documentation strength enables code development because sufficiently strong documentation is sufficient specification for implementation, documentation can progressively detail lower abstraction levels, and each level can be filled in independently. Architecture and code are opposing abstraction processes - writing consistent high-level architecture is abstract, conceptual, and interface-focused, while writing concrete implementation code is detailed, specific, and mechanism-focused, and trying to do both simultaneously creates cognitive overload and poor results, so DDD separates these into sequential phases.

Refactoring is dramatically easier when only documentation exists because before code exists architectural changes are just text edits, outside-in development from interface to implementation prevents major architectural breakage, and by the time you write code the architecture has been refined and validated. Traditional development guesses architecture upfront, discovers it doesn't fit, and requires expensive refactoring, while DDD documents high-level contracts, notices patterns as you realize "these all need X" and abstracts naturally, uses progressive refinement where each iteration narrows the solution space converging on correct architecture, and by implementation phase the architecture has already been pressure-tested against requirements.

Documentation being cheap to refactor means changing a function signature in docs takes 30 seconds while changing it in code requires refactoring all call sites, updating tests, and fixing breaks - design iteration happens in the cheap phase (documentation) rather than expensive phase (code). Implementation becomes mechanical because all dependencies are identified during planning, all interfaces are defined before coding, and the implementation task is simply "fill in this contract to pass these tests" with no architectural decisions, no scope creep, and clear success criteria.

True parallel development emerges because bounded contracts can be filled by different people or teams simultaneously with no coordination needed beyond honoring contracts and integration guaranteed if both sides satisfy their contracts. Refactoring freedom comes from black box tests validating contracts not implementation, meaning you can completely rewrite internals as long as the contract is preserved with no fear of breaking hidden assumptions.

Modern LLMs can implement to Natural Language Formal Specifications, which is the technological enabler making insights about documentation strength, opposing abstraction processes, and refactoring ease practically exploitable. With LLMs, documentation can be the specification without needing separate formal spec languages, implementation details abstract into successively deeper backend documentation, each layer has public contracts validated by black box tests, and LLMs excel at "implement to spec" but struggle with "design the spec". This means human designers work at the abstraction and architecture level (their strength) while LLMs and juniors work at the implementation level (bounded problem-solving), and the methodology ensures they integrate seamlessly.
