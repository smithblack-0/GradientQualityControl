# Development Practices

## Document-Driven Development (DDD)

This project uses documentation driven development.

### Big Picture

Document-Driven Development is a **contract-first development, outside-in methodology** where documentation serves as the design medium. Instead of designing in code and documenting afterward, you design by writing Natural Language Formal Specifications (which may include formal mathematics when appropriate), implementing tests while filling out the dependency public specification, then implement code to conform to those specifications.

**Core mechanism:** Work at only one abstraction level at a time, pushing lower-level concerns into future bounded contracts that can be filled later or by others.

**Why this works:**
- **Documentation is cheap to refactor** - Architecture emerges through progressive refinement during documentation phase, not during expensive implementation phase
- **Tests drive interface discovery** - APIs emerge when writing tests (not designed upfront), documentation formalizes them as contracts
- **Implementation becomes mechanical** - All dependencies identified during planning, implementation is just "fill this contract to pass tests"
- **LLM-friendly** - Modern LLMs can implement to Natural Language Formal Specifications, enabling true parallel development (humans design, LLMs/juniors implement)

**Key mechanism:** During testing you realize "I need something doing X, Y, Z" → Create interface → Test against it → Send back to documentation team to formalize → Continue at your abstraction level. **Documentation + Interfaces IS the specification.**

**Taxonomical classification:** Contract-first development, project-oriented (produces substantial documentation mass). Excellent for building tools and libraries with stable requirements, poor fit for rapidly changing requirements or throwaway prototypes.

### Workflow

**The iterative loop:**

1. **Document public behavior** - Write contract for the thing you're building (can be vague "does X somehow", just needs public API surface)
2. **Write tests** - As you write tests, you realize "I need to inject something doing Y, Z" → APIs emerge here
3. **FORK** - Send the needed dependency interface back to documentation role to flesh out as new contract
4. **Implement code** - Write implementation to pass tests (mechanical, bounded)
5. **Implement integration tests** - Verify the pieces work together
6. **Loop back to step 2** - Continue with next piece

**Key insight:** You don't write stub APIs first - they **emerge during test writing** when you realize what you need to inject. Testing phase discovers the necessary interfaces, documentation phase formalizes them as contracts.

**The inversion:** Tests drive interface discovery, documentation captures interfaces as contracts, implementation fills contracts. Traditional development discovers interfaces during implementation (too late, causes refactoring).

### The Core Mechanism: Single-Level Abstraction

**The fundamental principle that makes DDD work:**

You **never work at more than one level of abstraction simultaneously**. Instead, you isolate related abstractions at the same level, document them together, then push lower-level concerns into future bounded contracts.

**How this works in practice:**

1. **During documentation**: Identify what abstraction level you're working at (e.g., "public optimizer interface")
2. **Identify dependencies**: "This needs something that does X, Y, Z" - that's a lower abstraction level
3. **Contract it out**: Define the interface/contract for that dependency, stub the behavior
4. **Stay at your level**: Continue documenting at your current abstraction level
5. **Later**: Fill in the lower-level contract (or assign to someone else)

**During testing, this becomes:**
- "I need something doing THESE functions and will verify my class uses them"
- Create interface for the dependency
- Test against that interface (dependency injection + black box testing)
- Stub the dependency for now
- Pass back to documentation team to fill in later

**Key insight:** Documentation + Interfaces IS the specification. By including needed dependency behavior through dependency injection into black box testing contracts, you can slice off pieces of the problem and defer them.

**Why this works:**
- Each object/document contains things at the **same abstraction level**
- Lower-level details become contracted stubs
- Hierarchy starts at public interface, progressively deepens
- You're always solving a bounded problem at one abstraction level
- No cognitive overload from mixing abstraction levels

**When this fails:**
- Insisting on single objects instead of multiple abstraction layers
- Not contracting out sub-roles as separate abstractions
- Trying to work at multiple abstraction levels simultaneously
- Mixing high-level architecture with low-level implementation details

### Key Insights Enabling DDD

These insights make DDD practical with modern technology:

**1. Documentation strength enables code development**
- Sufficiently strong documentation is sufficient specification for implementation
- Documentation can progressively detail lower abstraction levels
- Each level can be filled in independently

**2. Architecture and code are opposing abstraction processes**
- Writing consistent high-level architecture: abstract, conceptual, interface-focused
- Writing concrete implementation code: detailed, specific, mechanism-focused
- Trying to do both simultaneously creates cognitive overload and poor results
- DDD separates these into sequential phases

**3. Refactoring is easier when only documentation exists**
- Before code exists, architectural changes are just text edits
- Outside-in development (interface → implementation) prevents major architectural breakage
- By the time you write code, architecture has been refined and validated

**4. Modern LLMs can implement to Natural Language Formal Specifications**
- This is the technological enabler that makes insights 1-3 practically exploitable
- With LLMs, documentation CAN BE the specification (no separate formal spec language needed)
- Implementation details abstract into successively deeper backend documentation
- Each layer has public contracts validated by black box tests
- LLMs excel at "implement to spec" but struggle with "design the spec"

**Result:** Human designers work at the abstraction/architecture level (their strength), LLMs/juniors work at the implementation level (bounded problem-solving), and the methodology ensures they integrate seamlessly.

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

DDD is **project-oriented** and produces substantial documentation. It's not "lightweight" - expect to write significant documentation mass.

**Excellent fit:**
- **Building tools and libraries** - Clear contracts, stable interfaces, reusable components
- Stable or well-understood requirements
- Projects needing parallel development (distributed teams, LLM assistance)
- Code that will be maintained long-term
- When architecture quality matters more than speed-to-first-prototype
- Complex systems requiring multiple abstraction layers

**Poor fit:**
- Rapidly changing requirements (contracts invalidate quickly, wasted documentation effort)
- Throwaway prototypes or exploratory code
- Research code where you're discovering what to build
- Very small single-file projects (overhead not justified)
- Projects that don't naturally decompose into multiple abstraction layers

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