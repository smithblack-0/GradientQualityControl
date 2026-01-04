# GradientQualityControl

A PyTorch library for gradient quality control using Sequential Binary Decision Controllers (SBDC).

## Project Structurees

```
src/gradient_quality_control/    # Source code
tests/                           # Tests
documentation/                   # API documentation
dev.md                           # Development methodology (READ THIS)
```

## Development Methodology

This project uses **Document-Driven Development (DDD)**. Read `dev.md` for the full methodology.

**Core principles:**
- Documentation is the source of truth
- Tests validate documented contracts (black-box only)
- Code implements what documentation specifies
- All three must stay synchronized - a mismatch is a bug

## Key Documentation

- `dev.md` - Full development methodology and workflow
- `documentation/base_object_api.md` - Core AbstractOptimizerWrapper specification
- `documentation/user_guide.md` - Usage patterns
- `documentation/api_guide.md` - Factory functions

## Making Changes Safely

### Before Any Change

1. Read the relevant documentation
2. Understand the existing contract
3. Identify what type of change this is (doc/test/code)

### Adding Features

1. **Document first** - Update documentation with the new API/behavior
2. **Test second** - Write black-box tests that validate the documented contract
3. **Implement last** - Write code to pass tests and match documentation

### Fixing Bugs

1. Identify the source: Is it a doc bug, test bug, or code bug?
2. Fix at the source - don't patch around it
3. Verify all three remain synchronized

### Refactoring

1. Ensure tests exist for current behavior
2. Refactor code without changing documented behavior
3. Tests should continue to pass unchanged

## Detecting and Reporting Problems

### Code Smells to Watch For

When working on this codebase, watch for these smells:

| Smell | Signs | Action |
|-------|-------|--------|
| **God Object** | Class has many unrelated responsibilities | Raise to user - needs decomposition |
| **Tight Coupling** | Need many mocks/dependencies to test one thing | Raise to user - needs decoupling |
| **Leaky Abstraction** | Tests need implementation details | Raise to user - API needs work |
| **Missing Abstraction** | Same pattern repeated in multiple places | Raise to user - needs extraction |
| **Doc/Code Mismatch** | Documentation doesn't match implementation | Fix immediately or raise to user |
| **Untestable Design** | Can't write black-box tests for documented behavior | Raise to user - needs redesign |

### When You Detect a Problem

1. **Stop and report** - Don't silently work around it
2. **Be specific** - Describe what you observed and where
3. **Explain impact** - Why is this a problem?
4. **Suggest direction** (optional) - What might fix it?

If uncertain whether something is a problem, ask. It's better to raise a false alarm than to propagate a bad design.

## Testing Philosophy

**Black-box testing only:**
- Test public methods and documented behavior
- Do NOT test implementation details
- Tests should pass for ANY correct implementation of the contract

**What you CAN test:**
- Public method inputs/outputs
- Observable state via public properties
- Documented error conditions
- Documented invariants

**What you CANNOT test:**
- Private methods or internal state
- Implementation algorithms
- Undocumented behavior

## Core Behavior

**The most important job is maintaining consistency while following best practices.**

1. **Maintain consistency** - docs, tests, and code must always match
2. **Raise issues** - if something seems wrong, say so
3. **Escalate when stuck** - if you can't resolve an issue, bring it to the user

### When You See Something Wrong

Don't silently work around problems. Instead:

- "I notice X doesn't match Y - should I fix it?"
- "This design seems to have issue X - can we discuss?"
- "I can't implement this as specified because X - what should we do?"

If you think something is missing or could be better, that's a design discussion, not something to suppress. Raise it.

### Code Style

- Follow existing patterns in the codebase
- Comments explain WHY, not WHAT
- **Never use numbered enumeration** in comments (Suite 1, Suite 2, Step 1, etc.) - it's brittle and breaks when things are reordered
- Instead, name the **role** of the block and describe what it does in a complete sentence

**Comment patterns:**
- Test suites: `"Name - tests that [complete sentence]"`
  - Example: `"Get State Test Suite - tests that get_state() retrieves wrapper state values"`
  - NOT: `"Suite 1: Get State Tests"` or `"Get state tests"`
- Customize descriptions to what each block actually does - don't copy/paste the same comment everywhere
- Use complete sentences, not fragments
