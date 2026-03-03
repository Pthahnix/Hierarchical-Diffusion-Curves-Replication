# Documentation Index

Welcome to the Hierarchical Diffusion Curves documentation!

## Quick Start

New to the project? Start here:
1. **[README.md](../README.md)** - Quick overview and installation
2. **[USER_GUIDE.md](USER_GUIDE.md)** - Complete usage guide with examples

## Documentation Structure

### For Users

- **[README.md](../README.md)** - Project overview, installation, basic usage
- **[USER_GUIDE.md](USER_GUIDE.md)** - Comprehensive usage guide
  - Installation instructions
  - Basic and advanced usage examples
  - Configuration options
  - Troubleshooting
  - Performance tips
  - API reference

### For Developers

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development workflow and guidelines
  - TDD methodology
  - Testing strategies
  - Code style and conventions
  - Git workflow
  - Debugging techniques
  - Performance optimization

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and architecture
  - Module design and interactions
  - Design decisions and trade-offs
  - Extensibility points
  - Performance considerations
  - Common pitfalls

### Technical Reference

- **[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)** - Implementation details
  - Simplifications from paper
  - Solver comparison
  - Future enhancements
  - Usage for experiments

- **[CLAUDE.md](../CLAUDE.md)** - Complete project memory
  - Architecture overview
  - Key design decisions
  - Testing strategy
  - Usage patterns
  - File structure
  - Troubleshooting

### Planning Documents

- **[plans/2026-03-03-hierarchical-diffusion-curves-implementation.md](plans/2026-03-03-hierarchical-diffusion-curves-implementation.md)** - Original implementation plan
  - Task breakdown (15 tasks)
  - TDD approach for each task
  - Verification steps

## Documentation by Topic

### Getting Started
- Installation → [README.md](../README.md#installation)
- First example → [USER_GUIDE.md](USER_GUIDE.md#basic-usage)
- Command line usage → [USER_GUIDE.md](USER_GUIDE.md#command-line-interface)

### Understanding the Code
- Architecture overview → [ARCHITECTURE.md](ARCHITECTURE.md#system-architecture)
- Module design → [ARCHITECTURE.md](ARCHITECTURE.md#module-design)
- Data flow → [ARCHITECTURE.md](ARCHITECTURE.md#high-level-flow)

### Using the Library
- Python API → [USER_GUIDE.md](USER_GUIDE.md#python-api)
- Configuration → [USER_GUIDE.md](USER_GUIDE.md#configuration-options)
- Advanced usage → [USER_GUIDE.md](USER_GUIDE.md#advanced-usage)
- Examples → [USER_GUIDE.md](USER_GUIDE.md#examples-gallery)

### Development
- Setting up dev environment → [DEVELOPMENT.md](DEVELOPMENT.md#development-setup)
- TDD workflow → [DEVELOPMENT.md](DEVELOPMENT.md#tdd-cycle)
- Adding features → [DEVELOPMENT.md](DEVELOPMENT.md#example-adding-a-new-feature)
- Testing → [DEVELOPMENT.md](DEVELOPMENT.md#testing-guidelines)
- Contributing → [DEVELOPMENT.md](DEVELOPMENT.md#contributing)

### Troubleshooting
- Common issues → [USER_GUIDE.md](USER_GUIDE.md#troubleshooting)
- Performance problems → [USER_GUIDE.md](USER_GUIDE.md#performance-tips)
- Development issues → [DEVELOPMENT.md](DEVELOPMENT.md#troubleshooting-development-issues)

### Technical Deep Dive
- Design patterns → [ARCHITECTURE.md](ARCHITECTURE.md#design-patterns-used)
- Solver comparison → [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md#solver-comparison)
- Performance analysis → [ARCHITECTURE.md](ARCHITECTURE.md#performance-considerations)
- Extensibility → [ARCHITECTURE.md](ARCHITECTURE.md#extensibility-points)

## Document Summaries

### README.md (1.1 KB)
Quick project overview with installation and basic usage. Start here if you're new.

### USER_GUIDE.md (15 KB)
Complete guide for using the library. Covers installation, configuration, examples, and troubleshooting.

### ARCHITECTURE.md (20 KB)
Detailed design documentation. Explains module design, trade-offs, and extensibility.

### DEVELOPMENT.md (18 KB)
Development workflow guide. TDD methodology, testing, code style, and contribution guidelines.

### IMPLEMENTATION_NOTES.md (2 KB)
Technical notes on implementation choices and future enhancements.

### CLAUDE.md (12 KB)
Comprehensive project memory. Architecture, usage patterns, and complete reference.

## Reading Paths

### Path 1: Quick Start User
1. README.md (5 min)
2. USER_GUIDE.md - Basic Usage section (10 min)
3. Try examples (15 min)

### Path 2: Thorough User
1. README.md (5 min)
2. USER_GUIDE.md (30 min)
3. IMPLEMENTATION_NOTES.md (10 min)
4. Experiment with code (1 hour)

### Path 3: New Developer
1. README.md (5 min)
2. ARCHITECTURE.md (45 min)
3. DEVELOPMENT.md (30 min)
4. Read source code with docs as reference (2 hours)

### Path 4: Contributor
1. DEVELOPMENT.md - Development Setup (15 min)
2. DEVELOPMENT.md - TDD Cycle (20 min)
3. ARCHITECTURE.md - Extensibility Points (15 min)
4. Pick an issue and start coding (ongoing)

### Path 5: Researcher
1. Original paper (1 hour)
2. IMPLEMENTATION_NOTES.md (10 min)
3. ARCHITECTURE.md - Module Design (30 min)
4. CLAUDE.md - Solver Comparison (15 min)
5. Run comparison experiments (1 hour)

## Maintenance

### Updating Documentation

When making changes to the codebase:

1. **Code changes** → Update relevant sections in:
   - ARCHITECTURE.md (if design changes)
   - USER_GUIDE.md (if API changes)
   - CLAUDE.md (if major changes)

2. **New features** → Add to:
   - USER_GUIDE.md (usage examples)
   - DEVELOPMENT.md (if affects dev workflow)
   - ARCHITECTURE.md (design decisions)

3. **Bug fixes** → Update:
   - USER_GUIDE.md (if affects usage)
   - DEVELOPMENT.md (add regression test example)

4. **Performance improvements** → Update:
   - ARCHITECTURE.md (performance section)
   - CLAUDE.md (performance characteristics)

### Documentation Standards

- Use clear, concise language
- Include code examples for concepts
- Keep examples runnable and tested
- Update all affected documents together
- Use consistent terminology across docs

## Getting Help

1. **Check documentation** - Use this index to find relevant docs
2. **Search issues** - Someone may have had the same question
3. **Read source code** - Code is well-commented
4. **Ask questions** - Create an issue with your question

## Contributing to Documentation

Documentation improvements are always welcome! See [DEVELOPMENT.md](DEVELOPMENT.md#contributing) for guidelines.

Good documentation contributions:
- Fix typos or unclear explanations
- Add missing examples
- Improve existing examples
- Add troubleshooting tips
- Update outdated information

---

**Last Updated:** 2026-03-03
**Documentation Version:** 1.0
**Project Version:** 0.1.0
