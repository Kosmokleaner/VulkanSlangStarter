# Contributing

Thank you for contributing to this project. This document describes repository rules, standards, and the process for contributing changes.

## Guidelines

- Be respectful and clear in PR descriptions and commit messages.
- Use small, focused pull requests when possible.
- Run project tests and verify a local build before opening a PR.
- Don't remove commented out code created by other contributors

## Coding Standards
- use uint8, int32, ... uint64 instead of _t variants
- Include globals.h in every cpp first line
- m_ for class members, struct should not have that
- Class and Strut names are pascal case
- Variable and Function names are camel case
- Tab size 4, use tabs, not spaces
- Function and method output parameters start with out, then upperscase
- Avoid lambda and modern C++ constructs, light std usage is ok 
- Member variables and local variables that use a type without full member initialization should use = {} to avoid uninitialized data
- Keep changes consistent with the existing code style.
- don't use C++ exceptions

## Modification Restrictions

- The `extern/` directory is off-limits. Do not modify, update, or remove any files under `extern/` in any branch or pull request unless you have explicit approval from a repository maintainer.
  - Rationale: `extern/` contains external third-party dependencies and binary/artifact files that must remain unchanged to ensure reproducible builds.
  - Exceptions: If an update is required (security fixes, critical bug fixes, version bumps), open an issue describing the change and request maintainer approval before creating a PR. The PR must clearly document the reason and the maintainer sign-off.

## Reporting Problems or Requesting Exceptions

- Open an issue describing the required change and the reason.
- Mention the maintainers and wait for approval before creating a PR that modifies `extern/`.

## Enforcement (optional)

If you want, we can add a pre-commit hook or CI check to automatically block changes under `extern/`. Ask if you want that added and I will prepare it.

---

By contributing you agree to follow these rules. Thank you for helping keep the repository stable and reproducible.