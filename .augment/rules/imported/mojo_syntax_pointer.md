---
type: "always_apply"
---

# Mojo Syntax Reference & Coding Standards
- Do not add comments that are unnecessary or overexplaining.
- Remove comments that are obsolete or necessary for understanding the code.
- When creating, modifying, or validating Mojo source files, adhere to design patterns/guidelines captured in `mojo_syntax.md`.
- Run `update_mojo_syntax.mojo` automation script to systematically detect Mojo syntax and design pattern issues.
- Fix all errors found by the automation script `update_mojo_syntax.mojo` using the built-in command line flags.
- Fix compiler errors and warnings not detected by the automation script.
- Fix compiler language server protocol (LSP) errors and warnings not detected by the automation script.
- When taking action, create brief and clear responses and summaries.