# Project Rules

- This project is NOT equilibrium simulation and NOT static frequency replay.
- Automata must be adaptive: maintain beliefs over opponent types and update online.
- Default policy must be explainable (belief-based). Neural net policy is optional extension.
- Keep UI minimal and experimental: instructions -> decisions -> results -> CSV download.
- Implement both modes: (1) human responder vs bot proposer, (2) human proposer vs bot responder.
- Add headless stress-test scripts and ensure outputs are reproducible.
- Modularize code: do not put everything in app.js.
- Log belief states and decision rationales every round.
