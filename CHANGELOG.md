# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffold: package layout with typed API stubs, `docs/DESIGN.md` (formulas,
  reference values, validation strategy, Monte Carlo and portfolio study design,
  milestones), quick and full Monte Carlo configs, portfolio config, CI.
- `scripts/reference_values.py`: independent recomputation of the published numeric
  examples (Bailey and Lopez de Prado 2012 and 2014; Lopez de Prado, Lipton and Zoonekynd
  2025) and derived closed-form values, pinned by tests.
