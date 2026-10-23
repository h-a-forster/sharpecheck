# Security policy

## Supported versions

Security fixes go into the latest release.

## Reporting a vulnerability

Report privately through the repository's **Security** tab ("Report a vulnerability").
Do not open a public issue.

Include the sharpecheck version, your OS, what goes wrong, and how to reproduce it. Expect
an acknowledgement within a week.

## Threat model

sharpecheck is a local statistics library and command-line tool.

- **No credentials.** sharpecheck needs no API keys. The data command fetches public CSV
  zips over HTTPS from `mba.tuck.dartmouth.edu` (Kenneth R. French Data Library) and
  nothing else.
- **Downloaded data is untrusted input.** Archives are read in memory and nothing from them
  is executed. The parser checks the expected layout (header width, consecutive six-digit
  months, finite numbers) and rejects anything else. The library publishes no checksums,
  so the SHA-256 of each download and its CRSP vintage line are recorded in a manifest;
  a changed hash is reported, never silently mixed into an existing run. Downloads use a
  timeout and are written atomically.
- **Cache and run directories** (`data/`, `runs/`) hold public data and intermediate
  results only. They are local, gitignored, and safe to delete.
