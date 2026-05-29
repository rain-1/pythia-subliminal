# Day 2 Length-Controlled Carrier Visibility Audit

This audits the exact matched datasets used in the current PolyPythia sports and legal replication reports. The main subliminality check is on generated continuations, because fixed template prompts can contain ordinary scaffold words such as `score`.

## Sports

| dataset | rows | continuation alpha rows | continuation exact blacklist rows | continuation substring blacklist rows | full-text exact blacklist rows | avg continuation chars |
|---|---:|---:|---:|---:|---:|---:|
| sports seed3 neutral | 5800 | 0 | 0 | 0 | 0 | 51.7 |
| sports seed3 steered | 5800 | 0 | 0 | 0 | 0 | 51.1 |
| sports seed4 neutral | 7478 | 0 | 0 | 0 | 0 | 51.5 |
| sports seed4 steered | 7478 | 0 | 0 | 0 | 0 | 51.0 |
| sports seed5 neutral | 7963 | 0 | 0 | 0 | 0 | 55.2 |
| sports seed5 steered | 7963 | 0 | 0 | 0 | 0 | 55.0 |
| sports seed6 neutral | 8638 | 0 | 0 | 0 | 0 | 56.1 |
| sports seed6 steered | 8638 | 0 | 0 | 0 | 0 | 56.0 |
| sports seed7 neutral | 8203 | 0 | 0 | 0 | 0 | 54.1 |
| sports seed7 steered | 8203 | 0 | 0 | 0 | 0 | 54.0 |


## Legal

| dataset | rows | continuation alpha rows | continuation exact blacklist rows | continuation substring blacklist rows | full-text exact blacklist rows | avg continuation chars |
|---|---:|---:|---:|---:|---:|---:|
| legal seed6 neutral | 9296 | 0 | 0 | 0 | 0 | 57.8 |
| legal seed6 steered | 9296 | 0 | 0 | 0 | 0 | 57.9 |
| legal seed7 neutral | 9383 | 0 | 0 | 0 | 0 | 56.6 |
| legal seed7 steered | 9383 | 0 | 0 | 0 | 0 | 56.6 |
| legal seed8 neutral | 9263 | 0 | 0 | 0 | 0 | 54.9 |
| legal seed8 steered | 9263 | 0 | 0 | 0 | 0 | 55.0 |
| legal seed9 neutral | 8922 | 0 | 0 | 0 | 0 | 58.1 |
| legal seed9 steered | 8922 | 0 | 0 | 0 | 0 | 58.1 |


## Readout

- All audited generated continuations have zero alphabetic rows and zero exact or substring trait-blacklist hits.
- Full-text exact blacklist hits are also zero for these matched sports and legal datasets.
- Full-text substring blacklist hits are zero in the JSON audit details as well.
- This supports the claim that the current sports/legal hard-token replications use visibly innocuous generated carrier continuations. It does not by itself rule out every possible non-obvious statistical cue in numeric formatting.

JSON details: `reports/day2_length_controlled_carrier_visibility_audit.json`
