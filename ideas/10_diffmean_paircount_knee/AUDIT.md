# AUDIT — H<NN>

> Adversarial self-critique. Treat the implementation as if you are
> reviewing it for a top venue. This document must be filed BEFORE the
> first experiment run. An empty AUDIT.md means the audit was lazy —
> the minimum is 3 weaknesses. Do not sign VERIFY.md until this is filed.

## Weaknesses found

1. **<one-line summary>** — <why it matters; which file/function; severity: HIGH/MEDIUM/LOW>
2. **<one-line summary>** — <why it matters; which file/function; severity>
3. **<one-line summary>** — <why it matters; which file/function; severity>

## Bugs caught by tests

- <list bugs caught by the tests.py suite>

## Bugs NOT caught by tests but suspected

- <edge cases, off-by-ones, silent shape errors, floating-point gotchas>

## Data / leakage concerns

- <does the implementation ever touch the eval split? where? how is it prevented?>

## Mitigations queued for IMPROVEMENTS.md

- <one line per fix; will be addressed before VERIFY.md is signed>

## Audit verdict

[ ] CLEARED — all HIGH-severity issues resolved; implementation safe to run
[ ] BLOCKED — HIGH-severity issues remain; do not run until resolved

Auditor: <identifier>
Date: <YYYY-MM-DD>
