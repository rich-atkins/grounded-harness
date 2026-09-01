---
title: Deploy Process
tags: [engineering, process]
---

# Deploy Process

Deploys go out through the pipeline, never by hand. Merge to `main` triggers
build, test, and a staged rollout: 5% canary for 30 minutes, then full.

## Rollback

Rollback is one click in the deploy dashboard and takes about 90 seconds.
If the canary error rate doubles, rollback happens automatically. Incidents
follow the [[incident-runbook]].

## Freeze windows

No deploys after 15:00 on Fridays, or during the Black Friday freeze
(the last full week of November).
