---
title: Incident Runbook
tags: [engineering, incidents]
---

# Incident Runbook

Declare early: a suspected incident IS an incident until proven otherwise.

## Severity levels

- SEV1: customer-facing outage. Page the on-call lead immediately.
- SEV2: degraded service or a broken internal tool with a workaround.
- SEV3: cosmetic or contained; fix in normal hours.

## During an incident

One person leads, one person scribes. Updates to the status channel every 20
minutes. Nobody debugs in silence. The deploy path is in [[deploy-process]].

## Afterwards

Blameless review within five working days. Actions get owners and dates or
they do not leave the room.
