# Errors

Command failures and integration errors.

---

## [ERR-20260427-001] prompt-key-env-mismatch

**Logged**: 2026-04-27T10:21:00Z
**Priority**: high
**Status**: resolved
**Area**: config

### Summary
Prompt-mode backend selection incorrectly reused NVIDIA_API_KEY as the default env var even when GPT was chosen.

### Error
```
RuntimeError: No API key found in environment variable NVIDIA_API_KEY
```

### Context
- User chose GPT from interactive prompt flow
- CLI had a generic --api-key-env default set to NVIDIA_API_KEY
- Config creation reused that default for GPT runs
- Fixed by inferring env var from selected backend: OPENAI_API_KEY for GPT, NVIDIA_API_KEY for NVIDIA

### Suggested Fix
Remove api-key-env from user-facing prompt flow and derive the secret variable automatically from the chosen backend.

### Metadata
- Reproducible: yes
- Related Files: run_discussion.py, README.md

### Resolution
- **Resolved**: 2026-04-27T10:22:00Z
- **Commit/PR**: pending
- **Notes**: Prompt flow now asks for the correct key based on backend choice, without requiring a parameter.

---
