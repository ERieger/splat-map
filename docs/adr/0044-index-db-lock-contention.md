# 0044. Index database: short write transactions and a longer busy timeout

## Status

Accepted. Builds on ADR 0037 (activity log) and ADR 0023 (queue).

## Context

A real queued extraction (888 frames, Insta360 footage on the EstoWines project) failed after
the thumbnail for frame 340 with `database is locked`. It happened just as more jobs were being
added to the queue from the GUI. 320 frame rows had been committed and the job was marked failed.
Its downstream projection, masks and pose jobs were left blocked. The Activity log entry stayed
"running" for the rest of the session.

Causes:
- **The worker held the write lock across slow work.** `extract_frames` inserted each frame row
  as it went and committed every 20 frames. The first insert after a commit opens a write
  transaction, so the lock was held across ~20 thumbnail ffmpeg calls (~40 s at 8K). Any other
  connection had to wait that long to write: the GUI thread saving a queue edit, or the logger.
  A reader blocked the worker's commit the same way. sqlite3's default wait is 5 s.
- **Mask building did the same thing in a smaller way.** It wrote each layer as soon as it was
  computed, so with sky + person the lock was held during the second layer's SAM 3 inference.
- **The activity log's finish write gave up silently.** It opens its own connection, hit the
  same lock and swallowed the error (by design, logging must never break the work). So the
  entry was never completed.

## Decision

- `extract_frames` buffers frame rows and writes each batch of 20 with one `executemany` and an
  immediate commit. The lock is now held for milliseconds, never across thumbnailing or
  checksums. Partial progress is still kept every 20 frames.
- `build_layers_for_views` computes every layer for a view before writing any of them, then
  writes, composes and commits that view together. Per-view consistency is unchanged.
- Every index connection is opened through `project.connect_index_db`, which waits up to
  `DB_BUSY_TIMEOUT_SECONDS` (30 s) for a lock instead of 5 s.
- `activity_log._finish_quietly` retries up to `FINISH_ATTEMPTS` (3) times on
  `sqlite3.OperationalError` before giving up. It still never raises.

## Alternatives considered

- **WAL journal mode.** This would let readers and the writer run concurrently. Rejected for
  now: projects normally live on a Windows drive mounted into WSL (`/mnt/e`, DrvFs). WAL needs
  a shared-memory `-shm` file mapped by every connection, and SQLite documents that WAL doesn't
  work over network-style filesystems. That's a risk of silent corruption, not just a lock error.
  Keeping write transactions short fixes the observed failure without that risk.
- **A longer timeout alone.** This hides the problem but doesn't fix it. A GUI-thread write
  could freeze the UI for up to the 40 s the worker held the lock.

## Consequences

- `tests/test_ingest_integration.py::test_extraction_never_holds_the_db_lock_across_thumbnailing`
  writes from a second connection (0.2 s timeout) during every thumbnail.
  `tests/test_activity_log.py::test_finish_is_retried_when_the_database_is_briefly_locked`
  covers the retry. Both fail on the previous code.
- Any new long-running worker that writes to the index should follow the same rule: do the slow
  work first, then write and commit together. Never leave an insert pending across a slow step.
