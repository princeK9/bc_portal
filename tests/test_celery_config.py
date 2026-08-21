"""
Celery settings that exist for correctness reasons.

These are assertions about configuration rather than behaviour, which is
unusual for a test suite -- but each of these defaults is wrong for this
workload in a way that loses work silently rather than failing loudly. A
regression here would not break any other test.
"""

import config
from celery_app import app as celery_app

# Confirmed live: the 429 payload names limit: 15 for this model/tier.
FREE_TIER_QUOTA_PER_MIN = 15


class TestDeliveryGuarantees:
    def test_acks_late_is_enabled(self):
        """Default acks-on-receipt loses a file entirely when a worker dies
        mid-extraction: the broker considers it delivered, nobody processes it."""
        assert celery_app.conf.task_acks_late is True

    def test_reject_on_worker_lost_is_enabled(self):
        """acks_late alone still drops work on a hard kill -- the message stays
        unacked but is never requeued. This is what actually returns it."""
        assert celery_app.conf.task_reject_on_worker_lost is True

    def test_prefetch_is_one_per_slot(self):
        """The default (4x concurrency) lets one worker reserve dozens of files
        it has not started; all of them stall if it dies."""
        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_broker_visibility_timeout_is_set_explicitly(self):
        """
        Regression guard for a bug only a live run exposed.

        Redis has no server-side ack. kombu emulates it by holding delivered
        messages in an `unacked` hash and returning them to the queue only
        after visibility_timeout. task_reject_on_worker_lost is therefore close
        to a no-op here -- there is no broker to reject to.

        With the setting unset, kombu's 3600s default applied: SIGKILLing a
        worker stranded its four in-flight tasks and froze the job at 19/23 for
        the full ten minutes it was observed. Setting this explicitly is what
        actually makes a dead worker's tasks recoverable.
        """
        options = celery_app.conf.broker_transport_options or {}
        assert "visibility_timeout" in options, (
            "unset means kombu's 3600s default: a killed worker's tasks sit "
            "stranded for an hour"
        )
        assert options["visibility_timeout"] == config.BROKER_VISIBILITY_TIMEOUT

    def test_visibility_timeout_exceeds_the_longest_legitimate_hold(self):
        """
        Must be longer than any healthy task can hold a message, or a slow but
        perfectly alive task gets redelivered and runs a second time.

        The two bounds are the hard task time limit and the longest retry
        backoff including jitter.
        """
        from tasks import MAX_RETRY_COUNTDOWN

        visibility = config.BROKER_VISIBILITY_TIMEOUT
        longest_backoff = MAX_RETRY_COUNTDOWN * 1.2  # jitter ceiling
        assert visibility > config.TASK_TIME_LIMIT
        assert visibility > longest_backoff

    def test_acks_late_is_paired_with_an_idempotent_write(self):
        """acks_late permits duplicate execution, which is only safe because
        the DB write upserts. If that pairing is broken, retries duplicate rows."""
        from db import SQL_UPSERT_ON_BATCH_FILE, SQL_UPSERT_ON_PHONE

        assert celery_app.conf.task_acks_late is True
        assert "ON CONFLICT" in SQL_UPSERT_ON_PHONE
        assert "ON CONFLICT" in SQL_UPSERT_ON_BATCH_FILE


class TestTimeLimits:
    def test_hard_time_limit_is_set(self):
        """One hung Gemini call must not own a worker slot indefinitely."""
        assert celery_app.conf.task_time_limit == config.TASK_TIME_LIMIT
        assert celery_app.conf.task_time_limit > 0

    def test_soft_limit_is_below_hard_limit(self):
        """Soft must fire first so the task can fail cleanly and be retried;
        if they were equal the process would be SIGKILLed with no cleanup."""
        assert celery_app.conf.task_soft_time_limit < celery_app.conf.task_time_limit

    def test_limits_leave_room_for_a_slow_call(self):
        """A 1-3s call with retries must not be killed by an over-tight limit."""
        assert celery_app.conf.task_soft_time_limit >= 30


class TestThroughput:
    def test_concurrency_matches_config(self):
        assert celery_app.conf.worker_concurrency == config.WORKER_CONCURRENCY

    def test_rate_limit_is_set(self):
        """Concurrency alone cannot hold the API budget when calls are fast."""
        assert celery_app.conf.task_default_rate_limit == config.GEMINI_RATE_LIMIT

    def test_rate_limit_stays_under_the_real_free_tier_quota(self):
        """
        The quota is 15 requests/minute, confirmed by a live 429 during load
        testing. An earlier version of this test asserted 35-104/min from an
        unverified assumption and happily passed while the configured 90/m was
        admitting ~5x the real allowance -- 8 of 46 files failed permanently.
        """
        limit = celery_app.conf.task_default_rate_limit
        assert limit.endswith("/m")
        per_minute = int(limit.removesuffix("/m"))
        assert per_minute < FREE_TIER_QUOTA_PER_MIN, (
            f"{per_minute}/m is at or above the {FREE_TIER_QUOTA_PER_MIN}/m quota"
        )
        assert per_minute >= 1

    def test_rate_limit_keeps_headroom_below_the_quota(self):
        """Our limiter and Google's window will not agree to the request; a
        limit exactly equal to the quota trips 429s on clock skew alone."""
        per_minute = int(celery_app.conf.task_default_rate_limit.removesuffix("/m"))
        assert per_minute <= FREE_TIER_QUOTA_PER_MIN - 1

    def test_free_tier_cannot_reach_the_50k_per_day_target(self):
        """Documents a capacity ceiling, not a bug: no amount of concurrency
        tuning gets the free tier to 50k/day, so that target implies a paid
        tier rather than a configuration change."""
        per_day = FREE_TIER_QUOTA_PER_MIN * 60 * 24
        assert per_day == 21_600
        assert per_day < 50_000


class TestResultBackend:
    def test_results_outlive_the_run(self):
        """/status reads per-file state from the backend after the job ends."""
        assert celery_app.conf.result_expires == config.BATCH_TTL_SECONDS

    def test_started_state_is_tracked(self):
        """Without this a running task is indistinguishable from a queued one,
        and /status cannot report meaningful progress."""
        assert celery_app.conf.task_track_started is True


class TestSerialization:
    def test_pickle_is_not_accepted(self):
        """Pickle over the broker means a compromised Redis executes arbitrary
        code inside a worker."""
        assert "pickle" not in celery_app.conf.accept_content
        assert celery_app.conf.task_serializer == "json"


class TestRedisSeparation:
    def test_broker_and_cache_use_different_logical_databases(self):
        """Flushing the disposable search cache must not be able to drop queued
        work. Different db index is what guarantees that."""
        assert config.REDIS_URL != config.REDIS_CACHE_URL
        assert config.REDIS_URL.rsplit("/", 1)[-1] != config.REDIS_CACHE_URL.rsplit("/", 1)[-1]

    def test_broker_and_backend_are_configured(self):
        assert celery_app.conf.broker_url
        assert celery_app.conf.result_backend


class TestTaskRegistration:
    def test_process_resume_is_registered(self):
        assert "tasks.process_resume" in celery_app.tasks

    def test_task_can_schedule_its_own_retry(self):
        """bind=True is what makes self.retry(countdown=...) available; without
        it the task would have to fall back to sleeping."""
        from tasks import process_resume

        assert hasattr(process_resume, "retry")

    def test_delivery_settings_are_set_on_the_task_itself(self):
        """Set per-task as well as globally, so moving this task to another
        app or queue cannot silently drop the guarantees."""
        from tasks import process_resume

        assert process_resume.acks_late is True
        assert process_resume.reject_on_worker_lost is True

    def test_task_retry_budget_is_bounded(self):
        """An unbounded retry loop on a permanently-broken file never drains."""
        from tasks import process_resume

        assert 0 < process_resume.max_retries <= 10
