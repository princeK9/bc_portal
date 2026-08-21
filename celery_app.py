"""
Celery application and its correctness settings.

Every option below is set deliberately; the Celery defaults are wrong for this
workload in ways that lose work silently.

WORKER CONCURRENCY -- corrected against the real quota
------------------------------------------------------
This block previously reasoned from a budget of "roughly 35-104 requests per
minute". That figure was never verified and is wrong. A 46-file load test
produced a live 429 stating the actual allowance:

    Quota exceeded for metric: generate_content_free_tier_requests,
    limit: 15, model: gemini-3.5-flash-lite
    quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier

The free tier permits **15 requests per minute**, per project per model.

That changes the conclusion, not just the arithmetic. A single resume
extraction takes about 1-3 seconds, so one worker slot alone sustains:
    3s/call -> 20 requests/minute
    1s/call -> 60 requests/minute

Both exceed 15. So on the free tier concurrency cannot be sized *up to* the
quota -- even one slot outruns it. The rate limiter is not a safety net on top
of a well-chosen concurrency; it is the entire throughput governor, and
concurrency only decides how much work sits ready behind it. That is why the
old 90/m limit was so damaging: it admitted ~5x the real allowance, and 8 of 46
files burned all five retries against a permanently saturated quota.

Concurrency stays at 4 regardless. It costs nothing (the work is I/O-bound,
waiting on HTTP), it absorbs latency variance so the limiter always has a task
ready to release, and it is the right number the moment the quota rises. On a
paid tier the original reasoning applies again: with a 2000/min allowance the
limiter stops binding and concurrency becomes the real constraint.

Capacity implication worth stating plainly: 15/min is 21,600 requests/day at a
perfect 100% duty cycle. A 50,000-resume/day target is therefore not reachable
on the free tier by any amount of tuning -- it is roughly 2.3x over the ceiling.
It is comfortably reachable on a paid tier, where 50,000 requests is about 25
minutes of budget rather than a day of it.
"""

import logging

from celery import Celery
from celery.signals import setup_logging as celery_setup_logging

from config import (
    BROKER_VISIBILITY_TIMEOUT,
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    GEMINI_RATE_LIMIT,
    BATCH_TTL_SECONDS,
    TASK_MAX_RETRIES,
    TASK_SOFT_TIME_LIMIT,
    TASK_TIME_LIMIT,
    WORKER_CONCURRENCY,
    setup_logging,
)

app = Celery("bc_portal", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

app.conf.update(
    # --- Delivery guarantees -------------------------------------------------
    # Acknowledge only after the task finishes. With the default (ack on
    # receipt) a worker killed mid-extraction loses the file silently: the
    # broker considers it delivered and nobody ever processes it. acks_late
    # moves the risk to the other side -- a task may run twice -- which is why
    # the database write is an upsert (see db.upsert_candidate).
    task_acks_late=True,
    # Meaningful on a real broker (AMQP), where a lost worker's unacked message
    # is rejected straight back to the queue. On Redis this flag is close to a
    # no-op -- see broker_transport_options below, which is what actually
    # governs recovery here. Kept because it costs nothing and is correct if the
    # broker is ever swapped for RabbitMQ.
    task_reject_on_worker_lost=True,
    # One unacked message per slot. Default prefetch (4x concurrency) would let
    # a single worker reserve dozens of files it has not started, all of which
    # sit idle if it dies and all of which delay other workers.
    worker_prefetch_multiplier=1,

    # The real redelivery mechanism on Redis. Without this, kombu's 3600s
    # default applies and a killed worker's in-flight tasks sit in the `unacked`
    # hash for an hour before anyone re-runs them -- the job simply stalls.
    # Verified by SIGKILLing a worker: four tasks stranded, job frozen at 19/23.
    broker_transport_options={"visibility_timeout": BROKER_VISIBILITY_TIMEOUT},
    result_backend_transport_options={"visibility_timeout": BROKER_VISIBILITY_TIMEOUT},

    # --- Time limits ---------------------------------------------------------
    # A hung Gemini call must not own a slot indefinitely. Soft limit raises
    # SoftTimeLimitExceeded inside the task so it can fail cleanly and be
    # retried; the hard limit SIGKILLs the child if it ignores that.
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT,
    task_time_limit=TASK_TIME_LIMIT,

    # --- Throughput ----------------------------------------------------------
    worker_concurrency=WORKER_CONCURRENCY,
    task_default_rate_limit=GEMINI_RATE_LIMIT,

    # --- Result backend ------------------------------------------------------
    # /status reads per-file state from here, so results must outlive the run.
    result_expires=BATCH_TTL_SECONDS,
    result_extended=True,
    task_track_started=True,

    # --- Serialization -------------------------------------------------------
    # JSON only. Pickle would let a compromised broker execute arbitrary code
    # in a worker.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    timezone="UTC",
    enable_utc=True,
    task_max_retries=TASK_MAX_RETRIES,
)

# Import task modules so the worker registers them.
app.autodiscover_tasks(["tasks"], force=True)


@celery_setup_logging.connect
def _configure_worker_logging(**kwargs):
    """
    Use the project's logging config inside workers.

    Connecting to this signal stops Celery from installing its own root handler,
    which is why library modules were changed to never call basicConfig at
    import time -- two handlers on the root logger duplicates every line.
    """
    setup_logging()
    logging.getLogger("celery").setLevel(logging.INFO)
