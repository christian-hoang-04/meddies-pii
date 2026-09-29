"""Even after a minute-window clears (all RPM back to 0).

The account that has served fewer requests *today* is preferred — so daily load fans across all accounts instead of
concentrating on the first one. Selecting purely by RPM would re-pick k1 here once its minute window emptied.

A hard trip (cooldown=None) takes an account out for the rest of the run, so choose routes to a sibling instead of
re-hammering the capped account.

A transient trip (e.g. a Retry-After value) blocks an account briefly. With no sibling free, choose returns it with a wait,
then frees it once elapsed.

"""

from __future__ import annotations

# ruff: file-ignore[float-equality-comparison]
# reason: these assertions pin the exact value the code under test produces from deterministic
# reason: inputs, so a tolerance would make the test accept a value the code does not produce.
from typing import TYPE_CHECKING

from meddies_pii.generation.account_ledger import AccountLedger, AccountLimits

if TYPE_CHECKING:
    from collections.abc import Mapping


def _ledger(limits: Mapping[str, AccountLimits], clock: list[float]) -> AccountLedger:
    return AccountLedger(limits, now=lambda: clock[0])


def test_empty_picks_first_key_no_wait() -> None:
    led = _ledger({"groq": AccountLimits(rpm=2, rpd=10)}, [1000.0])
    key, wait = led.choose("groq", ["k1", "k2"])
    assert key == "k1"
    assert wait == 0.0


def test_rotates_to_less_loaded_key_within_rpm() -> None:
    led = _ledger({"groq": AccountLimits(rpm=2, rpd=10)}, [1000.0])
    led.reserve("groq", "k1")
    key, wait = led.choose("groq", ["k1", "k2"])
    assert key == "k2"
    assert wait == 0.0


def test_spreads_by_daily_load_not_just_minute_window() -> None:
    clock = [1000.0]
    led = _ledger({"groq": AccountLimits(rpm=10, rpd=100)}, clock)
    led.reserve("groq", "k1")
    led.reserve("groq", "k2")
    clock[0] = 1000.0 + 61
    key, wait = led.choose("groq", ["k1", "k2", "k3"])
    assert key == "k3"
    assert wait == 0.0


def test_waits_when_all_keys_at_rpm() -> None:
    led = _ledger({"groq": AccountLimits(rpm=1, rpd=10)}, [1000.0])
    led.reserve("groq", "k1")
    led.reserve("groq", "k2")
    key, wait = led.choose("groq", ["k1", "k2"])
    assert key in {"k1", "k2"}
    assert wait > 0


def test_skips_rpd_exhausted_key() -> None:
    led = _ledger({"groq": AccountLimits(rpm=10, rpd=1)}, [1000.0])
    led.reserve("groq", "k1")
    key, _ = led.choose("groq", ["k1", "k2"])
    assert key == "k2"


def test_all_rpd_exhausted_returns_none() -> None:
    led = _ledger({"groq": AccountLimits(rpm=10, rpd=1)}, [1000.0])
    led.reserve("groq", "k1")
    led.reserve("groq", "k2")
    key, _ = led.choose("groq", ["k1", "k2"])
    assert key is None


def test_rpm_window_slides() -> None:
    clock = [1000.0]
    led = _ledger({"groq": AccountLimits(rpm=1, rpd=10)}, clock)
    led.reserve("groq", "k1")
    clock[0] = 1000.0 + 61
    key, wait = led.choose("groq", ["k1"])
    assert key == "k1"
    assert wait == 0.0


def test_uncapped_provider_never_blocks() -> None:
    led = _ledger({}, [1000.0])
    key, wait = led.choose("mistral", ["k1"])
    assert key == "k1"
    assert wait == 0.0


def test_snapshot_reports_per_account_usage() -> None:
    led = _ledger({"groq": AccountLimits(rpm=10, rpd=100)}, [1000.0])
    led.reserve("groq", "k1")
    led.add_tokens("groq", "k1", 500)
    snap = led.snapshot()
    account = snap["groq"]["k1"]
    assert account["requests"] == 1
    assert account["tokens"] == 500


def test_tpd_ceiling_stops_account_for_the_day() -> None:
    """Est 2000 would push 4000 + 2000 = 6000 over the 5000 daily token cap."""
    led = _ledger({"groq": AccountLimits(rpm=10, rpd=100, tpd=5000)}, [1000.0])
    led.add_tokens("groq", "k1", 4000)
    key, _ = led.choose("groq", ["k1"], est_tokens=2000)
    assert key is None


def test_tpm_ceiling_throttles_then_frees() -> None:
    """5000 + 2000 = 7000 > 6000 tpm -> wait (daily is fine, so not None).

    after the minute window slides, the token budget frees and it can send.

    """
    clock = [1000.0]
    led = _ledger({"groq": AccountLimits(rpm=60, tpm=6000)}, clock)
    led.add_tokens("groq", "k1", 5000)
    key, wait = led.choose("groq", ["k1"], est_tokens=2000)
    assert key == "k1"
    assert wait > 0
    clock[0] = 1000.0 + 61
    key, wait = led.choose("groq", ["k1"], est_tokens=2000)
    assert key == "k1"
    assert wait == 0.0


def test_request_larger_than_tpm_is_skipped() -> None:
    """A single 8000-token request can never fit under a 6000 TPM cap."""
    led = _ledger({"groq": AccountLimits(tpm=6000)}, [1000.0])
    key, _ = led.choose("groq", ["k1"], est_tokens=8000)
    assert key is None


def test_spreads_to_account_with_fewer_daily_tokens() -> None:
    """Tie on requests -> the one with fewer daily tokens wins."""
    led = _ledger({"groq": AccountLimits(rpm=10, tpd=100_000)}, [1000.0])
    led.reserve("groq", "k1")
    led.add_tokens("groq", "k1", 8000)
    led.reserve("groq", "k2")
    key, _ = led.choose("groq", ["k1", "k2"], est_tokens=2000)
    assert key == "k2"


def test_hard_trip_skips_account_like_daily_exhausted() -> None:
    led = _ledger({"groq": AccountLimits(rpm=10, rpd=100)}, [1000.0])
    led.trip("groq", "k1")
    key, _ = led.choose("groq", ["k1", "k2"])
    assert key == "k2"


def test_all_hard_tripped_returns_none() -> None:
    """Every account capped -> stop the provider for the day."""
    led = _ledger({"groq": AccountLimits(rpm=10, rpd=100)}, [1000.0])
    led.trip("groq", "k1")
    led.trip("groq", "k2")
    key, _ = led.choose("groq", ["k1", "k2"])
    assert key is None


def test_transient_trip_throttles_then_frees() -> None:
    clock = [1000.0]
    led = _ledger({"groq": AccountLimits(rpm=10, rpd=100)}, clock)
    led.trip("groq", "k1", cooldown=30.0)
    key, wait = led.choose("groq", ["k1"])
    assert key == "k1"
    assert 0 < wait <= 30.0
    clock[0] = 1000.0 + 31
    key, wait = led.choose("groq", ["k1"])
    assert key == "k1"
    assert wait == 0.0


def test_transient_trip_prefers_untripped_sibling() -> None:
    led = _ledger({"groq": AccountLimits(rpm=10, rpd=100)}, [1000.0])
    led.trip("groq", "k1", cooldown=30.0)
    key, wait = led.choose("groq", ["k1", "k2"])
    assert key == "k2"
    assert wait == 0.0
