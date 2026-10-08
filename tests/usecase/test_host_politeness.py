"""Publisher politeness for PDF fetches (benchmark-architecture.md §7.87, HQ 2026-10-09:
31 back-to-back pubs.rsc.org requests left the owner's browser on the same network
with HTTP 429; the run kept hitting MDPI after its 403s. Owner, same day: one publisher
request per minute, globally)."""

from real_chart_bench.usecase.host_politeness import HostPoliteFetcher
from real_chart_bench.usecase.pdf_fetch import PdfFetchResult, PdfFetchStatus


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def time(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


class _Inner:
    def __init__(self, results: dict[str, PdfFetchResult]):
        self.results = results
        self.calls: list[str] = []

    def fetch(self, url):
        self.calls.append(url)
        return self.results.get(url, PdfFetchResult(status=PdfFetchStatus.OK, content=b"%PDF"))


def _http(code):
    return PdfFetchResult(status=PdfFetchStatus.HTTP_ERROR, detail=f"HTTP {code}")


def _fetcher(inner, clock, gap=60.0):
    return HostPoliteFetcher(inner, min_gap_s=gap, clock=clock.time, sleep=clock.sleep)


def test_default_gap_is_one_minute():
    clock = _Clock()
    f = HostPoliteFetcher(_Inner({}), clock=clock.time, sleep=clock.sleep)
    f.fetch("https://a.org/1.pdf")
    f.fetch("https://b.org/1.pdf")
    assert clock.slept == [60.0]


def test_first_request_does_not_wait():
    clock = _Clock()
    _fetcher(_Inner({}), clock).fetch("https://a.org/1.pdf")
    assert clock.slept == []


def test_second_request_waits_until_the_gap_has_passed():
    clock = _Clock()
    f = _fetcher(_Inner({}), clock)
    f.fetch("https://a.org/1.pdf")
    clock.now += 3.0
    f.fetch("https://a.org/2.pdf")
    assert clock.slept == [57.0]


def test_gap_is_global_so_different_hosts_also_wait():
    clock = _Clock()
    f = _fetcher(_Inner({}), clock)
    f.fetch("https://a.org/1.pdf")
    f.fetch("https://b.org/1.pdf")
    assert clock.slept == [60.0]


def test_skipped_hosts_cost_no_wait_and_do_not_reset_the_gap():
    clock = _Clock()
    f = _fetcher(_Inner({}), clock)
    f.blocked_hosts["blocked.org"] = "HTTP 403"
    f.fetch("https://a.org/1.pdf")
    clock.now += 30.0
    f.fetch("https://blocked.org/1.pdf")
    assert clock.slept == []
    f.fetch("https://b.org/1.pdf")
    assert clock.slept == [30.0]


def test_no_wait_once_the_gap_has_already_elapsed():
    clock = _Clock()
    f = _fetcher(_Inner({}), clock)
    f.fetch("https://a.org/1.pdf")
    clock.now += 60.0
    f.fetch("https://a.org/2.pdf")
    assert clock.slept == []


def test_403_blocks_the_host_for_the_rest_of_the_run_without_new_requests():
    clock = _Clock()
    inner = _Inner({"https://a.org/1.pdf": _http(403)})
    f = _fetcher(inner, clock)
    f.fetch("https://a.org/1.pdf")
    result = f.fetch("https://a.org/2.pdf")
    assert inner.calls == ["https://a.org/1.pdf"]
    assert result.status is PdfFetchStatus.HTTP_ERROR
    assert "a.org" in result.detail and "403" in result.detail and "skipped" in result.detail
    assert f.blocked_hosts == {"a.org": "HTTP 403"}


def test_429_also_blocks_the_host():
    clock = _Clock()
    inner = _Inner({"https://a.org/1.pdf": _http(429)})
    f = _fetcher(inner, clock)
    f.fetch("https://a.org/1.pdf")
    f.fetch("https://a.org/2.pdf")
    assert inner.calls == ["https://a.org/1.pdf"]


def test_404_does_not_block_the_host():
    clock = _Clock()
    inner = _Inner({"https://a.org/1.pdf": _http(404)})
    f = _fetcher(inner, clock)
    f.fetch("https://a.org/1.pdf")
    f.fetch("https://a.org/2.pdf")
    assert inner.calls == ["https://a.org/1.pdf", "https://a.org/2.pdf"]


def test_a_blocked_host_does_not_block_other_hosts():
    clock = _Clock()
    inner = _Inner({"https://a.org/1.pdf": _http(403)})
    f = _fetcher(inner, clock)
    f.fetch("https://a.org/1.pdf")
    assert f.fetch("https://b.org/1.pdf").status is PdfFetchStatus.OK


def test_hosts_can_be_pre_blocked_from_an_earlier_run():
    clock = _Clock()
    inner = _Inner({})
    f = HostPoliteFetcher(inner, min_gap_s=60, clock=clock.time, sleep=clock.sleep,
                          blocked_hosts={"a.org": "HTTP 429"})
    assert f.fetch("https://a.org/1.pdf").status is PdfFetchStatus.HTTP_ERROR
    assert inner.calls == []


def test_missing_url_is_passed_through_to_the_inner_fetcher():
    clock = _Clock()
    inner = _Inner({})
    _fetcher(inner, clock).fetch(None)
    assert inner.calls == [None]
