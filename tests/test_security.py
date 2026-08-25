import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_events = load_script("fetch_events", "fetch-events.py")
google_auth = load_script("google_auth", "google-auth.py")


class SecureJsonMixin:
    module = None

    def test_secure_json_round_trip_uses_owner_only_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            self.module.write_secure_json(path, {"token": "secret"})
            self.assertEqual(self.module.safe_load_json(path), {"token": "secret"})
            file_stat = os.stat(path, follow_symlinks=False)
            self.assertTrue(stat.S_ISREG(file_stat.st_mode))
            self.assertEqual(stat.S_IMODE(file_stat.st_mode), 0o600)
            self.assertEqual(os.listdir(directory), ["state.json"])

    def test_secure_json_read_rejects_symlink_and_oversize_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "target.json")
            link = os.path.join(directory, "link.json")
            with open(target, "w", encoding="utf-8") as stream:
                json.dump({"secret": True}, stream)
            os.symlink(target, link)
            with self.assertRaises(OSError):
                self.module.safe_load_json(link)
            with self.assertRaises(ValueError):
                self.module.safe_load_json(target, max_bytes=2)

    def test_secure_json_write_refuses_replaceable_non_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "target.json")
            link = os.path.join(directory, "state.json")
            with open(target, "w", encoding="utf-8") as stream:
                stream.write("keep")
            os.symlink(target, link)
            with self.assertRaises(ValueError):
                self.module.write_secure_json(link, {"new": True})
            with open(target, "r", encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "keep")

    def test_secure_json_write_enforces_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            with self.assertRaises(ValueError):
                self.module.write_secure_json(path, {"large": "value"}, max_bytes=2)
            self.assertFalse(os.path.exists(path))

    def test_secure_json_temp_creation_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            collision = os.path.join(directory, ".state.json.tmp-aaaa")
            with open(collision, "w", encoding="utf-8") as stream:
                stream.write("sentinel")
            with mock.patch.object(self.module.secrets, "token_hex", side_effect=["aaaa", "bbbb"]):
                self.module.write_secure_json(path, {"ok": True})
            with open(collision, "r", encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "sentinel")


class FetchEventsSecureJsonTests(SecureJsonMixin, unittest.TestCase):
    module = fetch_events


class GoogleAuthSecureJsonTests(SecureJsonMixin, unittest.TestCase):
    module = google_auth


class JmapTransportTests(unittest.TestCase):
    class FakeResponse:
        def __init__(self, url, payload=b"{}"):
            self.url = url
            self.payload = payload
            self.offset = 0
            self.closed = False

        def geturl(self):
            return self.url

        def read(self, size=-1):
            if size < 0:
                size = len(self.payload) - self.offset
            chunk = self.payload[self.offset:self.offset + size]
            self.offset += len(chunk)
            return chunk

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def test_only_credential_free_https_urls_are_accepted(self):
        _, origin = fetch_events.validate_jmap_https_url("https://calendar.example/jmap/session")
        self.assertEqual(origin, ("https", "calendar.example", 443))
        for url in (
            "http://calendar.example/jmap/session",
            "https://user:pass@calendar.example/jmap/session",
            "https://calendar.example/jmap/session#fragment",
            "https://calendar.example\\@attacker.example/session",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                fetch_events.validate_jmap_https_url(url)

    def test_api_and_redirect_urls_must_keep_the_session_origin(self):
        _, origin = fetch_events.validate_jmap_https_url("https://calendar.example/session")
        fetch_events.validate_jmap_https_url("https://calendar.example/api", origin)
        with self.assertRaises(ValueError):
            fetch_events.validate_jmap_https_url("https://attacker.example/api", origin)
        handler = fetch_events.JmapSameOriginRedirectHandler(origin)
        with self.assertRaises(ValueError):
            handler.redirect_request(None, None, 302, "Found", {}, "https://attacker.example/api")

    def test_http_session_is_rejected_before_request_construction(self):
        calendar = {"name": "unsafe", "type": "jmap", "jmapUrl": "http://calendar.example/session", "jmapToken": "secret"}
        with mock.patch.object(fetch_events.urllib.request, "build_opener") as build_opener:
            result = fetch_events.fetch_jmap_calendar(calendar, fetch_events.datetime.now(), fetch_events.datetime.now())
        build_opener.assert_not_called()
        self.assertIn("must use HTTPS", result["status"])

    def test_untrusted_api_url_is_rejected_before_second_authenticated_request(self):
        session = json.dumps({"apiUrl": "https://attacker.example/api"}).encode("utf-8")
        opener = mock.Mock()
        opener.open.return_value = self.FakeResponse("https://calendar.example/session", session)
        calendar = {"name": "unsafe", "type": "jmap", "jmapUrl": "https://calendar.example/session", "jmapToken": "secret"}
        with mock.patch.object(fetch_events.urllib.request, "build_opener", return_value=opener):
            result = fetch_events.fetch_jmap_calendar(calendar, fetch_events.datetime.now(), fetch_events.datetime.now())
        self.assertEqual(opener.open.call_count, 1)
        self.assertIn("configured session origin", result["status"])

    def test_final_response_origin_is_checked_and_closed(self):
        _, origin = fetch_events.validate_jmap_https_url("https://calendar.example/session")
        response = self.FakeResponse("https://attacker.example/session")
        opener = mock.Mock()
        opener.open.return_value = response
        request = fetch_events.urllib.request.Request("https://calendar.example/session")
        with self.assertRaises(ValueError):
            fetch_events.open_trusted_jmap(opener, request, origin, timeout=1)
        self.assertTrue(response.closed)


class MeetingUrlSecurityTests(unittest.TestCase):
    def test_validate_meeting_url_accepts_clean_http_and_https_urls(self):
        valid_urls = [
            "https://meet.google.com/abc-defg-hij",
            "https://us02web.zoom.us/j/1234567890",
            "https://teams.microsoft.com/l/meetup-join/19%3ameeting",
            "https://custom.server.com/rooms/101",
            "http://local.meeting.net/join/test",
        ]
        for url in valid_urls:
            with self.subTest(url=url):
                self.assertEqual(fetch_events.validate_meeting_url(url), url)

    def test_validate_meeting_url_rejects_dangerous_and_malformed_schemes(self):
        invalid_urls = [
            "javascript:alert(1)",
            "file:///bin/sh",
            "data:text/html,<script>alert(1)</script>",
            "<h1>Meeting Title</h1>",
            "<script src='http://evil.com'></script>",
            "https://meet.google.com/<script>",
            "https://user:pass@meet.google.com/room",
            "https://zoom.us/j/123\nnewline",
            "ftp://files.example.com/meet",
            "",
            None,
            12345,
            "https://",
            "https://[invalid-host",
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                self.assertEqual(fetch_events.validate_meeting_url(url), "")

    def test_jmap_virtual_locations_unsafe_uri_is_sanitized(self):
        raw_events = [
            {
                "id": "jmap_unsafe_1",
                "title": "Malicious Meeting",
                "start": "2026-08-25T10:00:00Z",
                "virtualLocations": {
                    "loc1": {"uri": "file:///bin/sh"},
                    "loc2": {"uri": "<img src=x onerror=alert(1)>"},
                    "loc3": {"uri": "javascript:window.open()"},
                },
            },
            {
                "id": "jmap_safe_1",
                "title": "Safe Meeting",
                "start": "2026-08-25T11:00:00Z",
                "virtualLocations": {
                    "loc1": {"uri": "https://meet.jit.si/SafeRoom"},
                },
            },
        ]
        cal_info = {"name": "Test JMAP", "type": "jmap", "jmapToken": "secret"}
        start = fetch_events.datetime(2026, 8, 25, 0, 0, 0)
        end = fetch_events.datetime(2026, 8, 25, 23, 59, 59)

        # Mock JMAP network responses to return raw_events
        session_json = json.dumps({
            "apiUrl": "https://calendar.example/api",
            "accounts": {"acc1": {"accountCapabilities": {"urn:ietf:params:jmap:calendars": {}}}},
            "primaryAccounts": {"urn:ietf:params:jmap:calendars": "acc1"},
        }).encode("utf-8")

        query_get_json = json.dumps({
            "methodResponses": [
                ["CalendarEvent/get", {"list": raw_events}, "get0"]
            ]
        }).encode("utf-8")

        fake_session = JmapTransportTests.FakeResponse("https://calendar.example/session", session_json)
        fake_api = JmapTransportTests.FakeResponse("https://calendar.example/api", query_get_json)

        opener = mock.Mock()
        opener.open.side_effect = [fake_session, fake_api]

        with mock.patch.object(fetch_events.urllib.request, "build_opener", return_value=opener):
            cal_info["jmapUrl"] = "https://calendar.example/session"
            result = fetch_events.fetch_jmap_calendar(cal_info, start, end)

        self.assertEqual(result["status"], "ok")
        events = result["events"]
        self.assertEqual(len(events), 2)
        # Event 1 with dangerous URIs should have empty meetingUrl and meetingProvider
        self.assertEqual(events[0]["meetingUrl"], "")
        self.assertEqual(events[0]["meetingProvider"], "")
        # Event 2 with safe Jitsi URI should be extracted
        self.assertEqual(events[1]["meetingUrl"], "https://meet.jit.si/SafeRoom")
        self.assertEqual(events[1]["meetingProvider"], "Jitsi")


class RecurrenceCpuCeilingTests(unittest.TestCase):
    def test_multiday_expansion_on_multi_century_event_is_clamped_and_bounded(self):
        window_start = fetch_events.datetime(2026, 8, 1)
        window_end = fetch_events.datetime(2026, 8, 31)
        # Event spanning 1000 years
        event = {
            "id": "centuries_evt",
            "title": "Century Span",
            "start_dt": fetch_events.datetime(1900, 1, 1),
            "end_dt": fetch_events.datetime(2900, 1, 1),
            "all_day": True,
            "calendar": "Test",
            "color": "#4A90E2",
            "location": "",
            "description": "",
        }
        instances = fetch_events.expand_multiday_event(event, window_start, window_end)
        # Should be bounded to days within the window (31 days) and not loop millions of times
        self.assertLessEqual(len(instances), 31)
        self.assertGreater(len(instances), 0)
        for inst in instances:
            dt = fetch_events.datetime.strptime(inst["date_key"], "%Y-%m-%d")
            self.assertTrue(window_start <= dt <= window_end)

    def test_multiday_expansion_outside_window_returns_empty_immediately(self):
        window_start = fetch_events.datetime(2026, 8, 1)
        window_end = fetch_events.datetime(2026, 8, 31)
        event = {
            "id": "past_evt",
            "title": "Past",
            "start_dt": fetch_events.datetime(2020, 1, 1),
            "end_dt": fetch_events.datetime(2020, 1, 10),
            "all_day": True,
            "calendar": "Test",
            "color": "#4A90E2",
            "location": "",
            "description": "",
        }
        instances = fetch_events.expand_multiday_event(event, window_start, window_end)
        self.assertEqual(instances, [])

    def test_daily_recurrence_from_distant_past_terminates_under_cpu_ceiling(self):
        window_start = fetch_events.datetime(2026, 8, 1)
        window_end = fetch_events.datetime(2026, 8, 31)
        event = {
            "id": "old_rec",
            "title": "Ancient Daily Recurrence",
            "start_dt": fetch_events.datetime(1000, 1, 1, 9, 0, 0),
            "end_dt": fetch_events.datetime(1000, 1, 1, 10, 0, 0),
            "all_day": False,
            "calendar": "Test",
            "color": "#4A90E2",
            "location": "",
            "description": "",
            "rrule": {"FREQ": "DAILY", "INTERVAL": "1"},
        }
        # Should fast-forward and expand within window without timeout
        instances = fetch_events.expand_recurring_event(event, window_start, window_end)
        self.assertGreater(len(instances), 0)
        self.assertLessEqual(len(instances), 31)
        for inst in instances:
            self.assertTrue(window_start <= inst["start_dt"] <= window_end)

    def test_weekly_recurrence_from_distant_past_terminates_under_cpu_ceiling(self):
        window_start = fetch_events.datetime(2026, 8, 1)
        window_end = fetch_events.datetime(2026, 8, 31)
        event = {
            "id": "old_weekly",
            "title": "Ancient Weekly Recurrence",
            "start_dt": fetch_events.datetime(1500, 1, 1, 9, 0, 0),
            "end_dt": fetch_events.datetime(1500, 1, 1, 10, 0, 0),
            "all_day": False,
            "calendar": "Test",
            "color": "#4A90E2",
            "location": "",
            "description": "",
            "rrule": {"FREQ": "WEEKLY", "INTERVAL": "1"},
        }
        instances = fetch_events.expand_recurring_event(event, window_start, window_end)
        self.assertGreater(len(instances), 0)
        for inst in instances:
            self.assertTrue(window_start <= inst["start_dt"] <= window_end)


class ConfigAndStateSecurityTests(unittest.TestCase):
    def test_save_config_from_stdin(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "calendars.json")
            with mock.patch.object(fetch_events, "CONFIG_PATH", config_path):
                stdin_payload = json.dumps([{"name": "Fastmail", "type": "jmap", "jmapToken": "secret-token"}])
                with mock.patch("sys.argv", ["fetch-events.py", "--save-config"]), \
                     mock.patch("sys.stdin.readline", return_value=stdin_payload + "\n"), \
                     self.assertRaises(SystemExit) as cm:
                    fetch_events.main()
                self.assertEqual(cm.exception.code, 0)
                saved = fetch_events.safe_load_json(config_path)
                self.assertEqual(saved, [{"name": "Fastmail", "type": "jmap", "jmapToken": "secret-token"}])
                file_stat = os.stat(config_path)
                self.assertEqual(stat.S_IMODE(file_stat.st_mode), 0o600)

    def test_purge_plugin_data_removes_config_and_oauth_state(self):
        with tempfile.TemporaryDirectory() as directory:
            config_file = os.path.join(directory, "calendars.json")
            auth_file = os.path.join(directory, "google-auth.json")
            events_file = os.path.join(directory, "calendar-events.json")
            cache_file = os.path.join(directory, "translation-cache.json")

            with open(config_file, "w") as f:
                f.write('{"jmapToken": "secret"}')
            with open(auth_file, "w") as f:
                f.write('{"refresh_token": "secret"}')
            with open(events_file, "w") as f:
                f.write('{"events": []}')
            with open(cache_file, "w") as f:
                f.write('{}')

            with mock.patch.object(fetch_events, "CONFIG_PATH", config_file), \
                 mock.patch.object(fetch_events, "AUTH_FILE", auth_file), \
                 mock.patch.object(fetch_events, "OUTPUT_PATH", events_file), \
                 mock.patch.object(fetch_events, "TRANSLATION_CACHE_PATH", cache_file), \
                 mock.patch.object(fetch_events, "STATE_DIR", directory):
                res = fetch_events.purge_plugin_data()
                self.assertEqual(res["status"], "success")
                self.assertFalse(os.path.exists(config_file))
                self.assertFalse(os.path.exists(auth_file))
                self.assertFalse(os.path.exists(events_file))
                self.assertFalse(os.path.exists(cache_file))


if __name__ == "__main__":
    unittest.main()
