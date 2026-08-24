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


if __name__ == "__main__":
    unittest.main()
