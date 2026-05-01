from __future__ import annotations

import unittest

from steering.local_client import LocalServerError, parse_json_response


class LocalClientTests(unittest.TestCase):
    def test_parse_json_response_reports_invalid_json_as_local_server_error(self) -> None:
        with self.assertRaisesRegex(LocalServerError, "invalid JSON response from server"):
            parse_json_response("not-json", source="server")

    def test_parse_json_response_returns_decoded_data(self) -> None:
        self.assertEqual(parse_json_response('{"text": "ok"}', source="server"), {"text": "ok"})


if __name__ == "__main__":
    unittest.main()
