import json
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from refract.refactoring.providers import _post_json


def _fake_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _http_error(code: int) -> HTTPError:
    return HTTPError(url="https://example.test", code=code, msg="err", hdrs=None, fp=None)  # type: ignore[arg-type]


@patch("refract.refactoring.providers.time.sleep")
@patch("refract.refactoring.providers.request.urlopen")
def test_post_json_retries_on_503_then_succeeds(mock_urlopen, mock_sleep) -> None:
    # exactly what happened live: identical request, 503 then success with no changes
    mock_urlopen.side_effect = [_http_error(503), _fake_response({"ok": True})]

    result = _post_json("https://example.test", {}, {})

    assert result == {"ok": True}
    assert mock_urlopen.call_count == 2
    mock_sleep.assert_called_once()


@patch("refract.refactoring.providers.time.sleep")
@patch("refract.refactoring.providers.request.urlopen")
def test_post_json_gives_up_after_max_attempts(mock_urlopen, mock_sleep) -> None:
    mock_urlopen.side_effect = [_http_error(503), _http_error(503), _http_error(503)]

    with pytest.raises(HTTPError):
        _post_json("https://example.test", {}, {})

    assert mock_urlopen.call_count == 3


@patch("refract.refactoring.providers.time.sleep")
@patch("refract.refactoring.providers.request.urlopen")
def test_post_json_does_not_retry_non_retryable_status(mock_urlopen, mock_sleep) -> None:
    mock_urlopen.side_effect = [_http_error(400)]

    with pytest.raises(HTTPError):
        _post_json("https://example.test", {}, {})

    assert mock_urlopen.call_count == 1
    mock_sleep.assert_not_called()
