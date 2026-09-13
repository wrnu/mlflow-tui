from __future__ import annotations

import os
from collections.abc import Callable
from getpass import getpass
from urllib.parse import unquote, urlparse, urlunparse

_HTTP_PATCHED = False


def strip_userinfo(uri: str) -> str:
    """Return *uri* with any embedded user:password removed."""
    parsed = urlparse(uri)
    if not parsed.username and not parsed.password:
        return uri
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=host))


def drop_default_port(uri: str) -> str:
    """Drop :80 / :443 so the Host header matches typical --allowed-hosts values."""
    parsed = urlparse(uri)
    drop_https = parsed.scheme == "https" and parsed.port == 443
    drop_http = parsed.scheme == "http" and parsed.port == 80
    if not (drop_https or drop_http):
        return uri
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return urlunparse(parsed._replace(netloc=host))


def prefer_loopback_ipv4(uri: str) -> str:
    """Rewrite localhost to 127.0.0.1 so we do not hit IPv6 listeners on the same port.

    On macOS, ``localhost:5000`` often resolves to ``::1`` first, where Control Center
    AirPlay (AirTunes) is bound, while MLflow is on ``127.0.0.1:5000``.
    """
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        return uri
    if (parsed.hostname or "").lower() != "localhost":
        return uri
    host = "127.0.0.1"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=host))


def host_header_without_port(host_creds: object, extra_headers: dict | None) -> dict:
    """Send Host without :port so it matches MLflow --allowed-hosts like localhost,127.0.0.1."""
    extra = dict(extra_headers or {})
    if any(key.lower() == "host" for key in extra):
        return extra
    candidates = host_header_candidates(host_creds)
    if candidates:
        extra["Host"] = candidates[0]
    return extra


def host_header_candidates(host_creds: object) -> list[str]:
    """Host values to try. MLflow 3 --allowed-hosts is exact: `127.0.0.1` ≠ `127.0.0.1:5001`."""
    parsed = urlparse(getattr(host_creds, "host", "") or "")
    host = parsed.hostname
    if not host:
        return []
    if ":" in host:
        host = f"[{host}]"
    candidates = [host]
    default_port = {"http": 80, "https": 443}.get(parsed.scheme, None)
    if parsed.port and parsed.port != default_port:
        candidates.append(f"{host}:{parsed.port}")
    return candidates


def _invalid_host_response(response: object) -> bool:
    status = getattr(response, "status_code", None)
    if status != 403:
        return False
    text = (getattr(response, "text", None) or "").lower()
    return "invalid host" in text or "dns rebinding" in text


def _empty_api_response(response: object, endpoint: str) -> bool:
    if getattr(response, "status_code", None) != 200:
        return False
    if not str(endpoint).startswith("/api/"):
        return False
    return not (getattr(response, "text", None) or "").strip()


def _userinfo(uri: str | None) -> tuple[str | None, str | None]:
    if not uri:
        return None, None
    parsed = urlparse(uri)
    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None
    return username or None, password or None


def _first(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def should_strip_auth_on_redirect(old_url: str, new_url: str) -> bool:
    """Strip Authorization only when the hostname changes, not on scheme/port hops.

    urllib3 and requests both drop Authorization on ``http://host:5000`` →
    ``https://host`` redirects. Reverse proxies do that hop constantly, which
    shows up as 403 after a username/password was already supplied.
    """
    old_host = (urlparse(old_url).hostname or "").lower()
    new_host = (urlparse(new_url).hostname or "").lower()
    return old_host != new_host


def _preserve_auth_on_redirects(session: object) -> object:
    session.should_strip_auth = should_strip_auth_on_redirect  # type: ignore[attr-defined]
    for adapter in getattr(session, "adapters", {}).values():
        retry = getattr(adapter, "max_retries", None)
        if retry is None:
            continue
        # Let requests follow 3xx so our should_strip_auth hook runs.
        retry.redirect = False
        retry.remove_headers_on_redirect = frozenset()
    return session


def inject_requests_basic_auth(host_creds: object, kwargs: dict) -> dict:
    """Attach requests HTTP Basic auth in addition to MLflow's Authorization header."""
    if kwargs.get("auth") is not None:
        return kwargs
    if getattr(host_creds, "aws_sigv4", False) or getattr(host_creds, "auth", None):
        return kwargs
    username = getattr(host_creds, "username", None)
    password = getattr(host_creds, "password", None)
    if username and password:
        kwargs = dict(kwargs)
        kwargs["auth"] = (username, password)
    return kwargs


def install_redirect_safe_basic_auth() -> None:
    """Patch MLflow's HTTP client so Basic credentials survive reverse-proxy redirects."""
    global _HTTP_PATCHED
    if _HTTP_PATCHED:
        return
    try:
        from mlflow.utils import request_utils, rest_utils
    except ImportError:
        return

    original_http_request = rest_utils.http_request
    original_get_session = request_utils._get_request_session

    def http_request(host_creds, endpoint, method, *args, extra_headers=None, **kwargs):
        kwargs = inject_requests_basic_auth(host_creds, kwargs)
        extra = dict(extra_headers or {})
        if any(key.lower() == "host" for key in extra):
            return original_http_request(
                host_creds, endpoint, method, *args, extra_headers=extra, **kwargs
            )
        last = None
        for host in host_header_candidates(host_creds) or [None]:
            headers = dict(extra)
            if host:
                headers["Host"] = host
            last = original_http_request(
                host_creds, endpoint, method, *args, extra_headers=headers, **kwargs
            )
            if _invalid_host_response(last) or _empty_api_response(last, endpoint):
                continue
            return last
        return last

    def get_request_session(*args, **kwargs):
        return _preserve_auth_on_redirects(original_get_session(*args, **kwargs))

    http_request.__wrapped__ = original_http_request  # type: ignore[attr-defined]
    rest_utils.http_request = http_request
    request_utils._get_request_session = get_request_session
    _HTTP_PATCHED = True


def apply_tracking_auth(
    tracking_uri: str | None,
    *,
    username: str | None = None,
    password: str | None = None,
    token: str | None = None,
    prompt_password: bool = True,
    prompt: Callable[[str], str] = getpass,
) -> str | None:
    """Configure MLflow HTTP auth via the env vars the official client reads.

    Credentials are taken from CLI flags, then URI userinfo
    (``https://user:pass@host``), then existing environment variables. If a
    username is known and no password is, the user is prompted.
    """
    uri_user, uri_password = _userinfo(tracking_uri)
    username = _first(username, uri_user, os.environ.get("MLFLOW_TRACKING_USERNAME"))
    password = _first(password, uri_password, os.environ.get("MLFLOW_TRACKING_PASSWORD"))
    token = _first(token, os.environ.get("MLFLOW_TRACKING_TOKEN"))

    if username and not password and prompt_password:
        password = prompt("MLflow password: ") or None

    if username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
    if password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    if token and not (username and password):
        os.environ["MLFLOW_TRACKING_TOKEN"] = token
    elif username and password:
        # Basic auth is what we will send. A leftover bearer token would be
        # used instead if the password were missing on a later request.
        os.environ.pop("MLFLOW_TRACKING_TOKEN", None)

    install_redirect_safe_basic_auth()

    if tracking_uri:
        return drop_default_port(prefer_loopback_ipv4(strip_userinfo(tracking_uri)))
    return tracking_uri


def auth_hint(exc: BaseException) -> str | None:
    text = str(exc)
    lower = text.lower()
    if "invalid host header" in lower or "dns rebinding" in lower:
        return (
            "the tracking server rejected the Host header (MLflow 3 DNS-rebinding "
            "protection). this is not a username/password problem — the server must "
            "list this hostname in --allowed-hosts / MLFLOW_SERVER_ALLOWED_HOSTS"
        )
    if "permission denied" in lower:
        return (
            "the server authenticated the request but this user is not allowed to "
            "read experiments. check the MLflow auth ACL / admin settings"
        )
    if not any(marker in lower for marker in ("401", "403", "unauthorized", "forbidden")):
        return None

    has_basic = bool(
        os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD")
    )
    has_token = bool(os.environ.get("MLFLOW_TRACKING_TOKEN"))
    if "401" in lower or "unauthorized" in lower:
        if has_basic or has_token:
            return (
                "the tracking server rejected the credentials. "
                "confirm --username/--password (or MLFLOW_TRACKING_USERNAME / "
                "MLFLOW_TRACKING_PASSWORD). if you use uv, pass flags after the "
                "command: uv run mlflow-tui --password '…'  (uv run -p is --python)"
            )
        return (
            "the tracking server requires authentication. "
            "pass --username/--password or set MLFLOW_TRACKING_USERNAME "
            "and MLFLOW_TRACKING_PASSWORD"
        )
    # 403
    if has_basic or has_token:
        return (
            "credentials were sent but the server still returned 403. "
            "common causes: HTTP→HTTPS redirect dropping Authorization (use the "
            "https tracking URI), MLflow 3 --allowed-hosts rejecting this Host "
            "header, or an ACL that forbids this user"
        )
    return (
        "the tracking server rejected the request (403). "
        "pass --username/--password or set MLFLOW_TRACKING_USERNAME "
        "and MLFLOW_TRACKING_PASSWORD"
    )
