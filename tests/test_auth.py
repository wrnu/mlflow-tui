from __future__ import annotations

import base64
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from types import SimpleNamespace

from mlflow_tui.auth import (
    apply_tracking_auth,
    auth_hint,
    drop_default_port,
    host_header_candidates,
    host_header_without_port,
    inject_requests_basic_auth,
    install_redirect_safe_basic_auth,
    prefer_loopback_ipv4,
    should_strip_auth_on_redirect,
    strip_userinfo,
)


def test_strip_userinfo_removes_credentials() -> None:
    assert (
        strip_userinfo("https://alice:s3cret@mlflow.example:5000/path")
        == "https://mlflow.example:5000/path"
    )
    assert strip_userinfo("http://localhost:5000") == "http://localhost:5000"


def test_apply_auth_from_uri_sets_env(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    uri = apply_tracking_auth(
        "https://alice:s3cret@mlflow.example:5000",
        prompt_password=False,
    )
    assert uri == "https://mlflow.example:5000"
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "alice"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "s3cret"


def test_apply_auth_cli_overrides_uri(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    apply_tracking_auth(
        "https://alice:from-uri@mlflow.example",
        username="bob",
        password="cli-pass",
        prompt_password=False,
    )
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "bob"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "cli-pass"


def test_apply_auth_cli_password_does_not_prompt(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    prompted = {"called": False}

    def fail_prompt(_: str) -> str:
        prompted["called"] = True
        return "nope"

    apply_tracking_auth(
        "http://localhost:5000",
        username="alice",
        password="from-cli",
        prompt_password=True,
        prompt=fail_prompt,
    )
    assert prompted["called"] is False
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "from-cli"


def test_apply_auth_prompts_when_username_without_password(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    apply_tracking_auth(
        "http://localhost:5000",
        username="alice",
        prompt_password=True,
        prompt=lambda _: "prompted",
    )
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "prompted"


def test_apply_auth_does_not_prompt_when_env_has_password(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "alice")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "env-pass")
    prompted = {"called": False}

    def fail_prompt(_: str) -> str:
        prompted["called"] = True
        return "nope"

    apply_tracking_auth("http://localhost:5000", prompt_password=True, prompt=fail_prompt)
    assert prompted["called"] is False
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "env-pass"


def test_apply_auth_sets_token(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    apply_tracking_auth("http://localhost:5000", token="abc", prompt_password=False)
    assert os.environ["MLFLOW_TRACKING_TOKEN"] == "abc"


def test_apply_auth_basic_clears_stale_token(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "stale-bearer")
    apply_tracking_auth(
        "http://localhost:5000",
        username="alice",
        password="secret",
        prompt_password=False,
    )
    assert "MLFLOW_TRACKING_TOKEN" not in os.environ
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "secret"


def test_drop_default_https_port() -> None:
    assert drop_default_port("https://mlflow.example:443/path") == "https://mlflow.example/path"
    assert drop_default_port("http://localhost:5000") == "http://localhost:5000"


def test_prefer_loopback_ipv4() -> None:
    assert prefer_loopback_ipv4("http://localhost:5000") == "http://127.0.0.1:5000"
    assert prefer_loopback_ipv4("http://mlflow.example:5000") == "http://mlflow.example:5000"


def test_apply_auth_rewrites_localhost(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    uri = apply_tracking_auth(
        "http://localhost:5000",
        username="alice",
        password="secret",
        prompt_password=False,
    )
    assert uri == "http://127.0.0.1:5000"


def test_host_header_without_port() -> None:
    creds = SimpleNamespace(host="http://127.0.0.1:5000")
    assert host_header_without_port(creds, None)["Host"] == "127.0.0.1"
    assert host_header_without_port(creds, {"Host": "kept"})["Host"] == "kept"


def test_host_header_candidates_try_with_and_without_port() -> None:
    creds = SimpleNamespace(host="http://127.0.0.1:5001")
    assert host_header_candidates(creds) == ["127.0.0.1", "127.0.0.1:5001"]
    https = SimpleNamespace(host="https://mlflow.example")
    assert host_header_candidates(https) == ["mlflow.example"]


def test_should_strip_auth_only_on_hostname_change() -> None:
    assert not should_strip_auth_on_redirect(
        "http://mlflow.example:5000/api",
        "https://mlflow.example/api",
    )
    assert should_strip_auth_on_redirect(
        "http://mlflow.example/api",
        "https://evil.example/api",
    )


def test_inject_requests_basic_auth() -> None:
    creds = SimpleNamespace(username="alice", password="secret", auth=None, aws_sigv4=False)
    assert inject_requests_basic_auth(creds, {})["auth"] == ("alice", "secret")
    skipped = inject_requests_basic_auth(creds, {"auth": ("other", "x")})
    assert skipped["auth"] == ("other", "x")


def test_auth_hint_for_forbidden(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    hint = auth_hint(Exception("API request failed with error code 403 Forbidden"))
    assert hint is not None
    assert "MLFLOW_TRACKING_USERNAME" in hint
    assert auth_hint(Exception("connection refused")) is None


def test_auth_hint_for_invalid_host() -> None:
    hint = auth_hint(
        Exception(
            "API request failed with error code 403. "
            "Response body: 'Invalid Host header - possible DNS rebinding attack detected'"
        )
    )
    assert hint is not None
    assert "allowed-hosts" in hint


def test_auth_hint_for_acl_denial() -> None:
    hint = auth_hint(Exception("PERMISSION_DENIED: Permission denied."))
    assert hint is not None
    assert "ACL" in hint


def test_auth_hint_when_credentials_already_sent(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "alice")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret")
    hint = auth_hint(Exception("API request failed with error code 403 Forbidden"))
    assert hint is not None
    assert "credentials were sent" in hint


def test_basic_auth_survives_cross_port_redirect() -> None:
    expected = "Basic " + base64.standard_b64encode(b"alice:secret").decode()

    class AuthHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            if self.headers.get("Authorization") != expected:
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b"forbidden")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

    class RedirectHandler(BaseHTTPRequestHandler):
        target_port = 0

        def log_message(self, *_args) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            self.send_response(307)
            self.send_header("Location", f"http://127.0.0.1:{self.target_port}{self.path}")
            self.end_headers()

    auth_server = HTTPServer(("127.0.0.1", 0), AuthHandler)
    redirect_server = HTTPServer(("127.0.0.1", 0), RedirectHandler)
    RedirectHandler.target_port = auth_server.server_address[1]
    threads = [
        Thread(target=server.serve_forever, daemon=True)
        for server in (auth_server, redirect_server)
    ]
    for thread in threads:
        thread.start()
    try:
        from mlflow.utils import rest_utils
        from mlflow.utils.rest_utils import MlflowHostCreds

        install_redirect_safe_basic_auth()
        creds = MlflowHostCreds(
            host=f"http://127.0.0.1:{redirect_server.server_address[1]}",
            username="alice",
            password="secret",
        )
        patched = rest_utils.http_request(
            creds, "/api/2.0/mlflow/experiments/search", "POST", json={}
        )
        assert patched.status_code == 200
    finally:
        auth_server.shutdown()
        redirect_server.shutdown()
