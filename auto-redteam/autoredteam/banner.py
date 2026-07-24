"""banner.py -- the authorized-use gate and safety posture of the harness itself.

This tool automates the DISCOVERY of policy violations in LLMs so that they can be
REPORTED and FIXED. That is a defensive purpose, and it is only legitimate against
systems you are authorized to test. This module makes that contract explicit and
refuses to run a campaign unless the operator affirms it in config.

Design principles (mirrored from the harness plan, section 7):
  * The local attacker has NO network egress except the configured model server.
  * Defender API keys are read from the environment by name and NEVER logged.
  * Every run records a config hash + model versions for reproducible disclosure.
  * A campaign will not start unless `authorization.confirmed` is true AND an
    `authorization.scope` string (who authorized this, against what) is provided.
"""
from __future__ import annotations

BANNER = r"""
================================================================================
  auto-redteam  --  AUTHORIZED AI-SAFETY RESEARCH ONLY
--------------------------------------------------------------------------------
  This harness generates adversarial prompts to find and REPORT LLM policy
  violations. Use it only against models and endpoints you are authorized to
  test. Findings are for defensive remediation. Do not use discovered attacks
  to cause harm. API keys are never logged; the attacker has no network egress
  beyond its configured model server.
================================================================================
"""


class AuthorizationError(RuntimeError):
    """Raised when a campaign is launched without a confirmed authorization scope."""


def assert_authorized(authorization: dict | None) -> str:
    """Gate campaign launch on an explicit, logged authorization scope.

    `authorization` comes from the campaign config, e.g.::

        authorization:
          confirmed: true
          scope: "internal safety eval of our own Gemini deployment, ticket SAFE-1234"

    Returns the scope string (to be recorded in the run manifest) or raises.
    """
    authorization = authorization or {}
    if not authorization.get("confirmed"):
        raise AuthorizationError(
            "Refusing to launch: set authorization.confirmed=true in the campaign "
            "config to affirm this is an authorized test."
        )
    scope = (authorization.get("scope") or "").strip()
    if len(scope) < 10:
        raise AuthorizationError(
            "Refusing to launch: authorization.scope must describe who authorized "
            "this test and against what target (>=10 chars)."
        )
    return scope
