"""
Email notifier using Resend (https://resend.com).

``send_notification(subject, body)`` sends a single email to
``NOTIFICATION_TARGET`` and returns True on success, False otherwise.
Used by ``scripts.refresh_config`` to ping the operator after a
standard-mode run so they know to review and promote the new generation.

Required environment variables (read at call time):

    NOTIFICATION_TARGET   recipient email address
    RESEND_API_KEY        Resend API key from https://resend.com/api-keys
    EMAIL_FROM            sender address, e.g. "Refresh Bot <bot@yourdomain.com>".
                          Must be a verified Resend domain OR use Resend's
                          "onboarding@resend.dev" sandbox.

If ``RESEND_API_KEY`` or ``NOTIFICATION_TARGET`` is missing, the function
logs a warning and returns False without raising. This keeps the pipeline
safe on machines that aren't configured to send mail.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

DEFAULT_FROM = "onboarding@resend.dev"  # Resend sandbox; works without domain setup


def send_notification(subject: str, body: str) -> bool:
    """Send a plain-text email. Returns True if sent, False otherwise.

    Failure modes (missing config, auth error, network issues) are logged
    but never raised — notifications are best-effort.
    """
    target: Optional[str] = os.getenv("NOTIFICATION_TARGET")
    if not target:
        logging.info("NOTIFICATION_TARGET not set; skipping email")
        return False

    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        logging.warning("RESEND_API_KEY not set; skipping email notification")
        return False

    sender = os.getenv("EMAIL_FROM", DEFAULT_FROM)

    try:
        import resend  # imported lazily so missing dep doesn't break prod boot
    except ImportError:
        logging.warning("`resend` package not installed; run `pip install resend`")
        return False

    resend.api_key = api_key

    try:
        resp = resend.Emails.send({
            "from": sender,
            "to": [target],
            "subject": subject,
            "text": body,
        })
        email_id = resp.get("id") if isinstance(resp, dict) else None
        logging.info(f"Email sent to {target} (subject: {subject!r}, id: {email_id})")
        return True
    except Exception as ex:
        logging.error(f"Failed to send email notification: {ex}", exc_info=True)
        return False
