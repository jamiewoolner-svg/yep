#!/usr/bin/env python3
"""
King Kam Daily Market Briefing — emailed via GitHub Actions.

Runs the Kona v56 watchlist scanner, feeds results to King Kam (Claude),
and emails the Hawaiian Pidgin market briefing to the configured recipient.

Schedule (GitHub Actions cron, UTC → ET):
  14:00 UTC = 9 AM ET   — pre-market briefing
  17:00 UTC = 12 PM ET  — midday check-in
  20:00 UTC = 3 PM ET   — afternoon wrap-up
"""

import os
import sys
import json
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kam-briefing")

# ── Config from environment ──────────────────────────────────────────────────
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "")
TO_EMAIL = os.environ.get("TO_EMAIL", "")


def fetch_sp500_tickers() -> list[str]:
    """Grab S&P 500 tickers from Wikipedia."""
    import re
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = Request(url, headers={"User-Agent": "KonaScanner/1.0"})
    html = urlopen(req, timeout=15).read().decode()
    tickers = re.findall(r'<a[^>]*class="external text"[^>]*>([A-Z]{1,5})</a>', html)
    if not tickers:
        tickers = re.findall(r'nyse\.com/quote/[^"]*">([A-Z]{1,5})<', html)
    if not tickers:
        tickers = re.findall(r'nasdaq\.com/market-activity/stocks/[^"]*">([A-Z]{1,5})<', html)
    seen = set()
    unique = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:100]  # Top 100 for speed in CI


def run_watchlist_scan(tickers: list[str]) -> list[dict]:
    """Run Kona watchlist scan on the given tickers."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("POLYGON_API_KEY", POLYGON_API_KEY)
    from kailua_kona import scan_watchlist
    return scan_watchlist(tickers)


def format_scan_text(watchlist: list[dict]) -> str:
    """Format watchlist results as plain text for King Kam to read."""
    if not watchlist:
        return "No tickers approaching signal thresholds right now."
    lines = [f"{'Ticker':<8} {'Side':<6} {'Score':>5} {'BB%':>6} {'Stoch':>6} {'MA':<8} {'Status'}"]
    lines.append("-" * 60)
    for w in watchlist[:20]:
        flags = []
        if not w.get("earnings_ok", True):
            flags.append("earnings")
        if not w.get("month_ok", True):
            flags.append("month")
        if not w.get("strict_ok", True):
            flags.append("strict")
        status = ", ".join(flags) if flags else "OK"
        if w.get("triggered"):
            status = "TRIGGERED"
        lines.append(
            f"{w['ticker']:<8} {w['side']:<6} {w['score']:>5.0f} "
            f"{w['bb_pct']*100:>5.0f}% {w['stoch_k']:>5.0f} "
            f"{w['ma_state']:<8} {status}"
        )
    return "\n".join(lines)


def get_time_label() -> str:
    """Return a human-friendly label for the current scan time."""
    hour = datetime.utcnow().hour
    if hour <= 15:
        return "Pre-Market Morning"
    elif hour <= 18:
        return "Midday"
    else:
        return "Afternoon Wrap-Up"


def ask_king_kam(scan_text: str, time_label: str) -> str:
    """Ask King Kam to write the market briefing."""
    if not ANTHROPIC_API_KEY:
        return f"King Kam Briefing (no AI key — raw scan)\n\n{scan_text}"

    prompt = f"""Write a short market briefing email in your King Kam Hawaiian Pidgin style.

Time: {time_label} — {datetime.utcnow().strftime('%A, %B %d %Y')}

Here are today's watchlist scan results from the Kona v56 engine:

{scan_text}

Write the briefing covering:
1. Quick market vibe (1-2 sentences)
2. Top setups approaching signals and why they look interesting
3. Any warnings (earnings blocks, strict filter fails)
4. One piece of historical market wisdom

Keep it 3-5 short paragraphs. Sign off as King Kam. Use Hawaiian Pidgin naturally.
Do NOT give specific buy/sell advice — just context and color."""

    payload = json.dumps({
        "model": os.environ.get("KONA_AI_MODEL", "claude-sonnet-4-6"),
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}],
        "system": (
            "You are King Kam (King Kamehameha), the legendary Hawaiian trading uncle. "
            "You speak in Hawaiian Pidgin. You are writing a market briefing email."
        ),
    }).encode()

    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    resp = urlopen(req, timeout=60)
    body = json.loads(resp.read())
    return body["content"][0]["text"]


def send_email(subject: str, body_text: str) -> None:
    """Send the briefing email via SMTP."""
    if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, FROM_EMAIL, TO_EMAIL]):
        log.warning("SMTP not fully configured — printing to stdout instead.")
        print(f"\nSubject: {subject}\n\n{body_text}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL

    # Plain text
    msg.attach(MIMEText(body_text, "plain"))

    # Simple HTML version
    html_body = body_text.replace("\n", "<br>")
    html = f"""\
<html>
<body style="font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #f5f5f5;
            padding: 24px; border-radius: 12px;">
<h2 style="color: #f0c040; margin-top: 0;">🤙 King Kam Market Briefing</h2>
<p style="line-height: 1.6;">{html_body}</p>
<hr style="border: 1px solid #333; margin: 16px 0;">
<p style="font-size: 12px; color: #888;">Kona v56 — Not financial advice. Aloha! 🌺</p>
</div>
</body>
</html>"""
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())
    log.info(f"Email sent to {TO_EMAIL}")


def main():
    if not POLYGON_API_KEY:
        log.error("POLYGON_API_KEY not set")
        sys.exit(1)

    time_label = get_time_label()
    log.info(f"King Kam {time_label} Briefing — scanning...")

    # Get tickers and scan
    tickers = fetch_sp500_tickers()
    log.info(f"Scanning {len(tickers)} tickers...")
    watchlist = run_watchlist_scan(tickers)
    log.info(f"Found {len(watchlist)} approaching setups")

    # Format scan results
    scan_text = format_scan_text(watchlist)

    # Get King Kam's briefing
    log.info("Asking King Kam to write the briefing...")
    briefing = ask_king_kam(scan_text, time_label)

    # Send email
    today = datetime.utcnow().strftime("%b %d")
    subject = f"🤙 King Kam {time_label} Briefing — {today}"
    send_email(subject, briefing)

    log.info("Done!")


if __name__ == "__main__":
    main()
