# ── FILL THESE IN ───────────────────────────────────────────
SENDER_EMAIL   = "you@gmail.com"        # your Gmail address
RECEIVER_EMAIL = "recipient@example.com"  # where to send it
APP_PASSWORD   = "xxxx xxxx xxxx xxxx"  # Gmail App Password (see note below)
# ────────────────────────────────────────────────────────────
#
#  HOW TO GET A GMAIL APP PASSWORD (required if you use Gmail)
#  1. Go to myaccount.google.com → Security
#  2. Enable 2-Step Verification if not already on
#  3. Search for "App Passwords" → create one → copy the 16-char code
#  4. Paste it above (spaces are fine to keep)
#
#  For Outlook/Hotmail use smtp.office365.com  port 587
#  For Yahoo         use smtp.mail.yahoo.com   port 465

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

ATTACHMENT = "qqq_strategy_bqnt.py"
SUBJECT    = "QQQ BQNT Trading Strategy"
BODY       = "Hi,\n\nPlease find the QQQ BQNT trading strategy attached.\n"

SMTP_HOST  = "smtp.gmail.com"
SMTP_PORT  = 465   # SSL

if not os.path.exists(ATTACHMENT):
    raise FileNotFoundError(f"{ATTACHMENT} not found — run this from the repo directory")

msg = MIMEMultipart()
msg["From"]    = SENDER_EMAIL
msg["To"]      = RECEIVER_EMAIL
msg["Subject"] = SUBJECT
msg.attach(MIMEText(BODY, "plain"))

with open(ATTACHMENT, "rb") as f:
    part = MIMEBase("application", "octet-stream")
    part.set_payload(f.read())

encoders.encode_base64(part)
part.add_header(
    "Content-Disposition",
    f'attachment; filename="{ATTACHMENT}"'
)
msg.attach(part)

print(f"Connecting to {SMTP_HOST}:{SMTP_PORT} ...")
with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
    server.login(SENDER_EMAIL, APP_PASSWORD.replace(" ", ""))
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

print(f"Done — '{ATTACHMENT}' sent to {RECEIVER_EMAIL}")
