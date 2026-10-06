# The question to Anthropic sales

A paid hands needs written confirmation from Anthropic sales on two points that the
terms for products that run Claude Code
([legal and compliance](https://code.claude.com/docs/en/legal-and-compliance)) leave
open. [hosted-subscription.md](hosted-subscription.md#what-has-to-change-before-hands-charges-money)
says why. This is the message to send, ready as written. The answer settles
hands-monetization-76j.mhg.

---

**Subject: Does a paid voice app that runs Claude Code on the user's own Mac meet the Claude Code terms?**

Hello,

I build hands, a macOS app that lets someone talk to the Claude Code sessions they
run on their own Mac. I plan to sell it as a flat monthly fee for the software and
its updates. The fee includes no Claude usage. Every user signs in to Claude Code
with their own Claude plan or their own Console API key and is billed by you
directly. Nothing runs on a server of mine. Before I charge anyone, I want your
written confirmation on two parts of how it works.

**1. hands' own Claude Code, behind a local proxy that edits its requests.**
hands runs one Claude Code of its own, the unmodified published binary, under a
config directory of its own. The user logs it in with any login Claude Code offers
(`claude auth login`, either a Claude plan or `--console`). The user's spoken words
are typed into its interactive input. Its `ANTHROPIC_BASE_URL` is a proxy on
127.0.0.1, inside hands. For this one Claude Code, the proxy:

- appends a short text block to the newest message of each request: which Claude Code
  sessions are running and which one the user's words go to;
- replaces the full text of older tool results with a one-line summary;
- replaces Claude Code's summarisation prompt in a compaction request with one written
  for a spoken conversation;
- when the user interrupts, or the model has called a "stay silent" tool, answers the
  next request itself with a short canned reply instead of sending it to the API;
- turns an API error into a final 502, so Claude Code does not retry for minutes
  while the user waits.

Credentials pass through unchanged and are used for nothing but that request.

*Question:* Is this allowed when that Claude Code is logged in on the user's Claude
plan (Pro or Max)? If it is allowed only on a Console API key, I will require a key
for it in the paid app.

**2. A local TLS tap on the user's own Claude Code sessions.**
The user's ordinary Claude Code sessions run inside a pseudo-terminal wrapper that
leaves the binary unchanged. The wrapper decrypts each session's HTTPS traffic to
api.anthropic.com with a certificate authority generated on the user's machine. It
forwards every request and reply unchanged, and sends a copy, with the credential
headers removed, to hands on the same machine. That copy is how hands knows what a
session said, so it can read it aloud. Nothing leaves the machine except the
session's own request to you.

*Question:* Is this allowed when those sessions are logged in on the user's Claude
plan?

The app is not named after Claude Code or Anthropic. I can send more detail or a
recording of it running.

Thank you,
Brandon Fryslie
