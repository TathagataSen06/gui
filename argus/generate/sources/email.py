"""E-mail header metadata (no bodies).

Headers follow RFC 5322 conventions: ``Date`` in local time with a numeric
offset, ``From``/``To``/``Cc`` as ``"Display Name" <address>``, a ``Message-ID``
and attachment metadata. Display names vary per client/device and are often
nicknames or missing, which makes the e-mail feed a weak but useful
identity source. Bulk mail carries ``List-Unsubscribe``/``Precedence: bulk``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..identifiers import slug
from ..model import Person
from ..noise import nickname, typo
from ..reference import HOUR_PROFILES, NEWSLETTER_SUBJECTS, PERSONAL_SUBJECTS, WORK_SUBJECTS
from ..state import World, sample_hours

ATTACHMENT_TYPES: tuple[tuple[str, str, float], ...] = (
    ("application/pdf", "pdf", 0.4),
    ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx", 0.25),
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx", 0.2),
    ("image/jpeg", "jpg", 0.15),
)


class MailLog:
    def __init__(self, w: World) -> None:
        self.w = w
        self.rng = w.rng.stream("email")
        self._display: dict[str, list[str]] = {}

    def display_names(self, address: str) -> list[str]:
        """Stable per-address set of display names (one per mail client)."""
        if address in self._display:
            return self._display[address]
        w, rng = self.w, self.rng
        owner = w.email_owner.get(address)
        names: list[str] = []
        if owner and owner in w.person_by_id:
            p = w.person_by_id[owner]
            work = address == p.work_email
            for _ in range(1 if rng.random() < 0.7 else 2):
                if rng.random() < w.cfg.noise.display_name_missing and not work:
                    names.append("")
                    continue
                first = nickname(p.first, rng) if rng.random() < w.cfg.noise.nickname_email else p.first
                style = rng.choice(["first_last", "last_comma_first", "first_mi_last", "first_only", "lower"], p=[0.55, 0.15, 0.16, 0.04, 0.1] if work else [0.6, 0.04, 0.12, 0.12, 0.12])
                if style == "first_last":
                    n = f"{first} {p.last}"
                elif style == "last_comma_first":
                    n = f"{p.last}, {first}"
                elif style == "first_mi_last" and p.middle:
                    n = f"{first} {p.middle[0]}. {p.last}"
                elif style == "first_only":
                    n = first
                else:
                    n = f"{first} {p.last}".lower()
                if rng.random() < w.cfg.noise.name_typo / 2:
                    n = typo(n, rng)
                names.append(n)
        elif owner and owner in w.orgs:
            names.append(w.orgs[owner].name.split(",")[0])
        else:
            names.append("")
        self._display[address] = names
        return names

    def fmt(self, address: str) -> str:
        names = self.display_names(address)
        n = names[int(self.rng.integers(len(names)))]
        if not n:
            return address
        if "," in n:
            return f'"{n}" <{address}>'
        return f"{n} <{address}>"

    def add(
        self,
        ts: int,
        sender: str,
        to: list[str],
        cc: list[str] | None = None,
        subject: str = "",
        size: int | None = None,
        attachments: list[dict[str, Any]] | None = None,
        bulk: bool = False,
        tag: str | None = None,
    ) -> None:
        rng = self.rng
        domain = sender.split("@", 1)[1]
        att = attachments or []
        body = int(size if size is not None else rng.lognormal(np.log(14_000), 0.8))
        rec: dict[str, Any] = {
            "ts": int(ts),
            "message_id": f"<{int(rng.integers(16**11, 16**12)):012x}.{int(rng.integers(0, 2**31)):x}@{domain}>",
            "from": self.fmt(sender),
            "to": [self.fmt(a) for a in to],
            "cc": [self.fmt(a) for a in (cc or [])],
            "subject": subject,
            "size": body + sum(int(a["size"]) for a in att),
            "attachments": att,
            "_tag": tag,
        }
        if bulk:
            rec["list_unsubscribe"] = f"<mailto:unsubscribe@{domain}>"
            rec["precedence"] = "bulk"
        self.w.email_headers.append(rec)

    def random_attachments(self, k: int, size_scale: float = 1.0, prefix: str = "") -> list[dict[str, Any]]:
        rng = self.rng
        out = []
        cum = np.cumsum([a[2] for a in ATTACHMENT_TYPES])
        names = ("report", "invoice", "scan", "photo", "notes", "summary", "draft")
        for _ in range(k):
            ct, ext, _ = ATTACHMENT_TYPES[int(np.searchsorted(cum, rng.random() * cum[-1]))]
            name = prefix or names[int(rng.integers(len(names)))]
            out.append({"filename": f"{name}_{int(rng.integers(100, 9999))}.{ext}", "content_type": ct, "size": int(rng.lognormal(np.log(180_000 * size_scale), 0.9))})
        return out


_DAYS = ("Monday", "Tuesday", "Thursday", "Friday")


def _subject(rng: np.random.Generator, kind: str) -> str:
    if kind == "work":
        s = WORK_SUBJECTS[int(rng.integers(len(WORK_SUBJECTS)))]
        if "{" not in s:
            return s
        return s.format(q=int(rng.integers(1, 5)), day=_DAYS[int(rng.integers(4))], n=int(rng.integers(10000, 99999)))
    return PERSONAL_SUBJECTS[int(rng.integers(len(PERSONAL_SUBJECTS)))]


def generate_benign_email(w: World, log: MailLog) -> None:
    rng = log.rng
    biz = np.flatnonzero(w.clock.business)
    colleagues: dict[str, list[Person]] = {}
    for p in w.persons:
        if p.work_email and p.employer_id:
            colleagues.setdefault(p.employer_id, []).append(p)
    coworker_ties: dict[str, list[str]] = {}
    for r in w.relationships:
        if r.kind == "coworker":
            coworker_ties.setdefault(r.a, []).append(r.b)
            coworker_ties.setdefault(r.b, []).append(r.a)
    # Work mail
    for org_id, staff in colleagues.items():
        for p in staff:
            ties = [w.person_by_id[q] for q in coworker_ties.get(p.person_id, []) if w.person_by_id[q].work_email]
            pool = ties or [q for q in staff if q is not p]
            if not pool:
                continue
            n = int(rng.poisson(w.cfg.work_email_rate * len(biz)))
            days = rng.choice(biz, size=n)
            secs = sample_hours(rng, HOUR_PROFILES["office"], n)
            for i in range(n):
                k = 1 if rng.random() < 0.7 else int(rng.integers(2, 4))
                if k == 1 or len(pool) == 1:
                    to = [pool[int(rng.integers(len(pool)))].work_email]
                else:
                    to = [pool[int(j)].work_email for j in rng.permutation(len(pool))[: min(k, len(pool))]]
                cc = []
                if rng.random() < 0.2:
                    q = staff[int(rng.integers(len(staff)))]
                    if q is not p and q.work_email not in to:
                        cc = [q.work_email]
                att = log.random_attachments(1) if rng.random() < 0.18 else []
                log.add(int(w.clock.ts(int(days[i]), int(secs[i]))), p.work_email, to, cc, _subject(rng, "work"), attachments=att)  # type: ignore[arg-type]
    # Personal mail along social ties
    for r in w.relationships:
        if r.email_rate <= 0:
            continue
        a, b = w.person_by_id[r.a], w.person_by_id[r.b]
        if not a.personal_email or not b.personal_email:
            continue
        n = int(rng.poisson(r.email_rate * w.clock.days))
        for _ in range(n):
            s, t = (a, b) if rng.random() < 0.5 else (b, a)
            d = int(rng.integers(0, w.clock.days))
            att = log.random_attachments(int(rng.integers(1, 4))) if rng.random() < 0.15 else []
            log.add(int(w.clock.ts(d, sample_hours(rng, HOUR_PROFILES["evening"], 1)[0])), s.personal_email, [t.personal_email], subject=_subject(rng, "personal"), attachments=att)  # type: ignore[arg-type]
    # Bulk newsletters from merchants
    senders = [m for m in w.merchants[:: max(1, len(w.merchants) // 8)]][:8]
    for m in senders:
        sender = f"news@{slug(m.name)[:24].strip('-')}.example"
        w.email_owner.setdefault(sender, "")
        subs = [p for p in w.persons if p.personal_email and rng.random() < 0.22]
        for d in range(int(rng.integers(0, 14)), w.clock.days, 30):
            for p in subs:
                ts = int(w.clock.ts(d, int(rng.integers(5, 9)) * 3600 + int(rng.integers(0, 3600))))
                log.add(ts, sender, [p.personal_email], subject=str(rng.choice(NEWSLETTER_SUBJECTS)), size=int(rng.integers(40_000, 120_000)), bulk=True)  # type: ignore[list-item]
