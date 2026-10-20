"""E-mail header metadata -> address records (from display names) + e-mail events."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Iterator

import pandas as pd

from ...ids import normalize_email
from ..base import SourceAdapter, attrs_json, register_adapter

_ADDR = re.compile(r'^\s*(?:"?(?P<name>[^"<]*?)"?\s*)?<(?P<addr>[^>]+)>\s*$')


def parse_mailbox(header: str) -> tuple[str | None, str | None]:
    """Split ``"Display Name" <addr>`` into (display name, normalized address)."""
    m = _ADDR.match(header or "")
    if m:
        name = (m.group("name") or "").strip() or None
        return name, normalize_email(m.group("addr"))
    return None, normalize_email(header)


@register_adapter
class EmailAdapter(SourceAdapter):
    name = "email"

    def __init__(self, raw_root) -> None:  # type: ignore[no-untyped-def]
        super().__init__(raw_root)
        self._names: dict[str, Counter[str]] = defaultdict(Counter)
        self._counts: Counter[str] = Counter()
        self._bulk: set[str] = set()

    def events(self) -> Iterator[pd.DataFrame]:
        for f in sorted(self.root.glob("headers_*.jsonl")):
            recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.rows_read += len(recs)
            ids, tss, actors, roles, attrs = [], [], [], [], []
            dates = pd.to_datetime(pd.Series([r.get("date") for r in recs]), format="%a, %d %b %Y %H:%M:%S %z", errors="coerce", utc=True)
            for r, ts in zip(recs, dates):
                sender_name, sender = parse_mailbox(r.get("from", ""))
                if sender is None or pd.isna(ts):
                    self.reject("bad_sender_or_date")
                    continue
                a, ro = [f"email:{sender}"], ["sender"]
                self._observe(sender, sender_name)
                for field, role in (("to", "to"), ("cc", "cc")):
                    for h in r.get(field, []) or []:
                        n, addr = parse_mailbox(h)
                        if addr and f"email:{addr}" not in a:
                            a.append(f"email:{addr}")
                            ro.append(role)
                            self._observe(addr, n)
                if len(a) < 2:
                    self.reject("no_recipients")
                    continue
                bulk = bool(r.get("list_unsubscribe") or r.get("precedence") == "bulk")
                if bulk:
                    self._bulk.add(sender)
                att = r.get("attachments") or []
                ids.append(f"email:{r['message_id']}")
                tss.append(ts)
                actors.append(a)
                roles.append(ro)
                attrs.append(
                    {
                        "message_id": r["message_id"],
                        "subject": r.get("subject"),
                        "size": r.get("size"),
                        "n_attachments": len(att),
                        "attachment_bytes": sum(int(x.get("size", 0)) for x in att),
                        "attachment_types": sorted({x.get("content_type", "") for x in att}) or None,
                        "bulk": bulk or None,
                        "n_recipients": len(a) - 1,
                    }
                )
            yield pd.DataFrame({"event_id": ids, "ts": tss, "source": "email", "event_type": "email", "actor_ids": actors, "actor_roles": roles, "attributes": attrs_json(attrs)})

    def _observe(self, addr: str, name: str | None) -> None:
        self._counts[addr] += 1
        if name:
            self._names[addr][name] += 1

    def records(self) -> pd.DataFrame:
        """One record per observed address, named by its most frequent display name.

        Must be called after :meth:`events` (the pipeline guarantees this order).
        """
        rows = []
        for addr in sorted(self._counts):
            names = self._names.get(addr, Counter())
            top = sorted(names.items(), key=lambda kv: (-kv[1], kv[0]))
            rows.append(
                {
                    "record_id": f"email:{addr}",
                    "source": "email",
                    "entity_type": "org" if addr in self._bulk else "person",
                    "name": top[0][0] if top else None,
                    "dob": None,
                    "address": None,
                    "phone": None,
                    "email": addr,
                    "owned_ids": [f"email:{addr}"],
                    "attributes": {
                        "domain": addr.split("@", 1)[1],
                        "messages_seen": self._counts[addr],
                        "display_names": [n for n, _ in top[:3]] or None,
                        "bulk_sender": addr in self._bulk or None,
                    },
                }
            )
        out = pd.DataFrame(rows, columns=["record_id", "source", "entity_type", "name", "dob", "address", "phone", "email", "owned_ids", "attributes"])
        out["attributes"] = attrs_json(out["attributes"].tolist())
        return out

    records_after_events = True
