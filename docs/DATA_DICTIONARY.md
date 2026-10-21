# Data dictionary

Everything below is produced by `argus generate`. All values are fabricated.
Phone numbers sit in the NANP `NXX-555-01XX` block reserved for fiction,
e-mail domains end in `.example` (RFC 2606), bank routing numbers start with
the unassigned prefix `99`, and business tax IDs use the never-issued `00-`
prefix. `argus ingest` refuses any directory that breaks these rules.

## Raw feeds (`<world>/raw/`)

Each feed deliberately follows the conventions of a different kind of source
system, so ingestion has real normalization work to do.

### `bank/<BANK>/` - consortium core banking (3 fictional banks)

| File | Grain | Notes |
|---|---|---|
| `customers.csv` | one row per customer relationship | KYC profile. `full_name` casing and `date_of_birth` format differ per bank (`FMB`: `Katie M. Rush`, `1985-03-04`; `CFCU`: `MILLER, TIFFANY PATRICIA`, `06/29/1970`; `HVT`: `Lisa Wong`, `28-Oct-1986`). Business rows carry `tax_reg_no`, `incorporation_date`, `business_type` and the declared `beneficial_owner_name` / `_dob`. Phone and e-mail are optional and occasionally mistyped. |
| `accounts.csv` | one row per account | `account_number`, `customer_id`, `product`, `open_date`, `routing_number`. |
| `transactions_YYYYMM.csv` | one row per **ledger leg** | Double-entry: a transfer between two consortium accounts appears twice (a `DR` on the payer and a `CR` on the payee, sharing `end_to_end_id`). The `CR` leg settles later (P2P seconds, wire minutes, ACH next business day). `booking_datetime` is local time in the bank's own format (`2026-03-04 14:22:10`, `03/04/2026 14:22`, `2026-03-04T14:22:10-05:00`). Channels: `CARD`, `ACH`, `WIRE`, `P2P`, `BOOK`, `CHECK`, `CASH`, `ATM`. Card rows carry ISO 18245 `mcc` and merchant. |

### `telco/` - mobile carriers

| File | Notes |
|---|---|
| `subscribers.csv` | `msisdn` (no `+`), `carrier`, `account_type` (`POSTPAID`/`PREPAID`), `subscriber_name` as typed at the shop (nicknames, `LAST, FIRST`, lower case, initials; empty for anonymous prepaid lines), `subscriber_dob` as `DDMMYYYY` (often missing), `billing_address` (sometimes stale), `contact_email`. |
| `cdr/cdr_YYYYMM.csv` | Call-detail records: `record_type` (`MOC` voice / `SMS-MO`), `calling_number` (11-digit), `called_number` **as dialled** (10-digit, 11-digit or E.164), `start_time` UTC `YYYYMMDDHHMMSS`, `duration_sec`, serving `cell_id`, `call_result`. |
| `cell_sites.csv` | `cell_id`, coordinates, azimuth, technology. |

### `access_control/` - physical access control (PACS)

| File | Notes |
|---|---|
| `cardholders.csv` | `card_number` (Wiegand style `FC<facility>-<card>`), `cardholder_name` (`LAST, FIRST M` from HR, or as typed at a gym / co-working desk), `employee_id`, `company`, `department`, `work_email`, `mobile_phone`, `card_type`. |
| `readers.csv` | `reader_id`, site, `zone` (`TRANSIT`, `OFFICE`, `RESTRICTED`, `MEETING`, `AMENITY`, `FACILITY`), coordinates. |
| `events/access_YYYY-MM.jsonl` | PACS export: `EventId` (GUID), `EventTime` (UTC ISO with ms + `Z`), `CardNumber`, `ReaderId`, `SiteCode`, `Result` (`ACCESS_GRANTED`/`ACCESS_DENIED`), `Direction`, `DenyReason`. |

### `email/headers_YYYY-MM.jsonl` - message metadata (no bodies)

RFC 5322 style: `message_id`, `date` (`Sat, 28 Feb 2026 19:07:49 -0500`),
`from` / `to` / `cc` as `"Display Name" <address>` (display names vary by
mail client, are often nicknames, sometimes absent), `subject`, `size`,
`attachments` (filename, MIME type, size). Bulk mail carries
`list_unsubscribe` and `precedence: bulk`.

### `watchlist/watchlist.csv` - internal hot-list

`entry_id`, `full_name` (sometimes misspelt), `aliases` (`;`-separated),
`dob` (sometimes missing), `reason_code`, `added_date`. Contains one leader
of each planted network **plus stale entries for innocent people and entries
for people who do not exist in the world**.

### `labels/case_outcomes.csv` - historical investigation outcomes

`case_id`, `subject_record_id` (a bank customer), `disposition`
(`CONFIRMED_SUSPICIOUS`, `CLOSED_NO_ACTION`, `CLOSED_VICTIM`), `typology`,
dates. This is the only supervision the pipeline receives.

## Ground truth (`<world>/ground_truth/`) - evaluation only

| File | Content |
|---|---|
| `persons.csv`, `orgs.csv`, `accounts.csv`, `phones.csv` | the true world |
| `record_truth.csv` | `record_id` (in the ingest naming scheme) -> true person/org, with `entity_kind` (`person`, `org`, `burner`, `anonymous_line`) |
| `scenarios.json` | each planted network: members and roles, shell companies, burners, cycles, windows, victims |
| `decoys.json` | benign look-alikes and their participants |
| `person_labels.csv` | per person: label, ring, role, decoy role |
| `planted_events.csv` | every raw event (by native id) that a scenario produced |

## Warehouse (`<world>/argus.duckdb`)

| Table | Stage | Content |
|---|---|---|
| `events` | ingest | common schema: `event_id, ts (UTC), source, event_type, actor_ids[], actor_roles[], attributes JSON` |
| `event_actors` | ingest | exploded `(event_id, actor_id, role, ordinal)` |
| `identity_records` | ingest | `record_id, source, entity_type, name, dob (ISO), address, phone, email, owned_ids[], attributes` |
| `case_labels`, `ref_accounts`, `ref_readers`, `ref_cell_sites`, `ingest_log`, `manifest` | ingest | labels, reference data, data-quality log |
| `entities`, `record_entity`, `identifier_entity`, `entity_links`, `er_pairs` | resolve | resolved entities, membership, identifier ownership, `controls` / `employed_by` links, scored pairs with per-field evidence |
| `graph_identifier_edges`, `graph_edges`, `graph_nodes`, `communities`, `money_cycles`, `graph_hubs` | graph | observed interactions, entity-level multiplex edges, centrality, Louvain communities with money-flow profiles, temporal cycles |
| `entity_features`, `risk_scores`, `model_importances`, `model_contributions` | score | features, ranked explained scores, global and per-entity explanations |
| `run_meta` | all | per-stage statistics and configuration (JSON) |
