"""In-memory model of the synthetic world (ground-truth view)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class Address:
    number: str
    street: str
    unit: str | None
    city: str
    state: str
    zip: str
    lat: float
    lon: float

    def one_line(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.number} {self.street}{unit}, {self.city}, {self.state} {self.zip}"


@dataclass
class Town:
    name: str
    state: str
    zips: list[str]
    lat: float
    lon: float
    weight: float


@dataclass
class Person:
    person_id: str
    first: str
    middle: str | None
    last: str
    suffix: str | None
    sex: str
    dob: date
    household_id: str
    town: int
    address: Address
    prior_address: Address | None = None
    maiden_last: str | None = None
    employer_id: str | None = None
    department: str | None = None
    occupation: str = "Unemployed"
    salary: float = 0.0
    income_kind: str = "none"  # payroll | social_security | benefits | parents | self
    phones: list[str] = field(default_factory=list)  # E.164, first is primary
    prepaid_phones: set[str] = field(default_factory=set)
    personal_email: str | None = None
    work_email: str | None = None
    accounts: list[str] = field(default_factory=list)  # account keys "BANK:number"
    customer_ids: dict[str, str] = field(default_factory=dict)  # bank_code -> customer_id
    cards: list[str] = field(default_factory=list)  # access card numbers
    traits: set[str] = field(default_factory=set)
    housing: str = "rent"  # rent | own | family
    rent: float = 0.0
    landlord_account: str | None = None  # consortium account key or ext id

    def age_on(self, d: date) -> int:
        return d.year - self.dob.year - ((d.month, d.day) < (self.dob.month, self.dob.day))

    def full_name(self) -> str:
        parts = [self.first, self.middle, self.last]
        name = " ".join(p for p in parts if p)
        return f"{name} {self.suffix}" if self.suffix else name


@dataclass
class Org:
    org_id: str
    name: str
    kind: str  # employer | shell | property_llc | cash_business | front | crypto_otc | biller | acquirer | external
    reg_no: str | None
    address: Address | None
    inc_date: date | None
    owners: list[str] = field(default_factory=list)  # person ids (beneficial owners)
    accounts: list[str] = field(default_factory=list)
    domain: str | None = None
    customer_ids: dict[str, str] = field(default_factory=dict)
    ext_account: str | None = None  # for orgs outside the consortium
    phone: str | None = None
    major: bool = False
    site_id: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Account:
    key: str  # "BANK:number"
    bank_code: str
    number: str
    owner_kind: str  # person | org
    owner_id: str
    product: str
    open_date: date
    customer_id: str


@dataclass
class Reader:
    reader_id: str
    site_id: str
    name: str
    zone: str
    lat: float
    lon: float


@dataclass
class Site:
    site_id: str
    name: str
    kind: str  # corporate | coworking | gym | storage
    address: Address
    facility_code: int
    readers: list[Reader] = field(default_factory=list)
    employer_id: str | None = None
    cell_id: str | None = None

    def reader(self, suffix: str) -> Reader:
        for r in self.readers:
            if r.name.endswith(suffix):
                return r
        raise KeyError(suffix)


@dataclass
class Merchant:
    merchant_id: str
    name: str
    mcc: str
    category: str
    town: int | None  # None = online
    median_amount: float
    sigma: float
    hours: str
    city_label: str


@dataclass
class Card:
    number: str
    holder_id: str
    site_id: str
    employee_id: str | None
    company: str
    department: str | None
    issued: date
    kind: str  # employee | member


@dataclass
class Relationship:
    a: str
    b: str
    kind: str  # household | friend | coworker | ring | handler
    call_rate: float  # calls+sms per day
    email_rate: float  # personal e-mails per day
    p2p_rate: float  # P2P payments per month
