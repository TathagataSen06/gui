"""Synthetic population: towns, households, people, employers, merchants, sites."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .identifiers import slug
from .model import Address, Merchant, Org, Person, Reader, Relationship, Site, Town
from .reference import (
    BANKS,
    CORPORATE_READERS,
    COWORKING_READERS,
    DEPARTMENTS,
    GYM_READERS,
    MERCHANT_CATEGORIES,
    OCCUPATIONS_NO_EMPLOYER,
    QUASI_CASH_MCC,
    STORAGE_READERS,
)
from .state import World

# Metro bounding box for synthetic coordinates (an invented region).
_LAT0, _LAT1 = 40.25, 40.95
_LON0, _LON1 = -74.85, -73.95


def build_towns(w: World) -> None:
    rng = w.rng.stream("population")
    names: list[str] = []
    while len(names) < w.cfg.n_towns:
        n = w.fake.city()
        if n not in names:
            names.append(n)
    states = rng.choice(["NJ", "NY", "PA"], size=len(names), p=[0.6, 0.25, 0.15])
    weights = 1.0 / np.arange(1, len(names) + 1) ** 0.8  # a few big towns, many small
    zip_pool = rng.choice(np.arange(10000, 99999), size=len(names) * 3, replace=False)
    for i, name in enumerate(names):
        w.towns.append(
            Town(
                name=name,
                state=str(states[i]),
                zips=[f"{int(z):05d}" for z in zip_pool[i * 3 : i * 3 + 1 + int(rng.integers(0, 3))]],
                lat=float(rng.uniform(_LAT0, _LAT1)),
                lon=float(rng.uniform(_LON0, _LON1)),
                weight=float(weights[i]),
            )
        )


def random_address(w: World, town_idx: int, rng: np.random.Generator, unit_p: float = 0.35) -> Address:
    t = w.towns[town_idx]
    unit = None
    if rng.random() < unit_p:
        unit = str(rng.choice(["Apt", "Unit", "Suite"], p=[0.75, 0.2, 0.05])) + f" {int(rng.integers(1, 30))}{rng.choice(['', 'A', 'B', 'C'])}"
    return Address(
        number=str(int(rng.integers(1, 9999))),
        street=w.fake.street_name(),
        unit=unit,
        city=t.name,
        state=t.state,
        zip=str(rng.choice(t.zips)),
        lat=round(t.lat + float(rng.normal(0, 0.012)), 6),
        lon=round(t.lon + float(rng.normal(0, 0.012)), 6),
    )


def _pick_town(w: World, rng: np.random.Generator) -> int:
    p = np.array([t.weight for t in w.towns])
    return int(rng.choice(len(w.towns), p=p / p.sum()))


def _dob_for_age(age: int, ref: date, rng: np.random.Generator) -> date:
    return ref - timedelta(days=int(age * 365.25 + rng.integers(0, 365)))


def _new_name(w: World, sex: str) -> str:
    return w.fake.first_name_male() if sex == "M" else w.fake.first_name_female()


def build_people(w: World) -> None:
    """Create households of adults with realistic name/age/address structure."""
    rng = w.rng.stream("population")
    ref = w.cfg.start
    n_target = w.cfg.n_people
    hh_n = 0
    while len(w.persons) < n_target:
        hh_id = f"H{hh_n:05d}"
        hh_n += 1
        size = int(rng.choice([1, 2, 3, 4], p=[0.34, 0.46, 0.13, 0.07]))
        size = min(size, n_target - len(w.persons))
        town = _pick_town(w, rng)
        addr = random_address(w, town, rng)
        head_age = int(np.clip(rng.normal(47, 15), 22, 88))
        head_sex = "M" if rng.random() < 0.5 else "F"
        family_last = w.fake.last_name()
        members: list[Person] = []

        def make(first: str, sex: str, age: int, last: str, maiden: str | None = None, suffix: str | None = None) -> Person:
            pid = f"P{len(w.persons) + len(members):06d}"
            middle = _new_name(w, sex) if rng.random() < 0.68 else None
            return Person(
                person_id=pid,
                first=first,
                middle=middle,
                last=last,
                suffix=suffix,
                sex=sex,
                dob=_dob_for_age(age, ref, rng),
                household_id=hh_id,
                town=town,
                address=addr,
                maiden_last=maiden,
            )

        head = make(_new_name(w, head_sex), head_sex, head_age, family_last)
        members.append(head)
        roommates = size >= 2 and head_age < 35 and rng.random() < 0.45
        for k in range(1, size):
            if roommates:
                sex = "M" if rng.random() < 0.5 else "F"
                members.append(make(_new_name(w, sex), sex, int(np.clip(head_age + rng.normal(0, 3), 21, 40)), w.fake.last_name()))
            elif k == 1:
                sex = "F" if head_sex == "M" and rng.random() < 0.9 else ("M" if head_sex == "F" and rng.random() < 0.9 else head_sex)
                age = int(np.clip(head_age + rng.normal(-1.5, 4), 21, 90))
                own_last = w.fake.last_name()
                if rng.random() < (0.72 if sex == "F" else 0.05):  # took partner's surname
                    members.append(make(_new_name(w, sex), sex, age, family_last, maiden=own_last))
                else:
                    members.append(make(_new_name(w, sex), sex, age, own_last))
            else:
                child_age = head_age - int(rng.integers(22, 36))
                if child_age < 18:
                    child_age = 18 + int(rng.integers(0, 4))
                sex = "M" if rng.random() < 0.5 else "F"
                father = next((m for m in members[:2] if m.sex == "M"), None)
                if sex == "M" and father is not None and father.suffix is None and rng.random() < 0.22:
                    # Junior: same first (and often middle) name as father, same address.
                    child = make(father.first, "M", child_age, family_last, suffix="Jr")
                    child.middle = father.middle if rng.random() < 0.7 else child.middle
                    if rng.random() < 0.6:
                        father.suffix = "Sr"
                    members.append(child)
                else:
                    members.append(make(_new_name(w, sex), sex, child_age, family_last))
        for m in members:
            if rng.random() < 0.13:  # moved in the last ~2 years: sources may hold the old address
                m.prior_address = random_address(w, _pick_town(w, rng), rng)
        w.persons.extend(members)
        w.households[hh_id] = [m.person_id for m in members]


def build_employers(w: World) -> None:
    rng = w.rng.stream("population")
    used: set[str] = set()

    def company_name() -> str:
        while True:
            n = w.fake.company()
            if n not in used:
                used.add(n)
                return n

    n_major, n_minor = w.cfg.n_major_employers, w.cfg.n_minor_employers
    for i in range(n_major + n_minor):
        major = i < n_major
        name = company_name()
        town = _pick_town(w, rng)
        org = Org(
            org_id=f"ORG-EMP-{i:03d}",
            name=name,
            kind="employer",
            reg_no=w.ids.reg_no(),
            address=random_address(w, town, rng, unit_p=0.2),
            inc_date=w.cfg.start - timedelta(days=int(rng.integers(5 * 365, 40 * 365))),
            domain=f"{slug(name.split(',')[0])[:28].strip('-')}.example",
            major=major,
            attrs={"town": town, "size": int(rng.integers(60, 180)) if major else int(rng.integers(3, 25))},
        )
        w.orgs[org.org_id] = org
        if major:
            site = Site(
                site_id=f"SITE-{i:02d}",
                name=f"{name.split(',')[0]} Campus",
                kind="corporate",
                address=org.address,  # type: ignore[arg-type]
                facility_code=100 + i,
                employer_id=org.org_id,
            )
            for rname, zone in CORPORATE_READERS:
                site.readers.append(Reader(f"{site.site_id}-{rname}", site.site_id, rname, zone, site.address.lat, site.address.lon))
            org.site_id = site.site_id
            w.sites[site.site_id] = site


def build_third_places(w: World) -> None:
    """Co-working spaces, gyms and self-storage facilities with card readers."""
    rng = w.rng.stream("population")
    specs = [
        ("coworking", "Commonwork Suites", COWORKING_READERS, 2),
        ("gym", "IronPeak Fitness", GYM_READERS, 3),
        ("storage", "SecureBox Self Storage", STORAGE_READERS, 2),
    ]
    k = 0
    for kind, brand, readers, count in specs:
        for j in range(count):
            town = _pick_town(w, rng)
            addr = random_address(w, town, rng, unit_p=0.0)
            site = Site(
                site_id=f"SITE-{kind[:3].upper()}-{j}",
                name=f"{brand} - {w.towns[town].name}",
                kind=kind,
                address=addr,
                facility_code=200 + k,
            )
            k += 1
            for rname, zone in readers:
                site.readers.append(Reader(f"{site.site_id}-{rname}", site.site_id, rname, zone, addr.lat, addr.lon))
            w.sites[site.site_id] = site


def build_cell_sites(w: World) -> None:
    rng = w.rng.stream("population")
    for ti, t in enumerate(w.towns):
        cells = []
        for j in range(2 + int(rng.integers(0, 3))):
            cid = f"310-260-{4100 + ti * 10 + j}"
            cells.append(cid)
            w.cell_sites.append(
                {
                    "cell_id": cid,
                    "site_name": f"{t.name.upper()[:10]}-{j:02d}",
                    "lat": round(t.lat + float(rng.normal(0, 0.01)), 5),
                    "lon": round(t.lon + float(rng.normal(0, 0.01)), 5),
                    "azimuth_deg": int(rng.choice([0, 120, 240])),
                    "technology": str(rng.choice(["LTE", "NR"], p=[0.6, 0.4])),
                }
            )
        w.town_cells[ti] = cells
    for site in w.sites.values():
        town = next((i for i, t in enumerate(w.towns) if t.name == site.address.city), 0)
        site.cell_id = w.town_cells[town][0]


def assign_employment(w: World) -> None:
    rng = w.rng.stream("population")
    majors = [o for o in w.orgs.values() if o.kind == "employer" and o.major]
    minors = [o for o in w.orgs.values() if o.kind == "employer" and not o.major]
    dept_names = [d[0] for d in DEPARTMENTS]
    dept_p = np.array([d[1] for d in DEPARTMENTS])
    dept_salary = {d[0]: d[2] for d in DEPARTMENTS}
    major_p = np.array([o.attrs["size"] for o in majors], dtype=float)
    minor_p = np.array([o.attrs["size"] for o in minors], dtype=float)
    ref = w.cfg.start
    for p in w.persons:
        age = p.age_on(ref)
        if age >= 67 or (age >= 62 and rng.random() < 0.5):
            p.occupation, p.income_kind = "Retired", "social_security"
            p.salary = float(rng.normal(23_000, 6_000))
            continue
        if age < 23 and rng.random() < 0.55:
            p.occupation, p.income_kind = "Student", "parents"
            p.salary = float(rng.uniform(4_000, 12_000))
            continue
        u = rng.random()
        if u < 0.44:
            org = majors[int(rng.choice(len(majors), p=major_p / major_p.sum()))]
        elif u < 0.80:
            org = minors[int(rng.choice(len(minors), p=minor_p / minor_p.sum()))]
        else:
            occ = str(rng.choice(OCCUPATIONS_NO_EMPLOYER[:-1] if u < 0.95 else OCCUPATIONS_NO_EMPLOYER[-1:]))
            p.occupation = occ
            p.income_kind = "benefits" if occ == "Unemployed" else "self"
            p.salary = float(rng.lognormal(np.log(48_000), 0.45)) if occ != "Unemployed" else float(rng.uniform(12_000, 20_000))
            continue
        p.employer_id = org.org_id
        p.department = str(rng.choice(dept_names, p=dept_p))
        base = dept_salary[p.department] if org.major else 52_000
        p.salary = float(rng.lognormal(np.log(base), 0.28))
        p.occupation = f"{p.department} - {org.name.split(',')[0]}"
        p.income_kind = "payroll"


def build_merchants(w: World) -> None:
    rng = w.rng.stream("population")
    cat_p = np.array([c.weight for c in MERCHANT_CATEGORIES])
    used: set[str] = set()
    templates = ["{last}'s {noun}", "{city} {noun}", "{word} {noun}", "The {word} {noun}", "{word} {noun} {num}"]
    nouns = {
        "5411": "Market", "5541": "Fuel", "5812": "Bistro", "5814": "Coffee", "5912": "Pharmacy",
        "5311": "Department Store", "5999": "Goods Online", "5732": "Electronics", "5813": "Tavern",
        "4121": "Rides", "4111": "Transit", "5499": "Deli", "7230": "Salon", "8011": "Medical Group",
        "8021": "Dental", "5251": "Hardware", "5691": "Outfitters", "7832": "Cinemas", "4899": "Stream",
        "5300": "Warehouse Club", "7011": "Inn", "4511": "Air",
    }
    words = ["Summit", "Harbor", "Maple", "Union", "Liberty", "Cedar", "Golden", "Riverside", "Garden", "Lighthouse", "Pioneer", "Oak"]
    for i in range(w.cfg.n_merchants):
        cat = MERCHANT_CATEGORIES[int(rng.choice(len(MERCHANT_CATEGORIES), p=cat_p / cat_p.sum()))]
        town = None if cat.online else _pick_town(w, rng)
        for _ in range(20):
            tpl = str(rng.choice(templates))
            name = tpl.format(
                last=w.fake.last_name(),
                noun=nouns[cat.mcc],
                city=w.towns[town].name if town is not None else "Net",
                word=str(rng.choice(words)),
                num=int(rng.integers(1, 99)),
            )
            if name not in used:
                used.add(name)
                break
        w.merchants.append(
            Merchant(
                merchant_id=w.ids.merchant_id(),
                name=name,
                mcc=cat.mcc,
                category=cat.label,
                town=town,
                median_amount=cat.median_amount * float(rng.lognormal(0, 0.15)),
                sigma=cat.sigma,
                hours=cat.hours,
                city_label=(w.towns[town].name.upper()[:13] + " " + w.towns[town].state) if town is not None else "ONLINE",
            )
        )
    # One quasi-cash crypto on-ramp used both by hobbyists and by mule networks.
    w.merchants.append(
        Merchant(
            merchant_id=w.ids.merchant_id(),
            name="CoinHarbor Exchange",
            mcc=QUASI_CASH_MCC.mcc,
            category=QUASI_CASH_MCC.label,
            town=None,
            median_amount=QUASI_CASH_MCC.median_amount,
            sigma=QUASI_CASH_MCC.sigma,
            hours=QUASI_CASH_MCC.hours,
            city_label="ONLINE",
        )
    )


def build_billers(w: World) -> None:
    """External payees: utilities, wireless carrier, mortgage servicers, property managers, payroll origin."""
    rng = w.rng.stream("population")
    billers = [
        ("utility_power", "Tri-County Power & Light"),
        ("utility_water", "Metro Regional Water Authority"),
        ("utility_gas", "Keystone Natural Gas"),
        ("wireless", "Northlink Wireless"),
        ("isp", "BrightWave Internet"),
        ("insurance", "Guardian Mutual Auto Insurance"),
        ("mortgage", "Lakeshore Home Loans"),
        ("mortgage", "Pinnacle Mortgage Servicing"),
        ("property_mgmt", "Crescent Property Management"),
        ("property_mgmt", "Elm Street Residential LLC"),
        ("property_mgmt", "Parkview Apartments"),
        ("government", "SSA TREAS 310"),
        ("government", "STATE DOL UI BENEFITS"),
        ("acquirer", "Clearpoint Merchant Services"),
        ("crypto_otc", "CoinHarbor Exchange"),
        ("tax", "IRS TREAS 310 TAX REF"),
    ]
    for i, (kind, name) in enumerate(billers):
        w.orgs[f"ORG-EXT-{i:02d}"] = Org(
            org_id=f"ORG-EXT-{i:02d}",
            name=name,
            kind=kind,
            reg_no=None,
            address=None,
            inc_date=None,
            ext_account=w.ids.external_account(),
            attrs={"bank": str(rng.choice(["NATIONAL CITY BANK NA", "EXTERNAL - ACH NETWORK", "UNITED COMMERCE BANK"]))},
        )


def build_branches(w: World) -> None:
    for b in BANKS:
        w.branches[b.code] = [f"{b.code}-BR-{i:03d}" for i in range(1, b.n_branches + 1)]


def build_social_graph(w: World) -> None:
    """Household, friendship and coworker ties with per-tie interaction rates."""
    rng = w.rng.stream("social")
    scale = w.cfg.social_call_scale
    ref = w.cfg.start
    rels: list[Relationship] = []
    seen: set[tuple[str, str]] = set()

    def add(a: str, b: str, kind: str, call: float, email: float, p2p: float) -> None:
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return
        seen.add(key)
        rels.append(Relationship(key[0], key[1], kind, call * scale, email, p2p))

    for members in w.households.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                add(members[i], members[j], "household", float(rng.lognormal(np.log(1.1), 0.4)), 0.05, float(rng.uniform(0.3, 2.0)))

    by_town: dict[int, list[int]] = {}
    for idx, p in enumerate(w.persons):
        by_town.setdefault(p.town, []).append(idx)
    ages = np.array([p.age_on(ref) for p in w.persons])
    n = len(w.persons)
    for idx, p in enumerate(w.persons):
        k = int(rng.negative_binomial(3, 3 / (3 + 3.2)))  # mean ~3.2 friendships initiated per person
        for _ in range(k):
            pool = by_town[p.town] if rng.random() < 0.6 else range(n)
            cand = rng.choice(np.fromiter(pool, dtype=int), size=min(12, len(pool)), replace=False)
            weights = np.exp(-np.abs(ages[cand] - ages[idx]) / 8.0)
            j = int(cand[int(rng.choice(len(cand), p=weights / weights.sum()))])
            q = w.persons[j]
            add(p.person_id, q.person_id, "friend", float(rng.lognormal(np.log(0.12), 0.9)), float(rng.lognormal(np.log(0.03), 0.8)), float(rng.choice([0, 0, 0.2, 0.5, 1.0])))

    by_team: dict[tuple[str, str], list[Person]] = {}
    for p in w.persons:
        if p.employer_id and p.department:
            by_team.setdefault((p.employer_id, p.department), []).append(p)
    for team in by_team.values():
        for i, p in enumerate(team):
            k = min(len(team) - 1, int(rng.integers(2, 7)))
            if k <= 0:
                continue
            others = [q for q in team if q is not p]
            for q in rng.choice(others, size=k, replace=False):
                add(p.person_id, q.person_id, "coworker", float(rng.lognormal(np.log(0.06), 0.8)), 0.0, 0.0)
    w.relationships = rels
