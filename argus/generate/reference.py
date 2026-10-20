"""Static reference data used by the synthetic world generator.

Everything here is either public-domain convention (ISO 18245 merchant
category codes, US street-suffix abbreviations, common English nicknames) or
invented (bank names, site templates). Nothing identifies a real person.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

#: Formal given name -> common nicknames / short forms.
NICKNAMES: dict[str, tuple[str, ...]] = {
    "robert": ("bob", "rob", "bobby", "robbie"),
    "william": ("bill", "will", "billy", "liam"),
    "richard": ("rick", "rich", "richie", "dick"),
    "james": ("jim", "jimmy", "jamie"),
    "john": ("jack", "johnny"),
    "michael": ("mike", "mikey", "mick"),
    "joseph": ("joe", "joey"),
    "thomas": ("tom", "tommy"),
    "charles": ("charlie", "chuck"),
    "christopher": ("chris",),
    "daniel": ("dan", "danny"),
    "matthew": ("matt",),
    "anthony": ("tony",),
    "donald": ("don", "donnie"),
    "steven": ("steve",),
    "stephen": ("steve",),
    "andrew": ("andy", "drew"),
    "joshua": ("josh",),
    "kenneth": ("ken", "kenny"),
    "edward": ("ed", "eddie", "ted"),
    "ronald": ("ron", "ronnie"),
    "timothy": ("tim", "timmy"),
    "jeffrey": ("jeff",),
    "nicholas": ("nick",),
    "benjamin": ("ben",),
    "samuel": ("sam",),
    "alexander": ("alex",),
    "jonathan": ("jon",),
    "gregory": ("greg",),
    "patrick": ("pat",),
    "peter": ("pete",),
    "raymond": ("ray",),
    "lawrence": ("larry",),
    "gerald": ("gerry", "jerry"),
    "douglas": ("doug",),
    "frederick": ("fred",),
    "zachary": ("zach",),
    "nathan": ("nate",),
    "jacob": ("jake",),
    "david": ("dave",),
    "phillip": ("phil",),
    "philip": ("phil",),
    "eric": ("rick",),
    "elizabeth": ("liz", "beth", "betty", "lizzie"),
    "jennifer": ("jen", "jenny"),
    "katherine": ("kate", "katie", "kathy"),
    "kathryn": ("kate", "katie"),
    "catherine": ("cathy", "kate"),
    "margaret": ("maggie", "meg", "peggy"),
    "patricia": ("pat", "patty", "trish"),
    "barbara": ("barb",),
    "susan": ("sue", "susie"),
    "deborah": ("deb", "debbie"),
    "rebecca": ("becky", "becca"),
    "jessica": ("jess",),
    "victoria": ("vicky", "tori"),
    "christina": ("tina", "chris"),
    "christine": ("chris", "tina"),
    "samantha": ("sam",),
    "alexandra": ("alex", "lexi"),
    "abigail": ("abby",),
    "kimberly": ("kim",),
    "pamela": ("pam",),
    "cynthia": ("cindy",),
    "sandra": ("sandy",),
    "theresa": ("terri", "tess"),
    "amanda": ("mandy",),
    "melissa": ("missy", "mel"),
    "stephanie": ("steph",),
    "jacqueline": ("jackie",),
    "judith": ("judy",),
    "madison": ("maddie",),
    "olivia": ("liv",),
    "gabrielle": ("gabby",),
    "nicole": ("nikki",),
    "danielle": ("dani",),
    "michelle": ("shelly",),
    "angela": ("angie",),
    "valerie": ("val",),
}

#: Nickname -> formal names it can stand for (derived).
NICKNAME_TO_FORMAL: dict[str, tuple[str, ...]] = {}
for _formal, _nicks in sorted(NICKNAMES.items()):
    for _n in _nicks:
        NICKNAME_TO_FORMAL[_n] = tuple(sorted(set(NICKNAME_TO_FORMAL.get(_n, ())) | {_formal}))

NAME_SUFFIXES: tuple[str, ...] = ("jr", "sr", "ii", "iii", "iv")
NAME_TITLES: tuple[str, ...] = ("mr", "mrs", "ms", "miss", "dr", "prof", "mx")

# --------------------------------------------------------------------------
# Keyboard model for realistic typos (QWERTY adjacency)
# --------------------------------------------------------------------------

_KEY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
KEY_NEIGHBOURS: dict[str, str] = {}
for _r, _row in enumerate(_KEY_ROWS):
    for _c, _ch in enumerate(_row):
        _n: list[str] = []
        for _dr in (-1, 0, 1):
            _rr = _r + _dr
            if 0 <= _rr < len(_KEY_ROWS):
                for _dc in (-1, 0, 1):
                    _cc = _c + _dc
                    if (_dr, _dc) != (0, 0) and 0 <= _cc < len(_KEY_ROWS[_rr]):
                        _n.append(_KEY_ROWS[_rr][_cc])
        KEY_NEIGHBOURS[_ch] = "".join(_n)

# Characters commonly confused by OCR / manual transcription of forms.
OCR_CONFUSIONS: dict[str, str] = {"o": "0", "0": "o", "l": "1", "1": "l", "i": "l", "s": "5", "5": "s", "b": "8", "e": "c"}

# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------

STREET_SUFFIX_ABBREV: dict[str, str] = {
    "street": "st",
    "avenue": "ave",
    "road": "rd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "boulevard": "blvd",
    "place": "pl",
    "terrace": "ter",
    "parkway": "pkwy",
    "circle": "cir",
    "highway": "hwy",
    "square": "sq",
    "trail": "trl",
    "way": "way",
    "crossing": "xing",
    "point": "pt",
    "ridge": "rdg",
    "heights": "hts",
    "village": "vlg",
    "junction": "jct",
    "station": "sta",
    "mountain": "mtn",
    "estates": "ests",
    "springs": "spgs",
    "gardens": "gdns",
    "valley": "vly",
    "hill": "hl",
    "hills": "hls",
    "island": "is",
    "lake": "lk",
    "mills": "mls",
    "harbor": "hbr",
    "crescent": "cres",
    "extension": "ext",
    "fort": "ft",
    "ford": "frd",
    "port": "prt",
    "plaza": "plz",
    "creek": "crk",
    "center": "ctr",
    "club": "clb",
    "common": "cmn",
    "fields": "flds",
    "forest": "frst",
    "glen": "gln",
    "grove": "grv",
    "haven": "hvn",
    "manor": "mnr",
    "meadows": "mdws",
    "oval": "oval",
    "park": "park",
    "pass": "pass",
    "path": "path",
    "pike": "pike",
    "pines": "pnes",
    "prairie": "pr",
    "ranch": "rnch",
    "shores": "shrs",
    "summit": "smt",
    "turnpike": "tpke",
    "union": "un",
    "via": "via",
    "view": "vw",
    "vista": "vis",
    "walk": "walk",
    "wall": "wall",
    "wells": "wls",
}

DIRECTIONALS: dict[str, str] = {"north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw"}

UNIT_DESIGNATORS: dict[str, str] = {"apartment": "apt", "suite": "ste", "unit": "unit", "floor": "fl", "room": "rm", "building": "bldg", "#": "#"}

# --------------------------------------------------------------------------
# Banking
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BankProfile:
    """A fictional bank and the data conventions of its core system."""

    code: str
    name: str
    routing: str  # 9 digits; prefix 99 is not an assigned Fed routing symbol
    name_style: str  # how full_name is stored
    dob_format: str  # strftime pattern for date_of_birth
    customer_prefix: str
    market_share: float
    n_branches: int


BANKS: tuple[BankProfile, ...] = (
    BankProfile("FMB", "First Meridian Bank", "991000013", "title_middle_initial", "%Y-%m-%d", "FMB-", 0.47, 14),
    BankProfile("CFCU", "Cascade Federal Credit Union", "992000016", "upper_last_comma_first", "%m/%d/%Y", "CF", 0.31, 8),
    BankProfile("HVT", "Harborview Trust", "993000019", "title_first_last", "%d-%b-%Y", "HVT", 0.22, 6),
)

#: Reporting threshold used by the structuring typology (cf. US CTR $10,000).
CASH_REPORTING_THRESHOLD = 10_000.0


@dataclass(frozen=True)
class MerchantCategory:
    """An ISO 18245 merchant category with a spend profile."""

    mcc: str
    label: str
    median_amount: float
    sigma: float
    weight: float  # relative share of card transactions
    hours: str  # time-of-day profile key
    online: bool = False


MERCHANT_CATEGORIES: tuple[MerchantCategory, ...] = (
    MerchantCategory("5411", "Grocery Stores, Supermarkets", 54.0, 0.65, 0.17, "daytime"),
    MerchantCategory("5541", "Service Stations", 42.0, 0.40, 0.09, "commute"),
    MerchantCategory("5812", "Eating Places, Restaurants", 36.0, 0.55, 0.11, "meals"),
    MerchantCategory("5814", "Fast Food / Coffee", 11.5, 0.45, 0.16, "morning"),
    MerchantCategory("5912", "Drug Stores and Pharmacies", 21.0, 0.70, 0.05, "daytime"),
    MerchantCategory("5311", "Department Stores", 64.0, 0.80, 0.04, "daytime"),
    MerchantCategory("5999", "Misc. and Specialty Retail (online)", 37.0, 0.90, 0.10, "evening", True),
    MerchantCategory("5732", "Electronics Stores", 118.0, 1.00, 0.015, "daytime"),
    MerchantCategory("5813", "Drinking Places (Bars)", 29.0, 0.55, 0.035, "night"),
    MerchantCategory("4121", "Taxicabs and Rideshare", 19.0, 0.50, 0.045, "commute", True),
    MerchantCategory("4111", "Commuter Transport", 2.90, 0.20, 0.04, "commute"),
    MerchantCategory("5499", "Convenience and Specialty Food", 9.0, 0.60, 0.05, "daytime"),
    MerchantCategory("7230", "Barber and Beauty Shops", 38.0, 0.45, 0.015, "daytime"),
    MerchantCategory("8011", "Doctors", 85.0, 0.60, 0.008, "office"),
    MerchantCategory("8021", "Dentists", 140.0, 0.60, 0.004, "office"),
    MerchantCategory("5251", "Hardware Stores", 34.0, 0.80, 0.02, "daytime"),
    MerchantCategory("5691", "Clothing Stores", 57.0, 0.70, 0.03, "daytime"),
    MerchantCategory("7832", "Motion Picture Theaters", 24.0, 0.35, 0.01, "evening"),
    MerchantCategory("4899", "Cable and Streaming Services", 15.99, 0.30, 0.012, "night", True),
    MerchantCategory("5300", "Wholesale Clubs", 112.0, 0.60, 0.012, "daytime"),
    MerchantCategory("7011", "Hotels and Lodging", 178.0, 0.50, 0.003, "evening", True),
    MerchantCategory("4511", "Airlines", 310.0, 0.55, 0.002, "evening", True),
)

#: Quasi-cash MCC used by the mule typology (crypto/OTC exchange top-ups).
QUASI_CASH_MCC = MerchantCategory("6051", "Quasi Cash - Crypto Exchange", 900.0, 0.8, 0.0, "night", True)

# Hour-of-day probability profiles (local time). 24 weights each.
HOUR_PROFILES: dict[str, tuple[float, ...]] = {
    "daytime": (0, 0, 0, 0, 0, 0, 0.2, 0.5, 1, 1.5, 2, 2.2, 2.4, 2.2, 2, 2, 2.2, 2.6, 2.6, 2, 1.4, 0.8, 0.3, 0.1),
    "commute": (0, 0, 0, 0, 0.1, 0.5, 1.8, 3, 2.2, 1, 0.8, 0.9, 1, 0.9, 0.9, 1.2, 1.9, 2.8, 2.4, 1.3, 0.8, 0.5, 0.3, 0.1),
    "meals": (0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.4, 0.3, 0.4, 1.8, 3, 2, 0.6, 0.4, 0.5, 1.4, 3, 3, 2, 0.9, 0.3, 0.1),
    "morning": (0, 0, 0, 0, 0, 0.3, 1.5, 3.2, 3, 1.8, 1.2, 1.6, 2.2, 1.6, 1, 0.9, 0.9, 1, 1, 0.8, 0.5, 0.3, 0.2, 0.1),
    "evening": (0.3, 0.1, 0, 0, 0, 0, 0.1, 0.3, 0.6, 0.8, 0.9, 1, 1.1, 1.1, 1, 1, 1.1, 1.3, 1.6, 2, 2.4, 2.2, 1.6, 0.8),
    "night": (0.8, 0.5, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.2, 0.2, 0.3, 0.4, 0.8, 1.4, 2, 2.6, 2.8, 2.4, 1.5),
    "office": (0, 0, 0, 0, 0, 0, 0, 0.3, 1.5, 2, 2, 1.6, 1, 1.4, 1.8, 1.8, 1.4, 0.6, 0.1, 0, 0, 0, 0, 0),
    "social_call": (0.2, 0.1, 0.05, 0.02, 0.02, 0.05, 0.2, 0.6, 0.8, 0.9, 1, 1.1, 1.3, 1.1, 1, 1, 1.1, 1.4, 1.8, 2, 1.9, 1.5, 0.9, 0.5),
    "business_call": (0, 0, 0, 0, 0, 0, 0.05, 0.4, 1.6, 2.2, 2.3, 1.9, 1.2, 1.6, 2, 2, 1.7, 0.9, 0.3, 0.1, 0.05, 0, 0, 0),
}

# --------------------------------------------------------------------------
# Employment
# --------------------------------------------------------------------------

DEPARTMENTS: tuple[tuple[str, float, float], ...] = (
    # (department, share of headcount, median annual salary USD)
    ("Operations", 0.22, 58_000),
    ("Sales", 0.14, 67_000),
    ("Finance", 0.10, 82_000),
    ("Information Technology", 0.10, 96_000),
    ("Human Resources", 0.05, 64_000),
    ("Facilities", 0.06, 52_000),
    ("Legal and Compliance", 0.04, 105_000),
    ("Engineering", 0.12, 112_000),
    ("Warehouse", 0.12, 41_000),
    ("Customer Support", 0.05, 46_000),
)

#: Departments whose cards open restricted zones (server/records rooms).
RESTRICTED_ACCESS_DEPTS: frozenset[str] = frozenset({"Information Technology", "Facilities", "Legal and Compliance"})

OCCUPATIONS_NO_EMPLOYER: tuple[str, ...] = (
    "Self-employed",
    "Retired",
    "Student",
    "Homemaker",
    "Freelance designer",
    "Rideshare driver",
    "Contractor",
    "Unemployed",
)

# --------------------------------------------------------------------------
# Communications
# --------------------------------------------------------------------------

PERSONAL_EMAIL_DOMAINS: tuple[str, ...] = (
    "mailhaven.example",
    "inboxly.example",
    "postbox.example",
    "netmail.example",
    "quickmail.example",
)

#: RFC 2606 / RFC 6761 reserved suffixes considered "synthetic" by ingest.
SYNTHETIC_EMAIL_SUFFIXES: tuple[str, ...] = (".example", ".test", ".invalid", "@example.com", "@example.org", "@example.net")

NEWSLETTER_SUBJECTS: tuple[str, ...] = (
    "Your weekly deals are here",
    "Members save 20% this weekend",
    "Your monthly statement is ready",
    "New arrivals picked for you",
    "Last chance: offer ends tonight",
)

WORK_SUBJECTS: tuple[str, ...] = (
    "Re: Q{q} forecast",
    "Weekly ops sync",
    "Re: vendor invoice approval",
    "Updated slides",
    "Schedule change for {day}",
    "Re: ticket #{n}",
    "Shipping exceptions report",
    "FW: policy update",
    "Re: budget variance",
    "Meeting notes",
    "Out of office",
    "Re: customer escalation",
)

PERSONAL_SUBJECTS: tuple[str, ...] = (
    "Dinner Saturday?",
    "Re: photos",
    "Happy birthday!",
    "Re: weekend plans",
    "Fwd: recipe",
    "Kids pickup",
    "Re: trip itinerary",
    "Check this out",
    "",
)

# --------------------------------------------------------------------------
# Sites with access control
# --------------------------------------------------------------------------

#: Zone types. TRANSIT readers (turnstiles, parking gates) are high-traffic.
ZONES: tuple[str, ...] = ("TRANSIT", "OFFICE", "RESTRICTED", "MEETING", "AMENITY", "FACILITY")

CORPORATE_READERS: tuple[tuple[str, str], ...] = (
    ("LOBBY-TURNSTILE-01", "TRANSIT"),
    ("LOBBY-TURNSTILE-02", "TRANSIT"),
    ("PARKING-GATE-A", "TRANSIT"),
    ("FLOOR-02-EAST", "OFFICE"),
    ("FLOOR-03-WEST", "OFFICE"),
    ("FLOOR-04-EAST", "OFFICE"),
    ("SERVER-ROOM-B1", "RESTRICTED"),
    ("RECORDS-ROOM-3F", "RESTRICTED"),
    ("LOADING-DOCK-01", "RESTRICTED"),
    ("WAREHOUSE-FLOOR", "OFFICE"),
)

COWORKING_READERS: tuple[tuple[str, str], ...] = (
    ("FRONT-DOOR", "TRANSIT"),
    ("MEETING-ROOM-A", "MEETING"),
    ("MEETING-ROOM-B", "MEETING"),
    ("PRIVATE-OFFICE-WING", "OFFICE"),
)

GYM_READERS: tuple[tuple[str, str], ...] = (("MEMBER-ENTRANCE", "AMENITY"),)

STORAGE_READERS: tuple[tuple[str, str], ...] = (("GATE-KEYPAD", "FACILITY"), ("BUILDING-B-DOOR", "FACILITY"))

#: Name fragments for invented shell companies.
SHELL_WORDS_A: tuple[str, ...] = (
    "Bluefin", "Castellan", "Northgate", "Silverline", "Meridian", "Granite", "Halcyon", "Ironwood",
    "Larkspur", "Obsidian", "Pinecrest", "Redwater", "Sablewood", "Tidewater", "Whitestone", "Ambergate",
    "Corvid", "Driftwood", "Emberly", "Foxhollow",
)
SHELL_WORDS_B: tuple[str, ...] = (
    "Holdings", "Ventures", "Capital Partners", "Trading", "Consulting Group", "Global Logistics",
    "Imports", "Advisory", "Management", "Resources", "Solutions", "Investments",
)
CASH_BUSINESS_TYPES: tuple[str, ...] = ("Auto Wash", "Laundromat", "Deli & Grill", "Nail Salon", "Food Truck Co", "Vape & Smoke", "Pizzeria")
