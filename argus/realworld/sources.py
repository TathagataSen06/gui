"""Dataset catalog and a provenance-recording downloader.

Every file lists its official source first and public mirrors after it. A
mirror is only accepted if the file passes the dataset's integrity checks
(documented byte size where one exists, plus row and label counts checked by
the loaders), and the source actually used is written to ``PROVENANCE.json``
next to the data so every report can say where its numbers came from.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_ROOT = Path("data/external")
USER_AGENT = "argus-lab/0.1 (research benchmark; +https://github.com/tathagatasen06/gui)"


class DatasetUnavailable(RuntimeError):
    """No source for a dataset file could be reached or none passed the checks."""


@dataclass(frozen=True)
class Source:
    url: str
    kind: str = "official"  # "official" | "mirror"
    member: str | None = None  # file to extract when the URL is a zip archive
    note: str = ""


@dataclass(frozen=True)
class FileSpec:
    name: str
    sources: tuple[Source, ...]
    size: int | None = None  # expected byte size when independently documented


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    homepage: str
    license: str
    citation: str
    files: tuple[FileSpec, ...]
    manual: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def hosts(self) -> list[str]:
        return sorted({urlparse(s.url).hostname or "" for f in self.files for s in f.sources} - {""})


CREDITCARD = DatasetSpec(
    key="creditcard",
    title="Credit Card Fraud Detection (ULB Machine Learning Group / Worldline)",
    homepage="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    license="Open Database License (ODbL) v1.0 / Database Contents License (DbCL) v1.0",
    citation=(
        "A. Dal Pozzolo, O. Caelen, R. A. Johnson and G. Bontempi. Calibrating Probability with "
        "Undersampling for Unbalanced Classification. IEEE SSCI, 2015."
    ),
    files=(
        FileSpec(
            name="creditcard.csv",
            size=150_828_752,
            sources=(
                Source("https://gitlab.com/data/d/openml/1597/-/raw/master/dataset/tables/data.csv", "mirror", note="datagit read-only mirror of OpenML 1597"),
                Source("https://maxhalford.github.io/files/datasets/creditcardfraud.zip", "mirror", member="creditcard.csv", note="River (online-ml) dataset mirror"),
            ),
        ),
    ),
    manual=(
        "The official copy needs a Kaggle login: download creditcard.csv from the Kaggle page and place it in "
        "data/external/creditcard/. The automatic sources are mirrors, accepted only if the file is byte-for-byte "
        "the documented size of the original and the loader's row and label counts match."
    ),
    notes=(
        "284,807 transactions over two days in September 2013, 492 frauds (0.172%).",
        "V1-V28 are PCA components; Time and Amount are the only raw fields; there is no cardholder or merchant key.",
    ),
)

ELLIPTIC = DatasetSpec(
    key="elliptic",
    title="Elliptic Data Set (Bitcoin transaction graph)",
    homepage="https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
    license="CC BY-NC-ND 4.0 (non-commercial; do not redistribute modified copies)",
    citation=(
        "M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson and C. E. Leiserson. "
        "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial "
        "Forensics. KDD Workshop on Anomaly Detection in Finance, 2019. arXiv:1908.02591."
    ),
    files=tuple(
        FileSpec(
            name=f"elliptic_txs_{part}.csv",
            sources=(
                Source(f"https://data.pyg.org/datasets/elliptic/elliptic_txs_{part}.csv.zip", "official", member=f"elliptic_txs_{part}.csv", note="PyTorch Geometric mirror used by EllipticBitcoinDataset"),
            ),
        )
        for part in ("features", "edgelist", "classes")
    ),
    manual=(
        "Download the three CSVs from the Kaggle page (free account) and place elliptic_txs_features.csv, "
        "elliptic_txs_edgelist.csv and elliptic_txs_classes.csv in data/external/elliptic/."
    ),
    notes=(
        "203,769 transactions in 49 time steps, 234,355 payment-flow edges; 4,545 illicit, 42,019 licit, 157,205 unknown.",
        "Use only these three files. Do not join third-party de-anonymization mappings or wallet data.",
    ),
)

_LEIPZIG_MIRROR = "https://raw.githubusercontent.com/skogler/systemds-amls-project-data/master/DBLP-ACM"
_DITTO = "https://raw.githubusercontent.com/megagonlabs/ditto/master/data/er_magellan/Structured/DBLP-ACM"

DBLP_ACM = DatasetSpec(
    key="dblp_acm",
    title="DBLP-ACM bibliographic record linkage benchmark (Leipzig)",
    homepage="https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution",
    license="No explicit license published; distributed by the Database Group, University of Leipzig, for entity-resolution research",
    citation=(
        "H. Kopcke, A. Thor and E. Rahm. Evaluation of entity resolution approaches on real-world match "
        "problems. PVLDB 3(1), 2010. Pre-blocked split: S. Mudgal et al., Deep Learning for Entity Matching, "
        "SIGMOD 2018; serialized by Y. Li et al., Ditto, PVLDB 14(1), 2021."
    ),
    files=tuple(
        FileSpec(
            name=name,
            sources=(
                Source("https://dbs.uni-leipzig.de/files/datasets/DBLP-ACM.zip", "official", member=name),
                Source(f"{_LEIPZIG_MIRROR}/{name}", "mirror", note="copy of the Leipzig files used for Apache SystemDS course data"),
            ),
        )
        for name in ("DBLP2.csv", "ACM.csv", "DBLP-ACM_perfectMapping.csv")
    )
    + tuple(
        FileSpec(name=f"pre_blocked/{split}.txt", sources=(Source(f"{_DITTO}/{split}.txt", "official", note="DeepMatcher Structured/DBLP-ACM split as serialized in the Ditto repository"),))
        for split in ("train", "valid", "test")
    ),
    manual=(
        "Download DBLP-ACM.zip from the Leipzig benchmark page and extract its three CSVs into data/external/dblp_acm/; "
        "optionally put the Ditto train/valid/test.txt files in data/external/dblp_acm/pre_blocked/."
    ),
    notes=(
        "2,616 DBLP and 2,294 ACM records; 2,224 true matches in the perfect mapping.",
        "The entity is a publication. Author names are public bibliographic metadata, used only as a string field and never reported.",
    ),
)

CATALOG: dict[str, DatasetSpec] = {d.key: d for d in (ELLIPTIC, CREDITCARD, DBLP_ACM)}


def dataset_dir(key: str, root: str | Path = DEFAULT_ROOT) -> Path:
    return Path(root) / key


def _download(url: str, dest: Path, timeout: float) -> tuple[int, str]:
    """Stream ``url`` to ``dest``; return (bytes, sha256). Honors HTTPS_PROXY and SSL_CERT_FILE."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    h = hashlib.sha256()
    n = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
    tmp.replace(dest)
    return n, h.hexdigest()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _extract(archive: Path, member: str, dest: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.rsplit("/", 1)[-1] == member]
        if not names:
            raise DatasetUnavailable(f"{archive.name} does not contain {member}")
        with z.open(names[0]) as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)


def fetch(key: str, root: str | Path = DEFAULT_ROOT, force: bool = False, timeout: float = 120.0) -> dict:
    """Download every file of dataset ``key`` (if missing) and record provenance.

    Files already present (for example placed by hand from Kaggle) are kept and
    recorded as ``"local"``. Returns the provenance mapping.
    """
    spec = CATALOG[key]
    out = dataset_dir(key, root)
    out.mkdir(parents=True, exist_ok=True)
    prov_path = out / "PROVENANCE.json"
    prov: dict = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov.setdefault("dataset", spec.title)
    prov.setdefault("license", spec.license)
    prov.setdefault("files", {})
    archives: dict[str, Path] = {}
    for fs in spec.files:
        dest = out / fs.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not force:
            if fs.name not in prov["files"]:
                prov["files"][fs.name] = {"source": "local", "bytes": dest.stat().st_size, "sha256": sha256_of(dest)}
            continue
        errors: list[str] = []
        for src in fs.sources:
            try:
                if src.member:
                    arc = archives.get(src.url)
                    if arc is None:
                        arc = out / (Path(urlparse(src.url).path).name or "download.zip")
                        _download(src.url, arc, timeout)
                        archives[src.url] = arc
                    _extract(arc, src.member, dest)
                else:
                    _download(src.url, dest, timeout)
                size = dest.stat().st_size
                if fs.size is not None and size != fs.size:
                    dest.unlink()
                    raise DatasetUnavailable(f"size {size:,} != documented {fs.size:,}")
                prov["files"][fs.name] = {
                    "source": src.url,
                    "source_kind": src.kind,
                    "note": src.note,
                    "bytes": size,
                    "sha256": sha256_of(dest),
                    "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                break
            except (urllib.error.URLError, OSError, zipfile.BadZipFile, DatasetUnavailable) as e:
                errors.append(f"{src.url}: {e}")
        else:
            raise DatasetUnavailable(
                f"could not obtain {fs.name} for {key}.\n  " + "\n  ".join(errors)
                + f"\nAllow one of these hosts in the environment's network policy: {', '.join(spec.hosts())}"
                + f"\nor place the file by hand: {spec.manual}"
            )
    for arc in archives.values():
        arc.unlink(missing_ok=True)
    prov_path.write_text(json.dumps(prov, indent=2))
    return prov
