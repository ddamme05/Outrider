"""Structural eval scenario: OBSERVED-tier query precision (Cost Lever 3).

Per specs/2026-06-14-observed-query-library-v1.md: each seed OBSERVED
security query MUST fire on a positive fixture and must NOT fire on a
negative one. LLM-free — runs `queries.registry.match` directly (the
structural layer), validating the `query_match_id` + match behavior
without any model call.

The queries are signal_only structural approximations (they augment the
LLM, never skip it), so the fixtures pin the load-bearing precision
contract: the positive forms the producer relies on, and the lookalike
negatives that must NOT produce a false OBSERVED finding.
"""

from __future__ import annotations

import pytest

from outrider.queries import registry

# (query_id, positive_src, negative_src): positive must match >=1, negative 0.
_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "python.command_injection_subprocess_shell",
        "import subprocess\n"
        "subprocess.run(cmd, shell=True)\n"
        "subprocess.Popen(c, shell=True)\n"
        "subprocess.check_output(x, shell=True)\n",
        "import subprocess\nsubprocess.run(['ls', '-l'])\nsubprocess.run(cmd, shell=False)\n",
    ),
    (
        "python.command_injection_os_system",
        "import os\nos.system(user_cmd)\nos.popen(c)\n",
        "import os\nos.path.join('a', 'b')\nos.getenv('X')\nos.system_call(x)\n",
    ),
    (
        "python.command_injection_eval_exec",
        "eval(user_input)\nexec(payload)\neval(a + b)\n",
        "eval('1 + 1')\nexec('print(1)')\n",
    ),
    (
        "python.unsafe_deserialization_pickle",
        "import pickle\npickle.loads(data)\npickle.load(f)\ncPickle.loads(d)\n",
        "import pickle\npickle.dumps(obj)\njson.loads(data)\n",
    ),
    (
        "python.unsafe_deserialization_yaml",
        "import yaml\nyaml.load(data)\n",
        "import yaml\nyaml.safe_load(data)\nyaml.load(data, Loader=yaml.SafeLoader)\n",
    ),
    (
        "python.sql_injection_string_concat",
        'cur.execute(f"SELECT * FROM t WHERE id = {uid}")\n'
        'cur.execute("SELECT " + col)\n'
        'cur.execute("WHERE id={}".format(uid))\n',
        'cur.execute("SELECT * FROM t WHERE id = %s", (uid,))\n'
        'cur.execute("SELECT * FROM t WHERE id = ?", (uid,))\n',
    ),
    (
        "python.tls_verify_disabled",
        "import requests\nrequests.get(url, verify=False)\nrequests.post(u, verify=(False))\n",
        "import requests\nrequests.get(url)\nrequests.get(url, verify=True)\n",
    ),
    (
        "python.blocking_call_in_async",
        "import time\nasync def f():\n    time.sleep(1)\n",
        "import time\ndef f():\n    time.sleep(1)\n",
    ),
    (
        "python.weak_crypto_broken_cipher",
        # Single canonical form so `>=1` is non-vacuous; every advertised cipher
        # variant (DES/ARC4/RC4/Blowfish) is pinned individually by
        # test_weak_crypto_each_advertised_variant_fires.
        "DES.new(key)\n",
        # Strong cipher, an import-only DES (an import is not use), and a
        # lowercase non-crypto `des` (the signal is name-based + case-sensitive)
        # must all NOT fire.
        "from Crypto.Cipher import DES\nimport hashlib\nAES.new(key, AES.MODE_GCM)\n"
        "des = make_factory()\ndes.new(rows=3)\n",
    ),
    (
        "python.weak_crypto_ecb_mode",
        # Single canonical form; both ECB construction forms are pinned by
        # test_weak_crypto_each_advertised_variant_fires.
        "AES.new(key, AES.MODE_ECB)\n",
        # Strong mode, plus a guard/denylist reference and a log of the constant
        # (neither is a cipher construction) must NOT fire.
        "AES.new(key, AES.MODE_GCM)\nCipher(algorithms.AES(key), modes.GCM(iv))\n"
        "if mode == AES.MODE_ECB:\n    raise ValueError()\nlog.info(AES.MODE_ECB)\n",
    ),
)

# Each advertised weak-crypto form, pinned individually — the shared precision
# test asserts only `>=1`, which a query matching a single form would pass, so
# this guards against one variant silently regressing while another keeps the
# combined fixture green.
_WEAK_CRYPTO_VARIANTS: tuple[tuple[str, str], ...] = (
    ("python.weak_crypto_broken_cipher", "DES.new(key)\n"),
    ("python.weak_crypto_broken_cipher", "ARC4.new(key)\n"),
    ("python.weak_crypto_broken_cipher", "RC4.new(key)\n"),
    ("python.weak_crypto_broken_cipher", "Blowfish.new(key)\n"),
    ("python.weak_crypto_ecb_mode", "AES.new(key, AES.MODE_ECB)\n"),
    ("python.weak_crypto_ecb_mode", "Cipher(algorithms.AES(key), modes.ECB())\n"),
)


@pytest.mark.parametrize(
    ("query_id", "positive", "negative"),
    _CASES,
    ids=[c[0].removeprefix("python.") for c in _CASES],
)
def test_observed_query_precision(query_id: str, positive: str, negative: str) -> None:
    """Positive fixture matches (>=1); negative lookalike does not (0)."""
    pos = registry.match(query_id, positive.encode())
    neg = registry.match(query_id, negative.encode())
    assert len(pos) >= 1, f"{query_id} should match its positive fixture, got {len(pos)}"
    assert len(neg) == 0, f"{query_id} must NOT match its negative fixture, got {len(neg)}"


@pytest.mark.parametrize(
    ("query_id", "variant"),
    _WEAK_CRYPTO_VARIANTS,
    ids=[f"{q.removeprefix('python.')}::{v.strip()}" for q, v in _WEAK_CRYPTO_VARIANTS],
)
def test_weak_crypto_each_advertised_variant_fires(query_id: str, variant: str) -> None:
    """Each advertised weak-crypto form fires on its own.

    The shared precision test asserts only `>=1` on a combined fixture, which
    a query matching a single form would pass — this pins DES/ARC4/RC4/Blowfish
    AND both ECB construction forms individually so one variant can't silently
    regress behind another (the per-variant vacuity the combined fixture hides).
    """
    matches = registry.match(query_id, variant.encode())
    assert len(matches) >= 1, f"{query_id} must match the advertised variant {variant!r}"


def test_every_observed_query_has_a_precision_case() -> None:
    """No OBSERVED query ships without a positive/negative precision case."""
    covered = {c[0] for c in _CASES}
    assert covered == set(registry.OBSERVED_QUERY_IDS), (
        f"precision-case drift: untested={set(registry.OBSERVED_QUERY_IDS) - covered} "
        f"stale={covered - set(registry.OBSERVED_QUERY_IDS)}"
    )
