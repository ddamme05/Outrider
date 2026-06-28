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
        "import yaml\nyaml.load(data)\nyaml.load(cfg, Loader=yaml.UnsafeLoader)\n",
        "import yaml\nyaml.safe_load(data)\n"
        "yaml.load(data, Loader=yaml.SafeLoader)\nyaml.load(d, Loader=yaml.CSafeLoader)\n",
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
        # variant (bare + the other ciphers + the qualified form) is pinned
        # individually by test_weak_crypto_each_advertised_variant_fires.
        "DES.new(key)\n",
        # Strong cipher (bare AND qualified), an import-only DES (an import is not
        # use), a lowercase non-crypto `des` (name-based + case-sensitive), and a
        # QUALIFIED `.new` whose attribute is NOT a cipher name must all NOT fire —
        # the qualified pattern binds the cipher NAME, not any dotted `.new`.
        "from Crypto.Cipher import DES\nimport hashlib\nAES.new(key, AES.MODE_GCM)\n"
        "Crypto.Cipher.AES.new(key, AES.MODE_GCM)\n"
        "des = make_factory()\ndes.new(rows=3)\n"
        "models.User.new(data)\norm.Session.new(bind)\n",
    ),
    (
        "python.weak_crypto_ecb_mode",
        # Single canonical form; both ECB construction forms are pinned by
        # test_weak_crypto_each_advertised_variant_fires.
        "AES.new(key, AES.MODE_ECB)\n",
        # Strong mode (positional AND keyword), a guard/denylist reference and a log
        # of the constant (neither is a cipher construction), AND an ECB-valued
        # NON-`mode=` keyword must all NOT fire. The keyword path binds `name="mode"`,
        # so the proof is "constructed in ECB MODE", not merely "an ECB constant
        # appears in some keyword" — the proof-boundary tightening (Codex review).
        "AES.new(key, AES.MODE_GCM)\nCipher(algorithms.AES(key), modes.GCM(iv))\n"
        "AES.new(key, mode=AES.MODE_GCM)\nCipher(algorithms.AES(key), mode=modes.GCM(iv))\n"
        "AES.new(key, expected=AES.MODE_ECB)\nCipher(algorithms.AES(key), default=modes.ECB())\n"
        "if mode == AES.MODE_ECB:\n    raise ValueError()\nlog.info(AES.MODE_ECB)\n",
    ),
    (
        "python.weak_asymmetric_key_size",
        # Canonical weak form: a 1024-bit RSA key (value-predicate keeps < 2048).
        "RSA.generate(1024)\n",
        # The secure boundary (2048, value-predicate DROPS — strict `<`), a strong
        # size, a non-literal size (cannot be evaluated → not flagged), and a
        # non-key-gen call must all NOT produce an OBSERVED finding.
        "RSA.generate(2048)\nRSA.generate(4096)\nRSA.generate(bits)\nRSA.encrypt(data)\n",
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
    # FUP-193 step-1.5: the other weak/legacy PyCryptodome block/stream ciphers
    # (64-bit-block or broken) — each pinned so one can't silently regress.
    ("python.weak_crypto_broken_cipher", "DES3.new(key)\n"),
    ("python.weak_crypto_broken_cipher", "ARC2.new(key)\n"),
    ("python.weak_crypto_broken_cipher", "RC2.new(key)\n"),
    ("python.weak_crypto_broken_cipher", "CAST.new(key)\n"),
    ("python.weak_crypto_broken_cipher", "IDEA.new(key)\n"),
    # FUP-193 step-1.5: the qualified-call form (the cipher imported under its
    # dotted path) — the cipher name is the immediate attribute of the object.
    ("python.weak_crypto_broken_cipher", "Crypto.Cipher.DES.new(key)\n"),
    ("python.weak_crypto_broken_cipher", "ciphers.DES3.new(key)\n"),
    ("python.weak_crypto_ecb_mode", "AES.new(key, AES.MODE_ECB)\n"),
    ("python.weak_crypto_ecb_mode", "Cipher(algorithms.AES(key), modes.ECB())\n"),
    # FUP-193 step-1.5: the qualified ECB construction form already worked (the
    # ctor is `.new` regardless of object depth) — pinned so it stays covered.
    ("python.weak_crypto_ecb_mode", "Crypto.Cipher.AES.new(key, mode=AES.MODE_ECB)\n"),
    # FUP-193 step-1.5 recall: the keyword-argument mode form (very common real
    # code) — the mode lives in a `keyword_argument` node the positional patterns
    # above miss. Both the `MODE_ECB` constant and the `modes.ECB()` call forms.
    ("python.weak_crypto_ecb_mode", "AES.new(key, mode=AES.MODE_ECB)\n"),
    ("python.weak_crypto_ecb_mode", "Cipher(algorithms.AES(key), mode=modes.ECB())\n"),
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


# yaml: the `#not-match? @_args "SafeLoader"` predicate must MATCH unsafe/absent
# loaders (closing the old `Loader`-substring over-suppression that dropped
# `Loader=UnsafeLoader`) and SUPPRESS only genuinely safe loaders. The shared
# `>=1` case is vacuous here (`yaml.load(data)` alone passes it), so each loader
# form is pinned individually. This is a PRAGMATIC text-predicate fix, NOT FUP-184's
# exact structural closure (which needs ast_facts loader-value resolution). Because
# it is a substring predicate over the whole arg list, it is NOT skip-safe on its
# own: it has residual errors BOTH ways — an under-match (a non-loader token that
# contains `SafeLoader`, pinned below) AND an over-match (a genuinely safe loader
# *aliased* without the literal `SafeLoader` text, e.g. `Loader=SL`, would match =
# a false-positive). Exact alias resolution is deferred to FUP-184 (ast_facts); the
# Phase-2 no-loss eval is what actually gates any 3b skip_safe promotion.
_YAML_LOADER_CASES: tuple[tuple[str, bool], ...] = (
    ("import yaml\nyaml.load(data)\n", True),  # no loader -> unsafe
    ("import yaml\nyaml.load(data, Loader=yaml.UnsafeLoader)\n", True),  # over-suppression fix
    ("import yaml\nyaml.load(data, Loader=yaml.FullLoader)\n", True),  # FullLoader is not safe
    ("import yaml\nyaml.load(data, Loader=yaml.Loader)\n", True),  # base Loader is unsafe
    ("import yaml\nyaml.load(data, Loader=yaml.SafeLoader)\n", False),  # safe -> suppressed
    ("import yaml\nyaml.load(data, Loader=CSafeLoader)\n", False),  # C safe variant -> suppressed
    # Residual limits of the substring predicate, PINNED so they stay visible and a
    # test flips when FUP-184 (ast_facts exact loader-value resolution) closes them.
    # The predicate is NOT skip-safe on its own:
    #  - UNDER-match / fails OPEN (the sharper residual): a GENUINELY UNSAFE loader
    #    whose NAME contains `SafeLoader` is silently suppressed -> NO match = a
    #    missed vuln. NotSafeLoader is a real yaml.Loader subclass here (semantic
    #    fixture, not a bare placeholder), so the fail-open is on a true unsafe load.
    (
        "import yaml\nclass NotSafeLoader(yaml.Loader): pass\n"
        "yaml.load(data, Loader=NotSafeLoader)\n",
        False,
    ),
    #  - UNDER-match: a non-loader DATA token literally containing `SafeLoader` is
    #    suppressed -> NO match (the first positional arg is arbitrary data).
    ("import yaml\nyaml.load(SafeLoader_default_blob)\n", False),
    #  - OVER-match / false-positive: a GENUINELY SAFE loader aliased without the
    #    literal `SafeLoader` text still matches -> spurious finding. SL IS
    #    yaml.SafeLoader here (semantic fixture), so the match is a true false-positive.
    ("import yaml\nSL = yaml.SafeLoader\nyaml.load(data, Loader=SL)\n", True),
)


@pytest.mark.parametrize(
    ("src", "should_match"),
    _YAML_LOADER_CASES,
    ids=[s.strip().splitlines()[-1] for s, _ in _YAML_LOADER_CASES],
)
def test_unsafe_yaml_loader_precision(src: str, should_match: bool) -> None:
    """Per-loader-form pin for the `#not-match? SafeLoader` predicate: unsafe/absent
    loaders match (the closed over-suppression — the load-bearing case is
    `Loader=UnsafeLoader`); only literal `SafeLoader`/`CSafeLoader` text suppresses.
    It is a pragmatic precision improvement, NOT exact: residual under- AND
    over-matches remain (see the block comment) — exactness is FUP-184's gate."""
    matches = registry.match("python.unsafe_deserialization_yaml", src.encode())
    if should_match:
        assert len(matches) >= 1, f"{src!r} should match (unsafe/absent loader)"
    else:
        assert len(matches) == 0, f"{src!r} must be suppressed (safe loader / text limit)"


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


# FUP-193 value-predicate: weak_asymmetric_key_size fires iff the captured key
# size is a LITERAL below 2048. The structural .scm match is identical for every
# integer argument; the registry value-predicate (queries/value_predicates.py)
# is what flips the verdict on the value, so the shared `>=1` precision case is
# vacuous for a threshold — each form is pinned individually here. A revert of
# the value-predicate (every integer matches) flips the False rows to match,
# and a revert of the first-arg anchor flips the `e=`-exponent row.
_KEY_SIZE_CASES: tuple[tuple[str, bool], ...] = (
    # WEAK literals (< 2048) — the value-predicate keeps these.
    ("RSA.generate(1024)\n", True),  # weak positional
    ("RSA.generate(2047)\n", True),  # just under the floor
    ("DSA.generate(512)\n", True),  # weak DSA
    ("RSA.generate(bits=1024)\n", True),  # weak keyword `bits`
    ("RSA.generate(0x400)\n", True),  # hex literal 1024 -> weak
    ("rsa.generate_private_key(public_exponent=65537, key_size=1024)\n", True),  # cryptography
    ("dsa.generate_private_key(key_size=1024)\n", True),
    ("Crypto.PublicKey.RSA.generate(1024)\n", True),  # qualified object
    # SECURE / boundary — the value-predicate DROPS these (strict `<`, so 2048 is
    # the floor and is NOT flagged).
    ("RSA.generate(2048)\n", False),
    ("RSA.generate(4096)\n", False),
    ("RSA.generate(bits=2048)\n", False),
    ("rsa.generate_private_key(key_size=2048)\n", False),
    ("RSA.generate(2_048)\n", False),  # underscore literal 2048
    ("RSA.generate(0x800)\n", False),  # hex literal 2048
    # The `e=` exponent (17) must never be read as the key size — the first-arg
    # anchor pins @_keysize to 4096, which is secure, so this drops.
    ("RSA.generate(4096, randfunc, 17)\n", False),
    # NON-LITERAL size: cannot be evaluated -> not flagged (the OBSERVED proof
    # must prove its claim; a variable size is not deterministic evidence).
    ("RSA.generate(bits)\n", False),
    ("RSA.generate(KEY_SIZE)\n", False),
    # WRONG SHAPE: not an RSA/DSA key generation.
    ("RSA.encrypt(data)\n", False),
    ("foo.generate(1024)\n", False),
)


@pytest.mark.parametrize(
    ("src", "should_match"),
    _KEY_SIZE_CASES,
    ids=[s.strip() for s, _ in _KEY_SIZE_CASES],
)
def test_weak_asymmetric_key_size_value_predicate(src: str, should_match: bool) -> None:
    """The value-predicate fires iff the key size is a literal below 2048: weak
    literals match, the 2048 floor and stronger sizes drop, the `e=` exponent is
    never mistaken for the key size, and a non-literal size is never flagged."""
    matches = registry.match("python.weak_asymmetric_key_size", src.encode())
    if should_match:
        assert len(matches) >= 1, f"{src!r} should fire (weak literal key size)"
    else:
        assert len(matches) == 0, f"{src!r} must NOT fire (secure / non-literal / wrong shape)"


def test_every_observed_query_has_a_precision_case() -> None:
    """No OBSERVED query ships without a positive/negative precision case."""
    covered = {c[0] for c in _CASES}
    assert covered == set(registry.OBSERVED_QUERY_IDS), (
        f"precision-case drift: untested={set(registry.OBSERVED_QUERY_IDS) - covered} "
        f"stale={covered - set(registry.OBSERVED_QUERY_IDS)}"
    )
