"""Standalone check of md_audio_tiled tile schedules (needs: pip install lupa).
Seam = any tile boundary strictly inside (0, W). PASS needs: coverage exactly
[0, W), no seam when W <= max_single, equal tiles (len spread <= 1) otherwise,
every tile <= max_single.

TEETH CHECK: set mode to 'broken' to see the pre-fix ceil formula fail the
spread<=1 assertion on multi-tile cases before running the patched Lua.
"""
import math, os, sys, lupa

here = os.path.dirname(os.path.abspath(__file__))
lua  = lupa.LuaRuntime()
core = lua.execute(open(os.path.join(here, "..", "md_audio_tiled_core.lua"), encoding="utf-8").read())

# ---------------------------------------------------------------------------
# Inline OLD formula (ceil-based) for TEETH CHECK
# ---------------------------------------------------------------------------
def old_auto_schedule(W, max_single, overlap):
    """Pre-fix build_auto_schedule using math.ceil -- leaves last tile short."""
    if W <= 0:
        return []
    if not (max_single == max_single) or max_single < 16:  # NaN or bad
        max_single = 15000
    if W <= max_single:
        return [(0, W)]
    if overlap < 0:
        overlap = 0
    overlap = min(overlap, max_single // 4)
    n      = math.ceil((W - overlap) / (max_single - overlap))
    length = math.ceil((W + (n - 1) * overlap) / n)
    hop    = length - overlap
    sched  = []
    for i in range(n):
        s = i * hop
        e = W if i == n - 1 else min(W, s + length)
        sched.append((s, e))
    return sched

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
nan = float("nan")

# Baseline cases (max=15000, kept for regression)
BASELINE = [
    ("very short 10f",     10,    15000, 192),
    ("total<=tile 3000",  3000,   15000, 192),
    ("164s 4100",         4100,   15000, 192),
    ("user 4622",         4622,   15000, 192),
    ("exactly max",       15000,  15000, 192),
    ("max+1",             15001,  15000, 192),
    ("2.5x max",          37500,  15000, 192),
    ("overlap>=tile",     20000,  15000, 99999),
    ("NaN params",         4622,    nan,   nan),
    ("zero params",        4622,      0,     0),
]

# New multi-tile cases at max=6000 (Part A: 3+ tiles)
MULTI_TILE = [
    ("13000 @max6000",    13000,  6000, 192),
    ("18001 @max6000",    18001,  6000, 192),
    ("25000 @max6000",    25000,  6000, 192),
    ("37500 @max6000",    37500,  6000, 192),
]

# ---------------------------------------------------------------------------
# TEETH CHECK: show old formula failing the spread<=1 assertion
# ---------------------------------------------------------------------------
print("=" * 72)
print("TEETH CHECK — pre-fix (ceil) formula on multi-tile cases")
print("Expected: FAIL spread>1 for every case below")
print("=" * 72)
teeth_ok = True
for name, W, mx, ov in MULTI_TILE:
    tiles = old_auto_schedule(W, mx, ov)
    lens  = [e - b for b, e in tiles]
    spread = max(lens) - min(lens) if lens else 0
    spread_fail = spread > 1
    last_ok = tiles[-1][1] == W if tiles else False
    flag = "FAIL" if spread_fail else "PASS"
    print(f"  {flag} {name:20s}  tiles={len(tiles)}  lens={lens}  spread={spread}  last_end={tiles[-1][1] if tiles else 'N/A'}")
    if not spread_fail:
        print(f"    *** WARNING: expected this to fail but spread={spread} (teeth check incomplete) ***")
        teeth_ok = False
print()

# ---------------------------------------------------------------------------
# Main harness: Lua build_auto_schedule
# ---------------------------------------------------------------------------
def check_case(name, W, mx, ov, mode):
    if mode == "old":
        sched = core.build_fixed_schedule(W, 4096, 192, 0)
        lim   = mx if mx == mx and mx >= 16 else 15000
    else:
        sched = core.build_auto_schedule(W, mx, ov)
        lim   = mx if mx == mx and mx >= 16 else 6000

    n     = len(sched)
    tiles = [(sched[i].start, sched[i].end_) for i in range(1, n + 1)]
    lens  = [e - b for b, e in tiles]
    spread = max(lens) - min(lens) if lens else 0

    # Core assertions
    coverage_ok = tiles[0][0] == 0 and tiles[-1][1] == W if tiles else W == 0
    chain_ok    = all(tiles[i][0] < tiles[i-1][1] for i in range(1, len(tiles)))
    single_ok   = True
    multi_ok    = True
    if W <= lim:
        seams = [b for b, _ in tiles[1:]]
        single_ok = not seams
    else:
        multi_ok = spread <= 1 and max(lens) <= lim

    passed = coverage_ok and chain_ok and single_ok and multi_ok

    # Print tile list (TEETH CHECK requirement: always print actual numbers)
    print(f"{'PASS' if passed else 'FAIL'} {name:22s}  W={W:6d}  tiles={len(tiles):2d}  lens={lens[:8]}{'...' if len(lens)>8 else ''}  spread={spread}  covered=[{tiles[0][0] if tiles else '?'},{tiles[-1][1] if tiles else '?'})")
    if not passed:
        if not coverage_ok:
            print(f"       coverage FAIL: first={tiles[0][0]} last_end={tiles[-1][1]} W={W}")
        if not chain_ok:
            print(f"       chain FAIL (gap between tiles)")
        if not single_ok:
            print(f"       single-tile FAIL: seam detected for W={W} <= lim={lim}")
        if not multi_ok:
            print(f"       multi-tile FAIL: spread={spread} max_len={max(lens)} lim={lim}")
    return passed

mode = sys.argv[1] if len(sys.argv) > 1 else "new"

print(f"{'=' * 72}")
print(f"Lua build_auto_schedule — mode={mode}")
print(f"{'=' * 72}")
fails = 0
for name, W, mx, ov in BASELINE + MULTI_TILE:
    ok = check_case(name, W, mx, ov, mode)
    fails += not ok

print()
print(f"Result: {fails} FAIL  (mode={mode})")
if not teeth_ok:
    print("WARNING: teeth check did not demonstrate a pre-fix failure -- re-examine old_auto_schedule")
    sys.exit(2)
sys.exit(1 if fails else 0)
