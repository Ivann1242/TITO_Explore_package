"""Scaling benchmark for compute_tito_join_tito on seeded random incomparable pairs.

Usage:  python examples/benchmark_scaling.py [out.json]

Families (one independent RNG per family and period, base seed 20261109):
  R  each input independently: shuffle the residues 0..n-1, cut each of the n-1
     gaps with probability 0.35 into consecutive windows, map each entry r to
     r + n*s with s uniform in {-3, ..., 3}, and make each window waning with
     probability 1/2 (the generator of tests/test_orders.py).
  S  one such partition and block order shared by A and B; each input draws its
     own shifts and flags.
Only incomparable pairs are kept. W is the largest number of increments in one
cell of compute_closure(n, union edges). Times cover the full checked join
(comparison, closure, star propagation, reconstruction and verification);
input construction is excluded. Each (family, n) runs in a child process with a
wall-clock budget; completed pairs are kept. Timings are local measurements,
not a cross-machine guarantee.
"""
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import timeit

SEED, PAIRS, BUDGET = 20261109, 20, 50.0
PLAN = {'R': [4, 6, 8, 10, 12, 16, 24, 32, 48, 64], 'S': [8, 16, 24, 32, 48, 64]}


def child(family, n):
    from TITO_Explore import (TranslationInvariantTotalOrder as TITO, compare_titos_overall,
                              compute_tito_join_tito, tito_to_inversion_set)
    from TITO_Explore.join import compute_closure, inversion_matrix_to_edges
    rng = random.Random(SEED * 1000 + n + (0 if family == 'R' else 500))

    def partition():
        residues = list(range(n)); rng.shuffle(residues)
        cuts = [0] + [i for i in range(1, n) if rng.random() < 0.35] + [n]
        return [residues[a:b] for a, b in zip(cuts, cuts[1:])]

    def fill(parts):
        blocks = [[r + n * rng.randint(-3, 3) for r in p] for p in parts]
        return TITO(n, len(blocks), blocks, [rng.randint(0, 1) for _ in blocks])

    done = 0
    while done < PAIRS:
        if family == 'R':
            a, b = fill(partition()), fill(partition())
        else:
            shared = partition(); a, b = fill(shared), fill(shared)
        if compare_titos_overall(a, b) != 'incomparable':
            continue
        edges = (inversion_matrix_to_edges(tito_to_inversion_set(a), n)
                 + inversion_matrix_to_edges(tito_to_inversion_set(b), n))
        W = max(len(cell[0]) for row in compute_closure(n, edges) for cell in row)
        start = time.perf_counter(); J = compute_tito_join_tito(a, b); sec = time.perf_counter() - start
        if sec < 0.05:  # short joins: median of 3 batches of about 0.1 s each
            number = max(1, math.ceil(0.1 / max(sec, 1e-6)))
            sec = statistics.median(t / number for t in timeit.repeat(
                lambda: compute_tito_join_tito(a, b), repeat=3, number=number))
        print(json.dumps(dict(family=family, n=n, W=W, sec=sec, in_blocks=[a.num_blocks, b.num_blocks],
                              join_blocks=J.num_blocks, join_waning=sum(J.waxing_waning),
                              A=[a.vectors, a.waxing_waning], B=[b.vectors, b.waxing_waning])), flush=True)
        done += 1


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_scaling.json'
    try:
        cpu = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        cpu = ''
    import TITO_Explore
    env = dict(package_version=TITO_Explore.__version__, python=platform.python_version(),
               platform=platform.platform(), machine=platform.machine(), cpu=cpu,
               seed=SEED, pairs_target=PAIRS, budget_s=BUDGET)
    print(json.dumps(env), flush=True)
    rows, summary = [], []
    for family, periods in PLAN.items():
        for n in periods:
            start = time.time()
            proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), '--child', family, str(n)],
                                    stdout=subprocess.PIPE, text=True)
            try:
                stdout, _ = proc.communicate(timeout=BUDGET)
            except subprocess.TimeoutExpired:
                proc.kill(); stdout, _ = proc.communicate()
            got = [json.loads(line) for line in stdout.splitlines() if line.strip()]
            rows += got
            entry = dict(family=family, n=n, pairs=len(got), wall_s=round(time.time() - start, 1))
            if got:
                ms = [g['sec'] * 1e3 for g in got]; W = [g['W'] for g in got]
                entry.update(median_ms=statistics.median(ms), max_ms=max(ms),
                             median_W=statistics.median(W), max_W=max(W),
                             median_join_blocks=statistics.median(g['join_blocks'] for g in got))
            summary.append(entry); print(json.dumps(entry), flush=True)
            if len(got) < 3:
                break  # larger periods of this family would exceed the budget too
    with open(out, 'w') as handle:
        json.dump(dict(env=env, summary=summary, rows=rows), handle, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        child(sys.argv[2], int(sys.argv[3]))
    else:
        main()
