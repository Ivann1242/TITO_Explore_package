from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from TITO_Explore.comparison import compare_titos_overall
from TITO_Explore.inversion_set import tito_to_inversion_set
from TITO_Explore.reflection import reflection_rows_to_blocks_and_order
from TITO_Explore.types import TranslationInvariantTotalOrder

Edge = Tuple[int, int, List[int], int]
MatrixCell = Tuple[List[int], int]


def inversion_matrix_to_edges(
    matrix: List[List[MatrixCell | None]],
    n: int,
) -> List[Edge]:
    """
    Converts an inversion matrix entry ([vals], star) at matrix[i][j]
    into edge tuples (i, (i + v) % n, [v], star_indicator).
    """
    edges: List[Edge] = []
    for i in range(n):
        for j in range(n):
            cell = matrix[i][j]
            if cell is None:
                continue
            values, infinite = cell
            indicator = 1 if infinite else 0
            for value in values:
                edges.append((i, (i + value) % n, [value], indicator))
    return edges


def _tito_to_window(
    tito: TranslationInvariantTotalOrder,
    inv_matrix: List[List[MatrixCell | None]] | None = None,
) -> Tuple[List[List[int]], List[int], Optional[Dict[str, Any]]]:
    """
    Converts a single TITO into the same window/flag/report triple that
    compute_tito_join returns. Accepts an optional precomputed inversion
    matrix to avoid recomputation when available.
    """
    inv = inv_matrix if inv_matrix is not None else tito_to_inversion_set(tito)
    edges = inversion_matrix_to_edges(inv, tito.n)
    closure = compute_closure(tito.n, edges)
    final_matrix = update_indicators(closure)
    table_rows = generate_reflection_table(final_matrix)
    return reflection_rows_to_blocks_and_order(table_rows, tito.n)


def initialize_custom_matrix(n: int, initial_edges: List[Edge]) -> List[List[MatrixCell]]:
    """Initializes an n by n matrix with edge values and a 0 indicator."""
    matrix: List[List[MatrixCell]] = [[([], 0) for _ in range(n)] for _ in range(n)]
    for u, v, values, edge_indicator in initial_edges:
        current_values, indicator = matrix[u][v]
        matrix[u][v] = (
            list(set(current_values) | set(values)),
            1 if indicator == 1 or edge_indicator == 1 else 0,
        )
    return matrix


def update_path(matrix: List[List[MatrixCell]], i: int, k: int, j: int) -> bool:
    """
    Extends path i->k->j if k is strictly intermediate.
    This guard prevents self-loop compounding.
    """
    if i == k or j == k:
        return False

    list_ik, indicator_ik = matrix[i][k]
    list_kj, indicator_kj = matrix[k][j]

    if list_ik and list_kj:
        generated = {a + b for a in list_ik for b in list_kj}
        current = set(matrix[i][j][0])
        current_indicator = matrix[i][j][1]
        generated_indicator = 1 if indicator_ik == 1 or indicator_kj == 1 else 0
        combined = current | generated
        combined_indicator = 1 if current_indicator == 1 or generated_indicator == 1 else 0
        if combined != current or combined_indicator != current_indicator:
            matrix[i][j] = (list(combined), combined_indicator)
            return True
    return False


def final_star_pattern(matrix: List[List[MatrixCell]]) -> Tuple[List[List[bool]], List[List[bool]]]:
    """
    Predicts, from the union matrix alone, which cells of the closure are
    nonempty and which are starred.

    Returns (nonempty, starred). A cell (i, j) of the closure is nonempty
    exactly when the union digraph has a walk of positive length from i to j
    (for i == j: when i lies on a cycle). It is starred exactly when such a
    walk can traverse a starred union edge or a cycle; this is the R M R rule
    of update_indicators, evaluated on the union instead of the closure.
    Reachability is the same in both, and a closure diagonal is nonempty
    exactly when its vertex lies on a union cycle, so both rules agree.
    """
    n = len(matrix)
    reach = [[bool(matrix[i][j][0]) for j in range(n)] for i in range(n)]
    for k in range(n):
        row_k = reach[k]
        for i in range(n):
            if reach[i][k]:
                row_i = reach[i]
                for j in range(n):
                    if row_k[j]:
                        row_i[j] = True
    reflexive = [[i == j or reach[i][j] for j in range(n)] for i in range(n)]
    marked = [[(bool(matrix[a][b][0]) and bool(matrix[a][b][1])) or (a == b and reach[a][a])
               for b in range(n)] for a in range(n)]
    through_mark = [[any(reflexive[i][a] and marked[a][b] for a in range(n))
                     for b in range(n)] for i in range(n)]
    starred = [[reach[i][j] and any(through_mark[i][b] and reflexive[b][j] for b in range(n))
                for j in range(n)] for i in range(n)]
    return reach, starred


def compute_closure(n: int, initial_edges: List[Edge]) -> List[List[MatrixCell]]:
    """
    Runs the Floyd-Warshall-style closure over the edge matrix.

    Starred cells of the closure are full period-n progressions from their
    least increment, so only that least increment is kept for them. The star
    pattern is predicted before the pass (final_star_pattern), which lets the
    pass prune every eventually starred cell to its minimum as it goes.
    Unstarred cells only receive sums of unstarred cells (a sum through a
    starred cell would itself be starred), so they keep exactly the finite
    increments of the unpruned pass. The minimum of a sumset is the sum of
    the minima, so pruning never changes a least increment. Without pruning,
    redundant starred increments multiply along every path and the cell
    sizes grow rapidly with n.
    """
    matrix = initialize_custom_matrix(n, initial_edges)
    _nonempty, starred = final_star_pattern(matrix)
    values = [[set(matrix[i][j][0]) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if starred[i][j] and values[i][j]:
                values[i][j] = {min(values[i][j])}

    for k in range(n):
        for i in range(n):
            if i == k:
                continue
            values_ik = values[i][k]
            if not values_ik:
                continue
            min_ik = min(values_ik)
            starred_i = starred[i]
            values_i = values[i]
            for j in range(n):
                if j == k:
                    continue
                values_kj = values[k][j]
                if not values_kj:
                    continue
                if starred_i[j]:
                    candidate = min_ik + min(values_kj)
                    current = values_i[j]
                    if not current or candidate < next(iter(current)):
                        values_i[j] = {candidate}
                else:
                    values_i[j] |= {a + b for a in values_ik for b in values_kj}

    return [[(sorted(values[i][j]), int(starred[i][j])) for j in range(n)] for i in range(n)]


def update_indicators(matrix: List[List[MatrixCell]]) -> List[List[MatrixCell]]:
    """
    Complete infinite-family flags on a closed TITO inversion matrix.

    A walk is unbounded if it traverses a starred edge or a nonempty cycle,
    even when neither endpoint lies on that cycle. With reflexive reachability
    R and the matrix M of starred edges/cycles, the required flags are R M R.
    Two Boolean products cost O(n^3). Values from compute_closure are preserved.
    For unions of TITO inversion sets, every infinite cell is a full period-n
    progression from its minimum; this is not for arbitrary weighted graphs.
    """
    n = len(matrix)
    reach = [[i == j or bool(matrix[i][j][0]) for j in range(n)] for i in range(n)]
    marked = [[bool(matrix[i][j][0]) and (bool(matrix[i][j][1]) or i == j)
               for j in range(n)] for i in range(n)]
    through_mark = [[any(reach[i][k] and marked[k][j] for k in range(n))
                     for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            values, _ = matrix[i][j]
            infinite = any(through_mark[i][k] and reach[k][j] for k in range(n))
            matrix[i][j] = (values, int(bool(values) and infinite))
    return matrix


def generate_reflection_table(matrix: List[List[MatrixCell]]) -> List[Tuple[int, int, int]]:
    """
    Converts the closure matrix into reflection table rows:
    (start_node, start_node + value, indicator)
    """
    n = len(matrix)
    table_rows: List[Tuple[int, int, int]] = []
    for a in range(n):
        for b in range(n):
            values, indicator = matrix[a][b]
            for c in values:
                table_rows.append((a, a + c, indicator))
    return table_rows


def compute_join_reflection_table(
    tito1: TranslationInvariantTotalOrder,
    tito2: TranslationInvariantTotalOrder,
) -> List[Tuple[int, int, int]]:
    """
    Takes two TITO objects, converts them to inversion matrices,
    converts the matrices to edge lists, runs the closure + indicator logic,
    and returns the resulting reflection table rows.
    """
    if tito1.n != tito2.n:
        raise ValueError("Both TITOs must have the same modulus n.")

    n = tito1.n
    inv1 = tito_to_inversion_set(tito1)
    inv2 = tito_to_inversion_set(tito2)

    edges = inversion_matrix_to_edges(inv1, n) + inversion_matrix_to_edges(inv2, n)
    closure = compute_closure(n, edges)
    final_matrix = update_indicators(closure)
    return generate_reflection_table(final_matrix)


def _compute_tito_join_with_closure(
    tito1: TranslationInvariantTotalOrder,
    tito2: TranslationInvariantTotalOrder,
) -> Tuple[
    Union[TranslationInvariantTotalOrder, Tuple[List[List[int]], List[int], Optional[Dict[str, Any]]]],
    Optional[List[List[MatrixCell]]],
]:
    """Shared body of the join functions; also returns the closed matrix (None if comparable)."""
    if tito1.n != tito2.n:
        raise ValueError("Both TITOs must have the same modulus n.")

    comparison = compare_titos_overall(tito1, tito2)
    if comparison == "tito1 < tito2":
        return tito2, None
    if comparison == "tito1 > tito2":
        return tito1, None
    if comparison == "tito1 = tito2":
        return tito1, None

    inv1 = tito_to_inversion_set(tito1)
    inv2 = tito_to_inversion_set(tito2)
    edges = inversion_matrix_to_edges(inv1, tito1.n) + inversion_matrix_to_edges(inv2, tito1.n)
    closure = compute_closure(tito1.n, edges)
    final_matrix = update_indicators(closure)
    table_rows = generate_reflection_table(final_matrix)
    return reflection_rows_to_blocks_and_order(table_rows, tito1.n), final_matrix


def compute_tito_join(
    tito1: TranslationInvariantTotalOrder,
    tito2: TranslationInvariantTotalOrder,
) -> Union[TranslationInvariantTotalOrder, Tuple[List[List[int]], List[int], Optional[Dict[str, Any]]]]:
    """
    Computes the join of two TITOs. If the two TITOs are already comparable
    in the TITO order, the larger original TITO object is returned directly.
    If they are equal, either original TITO may be returned. Otherwise the join
    is computed via reflection-table closure and returned in window/flag/report
    form.
    """
    return _compute_tito_join_with_closure(tito1, tito2)[0]


def compute_tito_join_tito(
    tito1: TranslationInvariantTotalOrder,
    tito2: TranslationInvariantTotalOrder,
) -> TranslationInvariantTotalOrder:
    """
    Computes the join of two TITOs and always returns a TITO object.

    This is the object-returning counterpart to compute_tito_join. Comparable
    inputs return the larger original TITO; otherwise the closure result is
    reconstructed into a TranslationInvariantTotalOrder.
    """
    join_result, expected = _compute_tito_join_with_closure(tito1, tito2)

    if isinstance(join_result, TranslationInvariantTotalOrder):
        return join_result

    blocks, waxing_waning, report = join_result
    if report is not None:
        raise ValueError(f"Could not reconstruct join TITO cleanly: {report}")

    result = TranslationInvariantTotalOrder(
        n=tito1.n,
        num_blocks=len(blocks),
        vectors=blocks,
        waxing_waning=waxing_waning,
    )
    # Check the reconstruction against the closure computed above.
    actual = tito_to_inversion_set(result)

    def canonical(cell):
        if cell is None or not cell[0]:
            return (), 0
        values, star = cell
        return ((min(values),) if star else tuple(sorted(set(values)))), int(bool(star))

    if any(canonical(actual[i][j]) != canonical(expected[i][j])
           for i in range(tito1.n) for j in range(tito1.n)):
        raise ValueError("Reconstructed TITO inversion table differs from the computed closure.")
    return result
