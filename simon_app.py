import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

try:
    from qiskit_aer import Aer  # type: ignore
except Exception:  # pragma: no cover
    Aer = None  # fallback to statevector sampling


# ------------------------------
# Utilities: bit operations
# ------------------------------

def bits_le_to_str_msb(bits_le: Sequence[int]) -> str:
    """Convert little-endian bits [b0..b{n-1}] to MSB-left string."""
    return "".join(str(b) for b in reversed(bits_le))


def str_msb_to_bits_le(s: str, n: int) -> List[int]:
    s = s.strip().replace(" ", "")
    if len(s) != n or any(c not in "01" for c in s):
        raise ValueError("Secret must be an n-bit binary string")
    return [int(c) for c in reversed(s)]


def int_to_bits_le(x: int, n: int) -> List[int]:
    return [(x >> i) & 1 for i in range(n)]


def bits_le_to_int(bits_le: Sequence[int]) -> int:
    v = 0
    for i, b in enumerate(bits_le):
        if b:
            v |= (1 << i)
    return v


def xor_bits(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [(ai ^ bi) for ai, bi in zip(a, b)]


# ------------------------------
# GF(2) Linear Algebra
# ------------------------------

def gf2_rref(matrix: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
    """Row-reduced echelon form over GF(2). Returns (rref, pivot_cols)."""
    if not matrix:
        return [], []
    m = [row[:] for row in matrix]
    rows = len(m)
    cols = len(m[0])
    r = 0
    pivots: List[int] = []
    for c in range(cols):
        # Find a row with a 1 in column c at or below row r
        pivot_row = None
        for rr in range(r, rows):
            if m[rr][c] == 1:
                pivot_row = rr
                break
        if pivot_row is None:
            continue
        # Swap into position r
        if pivot_row != r:
            m[r], m[pivot_row] = m[pivot_row], m[r]
        # Eliminate all other rows in column c
        for rr in range(rows):
            if rr != r and m[rr][c] == 1:
                m[rr] = [a ^ b for a, b in zip(m[rr], m[r])]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return m, pivots


def gf2_rank(matrix: List[List[int]]) -> int:
    _, pivots = gf2_rref(matrix)
    return len(pivots)


def gf2_nullspace(matrix: List[List[int]], nvars: Optional[int] = None) -> List[List[int]]:
    """Return a basis for the right nullspace of matrix over GF(2).

    Solves matrix @ x = 0. If nvars None, inferred from number of columns.
    """
    if not matrix:
        return []
    cols = len(matrix[0]) if nvars is None else nvars
    rref, pivots = gf2_rref(matrix)
    pivot_set = set(pivots)
    free_cols = [c for c in range(cols) if c not in pivot_set]
    basis: List[List[int]] = []
    # Build basis vector per free variable
    for f in free_cols:
        vec = [0] * cols
        vec[f] = 1
        # For each pivot row, set pivot variable = rref[row][f]
        prow = 0
        for pcol in pivots:
            vec[pcol] = rref[prow][f]
            prow += 1
        basis.append(vec)
    return basis


# ------------------------------
# Construct Simon oracle y = M x (mod 2)
# with nullspace(M) = span{s}
# ------------------------------

def basis_of_s_orthogonal(s_bits: Sequence[int]) -> List[List[int]]:
    n = len(s_bits)
    ones_idx = [i for i, b in enumerate(s_bits) if b == 1]
    zeros_idx = [i for i, b in enumerate(s_bits) if b == 0]
    if len(ones_idx) == 0:
        # s = 0^n: degenerate; return standard basis except last
        rows = [[1 if j == i else 0 for j in range(n)] for i in range(n - 1)]
        return rows
    rows: List[List[int]] = []
    # Include e_i for zeros coordinates (orthogonal to s)
    for i in zeros_idx:
        row = [0] * n
        row[i] = 1
        rows.append(row)
    # For ones, add e_i XOR e_last to get orthogonal rows
    last = ones_idx[-1]
    for i in ones_idx[:-1]:
        row = [0] * n
        row[i] ^= 1
        row[last] ^= 1
        rows.append(row)
    # rows has exactly n-1 independent vectors, all orthogonal to s
    return rows


def build_M_from_secret(s_bits: Sequence[int]) -> List[List[int]]:
    n = len(s_bits)
    rows = basis_of_s_orthogonal(s_bits)
    # Pad to n rows by adding a zero row; rank remains n-1
    rows.append([0] * n)
    # Validate
    rank = gf2_rank(rows)
    if rank != n - 1:
        # Fallback: randomize additional linear combos to ensure rank n-1
        rows = basis_of_s_orthogonal(s_bits)
        while gf2_rank(rows) < n - 1:
            # Add another combination of rows (should not happen in theory)
            combo = [0] * n
            for r in rows:
                if np.random.rand() < 0.5:
                    combo = [a ^ b for a, b in zip(combo, r)]
            if any(combo):
                rows.append(combo)
        rows = rows[: n - 1] + [[0] * n]
    # Sanity: M @ s == 0
    for row in rows:
        dot = 0
        for a, b in zip(row, s_bits):
            dot ^= (a & b)
        if row is not rows[-1] and dot != 0:
            raise RuntimeError("Row not orthogonal to secret; construction failed")
    return rows


# ------------------------------
# Qiskit: Build and run Simon circuit
# ------------------------------

def build_simon_circuit(n: int, M: List[List[int]], measure: bool = True) -> QuantumCircuit:
    x = QuantumRegister(n, "x")
    y = QuantumRegister(n, "y")
    c = ClassicalRegister(n, "c") if measure else None
    qc = QuantumCircuit(x, y, c) if measure else QuantumCircuit(x, y)
    # Step 1: H on x
    qc.h(x)
    # Step 2: Oracle U_f: |x>|y> -> |x>|y ⊕ Mx>
    for i in range(n):
        for j in range(n):
            if M[i][j] == 1:
                qc.cx(x[j], y[i])
    # Step 3: H on x
    qc.h(x)
    if measure:
        qc.barrier()
        qc.measure(x, c)
    return qc


def run_counts_with_aer(qc: QuantumCircuit, shots: int) -> Optional[Dict[str, int]]:
    if Aer is None:
        return None
    try:
        backend = Aer.get_backend("aer_simulator")
        tqc = transpile(qc, backend)
        job = backend.run(tqc, shots=shots)
        res = job.result()
        counts = res.get_counts()
        if isinstance(counts, list):  # multiple experiments
            counts = counts[0]
        return dict(counts)
    except Exception:
        return None


def run_counts_statevector(qc_no_meas: QuantumCircuit, n_top: int, shots: int) -> Dict[str, int]:
    sv = Statevector.from_instruction(qc_no_meas)
    # Probabilities for top n qubits only
    qargs = list(range(n_top))  # x register was added first
    probs_dict = sv.probabilities_dict(qargs=qargs)
    # probs_dict keys are little-endian strings of length n_top
    keys = list(probs_dict.keys())
    probs = np.array([probs_dict[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    samples = np.random.choice(len(keys), size=shots, p=probs)
    counts: Dict[str, int] = {}
    for idx in samples:
        k = keys[idx]
        counts[k] = counts.get(k, 0) + 1
    return counts


def get_counts(n: int, M: List[List[int]], shots: int) -> Dict[str, int]:
    qc_meas = build_simon_circuit(n, M, measure=True)
    counts = run_counts_with_aer(qc_meas, shots)
    if counts is not None:
        return counts
    # Fallback: simulate statevector and sample top register
    qc_nom = build_simon_circuit(n, M, measure=False)
    return run_counts_statevector(qc_nom, n_top=n, shots=shots)


# ------------------------------
# Simon workflow helpers
# ------------------------------

def add_equations_from_counts(
    counts: Dict[str, int], Y_rows: List[List[int]], n: int
) -> Tuple[List[List[int]], int]:
    """Update Y_rows with new independent y vectors from counts.
    Returns (updated_rows, num_new). Bitstrings in counts are MSB-left from Qiskit; convert to little-endian.
    """
    added = 0
    rank_before = gf2_rank(Y_rows) if Y_rows else 0
    # Sort by frequency desc to try common outcomes first
    for k, _v in sorted(counts.items(), key=lambda kv: -kv[1]):
        y_bits = [int(b) for b in reversed(k)]  # little-endian list
        if all(b == 0 for b in y_bits):
            continue
        candidate = Y_rows + [y_bits]
        if gf2_rank(candidate) > rank_before:
            Y_rows.append(y_bits)
            rank_before += 1
            added += 1
            if rank_before >= n - 1:
                break
    return Y_rows, added


def try_solve_secret(Y_rows: List[List[int]], n: int) -> Optional[List[int]]:
    if not Y_rows:
        return None
    basis = gf2_nullspace(Y_rows, nvars=n)
    if len(basis) == 1:
        s = basis[0]
        if any(s):
            return s
    return None


def verify_secret(M: List[List[int]], s: Sequence[int]) -> bool:
    # Check M @ s == 0 mod 2
    for row in M:
        dot = 0
        for a, b in zip(row, s):
            dot ^= (a & b)
        if dot != 0:
            return False
    return any(s)


def truth_table_df(M: List[List[int]]) -> pd.DataFrame:
    n = len(M)
    rows = []
    for x in range(1 << n):
        xb = int_to_bits_le(x, n)
        yb = [0] * n
        for i in range(n):
            # y[i] = sum_j M[i][j]*x[j] mod 2
            acc = 0
            for j in range(n):
                acc ^= (M[i][j] & xb[j])
            yb[i] = acc
        rows.append(
            {
                "x": bits_le_to_str_msb(xb),
                "f(x)": bits_le_to_str_msb(yb),
                "x_int": x,
                "f_int": bits_le_to_int(yb),
            }
        )
    df = pd.DataFrame(rows).sort_values("x_int").drop(columns=["x_int", "f_int"]).reset_index(drop=True)
    return df


# ------------------------------
# Streamlit App
# ------------------------------

st.set_page_config(page_title="Simon's Algorithm — Qiskit + Streamlit", layout="wide")

st.title("Simon's Algorithm — Interactive Simulation (Qiskit + Streamlit)")

with st.sidebar:
    st.header("Controls")
    n = st.slider("Number of input bits n", min_value=2, max_value=8, value=4, help="Total qubits = 2n")

    secret_mode = st.radio(
        "Secret s",
        options=["Random (non-zero)", "Manual"],
        index=0,
        help="Choose the hidden mask s",
    )

    manual_s_str: Optional[str] = None
    if secret_mode == "Manual":
        manual_s_str = st.text_input(
            f"Enter {n}-bit s (MSB→LSB)",
            value=("1" + "0" * (n - 1)) if n > 1 else "1",
            max_chars=n,
        )

    shots = st.slider("Shots per run", min_value=16, max_value=4096, value=512, step=16)

    colb1, colb2, colb3 = st.columns(3)
    with colb1:
        btn_reset = st.button("Reset/Apply settings")
    with colb2:
        btn_run = st.button("Run circuit (collect)")
    with colb3:
        btn_auto = st.button("Auto-solve")

# Initialize / reset session state
if "init_key" not in st.session_state:
    st.session_state.init_key = None

init_key = f"n={n}|mode={secret_mode}|manual={manual_s_str or ''}"

if btn_reset or st.session_state.init_key != init_key:
    # Build or validate secret s
    if secret_mode == "Manual":
        try:
            s_bits = str_msb_to_bits_le(manual_s_str or "", n)
        except Exception as e:
            st.error(str(e))
            st.stop()
        if not any(s_bits):
            st.error("Secret s cannot be all zeros in Simon's promise (choose a non-zero).")
            st.stop()
    else:
        # Random non-zero
        rng = np.random.default_rng()
        while True:
            s_bits = [int(b) for b in rng.integers(0, 2, size=n)]
            if any(s_bits):
                break

    M = build_M_from_secret(s_bits)

    st.session_state.n = n
    st.session_state.s_bits = s_bits
    st.session_state.M = M
    st.session_state.counts_total = {}  # type: Dict[str, int]
    st.session_state.last_counts = {}
    st.session_state.Y_rows = []  # collected independent equations y · s = 0
    st.session_state.solution = None  # type: Optional[List[int]]
    st.session_state.revealed = False
    st.session_state.init_key = init_key

# Access current state
n = st.session_state.n
s_bits = st.session_state.s_bits
M = st.session_state.M

# Header chips
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("n (bits)", n)
with c2:
    st.metric("Total qubits", 2 * n)
with c3:
    st.metric("Collected equations", len(st.session_state.Y_rows))
with c4:
    s_str = bits_le_to_str_msb(s_bits)
    if st.session_state.revealed:
        st.metric("Hidden s (MSB→LSB)", s_str)
    else:
        st.metric("Hidden s", "••••")

# Explanations
with st.expander("How Simon's algorithm works (short)", expanded=False):
    st.markdown(
        "- Prepare |0^n>|0^n>.\n"
        "- Apply H^n on the first register to create a uniform superposition over x.\n"
        "- Query the oracle U_f: |x>|y> → |x>|y ⊕ f(x)>. We use f(x)=M·x (mod 2) with nullspace(M)=span{s}.\n"
        "- Apply H^n again on the first register. Measurement yields a random y orthogonal to s (y·s=0 mod 2).\n"
        "- Repeat to collect n−1 independent y's, then solve the linear system over GF(2) to recover s."
    )

# Build circuit for display
qc_display = build_simon_circuit(n, M, measure=True)
colA, colB = st.columns([1.1, 0.9])
with colA:
    st.subheader("Simon circuit")
    fig_circ = qc_display.draw(output="mpl")
    st.pyplot(fig_circ)
with colB:
    st.subheader("Oracle matrix M (y = Mx mod 2)")
    dfM = pd.DataFrame(M, columns=[f"x{j}" for j in range(n)])
    dfM.index = [f"y{i}" for i in range(n)]
    st.dataframe(dfM, use_container_width=True)
    st.caption("Nullspace(M) = span{s}. Rows are constraints orthogonal to s.")

# Truth table (limited n)
with st.expander("Truth table f(x) (only for n ≤ 6)", expanded=False):
    if n <= 6:
        df_tt = truth_table_df(M)
        st.dataframe(df_tt, use_container_width=True, height=320)
        st.caption("For all x, f(x) = f(x ⊕ s) holds.")
    else:
        st.info("Increase performance: truth table hidden for n > 6.")

# Run / collect
if btn_run:
    counts = get_counts(n, M, shots)
    st.session_state.last_counts = counts
    # Merge into totals
    for k, v in counts.items():
        st.session_state.counts_total[k] = st.session_state.counts_total.get(k, 0) + v
    # Add equations
    st.session_state.Y_rows, _ = add_equations_from_counts(counts, st.session_state.Y_rows, n)
    # Try solve
    st.session_state.solution = try_solve_secret(st.session_state.Y_rows, n)

if btn_auto and st.session_state.solution is None:
    # Collect until rank n-1 or up to safety cap
    safety_cap = 40 * n
    iters = 0
    prog = st.progress(0.0, text="Collecting equations…")
    while iters < safety_cap and len(st.session_state.Y_rows) < n - 1 and st.session_state.solution is None:
        counts = get_counts(n, M, shots)
        # Merge counts
        for k, v in counts.items():
            st.session_state.counts_total[k] = st.session_state.counts_total.get(k, 0) + v
        st.session_state.Y_rows, _ = add_equations_from_counts(counts, st.session_state.Y_rows, n)
        st.session_state.solution = try_solve_secret(st.session_state.Y_rows, n)
        iters += 1
        prog.progress(min(1.0, (len(st.session_state.Y_rows) / max(1, n - 1)))),
    prog.empty()

# Results + Visualization
col1, col2 = st.columns([0.8, 1.2])
with col1:
    st.subheader("Latest measurement histogram")
    if st.session_state.last_counts:
        fig_h = plot_histogram(st.session_state.last_counts)
        st.pyplot(fig_h)
    else:
        st.info("Run the circuit to see measurements.")

with col2:
    st.subheader("Aggregated histogram (all runs)")
    if st.session_state.counts_total:
        fig_h2 = plot_histogram(st.session_state.counts_total)
        st.pyplot(fig_h2)
    else:
        st.info("No aggregated data yet.")

# Collected equations and solver
st.subheader("Collected linear constraints (mod 2)")
if st.session_state.Y_rows:
    eq_rows = []
    for y in st.session_state.Y_rows:
        lhs_terms = [f"s{j}" for j, b in enumerate(y) if b == 1]
        pretty = " ⊕ ".join(lhs_terms) if lhs_terms else "0"
        eq_rows.append({"y (MSB→LSB)": bits_le_to_str_msb(y), "equation": f"{pretty} = 0"})
    st.table(pd.DataFrame(eq_rows))
else:
    st.write("No equations yet. Press Run or Auto-solve.")

# Solve status
sol = st.session_state.solution
sol_ok = verify_secret(M, sol) if sol is not None else False
colS1, colS2, colS3 = st.columns(3)
with colS1:
    if sol is not None:
        st.success(f"Recovered s: {bits_le_to_str_msb(sol)}")
    else:
        st.warning("Secret not recovered yet.")
with colS2:
    if sol is not None:
        st.metric("Verified M·s = 0?", "Yes" if sol_ok else "No")
with colS3:
    if st.button("Reveal hidden s"):
        st.session_state.revealed = True
        st.rerun()

st.divider()

st.markdown(
    "**Run locally:** in a terminal, execute:\n"
    "```bash\nstreamlit run simon_app.py\n```"
)

st.caption(
    "This app uses a linear Simon oracle f(x)=M·x (mod 2) such that nullspace(M)=span{s}.\n"
    "The algorithm still demonstrates Simon's exponential speedup: classically O(2^{n/2}) vs quantum O(n)."
)
