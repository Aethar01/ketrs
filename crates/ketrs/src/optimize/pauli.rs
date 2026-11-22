//! Lossless synthesis of `PauliRotation` runs into minimal `{CX, 1q}` circuits.
//!
//! Uses two exact, complementary techniques:
//! 1. **Simultaneous Clifford diagonalization + GraySynth** (Amy, Azimzadeh,
//!    Mosca 2018) for maximal mutually-commuting clusters of Pauli rotations:
//!    transforms all commuting strings into pure Z-strings via symplectic
//!    Clifford reduction, then synthesizes the resulting phase polynomial with
//!    near-minimal CNOTs (often 1 CNOT per phase term instead of `2(w-1)`).
//! 2. **Tree-aligned ladder synthesis** for non-commuting neighbors: chooses
//!    shared-support outer CNOT chains and differing-Pauli roots so adjacent
//!    uncompute/compute ladders and basis changes cancel directly.
//!
//! Works losslessly for both fixed and parametric rotations (linear bindings).

use crate::circuit::{Circuit, Operation};
use crate::gate::Gate;
use crate::observable::{Pauli, PauliString};
use crate::optimize::one_qubit::{wrap_2pi, wrap_pi};
use crate::optimize::passes::{Node, ParamBinding, Segment, reduce_gf2_greedy};
use crate::statevector::Statevector;
use std::collections::HashMap;

/// Lower all weight-≥-2 `PauliRotation` operations into `{CX, 1q}`.
///
/// At `smart = false` (Level 1), uses plain ladder decomposition.
/// At `smart = true` (Level 2/3), partitions contiguous `PauliRotation` runs
/// into maximal mutually-commuting clusters, synthesizes each cluster with the
/// best of Simultaneous Clifford Diagonalization + GraySynth vs Gray-ordered
/// tree-aligned ladders, and aligns single-term ladders with their neighbors.
pub(crate) fn lower_pauli_rotations(seg: &mut Segment, smart: bool, tol: f64) -> bool {
    let has_multi_pauli = seg.nodes.iter().any(|n| {
        matches!(&n.op, Operation::PauliRotation { pauli, .. } if pauli.weight() >= 2)
    });
    if !has_multi_pauli {
        return false;
    }

    let nodes = std::mem::take(&mut seg.nodes);
    let mut out = Vec::with_capacity(nodes.len() * 4);
    let mut i = 0;

    while i < nodes.len() {
        let is_multi_pauli = |n: &Node| {
            matches!(&n.op, Operation::PauliRotation { pauli, .. } if pauli.weight() >= 2)
        };
        let as_pauli_term = |n: &Node| -> Option<PauliTerm> {
            match &n.op {
                Operation::PauliRotation { pauli, theta } => Some(PauliTerm {
                    pauli: pauli.clone(),
                    theta: *theta,
                    binding: n.binding,
                }),
                Operation::Gate { gate, target } => {
                    let (p, theta) = match *gate {
                        Gate::Rx(t) => (Pauli::X, t),
                        Gate::Ry(t) => (Pauli::Y, t),
                        Gate::Rz(t) => (Pauli::Z, t),
                        _ => return None,
                    };
                    let pauli = PauliString::new([(*target, p)]).expect("single qubit");
                    Some(PauliTerm {
                        pauli,
                        theta,
                        binding: n.binding,
                    })
                }
                _ => None,
            }
        };

        if as_pauli_term(&nodes[i]).is_none() {
            out.push(nodes[i].clone());
            i += 1;
            continue;
        }

        // Collect a maximal contiguous run of PauliRotations and single-qubit
        // axis rotations (Rx/Ry/Rz) so weight-1 terms don't fragment a
        // Pauli-evolution run!
        let start = i;
        while i < nodes.len() && as_pauli_term(&nodes[i]).is_some() {
            i += 1;
        }
        let run = &nodes[start..i];
        if !run.iter().any(is_multi_pauli) {
            out.extend_from_slice(run);
            continue;
        }

        if !smart {
            for node in run {
                emit_plain_ladder(node, &mut out);
            }
            continue;
        }

        // Extract Pauli terms in this run (including weight-1 Rx/Ry/Rz).
        let items: Vec<PauliTerm> = run.iter().filter_map(as_pauli_term).collect();

        // Candidate A: per-cluster (Simultaneous Clifford Diag + GraySynth vs aligned ladders).
        let clusters = partition_commuting_clusters(items.clone(), tol);
        let mut cand_clusters = Vec::new();
        for (ci, cluster) in clusters.iter().enumerate() {
            if cluster.is_empty() {
                continue;
            }
            if cluster.len() == 1 {
                let prev_p = ci
                    .checked_sub(1)
                    .and_then(|p| clusters[p].last())
                    .map(|t| &t.pauli);
                let next_p = clusters
                    .get(ci + 1)
                    .and_then(|c| c.first())
                    .map(|t| &t.pauli);
                emit_aligned_ladder(&cluster[0], prev_p, next_p, &mut cand_clusters);
                continue;
            }
            let cand_ladder = synth_cluster_aligned_ladders(cluster);
            let cand_gray = synth_cluster_clifford_graysynth(cluster, tol);
            let best = match cand_gray {
                Some(g)
                    if cx_count(&g) < cx_count(&cand_ladder)
                        || (cx_count(&g) == cx_count(&cand_ladder)
                            && g.len() < cand_ladder.len()) =>
                {
                    g
                }
                _ => cand_ladder,
            };
            cand_clusters.extend(best);
        }
        let cand_clusters = quick_cancel_cx_1q(cand_clusters);

        // Candidate B: Clifford-tracked Pauli-network synthesis across the entire
        // (potentially non-commuting!) run, restoring the accumulated Clifford C†
        // once at the end via symplectic tableau synthesis!
        if items.len() >= 2
            && let Some((cand_net, net_phase)) = synth_pauli_network(&items, tol)
        {
            let net_cx = cx_count(&cand_net);
            let cl_cx = cx_count(&cand_clusters);
            if net_cx < cl_cx || (net_cx == cl_cx && cand_net.len() < cand_clusters.len()) {
                seg.global_phase = wrap_pi(seg.global_phase + net_phase);
                out.extend(cand_net);
                continue;
            }
        }

        out.extend(cand_clusters);
    }

    seg.nodes = out;
    true
}

#[derive(Clone, Debug)]
struct PauliTerm {
    pauli: PauliString,
    theta: f64,
    binding: Option<ParamBinding>,
}

fn cx_count(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .filter(|n| matches!(n.op, Operation::Controlled { gate: Gate::X, .. }))
        .count()
}

/// Partition `items` into maximal mutually-commuting clusters, preserving
/// exact unitary order (an item is pulled forward into the current cluster
/// only if it commutes with every skipped uncollected item AND with every
/// item already in the cluster). Identical Pauli strings within a cluster are
/// merged.
fn partition_commuting_clusters(items: Vec<PauliTerm>, tol: f64) -> Vec<Vec<PauliTerm>> {
    let n = items.len();
    let mut used = vec![false; n];
    let mut clusters = Vec::new();

    let mut head = 0;
    while head < n {
        while head < n && used[head] {
            head += 1;
        }
        if head >= n {
            break;
        }
        used[head] = true;
        let mut cluster = vec![items[head].clone()];
        let mut skipped: Vec<usize> = Vec::new();

        for j in (head + 1)..n {
            if used[j] {
                continue;
            }
            let commutes_with_cluster = cluster
                .iter()
                .all(|t| t.pauli.commutes_with(&items[j].pauli));
            let commutes_with_skipped = skipped
                .iter()
                .all(|&s| items[s].pauli.commutes_with(&items[j].pauli));
            if commutes_with_cluster && commutes_with_skipped {
                used[j] = true;
                cluster.push(items[j].clone());
            } else {
                skipped.push(j);
            }
        }

        // Merge duplicate Pauli strings within this commuting cluster.
        let merged = merge_cluster_duplicates(cluster, tol);
        if !merged.is_empty() {
            clusters.push(merged);
        }
    }

    clusters
}

fn merge_cluster_duplicates(cluster: Vec<PauliTerm>, tol: f64) -> Vec<PauliTerm> {
    let mut out: Vec<PauliTerm> = Vec::with_capacity(cluster.len());
    for term in cluster {
        if let Some(existing) = out.iter_mut().find(|e| {
            e.pauli == term.pauli
                && match (e.binding, term.binding) {
                    (None, None) | (Some(_), None) | (None, Some(_)) => true,
                    (Some(a), Some(b)) => a.parameter == b.parameter,
                }
        }) {
            match (existing.binding, term.binding) {
                (None, None) => {
                    existing.theta = wrap_2pi(existing.theta + term.theta);
                }
                (Some(mut a), None) => {
                    a.offset += term.theta;
                    existing.theta = a.offset;
                    existing.binding = Some(a);
                }
                (None, Some(mut b)) => {
                    b.offset += existing.theta;
                    existing.theta = b.offset;
                    existing.binding = Some(b);
                }
                (Some(a), Some(b)) => {
                    let scale = a.scale + b.scale;
                    let offset = a.offset + b.offset;
                    if scale.abs() <= tol {
                        existing.binding = None;
                        existing.theta = wrap_2pi(offset);
                    } else {
                        existing.binding = Some(ParamBinding {
                            parameter: a.parameter,
                            scale,
                            offset,
                        });
                        existing.theta = offset;
                    }
                }
            }
        } else {
            out.push(term);
        }
    }
    out.retain(|t| t.binding.is_some() || wrap_2pi(t.theta).abs() > tol);
    out
}

/// Plain decomposition of a single `PauliRotation` using sorted qubit order.
fn emit_plain_ladder(node: &Node, out: &mut Vec<Node>) {
    let Operation::PauliRotation { pauli, theta } = &node.op else {
        out.push(node.clone());
        return;
    };
    let ops = pauli.ops();
    if ops.is_empty() {
        return;
    }
    let order: Vec<(usize, Pauli)> = ops.to_vec();
    emit_ordered_ladder(&order, *theta, node.binding, out);
}

/// Emit a single `PauliRotation` with qubits ordered to maximize outer edge
/// cancellation with `prev` and `next` Pauli strings.
fn emit_aligned_ladder(
    term: &PauliTerm,
    prev: Option<&PauliString>,
    next: Option<&PauliString>,
    out: &mut Vec<Node>,
) {
    let ops = term.pauli.ops();
    if ops.is_empty() {
        return;
    }
    if ops.len() == 1 {
        let (q, p) = ops[0];
        let gate = match p {
            Pauli::X => Gate::Rx(term.theta),
            Pauli::Y => Gate::Ry(term.theta),
            Pauli::Z => Gate::Rz(term.theta),
            Pauli::I => return,
        };
        out.push(Node {
            op: Operation::Gate { gate, target: q },
            binding: term.binding,
        });
        return;
    }

    let lookup = |ps: Option<&PauliString>, q: usize| -> Pauli {
        ps.and_then(|s| s.ops().iter().find(|&&(qq, _)| qq == q).map(|&(_, p)| p))
            .unwrap_or(Pauli::I)
    };

    // Score each qubit: higher score = more shared with neighbors -> place
    // toward the outer end of the ladder so basis changes & outer CXs cancel!
    let mut scored: Vec<(usize, usize, Pauli)> = ops
        .iter()
        .map(|&(q, p)| {
            let mut score = 0usize;
            let lp = lookup(prev, q);
            let np = lookup(next, q);
            if lp == p {
                score += 4;
            } else if lp != Pauli::I {
                score += 1;
            }
            if np == p {
                score += 4;
            } else if np != Pauli::I {
                score += 1;
            }
            (score, q, p)
        })
        .collect();

    // Highest sharedness first (outermost); tie-break by qubit index so
    // adjacent ladders agree on their shared prefix order!
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let ordered: Vec<(usize, Pauli)> = scored.into_iter().map(|(_, q, p)| (q, p)).collect();
    emit_ordered_ladder(&ordered, term.theta, term.binding, out);
}

/// Emit basis changes + CNOT chain `q[0]->q[1]->...->q[w-1]` + `Rz(q[w-1])`
/// + reverse CNOT chain + inverse basis changes.
fn emit_ordered_ladder(
    order: &[(usize, Pauli)],
    theta: f64,
    binding: Option<ParamBinding>,
    out: &mut Vec<Node>,
) {
    if order.is_empty() {
        return;
    }
    if order.len() == 1 {
        let (q, p) = order[0];
        let gate = match p {
            Pauli::X => Gate::Rx(theta),
            Pauli::Y => Gate::Ry(theta),
            Pauli::Z => Gate::Rz(theta),
            Pauli::I => return,
        };
        out.push(Node {
            op: Operation::Gate { gate, target: q },
            binding,
        });
        return;
    }

    let push_g = |out: &mut Vec<Node>, gate: Gate, target: usize| {
        out.push(Node {
            op: Operation::Gate { gate, target },
            binding: None,
        });
    };

    // Basis change into Z.
    for &(q, p) in order {
        match p {
            Pauli::X => push_g(out, Gate::H, q),
            Pauli::Y => {
                push_g(out, Gate::Sdg, q);
                push_g(out, Gate::H, q);
            }
            Pauli::Z | Pauli::I => {}
        }
    }
    // Forward parity ladder.
    for w in order.windows(2) {
        out.push(Node {
            op: Operation::Controlled {
                gate: Gate::X,
                control: w[0].0,
                target: w[1].0,
            },
            binding: None,
        });
    }
    // Central Rz on the last qubit.
    let root = order.last().unwrap().0;
    out.push(Node {
        op: Operation::Gate {
            gate: Gate::Rz(theta),
            target: root,
        },
        binding,
    });
    // Reverse parity ladder.
    for w in order.windows(2).rev() {
        out.push(Node {
            op: Operation::Controlled {
                gate: Gate::X,
                control: w[0].0,
                target: w[1].0,
            },
            binding: None,
        });
    }
    // Inverse basis change.
    for &(q, p) in order {
        match p {
            Pauli::X => push_g(out, Gate::H, q),
            Pauli::Y => {
                push_g(out, Gate::H, q);
                push_g(out, Gate::S, q);
            }
            Pauli::Z | Pauli::I => {}
        }
    }
}

/// Order a commuting cluster greedily by maximum Pauli overlap and emit
/// tree-aligned ladders, then perform a quick local cancel pass to estimate
/// its true CNOT count accurately.
fn synth_cluster_aligned_ladders(cluster: &[PauliTerm]) -> Vec<Node> {
    let m = cluster.len();
    let mut used = vec![false; m];
    let mut order = Vec::with_capacity(m);
    // Start with highest-weight term.
    let first = (0..m).max_by_key(|&i| cluster[i].pauli.weight()).unwrap_or(0);
    used[first] = true;
    order.push(first);

    for _ in 1..m {
        let prev = &cluster[*order.last().unwrap()].pauli;
        let next = (0..m)
            .filter(|&i| !used[i])
            .max_by_key(|&i| pauli_overlap(prev, &cluster[i].pauli))
            .unwrap();
        used[next] = true;
        order.push(next);
    }

    let mut out = Vec::new();
    for (pos, &idx) in order.iter().enumerate() {
        let prev_p = pos.checked_sub(1).map(|p| &cluster[order[p]].pauli);
        let next_p = order.get(pos + 1).map(|&n| &cluster[n].pauli);
        emit_aligned_ladder(&cluster[idx], prev_p, next_p, &mut out);
    }
    quick_cancel_cx_1q(out)
}

fn pauli_overlap(a: &PauliString, b: &PauliString) -> usize {
    let mut score = 0;
    for &(qa, pa) in a.ops() {
        if let Some(&(_, pb)) = b.ops().iter().find(|&&(qb, _)| qb == qa) {
            score += if pa == pb { 4 } else { 1 };
        }
    }
    score
}

/// Elementary Clifford gate used during simultaneous diagonalization.
#[derive(Clone, Copy, Debug)]
enum CliffGate {
    H(usize),
    S(usize),
    Sdg(usize),
    X(usize),
    Y(usize),
    Z(usize),
    Cx(usize, usize),
}

impl CliffGate {
    fn to_node(self, qubits: &[usize]) -> Node {
        let op = match self {
            CliffGate::H(q) => Operation::Gate {
                gate: Gate::H,
                target: qubits[q],
            },
            CliffGate::S(q) => Operation::Gate {
                gate: Gate::S,
                target: qubits[q],
            },
            CliffGate::Sdg(q) => Operation::Gate {
                gate: Gate::Sdg,
                target: qubits[q],
            },
            CliffGate::X(q) => Operation::Gate {
                gate: Gate::X,
                target: qubits[q],
            },
            CliffGate::Y(q) => Operation::Gate {
                gate: Gate::Y,
                target: qubits[q],
            },
            CliffGate::Z(q) => Operation::Gate {
                gate: Gate::Z,
                target: qubits[q],
            },
            CliffGate::Cx(c, t) => Operation::Controlled {
                gate: Gate::X,
                control: qubits[c],
                target: qubits[t],
            },
        };
        Node { op, binding: None }
    }

    fn inverse(self) -> Self {
        match self {
            CliffGate::H(q) => CliffGate::H(q),
            CliffGate::S(q) => CliffGate::Sdg(q),
            CliffGate::Sdg(q) => CliffGate::S(q),
            CliffGate::X(q) => CliffGate::X(q),
            CliffGate::Y(q) => CliffGate::Y(q),
            CliffGate::Z(q) => CliffGate::Z(q),
            CliffGate::Cx(c, t) => CliffGate::Cx(c, t),
        }
    }
}

/// Conjugate a Pauli row `P` by `g`: `P <- g · P · g†`, updating `sign`.
fn conj_row(row: &mut [Pauli], sign: &mut f64, g: CliffGate) {
    match g {
        CliffGate::H(q) => match row[q] {
            Pauli::I => {}
            Pauli::X => row[q] = Pauli::Z,
            Pauli::Z => row[q] = Pauli::X,
            Pauli::Y => *sign = -*sign,
        },
        CliffGate::S(q) => match row[q] {
            Pauli::I | Pauli::Z => {}
            Pauli::X => row[q] = Pauli::Y,
            Pauli::Y => {
                row[q] = Pauli::X;
                *sign = -*sign;
            }
        },
        CliffGate::Sdg(q) => match row[q] {
            Pauli::I | Pauli::Z => {}
            Pauli::X => {
                row[q] = Pauli::Y;
                *sign = -*sign;
            }
            Pauli::Y => row[q] = Pauli::X,
        },
        CliffGate::X(q) => match row[q] {
            Pauli::Y | Pauli::Z => *sign = -*sign,
            Pauli::I | Pauli::X => {}
        },
        CliffGate::Y(q) => match row[q] {
            Pauli::X | Pauli::Z => *sign = -*sign,
            Pauli::I | Pauli::Y => {}
        },
        CliffGate::Z(q) => match row[q] {
            Pauli::X | Pauli::Y => *sign = -*sign,
            Pauli::I | Pauli::Z => {}
        },
        CliffGate::Cx(c, t) => {
            let (nc, nt, s) = match (row[c], row[t]) {
                (Pauli::I, Pauli::I) => (Pauli::I, Pauli::I, 1.0),
                (Pauli::I, Pauli::X) => (Pauli::I, Pauli::X, 1.0),
                (Pauli::I, Pauli::Y) => (Pauli::Z, Pauli::Y, 1.0),
                (Pauli::I, Pauli::Z) => (Pauli::Z, Pauli::Z, 1.0),
                (Pauli::X, Pauli::I) => (Pauli::X, Pauli::X, 1.0),
                (Pauli::X, Pauli::X) => (Pauli::X, Pauli::I, 1.0),
                (Pauli::X, Pauli::Y) => (Pauli::Y, Pauli::Z, 1.0),
                (Pauli::X, Pauli::Z) => (Pauli::Y, Pauli::Y, -1.0),
                (Pauli::Y, Pauli::I) => (Pauli::Y, Pauli::X, 1.0),
                (Pauli::Y, Pauli::X) => (Pauli::Y, Pauli::I, 1.0),
                (Pauli::Y, Pauli::Y) => (Pauli::X, Pauli::Z, -1.0),
                (Pauli::Y, Pauli::Z) => (Pauli::X, Pauli::Y, 1.0),
                (Pauli::Z, Pauli::I) => (Pauli::Z, Pauli::I, 1.0),
                (Pauli::Z, Pauli::X) => (Pauli::Z, Pauli::X, 1.0),
                (Pauli::Z, Pauli::Y) => (Pauli::I, Pauli::Y, 1.0),
                (Pauli::Z, Pauli::Z) => (Pauli::I, Pauli::Z, 1.0),
            };
            row[c] = nc;
            row[t] = nt;
            *sign *= s;
        }
    }
}

/// Simultaneous Clifford diagonalization + GraySynth for a mutually-commuting
/// cluster of Pauli rotations on at most 64 active qubits.
fn synth_cluster_clifford_graysynth(cluster: &[PauliTerm], tol: f64) -> Option<Vec<Node>> {
    // Collect sorted active qubits.
    let mut qubits: Vec<usize> = Vec::new();
    for t in cluster {
        for q in t.pauli.qubits() {
            if !qubits.contains(&q) {
                qubits.push(q);
            }
        }
    }
    qubits.sort_unstable();
    let k = qubits.len();
    if k == 0 || k > 64 {
        return None;
    }
    let q_to_local: HashMap<usize, usize> =
        qubits.iter().enumerate().map(|(i, &q)| (q, i)).collect();

    let r = cluster.len();
    let mut tableau: Vec<Vec<Pauli>> = vec![vec![Pauli::I; k]; r];
    let mut signs: Vec<f64> = vec![1.0; r];
    for (j, t) in cluster.iter().enumerate() {
        for &(q, p) in t.pauli.ops() {
            tableau[j][q_to_local[&q]] = p;
        }
    }

    let mut cliff: Vec<CliffGate> = Vec::new();
    let apply_all = |tableau: &mut [Vec<Pauli>],
                     signs: &mut [f64],
                     cliff: &mut Vec<CliffGate>,
                     g: CliffGate| {
        for (row, s) in tableau.iter_mut().zip(signs.iter_mut()) {
            conj_row(row, s, g);
        }
        cliff.push(g);
    };

    // Step 1: 0-CNOT single-qubit diagonalization on QWC columns.
    for q in 0..k {
        let mut has_x = false;
        let mut has_y = false;
        let mut has_z = false;
        for row in &tableau {
            match row[q] {
                Pauli::X => has_x = true,
                Pauli::Y => has_y = true,
                Pauli::Z => has_z = true,
                Pauli::I => {}
            }
        }
        if has_x && !has_y && !has_z {
            apply_all(&mut tableau, &mut signs, &mut cliff, CliffGate::H(q));
        } else if has_y && !has_x && !has_z {
            apply_all(&mut tableau, &mut signs, &mut cliff, CliffGate::Sdg(q));
            apply_all(&mut tableau, &mut signs, &mut cliff, CliffGate::H(q));
        }
    }

    // Step 2: Symplectic Gaussian elimination on the X-block.
    let has_x = |p: Pauli| matches!(p, Pauli::X | Pauli::Y);
    let mut piv_rows: Vec<usize> = Vec::new();
    let mut piv_cols: Vec<usize> = Vec::new();
    let mut is_piv_col = vec![false; k];

    loop {
        // Pick row with fewest X/Y entries that still has an X/Y in a non-pivot column.
        let best = (0..r)
            .filter(|j| !piv_rows.contains(j))
            .filter_map(|j| {
                let c = (0..k).find(|&q| !is_piv_col[q] && has_x(tableau[j][q]))?;
                let weight = (0..k).filter(|&q| has_x(tableau[j][q])).count();
                Some((weight, j, c))
            })
            .min_by_key(|&(w, j, _)| (w, j));

        let Some((_, r_idx, c_idx)) = best else {
            break;
        };

        // Clear X in every other column of `r_idx` via CX(c_idx, q).
        // Earlier pivot rows have X = 0 at `c_idx`, so they stay untouched!
        for q in 0..k {
            if q != c_idx && has_x(tableau[r_idx][q]) {
                apply_all(&mut tableau, &mut signs, &mut cliff, CliffGate::Cx(c_idx, q));
            }
        }
        piv_rows.push(r_idx);
        piv_cols.push(c_idx);
        is_piv_col[c_idx] = true;
    }

    // Step 2b: Clear symmetric Z-block on (piv_rows, piv_cols) with CZ + S, then H.
    let d = piv_rows.len();
    for a in 0..d {
        let ra = piv_rows[a];
        let ca = piv_cols[a];
        for &cb in &piv_cols[a + 1..d] {
            if tableau[ra][cb] == Pauli::Z {
                // CZ(ca, cb) = H(cb) CX(ca, cb) H(cb).
                apply_all(&mut tableau, &mut signs, &mut cliff, CliffGate::H(cb));
                apply_all(&mut tableau, &mut signs, &mut cliff, CliffGate::Cx(ca, cb));
                apply_all(&mut tableau, &mut signs, &mut cliff, CliffGate::H(cb));
            }
        }
    }
    for a in 0..d {
        let ra = piv_rows[a];
        let ca = piv_cols[a];
        if tableau[ra][ca] == Pauli::Y {
            apply_all(&mut tableau, &mut signs, &mut cliff, CliffGate::S(ca));
        }
    }
    for &ca in &piv_cols {
        apply_all(&mut tableau, &mut signs, &mut cliff, CliffGate::H(ca));
    }

    // Sanity check: every entry in every row must now be I or Z!
    debug_assert!(
        tableau
            .iter()
            .all(|row| row.iter().all(|&p| matches!(p, Pauli::I | Pauli::Z))),
        "simultaneous diagonalization left non-Z entry: {tableau:?}"
    );

    // Build pure-Z phase polynomial terms: (bitmask, theta, binding).
    let mut z_terms: Vec<ZTerm> = Vec::with_capacity(r);
    for (j, (row, &s)) in tableau.iter().zip(signs.iter()).enumerate() {
        let mut mask = 0u64;
        for (q, &p) in row.iter().enumerate() {
            if p == Pauli::Z {
                mask |= 1u64 << q;
            }
        }
        if mask == 0 {
            continue;
        }
        let theta = s * cluster[j].theta;
        let binding = cluster[j].binding.map(|b| ParamBinding {
            parameter: b.parameter,
            scale: s * b.scale,
            offset: s * b.offset,
        });
        if binding.is_none() && wrap_2pi(theta).abs() <= tol {
            continue;
        }
        z_terms.push(ZTerm {
            mask,
            theta,
            binding,
            done: false,
        });
    }

    // Synthesize the pure-Z phase polynomial with GraySynth.
    let poly_nodes = graysynth(&mut z_terms, &qubits, tol);

    // Assemble: C -> GraySynth -> C†.
    let mut out = Vec::with_capacity(cliff.len() * 2 + poly_nodes.len());
    for &g in &cliff {
        out.push(g.to_node(&qubits));
    }
    out.extend(poly_nodes);
    for &g in cliff.iter().rev() {
        out.push(g.inverse().to_node(&qubits));
    }

    Some(quick_cancel_cx_1q(out))
}

/// Clifford-tracked Pauli-network synthesis for an arbitrary (commuting or
/// non-commuting) sequence of `PauliRotation`s on at most 10 active qubits.
///
/// Reduces each front-layer Pauli rotation to weight 1 using lookahead-guided
/// CNOTs without uncomputing between terms, accumulating the running Clifford
/// `C` in a `2k × 2k` symplectic tableau and restoring `C†` once at the end.
/// Returns `Some((nodes, global_phase_adjustment))`.
fn synth_pauli_network(items: &[PauliTerm], tol: f64) -> Option<(Vec<Node>, f64)> {
    let mut qubits: Vec<usize> = Vec::new();
    for t in items {
        for q in t.pauli.qubits() {
            if !qubits.contains(&q) {
                qubits.push(q);
            }
        }
    }
    qubits.sort_unstable();
    let k = qubits.len();
    if !(2..=10).contains(&k) {
        return None;
    }
    let q_to_local: HashMap<usize, usize> =
        qubits.iter().enumerate().map(|(i, &q)| (q, i)).collect();

    struct LiveTerm {
        row: Vec<Pauli>,
        sign: f64,
        theta: f64,
        binding: Option<ParamBinding>,
        done: bool,
    }

    let mut terms: Vec<LiveTerm> = items
        .iter()
        .map(|t| {
            let mut row = vec![Pauli::I; k];
            for &(q, p) in t.pauli.ops() {
                row[q_to_local[&q]] = p;
            }
            LiveTerm {
                row,
                sign: 1.0,
                theta: t.theta,
                binding: t.binding,
                done: false,
            }
        })
        .collect();

    // Live Clifford tableau representing C = G_{L-1} ... G_0:
    // x_img[q] = C X_q C†, z_img[q] = C Z_q C†.
    let mut x_img: Vec<Vec<Pauli>> = (0..k)
        .map(|q| {
            let mut r = vec![Pauli::I; k];
            r[q] = Pauli::X;
            r
        })
        .collect();
    let mut z_img: Vec<Vec<Pauli>> = (0..k)
        .map(|q| {
            let mut r = vec![Pauli::I; k];
            r[q] = Pauli::Z;
            r
        })
        .collect();
    let mut x_sign = vec![1.0f64; k];
    let mut z_sign = vec![1.0f64; k];

    let mut forward_cliff: Vec<CliffGate> = Vec::new();
    let mut out: Vec<Node> = Vec::new();

    let rows_commute = |a: &[Pauli], b: &[Pauli]| -> bool {
        let mut diff = 0usize;
        for (&pa, &pb) in a.iter().zip(b.iter()) {
            if pa != Pauli::I && pb != Pauli::I && pa != pb {
                diff += 1;
            }
        }
        diff.is_multiple_of(2)
    };

    let row_weight = |r: &[Pauli]| -> usize { r.iter().filter(|&&p| p != Pauli::I).count() };

    let apply_cliff = |g: CliffGate,
                       terms: &mut [LiveTerm],
                       x_img: &mut [Vec<Pauli>],
                       x_sign: &mut [f64],
                       z_img: &mut [Vec<Pauli>],
                       z_sign: &mut [f64],
                       forward_cliff: &mut Vec<CliffGate>,
                       out: &mut Vec<Node>| {
        forward_cliff.push(g);
        out.push(g.to_node(&qubits));
        for q in 0..k {
            conj_row(&mut x_img[q], &mut x_sign[q], g);
            conj_row(&mut z_img[q], &mut z_sign[q], g);
        }
        for t in terms.iter_mut() {
            if !t.done {
                conj_row(&mut t.row, &mut t.sign, g);
            }
        }
    };

    // Compute front commuting layer of undone terms: indices `j` that commute
    // with every earlier undone index `i < j`.
    let front_layer = |terms: &[LiveTerm]| -> Vec<usize> {
        let mut front = Vec::new();
        let mut undone = Vec::new();
        for (j, t) in terms.iter().enumerate() {
            if t.done {
                continue;
            }
            if undone.iter().all(|&i: &usize| rows_commute(&terms[i].row, &t.row)) {
                front.push(j);
            }
            undone.push(j);
        }
        front
    };

    // Emit any front-layer term that has reached weight 0 or 1, and merge
    // duplicate Pauli strings in the front layer!
    let emit_ready = |terms: &mut [LiveTerm], out: &mut Vec<Node>, tol: f64| {
        loop {
            let mut progress = false;
            let front = front_layer(terms);

            // Merge identical Pauli strings in the front layer.
            for a_pos in 0..front.len() {
                let ia = front[a_pos];
                if terms[ia].done {
                    continue;
                }
                for &ib in &front[a_pos + 1..] {
                    if terms[ib].done || terms[ia].row != terms[ib].row {
                        continue;
                    }
                    let same_param = match (terms[ia].binding, terms[ib].binding) {
                        (None, None) | (Some(_), None) | (None, Some(_)) => true,
                        (Some(ba), Some(bb)) => ba.parameter == bb.parameter,
                    };
                    if !same_param {
                        continue;
                    }
                    let eff_a = terms[ia].sign * terms[ia].theta;
                    let eff_b = terms[ib].sign * terms[ib].theta;
                    match (terms[ia].binding, terms[ib].binding) {
                        (None, None) => {
                            terms[ia].sign = 1.0;
                            terms[ia].theta = wrap_2pi(eff_a + eff_b);
                        }
                        (Some(ba), None) => {
                            let scale = terms[ia].sign * ba.scale;
                            let offset = terms[ia].sign * ba.offset + eff_b;
                            terms[ia].sign = 1.0;
                            terms[ia].theta = offset;
                            terms[ia].binding = Some(ParamBinding {
                                parameter: ba.parameter,
                                scale,
                                offset,
                            });
                        }
                        (None, Some(bb)) => {
                            let scale = terms[ib].sign * bb.scale;
                            let offset = terms[ib].sign * bb.offset + eff_a;
                            terms[ia].sign = 1.0;
                            terms[ia].theta = offset;
                            terms[ia].binding = Some(ParamBinding {
                                parameter: bb.parameter,
                                scale,
                                offset,
                            });
                        }
                        (Some(ba), Some(bb)) => {
                            let scale = terms[ia].sign * ba.scale + terms[ib].sign * bb.scale;
                            let offset = terms[ia].sign * ba.offset + terms[ib].sign * bb.offset;
                            terms[ia].sign = 1.0;
                            if scale.abs() <= tol {
                                terms[ia].binding = None;
                                terms[ia].theta = wrap_2pi(offset);
                            } else {
                                terms[ia].theta = offset;
                                terms[ia].binding = Some(ParamBinding {
                                    parameter: ba.parameter,
                                    scale,
                                    offset,
                                });
                            }
                        }
                    }
                    terms[ib].done = true;
                    progress = true;
                }
            }

            // Emit any front-layer term of weight <= 1.
            for &j in &front {
                if terms[j].done {
                    continue;
                }
                let w = row_weight(&terms[j].row);
                if w == 0 {
                    terms[j].done = true;
                    progress = true;
                } else if w == 1 {
                    let (q, p) = terms[j]
                        .row
                        .iter()
                        .enumerate()
                        .find(|&(_, &p)| p != Pauli::I)
                        .map(|(q, &p)| (q, p))
                        .unwrap();
                    let s = terms[j].sign;
                    let theta = s * terms[j].theta;
                    let binding = terms[j].binding.map(|b| ParamBinding {
                        parameter: b.parameter,
                        scale: s * b.scale,
                        offset: s * b.offset,
                    });
                    terms[j].done = true;
                    if binding.is_some() || wrap_2pi(theta).abs() > tol {
                        let gate = match p {
                            Pauli::X => Gate::Rx(theta),
                            Pauli::Y => Gate::Ry(theta),
                            Pauli::Z => Gate::Rz(theta),
                            Pauli::I => unreachable!(),
                        };
                        out.push(Node {
                            op: Operation::Gate {
                                gate,
                                target: qubits[q],
                            },
                            binding,
                        });
                    }
                    progress = true;
                }
            }

            if !progress {
                break;
            }
        }
    };

    // Main loop: repeatedly reduce the best front-layer term to weight 1.
    loop {
        emit_ready(&mut terms, &mut out, tol);
        let front = front_layer(&terms);
        if front.is_empty() {
            break;
        }

        // Collect upcoming undone terms (lookahead window up to 24 terms) for scoring.
        let lookahead: Vec<usize> = terms
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.done)
            .map(|(idx, _)| idx)
            .take(24)
            .collect();

        // Pick best single-CNOT reduction across all minimum-weight front-layer terms!
        let min_w = front
            .iter()
            .map(|&j| row_weight(&terms[j].row))
            .min()
            .unwrap();
        let candidates_j: Vec<usize> = front
            .iter()
            .copied()
            .filter(|&j| row_weight(&terms[j].row) <= min_w + 1)
            .take(8)
            .collect();

        let mut best_seq: Option<Vec<CliffGate>> = None;
        let mut best_score = i64::MAX;

        for &j in &candidates_j {
            let supp: Vec<usize> = (0..k).filter(|&q| terms[j].row[q] != Pauli::I).collect();
            for &c in &supp {
                for &t in &supp {
                    if c == t {
                        continue;
                    }
                    // Try both Z-reduction (removes `c` via CX(c,t) when c=Z, t∈{Z,Y})
                    // and X-reduction (removes `t` via CX(c,t) when c∈{X,Y}, t=X)!
                    let mut z_seq = Vec::with_capacity(4);
                    match terms[j].row[c] {
                        Pauli::Z => {}
                        Pauli::X => z_seq.push(CliffGate::H(c)),
                        Pauli::Y => {
                            z_seq.push(CliffGate::Sdg(c));
                            z_seq.push(CliffGate::H(c));
                        }
                        Pauli::I => unreachable!(),
                    }
                    match terms[j].row[t] {
                        Pauli::Z | Pauli::Y => {}
                        Pauli::X => z_seq.push(CliffGate::H(t)),
                        Pauli::I => unreachable!(),
                    }
                    z_seq.push(CliffGate::Cx(c, t));

                    let mut x_seq = Vec::with_capacity(4);
                    match terms[j].row[c] {
                        Pauli::X | Pauli::Y => {}
                        Pauli::Z => x_seq.push(CliffGate::H(c)),
                        Pauli::I => unreachable!(),
                    }
                    match terms[j].row[t] {
                        Pauli::X => {}
                        Pauli::Z => x_seq.push(CliffGate::H(t)),
                        Pauli::Y => x_seq.push(CliffGate::Sdg(t)),
                        Pauli::I => unreachable!(),
                    }
                    x_seq.push(CliffGate::Cx(c, t));

                    for seq in [z_seq, x_seq] {
                        // Score `seq` on the lookahead window: smaller score = better!
                        let mut score = (seq.len() as i64) * 2;
                        for (rank, &idx) in lookahead.iter().enumerate() {
                            let w_before = row_weight(&terms[idx].row) as i64;
                            let mut sim = terms[idx].row.clone();
                            let mut dummy_sign = 1.0;
                            for &g in &seq {
                                conj_row(&mut sim, &mut dummy_sign, g);
                            }
                            let w_after = row_weight(&sim) as i64;
                            let is_front = front.contains(&idx);
                            let weight = if is_front {
                                16
                            } else {
                                (12i64 - (rank as i64 / 2)).max(2)
                            };
                            score += weight * (w_after - w_before);
                            if is_front && w_after == 1 {
                                score -= 40;
                            }
                        }

                        if score < best_score {
                            best_score = score;
                            best_seq = Some(seq);
                        }
                    }
                }
            }
        }

        let seq = best_seq.expect("front layer has at least one weight >= 2 term");
        for g in seq {
            apply_cliff(
                g,
                &mut terms,
                &mut x_img,
                &mut x_sign,
                &mut z_img,
                &mut z_sign,
                &mut forward_cliff,
                &mut out,
            );
        }
    }

    // Synthesize C† from the live tableau (x_img, z_img) with minimal CNOTs!
    let d_gates = synthesize_clifford_inverse(x_img, x_sign, z_img, z_sign);

    // Compute exact global phase of (D · C) = e^{iφ} I on k <= 10 qubits!
    let mut check_circ = Circuit::new(k);
    let local_qubits: Vec<usize> = (0..k).collect();
    for &g in &forward_cliff {
        check_circ.push(g.to_node(&local_qubits).op).unwrap();
    }
    for &g in &d_gates {
        check_circ.push(g.to_node(&local_qubits).op).unwrap();
    }
    let state = Statevector::from_circuit(&check_circ);
    let amp0 = state.amplitudes()[0];
    debug_assert!(
        (amp0.norm() - 1.0).abs() < 1e-10,
        "Clifford inverse verification failed: |⟨0|DC|0⟩| = {}",
        amp0.norm()
    );
    let phase_fix = wrap_pi(-amp0.arg());

    for g in d_gates {
        out.push(g.to_node(&qubits));
    }

    Some((quick_cancel_cx_1q(out), phase_fix))
}

/// Synthesize a minimal-CNOT Clifford sequence `D` that maps the tableau
/// `(x_img, x_sign, z_img, z_sign)` back to `(+X_q, +Z_q)` for every qubit
/// `q = 0..k-1`.
fn synthesize_clifford_inverse(
    mut x_img: Vec<Vec<Pauli>>,
    mut x_sign: Vec<f64>,
    mut z_img: Vec<Vec<Pauli>>,
    mut z_sign: Vec<f64>,
) -> Vec<CliffGate> {
    let k = x_img.len();
    let mut gates = Vec::new();
    let mut done = vec![false; k];

    let apply = |g: CliffGate,
                 x_img: &mut [Vec<Pauli>],
                 x_sign: &mut [f64],
                 z_img: &mut [Vec<Pauli>],
                 z_sign: &mut [f64],
                 gates: &mut Vec<CliffGate>| {
        gates.push(g);
        for i in 0..k {
            conj_row(&mut x_img[i], &mut x_sign[i], g);
            conj_row(&mut z_img[i], &mut z_sign[i], g);
        }
    };

    // Process qubits in greedy minimum-weight order so fewer CNOTs are emitted.
    for _ in 0..k {
        let q = (0..k)
            .filter(|&i| !done[i])
            .min_by_key(|&i| {
                let wz = z_img[i].iter().filter(|&&p| p != Pauli::I).count();
                let wx = x_img[i].iter().filter(|&&p| p != Pauli::I).count();
                (wz + wx, wz, i)
            })
            .unwrap();

        // Step 1: Turn every non-I entry of z_img[q] into Z using 1q Cliffords.
        for j in 0..k {
            if done[j] {
                continue;
            }
            match z_img[q][j] {
                Pauli::I | Pauli::Z => {}
                Pauli::X => apply(
                    CliffGate::H(j),
                    &mut x_img,
                    &mut x_sign,
                    &mut z_img,
                    &mut z_sign,
                    &mut gates,
                ),
                Pauli::Y => {
                    apply(
                        CliffGate::Sdg(j),
                        &mut x_img,
                        &mut x_sign,
                        &mut z_img,
                        &mut z_sign,
                        &mut gates,
                    );
                    apply(
                        CliffGate::H(j),
                        &mut x_img,
                        &mut x_sign,
                        &mut z_img,
                        &mut z_sign,
                        &mut gates,
                    );
                }
            }
        }

        // Ensure z_img[q][q] == Z.
        if z_img[q][q] != Pauli::Z {
            let p = (0..k)
                .find(|&j| !done[j] && z_img[q][j] == Pauli::Z)
                .expect("stabilizer row is non-identity");
            // CX(q, p) maps (I_q, Z_p) -> (Z_q, Z_p)!
            apply(
                CliffGate::Cx(q, p),
                &mut x_img,
                &mut x_sign,
                &mut z_img,
                &mut z_sign,
                &mut gates,
            );
        }
        // Clear all other Z entries in z_img[q]: CX(j, q) maps (Z_j, Z_q) -> (I_j, Z_q)!
        for j in 0..k {
            if !done[j] && j != q && z_img[q][j] == Pauli::Z {
                apply(
                    CliffGate::Cx(j, q),
                    &mut x_img,
                    &mut x_sign,
                    &mut z_img,
                    &mut z_sign,
                    &mut gates,
                );
            }
        }

        // Step 2: Reduce x_img[q] to ±X_q without touching z_img[q] == ±Z_q.
        if x_img[q][q] == Pauli::Y {
            apply(
                CliffGate::S(q),
                &mut x_img,
                &mut x_sign,
                &mut z_img,
                &mut z_sign,
                &mut gates,
            );
        }
        debug_assert_eq!(x_img[q][q], Pauli::X);

        for j in 0..k {
            if done[j] || j == q {
                continue;
            }
            match x_img[q][j] {
                Pauli::I => continue,
                Pauli::X => {}
                Pauli::Z => apply(
                    CliffGate::H(j),
                    &mut x_img,
                    &mut x_sign,
                    &mut z_img,
                    &mut z_sign,
                    &mut gates,
                ),
                Pauli::Y => apply(
                    CliffGate::Sdg(j),
                    &mut x_img,
                    &mut x_sign,
                    &mut z_img,
                    &mut z_sign,
                    &mut gates,
                ),
            }
            // CX(q, j) maps (X_q, X_j) -> (X_q, I_j) and leaves (Z_q, I_j) untouched!
            apply(
                CliffGate::Cx(q, j),
                &mut x_img,
                &mut x_sign,
                &mut z_img,
                &mut z_sign,
                &mut gates,
            );
        }

        done[q] = true;
    }

    // Step 3: Fix signs with single-qubit Paulis (0 CNOTs).
    for q in 0..k {
        match (x_sign[q] < 0.0, z_sign[q] < 0.0) {
            (true, true) => apply(
                CliffGate::Y(q),
                &mut x_img,
                &mut x_sign,
                &mut z_img,
                &mut z_sign,
                &mut gates,
            ),
            (true, false) => apply(
                CliffGate::Z(q),
                &mut x_img,
                &mut x_sign,
                &mut z_img,
                &mut z_sign,
                &mut gates,
            ),
            (false, true) => apply(
                CliffGate::X(q),
                &mut x_img,
                &mut x_sign,
                &mut z_img,
                &mut z_sign,
                &mut gates,
            ),
            (false, false) => {}
        }
    }

    gates
}

struct ZTerm {
    mask: u64,
    theta: f64,
    binding: Option<ParamBinding>,
    done: bool,
}

/// Synthesize `∏ exp(-i θ_j/2 Z^{mask_j})` on `qubits` via GraySynth
/// (Amy, Azimzadeh, Mosca 2018), returning `{CX, Rz}` nodes.
fn graysynth(terms: &mut [ZTerm], qubits: &[usize], tol: f64) -> Vec<Node> {
    let k = qubits.len();
    let mut out: Vec<Node> = Vec::new();
    let mut parity: Vec<u64> = (0..k).map(|i| 1u64 << i).collect();

    let emit_ready = |terms: &mut [ZTerm], out: &mut Vec<Node>| {
        for t in terms.iter_mut() {
            if !t.done && t.mask.count_ones() == 1 {
                let q = t.mask.trailing_zeros() as usize;
                t.done = true;
                if t.binding.is_some() || wrap_2pi(t.theta).abs() > tol {
                    out.push(Node {
                        op: Operation::Gate {
                            gate: Gate::Rz(t.theta),
                            target: qubits[q],
                        },
                        binding: t.binding,
                    });
                }
            }
        }
    };

    let apply_cx = |c: usize, t: usize, terms: &mut [ZTerm], parity: &mut [u64], out: &mut Vec<Node>| {
        out.push(Node {
            op: Operation::Controlled {
                gate: Gate::X,
                control: qubits[c],
                target: qubits[t],
            },
            binding: None,
        });
        parity[t] ^= parity[c];
        let c_bit = 1u64 << c;
        let t_bit = 1u64 << t;
        for term in terms.iter_mut() {
            if !term.done && (term.mask & t_bit) != 0 {
                term.mask ^= c_bit;
            }
        }
    };

    emit_ready(terms, &mut out);

    let all_cols = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let initial: Vec<usize> = (0..terms.len()).filter(|&i| !terms[i].done).collect();
    let mut stack: Vec<(Vec<usize>, u64, Option<usize>)> = vec![(initial, all_cols, None)];

    while let Some((mut indices, cols, target)) = stack.pop() {
        indices.retain(|&idx| !terms[idx].done);
        if indices.is_empty() {
            continue;
        }

        // If a target qubit `t` is fixed, any other qubit `c` that is 1 in EVERY
        // remaining term of this sub-problem can be cleared in ONE CNOT CX(c, t)!
        if let Some(t) = target {
            let common = indices
                .iter()
                .fold(u64::MAX, |acc, &idx| acc & terms[idx].mask);
            let mut to_clear = common & !(1u64 << t);
            while to_clear != 0 {
                let c = to_clear.trailing_zeros() as usize;
                to_clear &= to_clear - 1;
                apply_cx(c, t, terms, &mut parity, &mut out);
                emit_ready(terms, &mut out);
            }
            indices.retain(|&idx| !terms[idx].done);
            if indices.is_empty() {
                continue;
            }
        }

        // Pick a branching column `i` in `cols` that has at least one 1 among `indices`
        // and maximizes the larger partition size (Amy et al. heuristic).
        let mut best_col: Option<usize> = None;
        let mut best_score = 0usize;
        let mut rem = cols;
        while rem != 0 {
            let c = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            if target == Some(c) {
                continue;
            }
            let ones = indices
                .iter()
                .filter(|&&idx| (terms[idx].mask >> c) & 1 == 1)
                .count();
            if ones == 0 {
                continue;
            }
            let zeros = indices.len() - ones;
            let score = ones.max(zeros);
            if score > best_score {
                best_score = score;
                best_col = Some(c);
            }
        }

        if let Some(col) = best_col {
            let next_cols = cols & !(1u64 << col);
            let mut s0 = Vec::new();
            let mut s1 = Vec::new();
            for &idx in &indices {
                if (terms[idx].mask >> col) & 1 == 1 {
                    s1.push(idx);
                } else {
                    s0.push(idx);
                }
            }
            if !s1.is_empty() {
                stack.push((s1, next_cols, target.or(Some(col))));
            }
            if !s0.is_empty() {
                stack.push((s0, next_cols, target));
            }
        } else {
            // Direct fallback for any remaining multi-weight term.
            let idx = indices[0];
            let m = terms[idx].mask;
            let t = target
                .filter(|&t| (m >> t) & 1 == 1)
                .unwrap_or_else(|| m.trailing_zeros() as usize);
            let mut rest = m & !(1u64 << t);
            while rest != 0 {
                let c = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                apply_cx(c, t, terms, &mut parity, &mut out);
                emit_ready(terms, &mut out);
            }
            indices.retain(|&i| !terms[i].done);
            if !indices.is_empty() {
                stack.push((indices, cols, target));
            }
        }
    }

    // Restore the GF(2) parity matrix `parity` back to I!
    // Row operations reducing `parity` to I applied in FORWARD order undo the CNOTs.
    let restore = reduce_parity_to_identity(&parity);
    for (c, t) in restore {
        out.push(Node {
            op: Operation::Controlled {
                gate: Gate::X,
                control: qubits[c],
                target: qubits[t],
            },
            binding: None,
        });
    }

    out
}

/// Reduce a `k × k` invertible GF(2) matrix (`k <= 64`) to `I`, returning the
/// sequence of `(control, target)` CNOTs (`row[t] ^= row[c]`) in forward order.
fn reduce_parity_to_identity(parity: &[u64]) -> Vec<(usize, usize)> {
    let k = parity.len();
    if parity.iter().enumerate().all(|(i, &r)| r == (1u64 << i)) {
        return Vec::new();
    }
    if k <= 16 {
        let small: Vec<u16> = parity.iter().map(|&r| r as u16).collect();
        if let Some(ops) = reduce_gf2_greedy(&small, k * k + 1) {
            return ops;
        }
    }
    // General Gaussian elimination for k <= 64.
    let mut m = parity.to_vec();
    let mut ops = Vec::new();
    for col in 0..k {
        let bit = 1u64 << col;
        if m[col] & bit == 0 {
            let src = (col + 1..k)
                .find(|&r| m[r] & bit != 0)
                .or_else(|| (0..col).find(|&r| m[r] & bit != 0))
                .expect("parity matrix is invertible");
            m[col] ^= m[src];
            ops.push((src, col));
        }
        for r in 0..k {
            if r != col && m[r] & bit != 0 {
                m[r] ^= m[col];
                ops.push((col, r));
            }
        }
    }
    ops
}

/// Fast local cancellation of adjacent self-inverse gates (`CX`, `H`, `S/Sdg`)
/// so candidate cost comparisons see accurate CNOT counts.
fn quick_cancel_cx_1q(nodes: Vec<Node>) -> Vec<Node> {
    let max_q = nodes
        .iter()
        .map(|n| n.op.max_qubit())
        .max()
        .map_or(0, |m| m + 1);
    let mut out: Vec<Option<Node>> = Vec::with_capacity(nodes.len());
    let mut last_on: Vec<Option<usize>> = vec![None; max_q];

    for node in nodes {
        let qs = node.op.qubits();
        let same_pred = qs
            .first()
            .and_then(|&q0| last_on[q0])
            .filter(|&p| qs.iter().all(|&q| last_on[q] == Some(p)));
        if let Some(p) = same_pred
            && let Some(prev) = &out[p]
            && prev.binding.is_none()
            && node.binding.is_none()
            && is_exact_inverse_pair(&prev.op, &node.op)
        {
            out[p] = None;
            for &q in &qs {
                last_on[q] = None;
                for (idx, maybe) in out.iter().enumerate().rev() {
                    if let Some(n) = maybe
                        && n.op.qubits().contains(&q)
                    {
                        last_on[q] = Some(idx);
                        break;
                    }
                }
            }
            continue;
        }
        let idx = out.len();
        for &q in &qs {
            last_on[q] = Some(idx);
        }
        out.push(Some(node));
    }
    out.into_iter().flatten().collect()
}

fn is_exact_inverse_pair(a: &Operation, b: &Operation) -> bool {
    match (a, b) {
        (
            Operation::Controlled {
                gate: Gate::X,
                control: c1,
                target: t1,
            },
            Operation::Controlled {
                gate: Gate::X,
                control: c2,
                target: t2,
            },
        ) => c1 == c2 && t1 == t2,
        (
            Operation::Gate {
                gate: g1,
                target: t1,
            },
            Operation::Gate {
                gate: g2,
                target: t2,
            },
        ) if t1 == t2 => matches!(
            (g1, g2),
            (Gate::H, Gate::H)
                | (Gate::X, Gate::X)
                | (Gate::Y, Gate::Y)
                | (Gate::Z, Gate::Z)
                | (Gate::S, Gate::Sdg)
                | (Gate::Sdg, Gate::S)
                | (Gate::T, Gate::Tdg)
                | (Gate::Tdg, Gate::T)
        ),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::statevector::Statevector;

    fn assert_circuits_equal(a: &Circuit, b: &Circuit) {
        let n = a.n_qubits();
        for idx in 0..(1usize << n) {
            let mut sa = Statevector::basis(n, idx);
            let mut sb = Statevector::basis(n, idx);
            sa.run(a).unwrap();
            sb.run(b).unwrap();
            for (x, y) in sa.amplitudes().iter().zip(sb.amplitudes()) {
                assert!((x - y).norm() < 1e-10, "mismatch on |{idx:0n$b}⟩: {x} vs {y}");
            }
        }
    }

    #[test]
    fn simultaneous_diagonalization_and_graysynth_exact_on_commuting_4q_block() {
        // 8 mutually commuting weight-4 Pauli strings (fermionic / SU(3) double excitation!).
        let terms = [
            ("XXXX", 0.31),
            ("XXYY", -0.27),
            ("XYXY", 0.43),
            ("XYYX", -0.19),
            ("YXXY", 0.51),
            ("YXYX", -0.37),
            ("YYXX", 0.23),
            ("YYYY", -0.41),
        ];
        let mut orig = Circuit::new(4);
        let mut cluster = Vec::new();
        for (label, theta) in terms {
            let p = PauliString::from_label(label).unwrap();
            orig.pauli_rotation(theta, p.clone());
            cluster.push(PauliTerm {
                pauli: p,
                theta,
                binding: None,
            });
        }
        let nodes = synth_cluster_clifford_graysynth(&cluster, 1e-12).unwrap();
        let cxs = cx_count(&nodes);
        // Raw ladders = 8 * 6 = 48 CXs! GraySynth + Clifford diag should use <= 18 CXs!
        assert!(cxs <= 18, "expected <= 18 CXs, got {cxs}");

        let seg = Segment {
            n_qubits: 4,
            n_parameters: 0,
            n_clbits: 0,
            nodes,
            global_phase: 0.0,
        };
        let synth = seg.into_circuit();
        assert_circuits_equal(&orig, &synth);
    }

    #[test]
    fn graysynth_exact_on_all_15_4q_z_strings() {
        let mut orig = Circuit::new(4);
        let mut cluster = Vec::new();
        for mask in 1u8..16 {
            let ops = (0..4).filter_map(|q| ((mask >> q) & 1 == 1).then_some((q, Pauli::Z)));
            let p = PauliString::new(ops).unwrap();
            let theta = 0.13 * (mask as f64);
            orig.pauli_rotation(theta, p.clone());
            cluster.push(PauliTerm {
                pauli: p,
                theta,
                binding: None,
            });
        }
        let nodes = synth_cluster_clifford_graysynth(&cluster, 1e-12).unwrap();
        let cxs = cx_count(&nodes);
        // Raw ladders = 64 CXs; GraySynth should use <= 20 CXs!
        assert!(cxs <= 20, "expected <= 20 CXs, got {cxs}");
        let seg = Segment {
            n_qubits: 4,
            n_parameters: 0,
            n_clbits: 0,
            nodes,
            global_phase: 0.0,
        };
        assert_circuits_equal(&orig, &seg.into_circuit());
    }
}
