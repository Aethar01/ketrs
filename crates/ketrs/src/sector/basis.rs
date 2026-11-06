//! Indexing of symmetry sectors: the basis states with fixed Hamming weights
//! on chosen sets of qubits.
//!
//! Uses Lin's two-table method. A basis index `j` splits into high and low
//! halves `h` and `l`, and sector states are ordered by `(h, l)`. A state's
//! position is then `offset[h] + rank[l]`, where `rank[l]` counts the smaller
//! low halves with the same weights as `l`, and `offset[h]` counts every
//! sector state with a smaller high half. Both tables have `2^(n/2)`
//! entries, so they are negligible next to the amplitudes, and a lookup is
//! two reads.

use crate::error::Error;
use std::collections::HashMap;

const INVALID: usize = usize::MAX;

/// The basis states of `n_qubits` qubits whose Hamming weight on each
/// constraint's qubits equals that constraint's weight. Qubits in no
/// constraint are unrestricted.
///
/// For example, `[(0..8, 4)]` is half filling of 8 sites, and
/// `[(even sites, 2), (odd sites, 2)]` fixes the number of each spin.
#[derive(Debug)]
pub struct SectorBasis {
    n_qubits: usize,
    /// `(qubit mask, weight)` per constraint.
    constraints: Vec<(usize, usize)>,
    low_bits: usize,
    /// Position of each high half's first state, or [`INVALID`] if no low
    /// half completes it.
    offset: Vec<usize>,
    /// Weight class each high half needs from its low half.
    high_class: Vec<u32>,
    /// Weight class of each low half.
    low_class: Vec<u32>,
    /// Rank of each low half within its class.
    rank: Vec<u32>,
    /// The low halves of each class, in increasing order.
    classes: Vec<Vec<u32>>,
    /// `(high half, first position)` of every high half with states, in
    /// order; used to iterate and to find a position's state.
    segments: Vec<(usize, usize)>,
    dim: usize,
}

impl SectorBasis {
    /// The sector of `n_qubits` qubits where, for each `(qubits, weight)`
    /// constraint, exactly `weight` of `qubits` are `|1⟩`. Constraints must
    /// not share qubits.
    pub fn new(n_qubits: usize, constraints: &[(Vec<usize>, usize)]) -> Result<Self, Error> {
        if n_qubits >= usize::BITS as usize - 1 {
            return Err(Error::InvalidArgument(format!(
                "{n_qubits} qubits is too many to index"
            )));
        }
        let mut used = 0usize;
        let mut masks = Vec::with_capacity(constraints.len());
        for (qubits, weight) in constraints {
            let mut mask = 0usize;
            for &q in qubits {
                if q >= n_qubits {
                    return Err(Error::QubitOutOfRange { qubit: q, n_qubits });
                }
                if (used | mask) & (1 << q) != 0 {
                    return Err(Error::DuplicateQubit(q));
                }
                mask |= 1 << q;
            }
            if *weight > qubits.len() {
                return Err(Error::InvalidArgument(format!(
                    "weight {weight} exceeds the {} qubits it constrains",
                    qubits.len()
                )));
            }
            used |= mask;
            masks.push((mask, *weight));
        }

        let low_bits = n_qubits / 2;
        let weights = |bits: usize, masks: &[(usize, usize)], shift: usize| -> Vec<u8> {
            masks
                .iter()
                .map(|&(m, _)| ((bits << shift) & m).count_ones() as u8)
                .collect()
        };

        // Classify low halves by their weight vector.
        let mut class_id: HashMap<Vec<u8>, u32> = HashMap::new();
        let mut classes: Vec<Vec<u32>> = Vec::new();
        let mut low_class = vec![0u32; 1 << low_bits];
        let mut rank = vec![0u32; 1 << low_bits];
        for l in 0..1usize << low_bits {
            let id = *class_id.entry(weights(l, &masks, 0)).or_insert_with(|| {
                classes.push(Vec::new());
                classes.len() as u32 - 1
            });
            rank[l] = classes[id as usize].len() as u32;
            classes[id as usize].push(l as u32);
            low_class[l] = id;
        }

        // Each high half needs the remaining weight from its low half.
        let n_high = 1usize << (n_qubits - low_bits);
        let mut offset = vec![INVALID; n_high];
        let mut high_class = vec![0u32; n_high];
        let mut segments = Vec::new();
        let mut dim = 0usize;
        for h in 0..n_high {
            let have = weights(h, &masks, low_bits);
            let need: Option<Vec<u8>> = masks
                .iter()
                .zip(&have)
                .map(|(&(_, w), &got)| (w as u8).checked_sub(got))
                .collect();
            if let Some(id) = need.and_then(|need| class_id.get(&need)) {
                offset[h] = dim;
                high_class[h] = *id;
                segments.push((h, dim));
                dim += classes[*id as usize].len();
            }
        }
        Ok(SectorBasis {
            n_qubits,
            constraints: masks,
            low_bits,
            offset,
            high_class,
            low_class,
            rank,
            classes,
            segments,
            dim,
        })
    }

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of basis states in the sector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// `(qubit mask, weight)` for each constraint.
    pub fn constraints(&self) -> &[(usize, usize)] {
        &self.constraints
    }

    /// Whether basis state `j` is in the sector.
    pub fn contains(&self, j: usize) -> bool {
        self.index(j).is_some()
    }

    /// The position of basis state `j` in the sector, if it is in it.
    pub fn index(&self, j: usize) -> Option<usize> {
        let (h, l) = (j >> self.low_bits, j & ((1 << self.low_bits) - 1));
        let start = *self.offset.get(h)?;
        (start != INVALID && self.low_class[l] == self.high_class[h])
            .then(|| start + self.rank[l] as usize)
    }

    /// [`index`](Self::index) for a state known to be in the sector.
    #[inline]
    pub(crate) fn index_unchecked(&self, j: usize) -> usize {
        self.offset[j >> self.low_bits] + self.rank[j & ((1 << self.low_bits) - 1)] as usize
    }

    /// The basis state at `position` in the sector.
    pub fn state(&self, position: usize) -> usize {
        assert!(
            position < self.dim,
            "position {position} outside the sector"
        );
        let s = self
            .segments
            .partition_point(|&(_, start)| start <= position)
            - 1;
        let (h, start) = self.segments[s];
        let low = self.classes[self.high_class[h] as usize][position - start] as usize;
        h << self.low_bits | low
    }

    /// Whether every change from local configuration `c` to `r` of `qubits`
    /// keeps all constraint weights, i.e. a matrix entry `(r, c)` is
    /// allowed. Bit `t` of `r` and `c` is the value of `qubits[t]`.
    pub(crate) fn allows(&self, qubits: &[usize], r: usize, c: usize) -> bool {
        let spread = |local: usize| {
            qubits
                .iter()
                .enumerate()
                .fold(0usize, |acc, (t, &q)| acc | ((local >> t) & 1) << q)
        };
        let (r, c) = (spread(r), spread(c));
        self.constraints
            .iter()
            .all(|&(m, _)| (r & m).count_ones() == (c & m).count_ones())
    }

    /// The sector's states in order, one segment per high half: each item is
    /// `(first position, the states)`.
    pub(crate) fn segments(&self) -> impl Iterator<Item = (usize, SegmentStates<'_>)> + '_ {
        self.segments.iter().map(move |&(h, start)| {
            let lows = &self.classes[self.high_class[h] as usize];
            (
                start,
                SegmentStates {
                    high: h << self.low_bits,
                    lows,
                },
            )
        })
    }
}

/// The basis states of one segment: a fixed high half with each allowed low
/// half.
#[derive(Clone, Copy)]
pub(crate) struct SegmentStates<'a> {
    high: usize,
    lows: &'a [u32],
}

impl<'a> SegmentStates<'a> {
    pub(crate) fn len(&self) -> usize {
        self.lows.len()
    }

    pub(crate) fn iter(self) -> impl Iterator<Item = usize> + 'a {
        let high = self.high;
        self.lows.iter().map(move |&l| high | l as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force(n: usize, constraints: &[(Vec<usize>, usize)]) -> Vec<usize> {
        (0..1usize << n)
            .filter(|&j| {
                constraints
                    .iter()
                    .all(|(qs, w)| qs.iter().filter(|&&q| j >> q & 1 == 1).count() == *w)
            })
            .collect()
    }

    type Constraints = Vec<(Vec<usize>, usize)>;

    #[test]
    fn indexes_match_brute_force_ordering() {
        let cases: Vec<(usize, Constraints)> = vec![
            (6, vec![((0..6).collect(), 3)]),
            (7, vec![(vec![0, 2, 4, 6], 2), (vec![1, 3, 5], 1)]),
            (8, vec![(vec![1, 2, 5], 1)]),
            (5, vec![]),
            (9, vec![((0..9).collect(), 0)]),
        ];
        for (n, constraints) in cases {
            let basis = SectorBasis::new(n, &constraints).unwrap();
            let want = brute_force(n, &constraints);
            assert_eq!(basis.dim(), want.len());
            for (position, &j) in want.iter().enumerate() {
                assert_eq!(basis.index(j), Some(position), "n={n} j={j:b}");
                assert_eq!(basis.state(position), j);
            }
            let in_order: Vec<usize> = basis.segments().flat_map(|(_, s)| s.iter()).collect();
            assert_eq!(in_order, want);
            for j in 0..1usize << n {
                assert_eq!(basis.contains(j), want.binary_search(&j).is_ok());
            }
        }
    }

    #[test]
    fn rejects_bad_constraints() {
        assert!(SectorBasis::new(4, &[(vec![0, 1], 3)]).is_err());
        assert!(SectorBasis::new(4, &[(vec![0, 1], 1), (vec![1, 2], 1)]).is_err());
        assert!(SectorBasis::new(4, &[(vec![4], 0)]).is_err());
    }
}
