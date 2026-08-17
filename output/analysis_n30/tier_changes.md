# Tier changes (n=10 vs merged n)

Tiers: **robust** = Holm p < 0.05; **suggestive** = raw t p < 0.05 but Holm n.s.; **marginal** = 0.05 ≤ t p < 0.10; **n.s.** otherwise.
Holm is the pre-registered 19-comparison accuracy family vs `full`.
Comparisons not in the n=30 expansion keep their n=10 p-values; Holm uses each row's own p (realized n is in the CSV).

## Changes
- `w/o_spectral`: suggestive (n=10, t_p=0.01593, holm=0.2071) → robust (n=30, t_p=1.065e-06, holm=1.917e-05)

## Realized n (accuracy vs full)
- `w/o_spectral`: n=30  tier=robust
- `w/o_temporal`: n=10  tier=robust
- `w/o_fusion`: n=10  tier=n.s.
- `w/o_dynamic_gating`: n=10  tier=n.s.
- `w/o_gating`: n=30  tier=suggestive
- `full_r1`: n=10  tier=n.s.
- `full_r3_stft`: n=10  tier=suggestive
- `full_r3_sinc`: n=10  tier=suggestive
- `full_r1_r3`: n=10  tier=suggestive
- `full_r2`: n=10  tier=n.s.
- `full_r2_gumbel`: n=30  tier=robust
- `full_r4_sparsemax`: n=10  tier=marginal
- `full_r4_entmax`: n=10  tier=n.s.
- `full_r4_lstm`: n=10  tier=n.s.
- `full_r1_r2_r3`: n=10  tier=n.s.
- `full_all`: n=10  tier=n.s.
- `r1_w/o_spectral`: n=10  tier=suggestive
- `full_fixed_global`: n=10  tier=n.s.
- `full_fixed_class`: n=10  tier=robust
