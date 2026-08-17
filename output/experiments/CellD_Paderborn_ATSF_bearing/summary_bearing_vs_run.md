# Cell D: bearing-level split vs run-level split

Run-level dir: `/mnt/workspace/output/experiments/CellD_Paderborn_ATSF`
Bearing-level dir: `/mnt/workspace/output/experiments/CellD_Paderborn_ATSF_bearing`

Code labels KB as **BothRings**. The supplement may say outer-ring for KB; do not change code labels.

## Protocol (bearing split)

- split_unit: `bearing`
- train bearings (n=19): ['K001', 'K002', 'K005', 'K006', 'KA01', 'KA03', 'KA05', 'KA07', 'KA09', 'KA16', 'KA30', 'KB23', 'KI03', 'KI04', 'KI05', 'KI14', 'KI16', 'KI17', 'KI18']
- val bearings (n=7): ['K003', 'KA04', 'KA08', 'KA22', 'KB27', 'KI07', 'KI08']
- test bearings (n=6): ['K004', 'KA06', 'KA15', 'KB24', 'KI01', 'KI21']
- KB 1/1/1: `True`
- excluded files: `['KA08/N15_M01_F10_KA08_2.mat']`

## Clean accuracy / F1 (bearing split)

| config | acc mean±SD | F1 mean±SD | n |
|---|---|---|---|
| full | 0.4933±0.0276 | 0.5317±0.0301 | 10 |
| w/o_spectral | 0.5155±0.0160 | 0.5464±0.0206 | 10 |
| w/o_temporal | 0.3867±0.0427 | 0.4124±0.0499 | 10 |
| w/o_fusion | 0.4945±0.0225 | 0.5240±0.0275 | 10 |
| w/o_gating | 0.5023±0.0321 | 0.5339±0.0345 | 10 |
| full_r1 | 0.4967±0.0332 | 0.5342±0.0339 | 10 |
| full_r2_gumbel | 0.4977±0.0182 | 0.5326±0.0213 | 10 |

## Paired w/o_spectral − full (bearing split)

- n=10
- accuracy: Δ=+2.224 pp, t p=0.05196, Wilcoxon p=0.06445, dz=+0.708
- macro-F1: Δ=+1.477 pp, t p=0.2355, Wilcoxon p=0.2754, dz=+0.402

**F1 reversal (w/o_spectral < full) persists:** no.

## Degradation matrix (bearing split)

```
degradation,full,w/o_spectral,w/o_temporal,w/o_fusion,w/o_gating,full_r1,full_r2_gumbel,delta_vs_clean_pp
clean,0.4933±0.0276,0.5155±0.0160,0.3867±0.0427,0.4945±0.0225,0.5023±0.0321,0.4967±0.0332,0.4977±0.0182,0.0
gaussian_noise_snr20,0.5339±0.0196,0.5351±0.0167,0.4281±0.1215,0.5211±0.0246,0.5376±0.0243,0.4082±0.1296,0.5276±0.0201,1.5
gaussian_noise_snr10,0.4906±0.0704,0.5867±0.0178,0.3159±0.1097,0.5031±0.0725,0.5040±0.0714,0.3546±0.0746,0.5427±0.0526,-1.27
drift,0.4813±0.0877,0.4798±0.0721,0.3499±0.0839,0.4702±0.0798,0.4762±0.1025,0.4046±0.0737,0.4964±0.0989,-3.26
bias,0.4594±0.0185,0.4712±0.0145,0.3867±0.0427,0.4503±0.0173,0.4566±0.0218,0.4684±0.0244,0.4569±0.0177,-3.39
stuck,0.4508±0.0149,0.4726±0.0119,0.3881±0.0341,0.4450±0.0221,0.4496±0.0227,0.4484±0.0183,0.4459±0.0185,-4.09
dropout,0.4460±0.0514,0.4752±0.0215,0.3124±0.0898,0.4782±0.0401,0.4640±0.0407,0.3564±0.0912,0.5029±0.0532,-5.02
downsample,0.4621±0.0356,0.5144±0.0209,0.4924±0.0547,0.4486±0.0202,0.4608±0.0299,0.4534±0.0245,0.4553±0.0263,-1.42
```

## S(0.9) / ρ (bearing split, config `full`)

- S(0.9) mean±SD: 0.2956±0.0307 (n=10)
- ρ mean±SD: 4.4488±0.9029 (n=10)

## Run-level reference (existing Cell D, not overwritten)

- w/o_spectral − full F1 Δ=-0.073 pp (n=10)
