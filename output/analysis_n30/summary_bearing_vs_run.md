# Cell D: bearing-level split vs run-level split

Run-level dir: `C:\Users\xuyix\Desktop\02.核电\code\code01\output\experiments\CellD_Paderborn_ATSF`
Bearing-level dir: `C:\Users\xuyix\Desktop\02.核电\code\code01\output\experiments\CellD_Paderborn_ATSF_bearing`

Code labels KB as **BothRings**. The supplement may say outer-ring for KB; do not change code labels.

Bearing-level experiment has not been run yet. From `code01/`:

```bash
python -m atsf_dcg.run_experiments --synthetic --smoke --cell replication --dataset paderborn --arch atsf --split-unit bearing --configs full

python -m atsf_dcg.run_experiments --cell replication --dataset paderborn --arch atsf --split-unit bearing \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --degradation --log-epoch-indicators \
  --out-dir output/experiments/CellD_Paderborn_ATSF_bearing
```

Do **not** point `--out-dir` at `CellD_Paderborn_ATSF`.
