# Model comparison

| Model | Params (M) | Best epoch | Val loss | s/epoch | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|---|---|---|---|
| lstm | 3.7 | 9 | 0.607 | 1.0 | 0.8095 | 0.7034 | 0.5484 | 0.3424 |
| attention | 7.1 | 38 | 1.018 | 3.0 | 0.2078 | 0.0937 | 0.0513 | 0.0 |
| transformer | 13.7 | 4 | 0.534 | 3.5 | 0.7778 | 0.7201 | 0.6713 | 0.6055 |

Missing values (—): train the model (`train.py --model X`) and/or run `evaluate.py --model X` first.