# KB-EWS

Knowledge-based early warning of entrepreneurial failure during digital transformation. A short communication for *Knowledge-Based Systems*, plus the one program that produces its figures and tables.

The decision time is a digital venture’s first priced funding round. The issued score is a depth-3 gradient booster. Four certainty-factor rules are attached as an audit trail. Three other rules were withdrawn because they pointed the wrong way on the 2006–2010 training years. On 737 ventures first funded in 2012–2013, the top 30% of scores had a closure precision of 0.760 against a base rate of 0.510 (ROC-AUC 0.740). Stacking the rule score onto the booster did not improve ranking. The full argument, Table 1, Table 2, and the figure captions are in `MANUSCRIPT.md`.

## Run

```bash
python -m pip install -r requirements.txt
python kbews.py
```

The script downloads two Crunchbase CSV files into `.cache/` (about 30 MB), fits the models, and writes:

- `outputs/fig1_architecture.png`
- `outputs/fig2_performance.png`
- `outputs/metrics.json`

Every number in Table 2 of the manuscript is copied from `metrics.json`. The random seed is 7. A full run takes about half a minute after the download.

## Data license

The December 4, 2015 Crunchbase export is used under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The CSV conversion is from [notpeter/crunchbase-data](https://github.com/notpeter/crunchbase-data) at commit `be8711fd`. Attribute Crunchbase. Non-commercial research use is allowed; commercial use needs a separate license from Crunchbase. This repository does not contain the raw files.

The code in `kbews.py` is available under the MIT license.

## What the program refuses to use

`funding_total_usd`, lifetime `funding_rounds`, `last_funding_at`, and `status` are banned as predictors. Those fields keep changing until a venture closes, so a model that sees them is not an early warning. The script stops if any of them appears in the design matrix.
