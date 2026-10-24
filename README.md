# KB-EWS

Knowledge-based early warning of entrepreneurial failure during digital transformation. A short communication for *Knowledge-Based Systems*, plus the one program that produces its tables.

The decision time is a digital venture’s first priced funding round. The pre-specified score is a depth-3 gradient booster. Four certainty-factor rules are attached as an audit trail; three other rules were withdrawn because they pointed the wrong way on the 2006–2010 training years. On 737 ventures first funded in 2012–2013, the top 30% of booster scores had closure precision 0.760 against a base rate of 0.510 (ROC-AUC 0.740). Paired tests do not separate that score from logistic regression. Stacking the rules into the probability lowers the ROC area by 0.004. The argument and both tables are in `MANUSCRIPT.md`.

## Run

```bash
python -m pip install -r requirements.txt
python kbews.py
```

The script downloads two Crunchbase CSV files into `.cache/` (about 30 MB), fits the models, and writes:

- `outputs/metrics.json`
- `outputs/warning_cards.csv` — one card per 2012–2013 venture, with the calibrated closure probability, the retained rules whose strength reached 0.5, and a flag for the top 30% review queue. `later_closed` is the held-out outcome used to audit that queue. It is not a model input.

Every number in Table 2 of the manuscript is copied from `metrics.json`. The random seed is 7. A full run takes about half a minute after the download.

## Data license

The December 4, 2015 Crunchbase export is used under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The CSV conversion is from [notpeter/crunchbase-data](https://github.com/notpeter/crunchbase-data) at commit `be8711fd`. Attribute Crunchbase. Non-commercial research use is allowed; commercial use needs a separate license from Crunchbase. This repository does not contain the raw files.

The code in `kbews.py` is available under the MIT license.

## What the program refuses to use

`funding_total_usd`, lifetime `funding_rounds`, `last_funding_at`, and `status` are banned as predictors. Those fields keep changing until a venture closes, so a model that sees them is not an early warning. The script stops if any of them appears in the design matrix.
