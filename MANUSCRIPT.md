# A knowledge-based early-warning system for entrepreneurial failure during digital transformation

Hemasri Babburu  
Doctor of Business Administration candidate

**Article type.** Short communication, prepared for *Knowledge-Based Systems*.

## Abstract

A warning that uses total capital raised, later rounds, or the date of the last round is partly a post-mortem: those fields keep moving until a firm closes. This communication specifies KB-EWS, which scores closure risk from the first priced funding round of a digitally oriented venture and lists the certainty-factor rules that fired. Seven rules were written before estimation. Three were withdrawn because, on ventures first funded in 2006–2010, they pointed against their own mechanism. On 737 ventures first funded in 2012–2013, a depth-3 gradient booster reached a ROC area of 0.740 (95% bootstrap interval 0.704–0.776). Reviewing the 221 highest scores, 30% of that cohort, gave precision 0.760 (0.706–0.824) against a base rate of 0.510 and recovered 44.7% of later closures. The retained rules alone reached 0.618. Stacking them onto the booster lowered the ROC area by 0.004. The issued warning is the booster’s probability, with the fired rules kept as the audit trail.

**Keywords:** artificial intelligence; data-driven prediction; knowledge-based systems; digital transformation; innovation; entrepreneurship; early warning; certainty factors

**Highlights**

- Closure risk is scored at the first priced round, without later funding totals.
- Four certainty-factor rules survive a training-period direction check; three are withdrawn.
- The top 30% of booster scores contain later closures at precision 0.760, against a base rate of 0.510.

## 1. Introduction

Closure, a change of ownership, and insolvency are different events, and survival studies often mix them [1]. For a digital venture the practical question is earlier: at the first priced financing, should a reviewer look more closely? Digital ventures build the offer, the channel, and the operating system together [2], and digital transformation is a renewal of the business model rather than a single installation [3,4]. Credit scoring has a long early-warning tradition [5]. Venture models often leave that tradition by using total capital, the number of rounds, or the recency of the last round. Crunchbase-style extracts are common in management research, with known limits of coverage and outcome coding [6].

Certainty factors were built so a rule base could accumulate graded evidence and remain readable [7]. KB-EWS uses that idea for one decision time. The score is a gradient booster [8] trained on first-round signals. The explanation is a short rule base. Rules that point the wrong way on the training years are withdrawn before any test venture is scored. A logistic stack of the two outputs is fit on a validation year and is not used as the issued score when it fails to improve held-out ranking.

## 2. Method

### 2.1 Decision time, cohort, and label

The source is the Crunchbase export of 4 December 2015, converted to CSV by Tripp (`notpeter/crunchbase-data`, commit `be8711fd`) under CC BY-NC 4.0. Commercial use of the extract requires a Crunchbase license. The program downloads the files at runtime and does not redistribute them.

The decision time is the earliest round with a positive disclosed amount. Allowed features are that amount, round type (venture, seed, angel, other), days from founding, category count, the share of categories in a fixed digital lexicon, country, and a trailing ecosystem count: other digital ventures in the same region first funded in the previous three calendar years, excluding the focal year. Total capital, lifetime round count, last-round date, and status are banned. The code stops if any banned field enters the design matrix.

A venture is included if a category is in the digital lexicon (software, mobile, e-commerce, analytics, cloud, and the related labels in `kbews.py`), dates parse, and the first priced round is within ten years of founding. The label is 1 for status closed and 0 for acquired or IPO. Ventures still operating in December 2015 are omitted. An acquisition can be a distress sale; the export does not say which [1]. The label is closure versus a recorded exit, among ventures that had reached one of those states.

Training years are 2006–2010. Platt scaling and stacking use 2011. The test years are 2012–2013, scored once.

### 2.2 Knowledge base

Conditions are memberships in [0, 1], ramping linearly between two cuts. Antecedents combine by the minimum. Cuts come from the training years only. Thin capital ramps from the 25th to the 60th percentile of log first-round amount, about $475,000 to $2.8 million. Slow funding ramps from the 40th to the 75th percentile of log days since founding, about 314 to 1,088 days. Broad scope ramps from 2 to 4 categories. Thin ecosystem ramps from about 11 to about 98 prior regional peers. Shallow digital depth uses the fixed band 0.40–0.80, and deep digital depth uses 0.50–0.90. Complements supply the “adequate,” “timely,” “narrow,” and “dense” conditions.

Table 1 lists the rules. Positive certainty factors support later closure. Evidence is combined with the Shortliffe–Buchanan rule [7], in order R1–R7, with inputs clipped to [−0.99, 0.99]. The code checks the textbook cases: 0.50 combined with 0.40 returns 0.70, and −0.20 combined with −0.30 returns −0.44.

A rule is kept only if its mean training strength is at least 0.02 higher in the class it was written to support. That comparison uses 2006–2010 only. R4, R5, and R7 fail it. The retained base is R1, R2, R3, and R6.

**Table 1.** Rules specified before estimation.

| Rule | Antecedents (fuzzy AND) | CF | Training decision | Mechanism |
|---|---|---:|---|---|
| R1 Resource strain | Thin capital AND slow first funding | +0.55 | Retained | Scaling without slack |
| R2 Unfunded scope | Thin capital AND broad category scope | +0.45 | Retained | Scope ahead of the first round |
| R3 Shallow digital base | Thin capital AND shallow digital depth | +0.35 | Retained | Thin capital, peripheral digital position |
| R4 Isolated entry | Thin ecosystem AND slow first funding | +0.30 | Withdrawn | Stronger among exits in training |
| R5 Unfocused entry | Broad scope AND shallow digital depth | +0.25 | Withdrawn | Stronger among exits in training |
| R6 Resourced focus | Adequate capital AND narrow scope AND deep digital depth | −0.50 | Retained | Capital, focus, and depth aligned |
| R7 Embedded start | Dense ecosystem AND timely funding | −0.30 | Withdrawn | Training gap below 0.02 |

### 2.3 Scores and the stacking test

Logistic regression uses a scaler fit on the training years. The eight most common training countries are kept, with the United States as the reference and other countries pooled. Venture rounds are the reference round type. The same columns are applied to 2011 and to the test years.

On 2006–2010 the program fits a class-balanced logistic regression, a class-balanced tree of depth at most 4 with at least 40 ventures per leaf, and a histogram gradient booster of depth 3, learning rate 0.06, L2 penalty 0.1, at least 40 ventures per leaf, and balanced class weights [8]. Early stopping on an internal 15% of the training years stopped at 84 trees. The 2011 year does not grow those trees.

Each of the four scores is mapped to a probability by a one-variable logistic fit on 2011 [9]. Stacked fusion is a logistic regression of closure on the booster logit and the raw certainty factor, also fit on 2011, with a free coefficient so the stack can ignore or reverse the rule base.

### 2.4 Test report

The test report is the ROC area, the precision-recall area, and the Brier score. The operating point is a review budget: the highest-scoring 20% and 30% of test ventures (147 and 221 alerts). The budget does not use test labels to pick a threshold. Precision is the closure share inside the reviewed set. Recall is the share of all test closures that the set contains. Intervals are 2.5 and 97.5 percentiles from 1,000 bootstrap draws. The seed is 7.

## 3. Results

The export has 66,368 companies and 114,949 rounds. Priced first rounds cover 53,583 companies, and 28,482 of those match the lexicon. Usable dates and the ten-year window leave 21,407 ventures, of which 4,290 had a resolved status by December 2015. The model uses the 3,150 resolved ventures first funded in 2006–2013: 1,901 for training (closure share 0.426), 512 in 2011 (0.525), and 737 for test (0.510; 376 closures). These shares are not failure rates for all digital startups. They are closure shares among resolved outcomes. The median first round is $1.65 million in training and $750,000 in the test years.

**Table 2.** Held-out results for 737 ventures first funded in 2012–2013. Intervals are 95% bootstrap percentiles. The 30% budget is the 221 highest scores.

| System | ROC-AUC | PR-AUC | Brier | Precision at 30% | Recall at 30% |
|---|---:|---:|---:|---:|---:|
| Retained knowledge base | 0.618 (0.576–0.657) | 0.601 | 0.242 | 0.633 (0.575–0.701) | 0.372 |
| Logistic regression | 0.735 (0.698–0.771) | 0.743 | 0.209 | 0.751 (0.692–0.810) | 0.441 |
| Decision tree | 0.729 (0.691–0.764) | 0.702 | 0.212 | 0.751 (0.697–0.810) | 0.441 |
| Gradient boosting | 0.740 (0.704–0.776) | 0.745 | 0.207 | 0.760 (0.706–0.824) | 0.447 |
| Stacked fusion | 0.736 (0.699–0.771) | 0.743 | 0.208 | 0.769 (0.715–0.828) | 0.452 |

Figure 1 is the issued system: the booster’s probability plus the rules that fired. At a 20% budget the booster’s precision is 0.810 (0.735–0.871) and its recall is 0.316. At 30%, precision 0.760 is 1.49 times the test base rate, and the review set holds 168 of 376 closures. Logistic regression is close. Most of the rank signal is linear.

In the class-balanced training regression, one standard deviation more log first-round amount lowers closure log-odds by 0.511, and one standard deviation more log ecosystem size lowers them by 0.328. Angel and seed rounds, relative to venture, raise log-odds by 0.333 and 0.163. Two linear signs disagree with a casual reading of the rules. A longer wait for the first round lowers closure log-odds by 0.276 per standard deviation, and a longer category list lowers them by 0.154, while digital depth raises them by 0.132. The retained rules are conjunctions. R1 survives only because a small round together with a long wait was slightly more common among training closures (mean strength 0.132 versus 0.096). R2 and R3, which both require thin capital, separated more clearly (0.282 versus 0.158, and 0.322 versus 0.168).

Figure 2 shows the test years, which were not used to keep or drop rules. R2 fired for 47.9% of later closures and 29.9% of exits. R3 fired for 46.5% and 25.8%. R6 fired for 8.5% of closures and 18.8% of exits. R1 fired for 15.7% and 9.7%. The withdrawn rules do not separate the test outcomes in the direction originally written.

Stacked fusion puts weight 1.034 on the booster logit and −0.233 on the certainty factor. Once the booster is known, a higher failure certainty factor slightly lowers the stacked probability. ROC area falls by 0.004 (interval −0.008 to −0.0004). Precision at 30% is 0.769 rather than 0.760, and the intervals overlap. The stack is not the issued score.

## 4. Conclusion

At the first priced round, KB-EWS returns a closure probability from a depth-3 booster and the retained rules that fired. On digital ventures first funded in 2012–2013, the top 30% of scores raise closure precision from 0.510 to 0.760. The rules are the auditable part. Alone they rank weakly, and stacking them into the probability does not help.

The association is not an effect of a digital-transformation program; the export does not record one. Operating ventures are omitted, so the base rate is not a population failure rate. Some acquisitions are failures under another name. Categories can be edited after founding. Later cohorts sit in denser ecosystems than the training years. A later warning could be sharper and would answer a different question.

Reproduce the tables and figures with `python kbews.py`. The numbers in Table 2 are taken from `outputs/metrics.json`.

## References

[1] M.A. Josefy, J.S. Harrison, D.G. Sirmon, C. Matz Carnes, Living and dying: synthesizing the literature on firm survival and failure across stages of development, Academy of Management Annals 11 (2) (2017) 770–799. https://doi.org/10.5465/annals.2015.0148

[2] S. Nambisan, Digital entrepreneurship: toward a digital technology perspective of entrepreneurship, Entrepreneurship Theory and Practice 41 (6) (2017) 1029–1055. https://doi.org/10.1111/etap.12254

[3] G. Vial, Understanding digital transformation: a review and a research agenda, Journal of Strategic Information Systems 28 (2) (2019) 118–144. https://doi.org/10.1016/j.jsis.2019.01.003

[4] K.S.R. Warner, M. Wäger, Building dynamic capabilities for digital transformation: an ongoing process of strategic renewal, Long Range Planning 52 (3) (2019) 326–349. https://doi.org/10.1016/j.lrp.2018.12.001

[5] E.I. Altman, Financial ratios, discriminant analysis and the prediction of corporate bankruptcy, Journal of Finance 23 (4) (1968) 589–609. https://doi.org/10.1111/j.1540-6261.1968.tb00843.x

[6] J.-M. Dalle, M. den Besten, C. Menon, Using Crunchbase for economic and managerial research, OECD Science, Technology and Industry Working Papers 2017/08 (2017). https://doi.org/10.1787/6c418d60-en

[7] E.H. Shortliffe, B.G. Buchanan, A model of inexact reasoning in medicine, Mathematical Biosciences 23 (3–4) (1975) 351–379. https://doi.org/10.1016/0025-5564(75)90047-4

[8] J.H. Friedman, Greedy function approximation: a gradient boosting machine, Annals of Statistics 29 (5) (2001) 1189–1232. https://doi.org/10.1214/aos/1013203451

[9] J.C. Platt, Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods, in: A.J. Smola, P. Bartlett, B. Schölkopf, D. Schuurmans (Eds.), Advances in Large Margin Classifiers, MIT Press, Cambridge, MA, 1999, pp. 61–74.

## Figures

**Figure 1.** KB-EWS at the first priced round. The issued score is the gradient booster. Retained rules are attached to the warning. Stacking the two into one probability was tested on 2011 and did not improve the 2012–2013 ranking.

![Figure 1. Architecture of the early-warning card](outputs/fig1_architecture.png)

**Figure 2.** Test years 2012–2013. (a) ROC curves. The solid navy curve is the issued score. The dashed curve stacks the certainty factor onto that score. (b) Share of ventures whose rule strength reached 0.5. Withdrawn rules failed the 2006–2010 direction check and are not in the knowledge-base score.

![Figure 2. Held-out ranking and rule firing](outputs/fig2_performance.png)
