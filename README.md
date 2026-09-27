# Comparison of feedforward network training algorithms

Stochastic gradient descent, scaled conjugate gradient (Moller 1993) and
LeapFrog LFOP1(b) (Snyman 1982, 1983) trained on the same one hidden layer
networks, over five classification and four function approximation problems.

## Running

```bash
python src/main.py all # the full study, writes out/ (about an hour)
python src/main.py all --quick # small grids, five problems, a few minutes
pytho src/main.py analyse # statistics and figures from existing csv

```

The same study with ReLU instead of sigmoid hidden units, and the comparison
of the two:

```bash
python3 src/main.py all --activation relu --out out/out_relu
python3 src/compare_activations.py
```