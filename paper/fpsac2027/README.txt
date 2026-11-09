FPSAC 2027 reproducible demonstration and editable figures

From the package root, install this exact revision into a virtual environment
with python -m pip install . and run:
  python -m unittest discover -s tests -v
  python examples/fpsac2027_join.py
  python examples/benchmark_join.py
  python examples/benchmark_scaling.py
  python paper/fpsac2027/fig/software-layout/source/compute-example.py

The last script independently checks order ranks and bounded integer paths,
including all six initial residues and increments 1 through 240. It regenerates
the raw and reduced table data and records SHA-256 hashes of the implementation.
It does not require the LaTeX manuscript. See docs/closure-correction.txt for
the general closure argument and tests/test_orders.py for regression inputs.

To rebuild the figures:
  cd paper/fpsac2027/fig
  npm install
  npx playwright install chromium
  npm run build
Alternatively set CHROME_PATH to a local Chrome/Chromium executable.
See fig/README.txt for the complete source map and output checks.

verification/benchmark.json (the period-6 demonstration) and
verification/benchmark_scaling.json (seeded random incomparable pairs at
periods 4-64) record local measurements; they are not guarantees for other
machines. tests-python313.txt and tests-python314.txt are local test logs.
Layout and collision JSON files describe the included vector PDFs; text and
line overlaps were also reviewed visually. Neutral cell backgrounds and
outlined, shaded changed cells are intentional contained overlays.
The figures embed unmodified TeX Gyre Pagella fonts, distributed under the
GUST Font License in fig/software-layout/fonts/GUST-FONT-LICENSE.txt.

The earlier project_poster.pdf is historical material, not a specification of
the corrected algorithm. Refer to docs/closure-correction.txt for the current rule.
