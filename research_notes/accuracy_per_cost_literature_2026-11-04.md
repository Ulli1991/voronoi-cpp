# Accuracy per cost for hydro codes: literature check (to Nov 2026)

Question: has anyone defined a standard accuracy-per-cost metric for hydro/CFD
codes that puts moving-mesh, AMR, high-order and fixed-grid codes in one unit?
Our proposal: E = W_ref(eps) / T, "effective cell updates per second".

## Evidence tags

- [P] read in primary text this session (arXiv LaTeX source, or PDF text via Ghostscript)
- [A] abstract, metadata or official web page read; body not read
- [S] search-engine excerpt only; wording not checked against the source
- [U] not verified this session; treat as a pointer, not a citation
- INFERENCE = our reasoning, not stated by any cited paper

Sections 1-5 are what sources state. Sections 6-8 contain inference and say so.

## Search limits (read before citing "not found")

- Tools: web search plus arXiv LaTeX source grep for about 45 papers. No ADS
  full-text search (no token). No paywalled journal text.
- "grep found nothing" means no match for the cost terms in the arXiv source
  (CPU, wall-clock, cost, and similar). It does not prove absence of the concept.
- NOT read in the body: Hairer/Norsett/Wanner books; Wang et al. 2013 (body text
  would not extract); Fleming & Wallace; Mashey 2004; Dolan & More; Vermeire 2017;
  Capuano 2023; Felker's thesis; Green500 and HPCG papers; Banks et al. 2008.
- The negative result for the name and metric is a search result, not a proof.

## 0. Verdict

(a) Exactly: NOT FOUND. No source found defines a cross-mesh metric in throughput
    units, normalised by a published reference-scheme work table, at fixed error,
    with a suite geomean.
(b) In part: YES. Every ingredient exists, separately (section 7).
(c) Name: no hit for "effective cell updates per second", "ECUPS", "eCUPS",
    "effective CUPS" or "equivalent cell updates" in any search run.

Closest prior work, by aspect:
- Formal metric: Chang, Fabien, Knepley & Mills 2018 (TAS, "true DoF/s").
- Factorisation: Fehn, Wall & Kronbichler 2018 (E = accuracy / cost, factored).
- Stated principle in astro: Bauer & Springel 2012; Hopkins 2015; Bauer+ 2016.
- Procedure in CFD: High-Order CFD Workshop (error vs TauBench work units).
- Accuracy per CPU time on shock tests: Greenough & Rider 2004.
- Benchmark design: MLPerf (time to quality), HPGMG ("solved" gate), SPEC.

## 1. Work-precision, time to solution, CFD

ODE tradition
- Hairer/Norsett/Wanner, Solving ODEs I (2nd ed. 1993, Springer SSCM 8), II (2nd
  ed. 1996, SSCM 14): book existence [A]. That the term "work-precision diagram"
  appears there or originates there: [U]. Found no text confirming it. Do not
  claim coinage.
- Hull, Enright, Fellen & Sedgwick 1972, SIAM J. Numer. Anal. 9 (DETEST lineage),
  and Enright, Hull & Lindberg 1975, BIT 15: DETEST gives two cost measures
  (function evaluations, overhead) plus reliability statistics [S]. This predates
  the term as far as we can tell; comparing ODE codes on a problem set is old.
- Bari IVP test set: work-precision diagrams plot significant correct digits
  (scd/mescd) against CPU seconds. It warns that other settings, another computer
  or compiler, or another tolerance range "might change the diagrams
  considerably" [S].
- SciMLBenchmarks: same idea, includes PDE semi-discretisations [S].

PDE and CFD
- Wang et al. 2013, IJNMF 72:811 (1st High-Order CFD Workshop). Excerpts [S]:
  work unit = CPU time to run TauBench for 10 steps; methods compared by error vs
  cost; authors "caution that the definition of cost is not fool-proof"; high-order
  better than 2nd-order FV "based on error versus cost". Excerpts are from the
  paper, body text did not extract, so check wording before quoting.
- XFlow workshop entry C1.6 [P]: error plotted vs work units; "one TauBench unit
  is equivalent to 16.5 seconds of compute time" on one machine. This confirms
  what the unit normalises: HARDWARE, not scheme.
- Vermeire, Witherden & Vincent 2017, JCP 334:497: compares accuracy and cost of
  PyFR (GPU) and STAR-CCM+ (CPU). No GPU TauBench exists to normalise with [S].
- Fehn, Wall & Kronbichler 2018, IJNMF, arXiv:1802.01439 [P]:
  E = accuracy / computational cost, accuracy = inverse of a "suitable and problem
  specific error measure", cost = t_wall * N_cores. Factored as
  (accuracy / (DoFs*timesteps)) * (1/iterations) * (DoFs*timesteps*iterations/cost).
  Best efficiency at moderate k=5,7 on under-resolved Taylor-Green.
- Chang, Fabien, Knepley & Mills 2018, SISC 40(6):C779, arXiv:1802.07832 [P]:
  Time-Accuracy-Size spectrum. DoA = -log10(err); DoS = log10(N); "efficacy" =
  error * time, DoE its log. "True static-scaling" scales DoF by the mesh
  convergence ratio; they state it yields the same measure. Tested on Poisson FE
  (FEniCS, deal.II and others), with CG vs DG. They note a code can look fast in
  DoF per time yet have low DoE and true DoF per time once DoA is included.
  Earlier: Chang et al., arXiv:1705.03625.
- Greenough & Rider 2004, JCP 196:259 [P]: "accuracy per unit CPU time" explicit.
  Common test bed, 1D. Linear advection: WENO5 "enormous advantage". Nonlinear:
  convergence falls to first order; PLMDE six times faster on a given mesh; "when
  the computational expense ... is equated, PLMDE produces solutions with uniformly
  lower errors". No named metric, no suite score.
- Motheau & Wakefield 2019, arXiv:1902.06665 [P]: "A more realistic way to assess a
  numerical method is to determine the cost needed to obtain a desired accuracy."
- Capuano et al. 2023, EJMB 102:91 "Cost vs Accuracy": Nek5000, OpenFOAM, Cartesian
  IBM perform in a "similar computational performance range" for DNS of a sphere
  at Re=3700 [S].
- Ranocha et al. 2022, arXiv:2112.10517 [P]: PID = runtime of one RHS evaluation /
  #DOF. A cost-only measure, per RHS rather than per step.

## 2. Astrophysical hydro

Throughput conventions: none of these carries accuracy
| Code | Unit | Source |
|---|---|---|
| Athena++ | "zone updates per second per core", zone-cycles/s, cell-updates/s | Stone+ 2020, ApJS 249:4 [P] |
| AthenaK | "cell-updates-per-device-second", linear wave test | arXiv:2409.16053 [P] |
| Parthenon | zone-cycles/s, per node | Grete+ 2023, arXiv:2202.12309 [P] |
| GAMER-2 | cell updates per second, per node | Schive+ 2018, arXiv:1712.07070 [P] |
| Enzo | cell updates per second per MPI process | Bryan+ 2014, arXiv:1307.2265 [P] |
| IDEFIX | cell/s (">10^8 cell/s in MHD" on a V100) | Lesur+ 2023, arXiv:2304.13746 [P] |
| Cholla | cell updates per GPU-second | Schneider & Robertson 2015 [P/A] |

- Definition drift: AGILE 1.0 (arXiv:2607.19277) counts one update as one RK
  substep [S]. A review (arXiv:2007.05218) warns against direct comparison between
  codes with different algorithms [S].
- Stone+ 2020 [P]: PPM drops throughput by about a factor of two vs PLM. AMR with
  16^3 MeshBlocks: "cell-updates per second is about half of the uniform grid's
  efficiency, but its time to solution is about five times shorter". This is the
  cleanest published example of raw CUPS and time to solution disagreeing.

Statements closest to our metric
- Bauer & Springel 2012, MNRAS 423:2558, arXiv:1109.4413 [P]. Moving mesh about 5x
  slower than fixed mesh at equal element count (mesh construction, about twice as
  many Riemann problems). "What should really be considered is the computational
  cost to reach a desired level of accuracy." 128^3 AREPO is at least as good as
  256^3 SPH on the Kolmogorov spectrum, so "about 4 times as efficient". And: "such
  statements ... are in general resolution and problem dependent". Relative cost
  of moving vs fixed depends on Mach number.
- Hopkins 2015, MNRAS 450:53, arXiv:1409.7395 [P]. Speed "always difficult to
  quantify" because non-linear solutions differ, so "it is not obvious that we are
  comparing the 'same' test". "we should really compare performance at fixed
  accuracy." For grid/AMR: "it is not clear how to make a fair speed comparison".
  Runtimes given at fixed resolution, normalised to TSPH. Claims grid codes need
  timesteps smaller by 100-1000 to keep accuracy in supersonic flow; this is a
  statement, not a measured time-to-error. Also: how long a Keplerian disk evolves
  before L1 exceeds a tolerance (error threshold reached in simulated time).
- Bauer+ 2016, arXiv:1602.09079 [P]. "efficiency for a given accuracy". DG-3 vs
  FV-2 on subsonic turbulence: 1.14, 1.34, 1.53 times faster at equal spectra; FV
  needs 1.86 times the DOF. Warns that optimisation level differs between solvers
  and CPU counts differ. This is an equal-accuracy equivalence, ad hoc, by eye.
- Schaal+ 2015, MNRAS 453:4278, arXiv:1506.06140 [P]. L1 vs measured wall-clock.
  FV-2 and DG-2 about equal; higher order "easily faster by more than an order of
  magnitude" for the smooth test.
- Hubber, Falle & Goodwin 2013, MNRAS 432:711, arXiv:1304.3456 [P]. Error-work law:
  3D FV E ~ W^-1/2, SPH E ~ W^-1/4. FV beats SPH by an order of magnitude in
  wall-clock "assuming the effective resolution of both codes is the same".
  Timings stated to apply to the two implementations used.
- Robertson+ 2010, MNRAS 401:2463, arXiv:0909.0513 [P]. A fitted L1 scaling can
  "calculate the necessary resolution ... to reach an equivalent error" under bulk
  flow, with a caveat that scaling could differ on other problems.
- Tasker+ 2008, MNRAS 390:1267, arXiv:0808.1844 [P]. Enzo, Flash, Gadget, Hydra.
  "roughly matching accuracy" at about one SPH particle per grid cell on Sod.
  Timing table caveat: comparing CPU times across disparate computers "is fraught".
- O'Shea+ 2005, ApJS 160:1 [P]. Dedicated timing section; warns not to extrapolate
  to other problems or machines.
- Kitsionas+ 2009, A&A 508:541, arXiv:0810.4599 [P]. CPU-h normalised to 256^3 and
  1.6 GHz, "accurate to within factors of a few". SPH extra cost reflects resolution
  following mass. Cost at a matched setup, not at matched error.
- Felker & Stone 2018, JCP 375:1365, arXiv:1711.07439 [P]. Fourth-order Athena++.
  NO error-per-cost number in the paper; says tradeoffs "will be quantitatively
  evaluated". The efficiency wording ("time-to-solution for a given level of
  accuracy") is in a Felker conference abstract [S], not this paper.
- Springel 2010, arXiv:0901.4107 [P]: per-step cost vs SPH (about 2.8x), no error
  per cost. Pakmor+ 2016 [P]: convergence fix at "insignificant" cost, no metric.
  Weinberger+ 2020 [P]: only a passing "accuracy for a given computational expense".
- Lemaster & Stone 2009; Federrath+ 2010; Kim+ 2014, 2016 (AGORA); Hayward+ 2014;
  McNally+ 2012; Lecoanet+ 2016: grep found no error-per-cost comparison. Frenk+
  1999 (Santa Barbara): lists per-code CPU times only [P].

Dissipation and effective-Reynolds measures
- Kritsuk+ 2011, ApJ 737:13 [P]: effective spectral bandwidth, ranking by relative
  effective Reynolds number. Weaker codes "must compensate higher numerical
  dissipation with higher numerical resolution". ILES "does not imply convergence".
- Malvadi Shivakumar & Federrath 2025 (arXiv:2311.10350) [A]: Re, Rm as power laws of
  N. Review arXiv:2510.12203 [P]: Re = (N/N_Re)^p_Re, bands include code differences.
- Grete, O'Shea & Beckwith 2023, arXiv:2211.09750 [A]: effective viscosity from energy
  transfer; no convergence at 2048^3.
- Lecoanet+ 2016 [P/S]: effective Re for Athena "extremely problem dependent". More
  small-scale structure as a proxy for accuracy "can, in some cases, lead to false
  conclusions" [P].

## 3. Benchmark standards

- SPEC CPU 2017 [A, official docs]: ratio = time on reference machine / time on SUT;
  reference is a historical Sun Fire V490; median of three runs (or slower of two);
  geometric mean overall; base required, peak optional.
- Fleming & Wallace 1986, CACM 29(3):218 [A]: arithmetic mean of normalised results is
  misleading; the geometric mean is normalisation independent.
- Smith 1988, CACM 31(10):1202 [P]: time is the ultimate measure; total time and
  harmonic mean of rates agree.
- John 2004, UT TR-040126-01 [P]: geometric mean does not give overall speedup;
  weighted arithmetic or harmonic means work. Mashey 2004, CAN 32(4): title [A].
- Hoefler & Belli SC15 [P]: Rule 3: arithmetic mean only for costs, harmonic for
  rates. Rule 4: avoid summarizing ratios, summarize the underlying costs or rates,
  and use the geometric mean only if those are unavailable; it has "no simple
  interpretation". Rule 9: document all varying factors.
- MLPerf Training, Mattson+ 2020, arXiv:1910.01500 [P]: metric is time to train to a
  defined quality target. Timing starts when data is touched. Thresholds near state
  of the art, chosen from reference runs. Vision 5 runs, others 10. Reference
  implementations define the task and are NOT for performance. EXPLICITLY NO SUMMARY
  SCORE: it implies a weighting and fails when submitters omit benchmarks.
- DAWNBench analysis, Coleman+ 2018, arXiv:1806.01427 [A/P of HTML]: time to accuracy
  "largely stable"; asks whether entries are best of many trials; MLPerf CoV below 14%.
- HPGMG, Adams, Brown, Shalf+ 2014 [P]: "Our metric is equations solved per second".
  "we face the challenge of defining 'solved'": the FMG Cauchy sequence verifies
  same-order convergence as the discretization. Manufactured solution; reference
  kernels; rejects submissions not using its harness. Metric is deliberately defined
  so the undesirable quantity is a function of the desirable one (HPL's trick).
- CEED BPs, Fischer+ 2020, IJHPCA 34(5), arXiv:2004.06722 [P]: DOFs/s, n_0.8, t_0.8,
  r_0.8 at 80% efficiency. Compares across platforms at set p. No error in the metric.
- ECP KPP-1, Alexander+ 2020, Phil Trans A 378:20190056 [A]: FOM = rate of "science
  work", "specific to an application area"; ratio to own baseline.
- STREAM [A, official page]: four kernels, array at least 4x caches, wall clock only,
  best of about 10 trials, non-conforming runs must be labelled "tuned"; the owner
  rules on conformity.
- HPCG [A]: "intended as a complement to" HPL. Green500 [S]: flops/W on HPL; authors
  note it may favour smaller systems.
- Pennycook+ arXiv:1611.07409 [A/S]: harmonic mean of efficiencies; 0 if any platform
  fails. Criticised as unintuitive and lossy [S].
- Dolan & More 2002, Math. Prog. 91 [S]: performance profiles, a distribution rather
  than a mean, so outliers do not dominate.

## 4. Effective DOF and equivalent grid

- AMR "effective resolution" is the finest-level equivalent: 2^l per dimension
  (arXiv:1905.01948) [S]. It measures geometry, not accuracy.
- AMR "may not achieve the convergence of equivalent finest uniform grid" (LANL
  report) [S].
- Kitsionas+ 2009 [P]: SPH effective resolution rises with density.
- Fehn 2018 (DoF vs error) and TAS (mesh-convergence ratio) [P] are the formal
  accuracy-weighted cousins.
- Climate: CPMIP (Balaji+ 2017, GMD 10:19): SYPD, core-hours per simulated year [A].
  Cost only.

## 5. Error norms across meshes

- Springel 2010 and Hopkins 2015 [P]: L1 = (1/N) sum |rho_i - rho(x_i)|, per cell or
  particle, evaluated at x_i, unweighted.
- Hopkins 2015 [P]: SPH L1 depends on start-up; E0 error can dominate. Reports
  velocity L1 and density L1 separately. Bilinear interpolation of meshless ICs back
  to the grid to match seed modes.
- Bauer & Springel 2012 [P]: power spectra by nearest-neighbour sampling onto a grid
  twice as fine as the simulation; checked aliasing and energy.
- Lecoanet+ 2016 [P]: L2 of passive dye, integrated dV; spectral interpolation onto
  the Athena grid; high-resolution Dedalus as truth. L1 gave similar results.
- McNally+ 2012 [P]: mode amplitude vs a converged reference with stated uncertainty;
  six codes including moving mesh and SPH.
- Price & Federrath 2010 [P]: volume-weighted SPH PDF by m/rho. Kitsionas+ 2009 [P]:
  p_v = p_m/rho, and grid-interpolated PDF matches the particle PDF.
- Kritsuk 2011 [P]: statistics rather than pointwise error for turbulence.
- Remapping: Dukowicz & Kodis 1987, Margolin & Shashkov 2003 [U, bibliography only].
  Powell & Abel 2015 [A]: exact conservative voxelization of polynomial density by
  polyhedron clipping. VorAMR (arXiv:2511.14697) [S]: Voronoi to AMR with a few
  percent mass and energy error from nearest-neighbour.

## 6. INFERENCE: properties of our metric

1. Factorisation. E = (W_ref/W_code) * (W_code/T): scheme equivalence times raw
   throughput. Same structure as Fehn 2018, in throughput units.
2. Under a geomean the ranking of two codes is independent of W_ref:
   GM(E_A)/GM(E_B) = GM(T_B,i / T_A,i). This is the Fleming-Wallace property. The
   table sets the absolute scale, and makes a reference-scheme code score its raw
   rate; it does not change rankings. Errors in the table cannot reorder codes,
   given fixed tests and eps. Under an arithmetic mean they could.
3. Hoefler Rule 4 needs an answer: costs (T) are available, so geomean is not their
   "fallback" case. Justification must be normalisation independence and the absence
   of a defensible workload weighting. Report per-test values too.
4. W_ref is a count, so it is hardware free, unlike SPEC reference times. The table
   ages only when scheme definition or eps changes.
5. TAS efficacy (error*time) is not fixed-error: it changes along a convergence
   curve with slope d-alpha. Our fixed eps removes that. TAS had no reference scheme.
6. The "plain code scores its raw rate" identity holds only if that code reproduces
   the table's error at the same N and step count (same CFL, limiter, splitting) and
   counts updates identically.

## 7. What exists and whom to credit

| Ingredient | Prior work | Status |
|---|---|---|
| Error vs cost as the comparison | ODE work-precision; Wang 2013 | exists |
| Accuracy per CPU time on shock tests | Greenough & Rider 2004 | exists |
| Fixed quality, measure time | MLPerf, DAWNBench | exists outside CFD |
| Accuracy-weighted DoF rate | Chang+ 2018 TAS | exists, Poisson FE |
| Accuracy/cost factorisation | Fehn+ 2018 | exists |
| Equal-accuracy equivalence between schemes | Bauer & Springel 2012; Bauer+ 2016; Hubber+ 2013; Robertson+ 2010 | ad hoc, per paper |
| Normalise to a reference | SPEC (machine); TauBench (machine) | hardware only |
| "Solved" gate | HPGMG | exists, no credit for lower error |
| Geomean, versioned suite, run rules | SPEC, STREAM | exists |
| Cross-mesh comparison of error | Lecoanet 2016; McNally 2012; Bauer & Springel 2012 | methods exist, not standardised |

Apparently new (INFERENCE): reference SCHEME work as the unit; a versioned W_ref(eps)
table; output in cell updates per second; the combination across mesh types.

## 8. Copy and avoid

Copy
- Fixed quality target, then time: MLPerf.
- Reference is a definition, not a performance claim: MLPerf.
- Versioned suite, reference ratio, median of runs, published per-test values: SPEC.
- Base (fixed rules) plus peak (tuned): SPEC.
- Verification harness and rejection right: HPGMG. "Tuned" label: STREAM.
- Report the full curve, not only the score: Bari, workshop practice.
- Published uncertainty on reference solutions: McNally 2012.
- Tests with converged reference solutions and passive scalar norm: Lecoanet 2016.
- Evaluate at two eps: justified by Bauer & Springel 2012 ("resolution and problem
  dependent") and Greenough & Rider.
- Cover both regimes: Greenough & Rider show ranking reverses between smooth and shock.
- Pin down "update": Stone 2020 vs AGILE show drift. Define per full step, all stages.
- Count everything in T, including mesh build and setup rule: MLPerf timing rules.

Avoid
- Normalising hardware only (TauBench): no GPU version existed.
- Mean hides reversals: Hoefler & Belli example (geomean 0.29 misleading); publish
  per-test values and consider performance profiles: Dolan & More.
- Suite composition is the weighting: MLPerf's reason for no summary score. Smooth
  tests can dominate via large ratios (Schaal >10x; Greenough "enormous").
- Failed tests: decide up front. Pennycook sets 0. MLPerf says omissions weaken a
  summary. INFERENCE: geomean cannot absorb a zero, so a failure rule is required.
- Effective Re as the accuracy: Lecoanet ("extremely problem dependent").
- Small-scale structure as accuracy proxy: Lecoanet.
- Pointwise error in ILES turbulence: Kritsuk footnote, Grete 2023 (no convergence).
- Unweighted per-cell L1 across meshes: Springel/Hopkins form. INFERENCE: with equal-mass
  cells it behaves like a mass-weighted norm, on a fixed grid like a volume-weighted
  one, so it rewards different things. Use one integral norm for all codes.
- Lossy remap in the norm: VorAMR few-percent errors would set an eps floor.
- Cross-machine CPU time: Tasker ("fraught"), Kitsionas ("factors of a few").
- Comparing unequally tuned solvers without saying so: Bauer+ 2016.
- Sub-linear contact convergence, p/(p+1): Banks, Aslam & Rider 2008, JCP 227 [A/S]. Do
  not extrapolate W_ref with a smooth-flow slope on contact tests.
- Moving-mesh-favourable bias: fixed mesh wins on stationary flow (Hopkins), and about
  5x per element (Bauer & Springel); RICH static-mesh result [S].

## 9. Name collisions

- "ECUPS", "eCUPS": none found.
- "CUPS": established in DNA alignment (GCUPS) [S]; earlier neural-net "connection
  updates" (Wikipedia SUPS) [S]; MLUPS in lattice Boltzmann [S].
- "Effective" carries fixed meanings here: AMR effective resolution (finest-level
  equivalent) and effective Reynolds number. INFERENCE: "effective cell updates" may
  be read as finest-level-equivalent cell updates of an AMR run, which is exactly not
  our meaning. "Reference-equivalent cell updates per second" avoids that.

## 10. Not done

No 2025-26 proposal for a standard astro hydro accuracy-per-cost suite was found,
in two searches. A check of ADS full text for "cost to reach" and "equivalent error"
would strengthen the negative claim before publication.
