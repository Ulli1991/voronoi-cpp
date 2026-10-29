# Stage splitting, queue-based kernels, branch-free formulations, and thread compaction/regrouping for divergent GPU kernels

Scope: (4) wavefront, stage-splitting and queue kernels compared with megakernels; (6) branch-free, predicated and fixed-trip-count formulations, plus hardware and software thread compaction and regrouping. The last section applies these to our Voronoi cell-clipping kernel (Kokkos/CUDA, A100, FP64).

Source-quality labels: [PR] = peer-reviewed; [VENDOR] = vendor document or forum; [PREPRINT] = arXiv, not reviewed; [BLOG] = independent technical blog. For [PR*] I quote the abstract or a search summary and did not read the full text in this session.

---

## Q1. How much does stage splitting improve SIMD efficiency in practice, and when does the state-spill cost outweigh it?

### Takeaway
Laine et al. measured the case directly. On a Kepler K20, wavefront path tracing raised thread utilization from 23% to 53% in a scene with divergent materials, and gave a 36% speedup even in the simplest scene. It paid off because each stage did a lot of work per item (ray casts, material shading) compared with a small 212 B path state stored in SOA layout. Leaner register use for the ray-cast kernel mattered as much as the divergence reduction. Stage splitting stops paying off when per-stage work is small compared with the state each item must write and re-read. That is our situation: about 1.7 KB of state per cell against a few dozen flops per cut.

### Cited findings
- **Mechanism (Laine, Karras & Aila, HPG 2013).**
  - The path tracer is split into a "logic" kernel that runs over all paths, plus "new path", per-material and ray-cast (extension and shadow) kernels.
  - Paths flow between kernels through queues in global memory. The paper describes them as "simple preallocated global memory buffers sized so that they can contain indices of every path in the pool", each with an atomic item counter.
  - Queue pushes are coalesced within each warp using warp-wide ballot and popc before a single atomic per warp. This also keeps paths from one warp contiguous in the queue.
  - Expensive materials each get their own kernel; cheap materials share one.
  - [PR] [Laine, Karras, Aila 2013, "Megakernels Considered Harmful: Wavefront Path Tracing on GPUs", HPG '13, DOI 10.1145/2492045.2492060](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- **Hardware and state size (same paper).**
  - Hardware: NVIDIA Tesla K20 (GK110 Kepler, 5 GB).
  - Pool of 1M (2^20) paths alive at all times. Path state is 212 bytes per path, including extension and shadow rays and space for ray-cast results, so 212 MB in total. Each queue takes 4 MB.
  - Growing the pool from 1M to 8M paths (1.7 GB) gave only about 5% more performance. [Laine 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- **SIMD efficiency, exact numbers (same paper).**
  - In the CONFERENCE scene, Nsight showed megakernel thread utilization of only 23%. The wavefront variant reached 53% overall.
  - Per kernel: 60% in logic, 99% in new-path generation, 71% in materials (lowered by a variable iteration count in a Mandelbrot shader) and 35% in ray casts.
  - Ray casts remained 44% to 56% of the time. [Laine 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- **Speedup and where it came from (same paper).**
  - Even in the "very simple" CARPAINT scene the wavefront version was 36% faster.
  - The authors say: "The overhead of storing the path state in GPU memory is more than compensated for by the faster ray casts enabled by running the ray cast kernels with low register counts and hence better latency hiding capability, while the entire ray cast code fits comfortably in the instruction caches."
  - Speedups were higher in the scenes with several materials.
  - I could not extract the per-scene Table 1 numbers from the PDF; see Gaps. [Laine 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- **Memory layout cost (same paper).** SOA layout of path state gave an 80% total speedup over AOS. Per kernel the gains were: logic +147%, new-path +790% ("presumably due to high number of memory writes"), with material kernels also faster. This shows that the cost of storing state between stages depends heavily on coalescing. [Laine 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- **Refill inside a megakernel (same paper).** Regenerating terminated paths inside the megakernel "whenever more than half of the threads in a warp are idling" gave about a 15% improvement. Regenerating too often lowers coherence and runs the initialisation code at low utilisation. [Laine 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- **Later, modest result.**
  - Padilla, Webster & Kim (Utah, arXiv May 2026) compared a Vulkan megakernel with a wavefront path tracer on an RTX 3060 Ti. Wavefront ran at 73.6 FPS against 64.7 FPS, about 16% faster.
  - Wavefront VRAM throughput was 41.4% against 19.3%, and L2 throughput 22.5% against 15.8%. SM throughput was slightly lower for wavefront (34.1% against 37.1%).
  - The authors attribute the gain to texture and cache coherence. They report no SIMD-efficiency number.
  - [PREPRINT] [Padilla et al. 2026, "Megakernel vs Wavefront GPU Path Tracing", arXiv:2605.27323](https://www.alphaxiv.org/overview/2605.27323v1)
- **Adoption.** Blender Cycles uses wavefront path tracing on the GPU and cites Laine 2013 as the main reference. [VENDOR/OSS docs] [Blender Developer Docs, Cycles Kernel Scheduling](https://developer.blender.org/docs/features/cycles/kernel_scheduling/)
- **Hardware reordering instead of splitting kernels.** NVIDIA's SER whitepaper says that reordering threads across execution units "requires migrating their live state through the memory system". Applications "should strive to reduce live state as much as possible": the less live state, the lower the overhead of ReorderThread. Reordering is not advised when the code that follows is trivial (for example shadow or AO hit shaders) or when the rays are already coherent. This is the same state-spill trade-off as wavefront, stated by the vendor. [VENDOR] [NVIDIA, "Shader Execution Reordering" whitepaper v1.0](https://developer.nvidia.cn/sites/default/files/akamai/gameworks/ser-whitepaper.pdf)

### Inferences
- Laine's improvement from 23% to 53% (about 2.3x in lane use) did not turn into a 2.3x speedup. Ray casts stayed at 35% utilisation and dominated the time, so the end-to-end gain is bounded by the least-coherent stage.
- The paper's own analysis credits part of the gain to occupancy: register-lean ray-cast kernels hid latency better. That part of the gain transfers more readily to our kernel than the divergence part.
- A rough rule from these sources: stage splitting wins when (work per item per stage) is much larger than (state bytes moved per item per stage ÷ bandwidth), and when the item pool fits in memory with enough parallelism.
  - Laine moves 212 B per path per bounce against thousands of instructions of traversal and shading.
  - The SER whitepaper's advice ("don't reorder before trivial work") states the same condition.

### Gaps
- I could not extract Laine's per-scene Table 1 timings (Mpaths/s) from the PDF, so speedups for CITY and CONFERENCE are not quoted.
- I found no peer-reviewed measured SIMD-efficiency numbers for pbrt-v4's GPU wavefront renderer or for OptiX-based wavefront renderers. pbrt-v4 is documented as a wavefront design (pbr-book.org 4th ed., "Wavefront Rendering on GPUs"), but I did not fetch it in this session.
- I found no general "kernel fission" study with a quantitative break-even model for state traffic against divergence.

---

## Q2. Which software-only compaction/regrouping techniques work on A100-class GPUs, and with what measured gains?

### Takeaway
Hardware proposals (dynamic warp formation, thread block compaction, intra-warp compaction, multi-path) report average gains of about 20% in simulation. None of them exists on A100. What does work on real hardware (Volta and later, including A100):
- persistent threads with work refill or "regeneration";
- global or shared-memory queues with warp-aggregated atomics (Laine);
- compiler or user-guided "speculative reconvergence", i.e. Loop Merge and Iteration Delay (Damani et al., measured on V100: 10% to 3x SIMT efficiency and speedup);
- thread-data remapping (up to 2.7x);
- shared-memory context stashing (CCC, 1.69x mean).

SER performs the hardware regrouping, but only on Ada or later and only inside ray-tracing pipelines (OptiX, DXR, Vulkan). On A100 it is a no-op.

### Cited findings
**Hardware proposals (simulation only, none in shipping NVIDIA GPUs):**
- **Thread block compaction (TBC).** Warps in a block share a block-wide reconvergence stack, and threads are compacted into new warps at divergent branches. Simulated average speedup was 22% over the per-warp stack baseline and 17% over dynamic warp formation, on divergent CUDA applications, with no penalty for non-divergent ones. [PR] [Fung & Aamodt, "Thread Block Compaction for Efficient SIMT Control Flow", HPCA 2011, DOI 10.1109/HPCA.2011.5749714](https://people.ece.ubc.ca/aamodt/papers/wwlfung.hpca2011.pdf)
- **Dynamic warp formation (DWF).** Fung, Sham, Yuan & Aamodt, MICRO 2007, regroup threads from different warps that share a PC into new warps. TBC improves on it by 17%, per the TBC abstract above. I did not re-verify DWF's own headline number (often quoted as about 20.7% average) in this session; see Gaps. [PR*] [Fung & Aamodt 2011](https://people.ece.ubc.ca/aamodt/papers/wwlfung.hpca2011.pdf)
- **Intra-warp compaction (BCC/SCC).** Execution cycles in divergent workloads fell by up to 42%, 20% on average. The paper defines SIMD efficiency as the ratio of the average number of enabled channels per executed instruction to the average SIMD width. [PR*] [Vaidya et al., "SIMD Divergence Optimization through Intra-Warp Compaction", ISCA 2013](https://class.ece.iastate.edu/tyagi/cpre581/papers/ISCA13Divergence.pdf)
- **Multi-path execution.** Speedups reached 7x on divergent benchmarks, with a harmonic mean of 32%. [PR*] [ElTantawy et al., HPCA 2014](https://people.ece.ubc.ca/aamodt/papers/eltantawy.hpca2014.pdf)
- **SIMD lane permutation.** A related hardware proposal. [PR*] [Rhu & Erez, ISCA 2013](https://lph.ece.utexas.edu/merez/uploads/MattanErez/isca2013_slp.pdf)

**Software-only on real hardware:**
- **Speculative Reconvergence (Damani et al., CGO 2020).**
  - What it is: a user-guided compiler mechanism built on Volta's convergence barriers (independent thread scheduling, BSSY/BSYNC/BREAK). It forces threads that execute common code (loop bodies, common calls) to arrive together at user-chosen reconvergence points.
  - Patterns covered: Loop Merge, where a thread whose inner loop finished pulls its next outer iteration and rejoins the inner loop with the others; Iteration Delay; and common function calls.
  - Results were measured on a real Volta V100 with nvprof. SIMT efficiency and performance improved by 10% to 3x ("speedups of up to 200%"). Workloads were mostly Monte Carlo codes with variable, unpredictable loop lengths, plus OptiX-based apps and MeiyaMD5.
  - A "soft barrier" threshold lets you trade idle lanes against serialised prologue/epilogue. The authors state that "early reconvergence is not guaranteed to improve performance". In their words, "SIMT efficiency improvement serves roughly as an upper bound on speedup."
  - Full unrolling prevents Loop Merge and Iteration Delay, and inlining a common function prevents common-call reconvergence.
  - In an automatic scan of 520 CUDA applications, 75 had SIMT efficiency below about 80%, and 16 showed non-trivial opportunity.
  - [PR] [Damani, Johnson, Stephenson, Yan, Giroux, McKeown, Keckler, CGO 2020, DOI 10.1145/3368826.3377911](https://d1qx31qr3h6wln.cloudfront.net/publications/speculativereconvergence.pdf); [NVIDIA Research page](https://research.nvidia.com/publication/2020-02_speculative-reconvergence-improved-simt-efficiency)
- **Iteration delaying and branch distribution (Han & Abdelrahman, GPGPU-4 2011).** Software-only. Iteration delaying runs the loop iterations that take the majority direction and postpones the others. Branch distribution factors common code out of both branch paths. Synthetic benchmarks improved by up to 30% and 80% respectively, and one real application by 12% and 16%. [PR] [Han & Abdelrahman, "Reducing Branch Divergence in GPU Programs"](https://www.eecis.udel.edu/~cavazos/cisc879-spring2012/papers/a3-han.pdf)
- **On-GPU thread-data remapping (Lin, Wang & Liu, TACO 2018).** Inserted code collects runtime divergence information before target branches and remaps threads to data, possibly several times per kernel. Speedups reached 2.718x on third-party benchmarks, on one NVIDIA and one AMD GPU generation (models not named on the abstract page). [PR*] [Lin et al., TACO 15(3):39, DOI 10.1145/3242089](https://researchwith.stevens.edu/en/publications/on-gpu-thread-data-remapping-for-branch-divergence-reduction/)
- **Collaborative Context Collection (CCC, Khorasani et al., MICRO-48 2015).** Registers of divergent threads are stashed in a per-warp stack in shared memory and restored once full warps of work can be formed. Arithmetic mean speedup was 1.69x across their benchmarks. [PR*] [Khorasani, Gupta, Bhuyan, "Efficient Warp Execution in Presence of Divergence with Collaborative Context Collection", MICRO 2015](http://www.cs.ucr.edu/~gupta/research/Publications/Comp/micro-48.pdf)
- **Active thread compaction for path tracing (Wald, Intel, 2011).** Software compaction of partly filled warps into fewer full ones. Wald reports that "certain inadequacies of today's hardware wipe out most of the achieved gains", leaving speedups of only 12% to 16%. Hardware of that era was Fermi-class. [PR*] [Wald 2011, "Active Thread Compaction for GPU Path Tracing", HPG 2011](https://www.sci.utah.edu/~wald/Publications/2011/PathCompaction/compact.pdf)
- **Path regeneration in a megakernel.** About 15% gain (Laine 2013; see Q1). [PR] [Laine 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)

**Shader Execution Reordering (SER):**
- **Availability.** The API runs on all ray-tracing-capable NVIDIA GPUs, but "on Ada Lovelace GPUs, ReorderThread will perform reordering, while on previous generation GPUs, it will behave as a no-op". It is exposed through DX12/NVAPI, a Vulkan extension and OptiX. It is part of the ray-tracing pipeline (HitObject/ReorderThread in raygen), not a plain CUDA compute intrinsic. [VENDOR] [NVIDIA SER whitepaper](https://developer.nvidia.cn/sites/default/files/akamai/gameworks/ser-whitepaper.pdf)
- **Measured gains.**
  - Cyberpunk 2077 path tracing on an RTX 4070 Ti: DispatchRays went from about 24.1 ms to 18.2 ms (about 24% less time), with roughly 46% more active lanes per warp. [BLOG] [Chips and Cheese 2023](https://chipsandcheese.com/p/shader-execution-reordering-nvidia-tackles-divergence)
  - OptiX MDL sample: 88 to 137 fps. [VENDOR forum] [NVIDIA dev forum](https://forums.developer.nvidia.com/t/optix-advanced-samples-on-github/48410/14)
  - Optical-photon Monte Carlo in an OptiX kernel with one thread per photon and highly variable lifetimes: 16.55x in propagation-kernel cycles and 14.72x end-to-end, with bit-identical output.
    - About 3.2x came from the launch-configuration changes that enabling SER requires, and a further 5.1x from the reorder itself.
    - The gain is attributed to recovering active lanes, with "branch efficiency and memory bandwidth change little".
    - [PREPRINT] [Galgóczi, arXiv:2608.21396, Aug 2026](https://arxiv.org/abs/2608.21396)
- **Volta+ primitives for building software regrouping.** Independent thread scheduling, `__activemask`, `__ballot_sync`, `__match_any_sync`, warp shuffles and cooperative-groups `coalesced_threads()`. These are documented in the [CUDA C++ Programming Guide (warp-level primitives, cooperative groups)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) [VENDOR; not fetched in this session]. Damani et al. note that Volta's convergence optimizer and barriers are what make user-directed reconvergence possible. [PR] [Damani 2020](https://d1qx31qr3h6wln.cloudfront.net/publications/speculativereconvergence.pdf)

### Inferences
- On A100 the realistic options are:
  - (a) persistent threads or warps that pull the next cell from an atomic counter when done (a refill or "Loop Merge" over cells);
  - (b) splitting by difficulty: a fast kernel, then a queue of hard or leftover cells for a second kernel;
  - (c) speculative-reconvergence-style restructuring of the cut loop, so that all lanes run the "process next plane" body together.
  SER is not available: A100 is not Ada, and the kernel is not an OptiX pipeline.
- The largest measured software gains (2.7x, 3x, and about 16x with SER) all come from workloads with heavily variable per-item loop lengths (Monte Carlo transport, photon lifetimes). Our uniform-random-point long tail has that same profile.

### Gaps
- I found no published measurement of warp-shuffle-based in-register compaction on A100 specifically. The Damani results are on V100, which has the same ITS model.
- I did not verify DWF's (MICRO 2007) exact average speedup in this session.
- CCC's hardware and its per-benchmark lane-utilization figures were not verified beyond the search summary.

---

## Q3. Is there evidence that branch-free formulations help geometric kernels with small loops?

### Takeaway
I found little direct peer-reviewed evidence on predication or fixed trip counts in small geometric loops such as polygon or polyhedron clipping. The indirect evidence points both ways:
- Fixed trip counts remove the variable-iteration divergence that causes the losses in Laine and Damani (for example, 71% material utilisation caused by a variable-iteration-count shader).
- Full unrolling or predication raises register pressure and executes work for dead lanes. Damani notes that it also disables reconvergence transforms.

### Cited findings
- Variable per-thread iteration counts lowered utilisation: the materials kernel reached only 71% "lowered by variable iteration count in Mandelbrot shader". [PR] [Laine 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- Divergent loops with "variable and unpredictable length" were the main source of low SIMT efficiency in the workloads Damani et al. studied. Completely unrolled loops cannot use Loop Merge or Iteration Delay. Loop Merge also turns previously warp-uniform loop indices into per-thread registers, which raises register count. [PR] [Damani 2020](https://d1qx31qr3h6wln.cloudfront.net/publications/speculativereconvergence.pdf)
- Branch distribution (factoring common code out of both sides of a branch so that only the small differing part diverges) gave up to 80% on synthetic and 16% on a real application. [PR] [Han & Abdelrahman 2011](https://www.eecis.udel.edu/~cavazos/cisc879-spring2012/papers/a3-han.pdf)

### Inferences (not directly sourced)
- For our kernel, the "test plane against all current vertices" step is a natural fixed-trip, branch-free loop:
  - compute signed distance;
  - ballot or mask in warp-per-cell;
  - early-out if no vertex is clipped (all positive).
  It is already SIMD-friendly in warp-per-cell. The divergent parts are the topology edit (cut) and the number of candidates processed before the security radius certifies the cell.
- Making the candidate loop fixed-trip (always process K = 24 planes, masking the rest) would remove tail divergence. It would waste work on uniform-random cells that certify early, and fail on the cells that need more than K. That is why a fixed-K fast path plus an overflow queue (Q4) looks better.

### Gaps
- I found no peer-reviewed study that measures predicated against branchy polyhedron or polygon clipping on GPUs. The closest relevant literature (GPU Voronoi by clipping, e.g. Ray et al. "Meshless Voronoi on the GPU", 2018) was not covered in this sub-task.

---

## Q4. Applicability to our kernel: split per-cell Voronoi clipping into stages with queues? What would per-cell state cost?

### Takeaway
(Our inference, based on the sources above.) Fine-grained stage splitting is very unlikely to give 2x on A100, because it ships about 1.7 KB of polyhedron state per cell through memory for each cut, against a few hundred flops of work per cut. The stages would be: fetch candidate, test plane, apply cut, finalize.

The literature points to three better levers:
1. **Coarse splitting by difficulty.** A fast kernel handles the common case with small state and high occupancy. Uncertified or overflowing cells are pushed, using warp-aggregated atomics, to a queue for a second "hard-cell" kernel. This is the Laine queue pattern applied once rather than per cut.
2. **Persistent threads or warps with refill.** When a cell certifies, the lane or warp immediately takes the next cell index. This is Loop Merge over cells (Damani: up to 3x on variable-length loops; Laine regeneration: about 15%).
3. **Shrinking the per-cell state** to raise occupancy above 8% or 17%. In Laine's analysis, register- and resource-lean kernels were the main reason the split paid off.

### Cited findings that the analysis rests on
- Laine's split paid off with 212 B per item in SOA layout, and SOA against AOS alone was worth 80%. [PR] [Laine 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- Laine's own split gains were partly due to "low register counts and hence better latency hiding". [PR] [Laine 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- SER and wavefront regrouping cost grows with live state, and reordering is not worth it before trivial work. [VENDOR] [NVIDIA SER whitepaper](https://developer.nvidia.cn/sites/default/files/akamai/gameworks/ser-whitepaper.pdf)
- Loop Merge across outer iterations raises SIMT efficiency, and a soft-barrier threshold trades idle lanes against prologue/epilogue serialisation. [PR] [Damani 2020](https://d1qx31qr3h6wln.cloudfront.net/publications/speculativereconvergence.pdf)
- Queues work best with warp-aggregated atomics (ballot plus popc, one atomic per warp), which also keep items contiguous. [PR] [Laine 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)

### Inferences (quantitative estimates are ours, not from sources)
**1. State cost of per-cut stage splitting.**
- Per-cell state is about 1.7 KB (vertices plus adjacency) plus the candidate cursor and security radius.
- With about 20 cuts per cell and two stages per cut that touch the polyhedron, each round trip reads and writes about 3.4 KB, giving roughly 70 KB of traffic per cell.
- A100 HBM is about 1.5 to 2 TB/s, so state traffic alone caps throughput at roughly 2 to 3 x 10^7 cells/s, unless the live pool fits in the 40 MB L2. That allows only about 24K resident cells at 1.7 KB each, which is too few to fill 108 SMs with useful coherence pools.
- Today DRAM is below 1% and L2 hits are 99%. Stage splitting would turn a compute- and latency-bound kernel into a bandwidth-bound one.
- Pool memory itself is affordable. At Laine-like 1M items it would be about 1.7 GB, against 212 MB for Laine.
- Only coarse splits (one or two queue hand-offs per cell, not one per cut) keep the traffic around 3 to 7 KB per cell, which is negligible.

**2. Where the 2x is likely to come from.**
- Both variants run at low occupancy (8% and 17%) with issue utilisation of 19% and 42%. Lane use is 5.4 of 32 for thread-per-cell and 15.6 of 32 for warp-per-cell.
- In warp-per-cell, half the lanes idle mostly because about 30 vertices per polyhedron do not map onto 32 lanes, and the topology edit is serial. Stage splitting does not fix that.
- Options:
  - **(a) Thread-per-cell with a compact state** (for example, an fp32 or 16-bit adjacency and vertex-index encoding in registers or a smaller shared-memory footprint), combined with persistent refill. This attacks both the 8% occupancy and the long tail.
  - **(b) Sub-warp groups.** In warp-per-cell, use 8- or 16-lane tiles per cell (cooperative-groups `tiled_partition`) so that two to four cells share a warp. This raises lanes per useful vertex and amortises serial edits.
  - **(c) Difficulty split.** A kernel with a fixed cap (for example K ≤ 24 planes and V ≤ 32 vertices, small state) certifies most cells. The rest go through a queue to a big-capacity kernel. The repository already has an overflow-rebuild path ("Rebuild overflowing cells with big capacities"), which is this pattern. Extending the queue criterion from "overflow" to "not certified within K planes" would move the uniform-random long tail out of the fast kernel.
- I would expect (a), (b) and (c) to beat a full wavefront split for this kernel. Nothing in the literature reviewed here establishes a 2x for any of them on a Voronoi kernel; that would have to be measured.

**3. Indicators to watch in Nsight Compute.**
- `smsp__thread_inst_executed_per_inst_executed` (lanes per instruction) for the divergence part.
- Achieved occupancy and stall reasons (`long_scoreboard`, barrier, `no_instruction`) for the latency part.
- If (c) is implemented, check that queue-push atomics are warp-aggregated and that the queue is consumed in Morton order to keep the 99% L2 hit rate.

### Gaps
- I found no published GPU Voronoi or polyhedron-clipping implementation that uses wavefront or queue staging with measured results, so the applicability judgement is our inference.
- The traffic estimate assumes the whole polyhedron is re-read and re-written per stage. A design that sends only plane indices and keeps the polyhedron resident (persistent warp per cell) has no spill but also does no regrouping.
