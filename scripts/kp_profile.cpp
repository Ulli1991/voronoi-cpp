// A Kokkos Tools connector for profiling without touching the code under test (research_notes/profile_2026-11-18.md).
//   KOKKOS_TOOLS_LIBS=libkp_profile.so  KP_NVTX=1  KP_LOG=<file>  <program>
// KP_NVTX=1: an NVTX range per parallel_for/reduce/scan named by its Kokkos label, and per Kokkos profile region, so
//   Nsight Systems and Nsight Compute name each kernel. It asks Kokkos for no global fences (requires_global_fencing =
//   false): the timeline keeps the program's own synchronisation.
// KP_LOG=<file>: one line per event, time in us since start:
//   A <t> <space> <bytes> <ptr> <label> <device bytes live>  allocation
//   D <t> <space> <bytes> <ptr> <label> <device bytes live>  deallocation
//   K <t> <kind> <label>                                     kernel launch (host side)
//   F <t> <label>                                            fence
//   C <t> <dst space> <src space> <bytes> <dst label> <src label>  deep copy
//   P <t> <device peak bytes>                                each new device peak (rate-limited to 1 % steps)
#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
struct SpaceHandle {
    char name[64];
};
struct ToolSettings {
    bool requires_global_fencing;
    bool padding[255];
};

bool nvtx = false;
std::FILE* out = nullptr;
std::mutex lock;
std::uint64_t live = 0, peak = 0, logged_peak = 0;
std::unordered_map<const void*, std::uint64_t> sizes;
const auto t0 = std::chrono::steady_clock::now();

double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(); }
bool device(const SpaceHandle& h) { return std::strcmp(h.name, "Cuda") == 0 || std::strcmp(h.name, "CudaUVM") == 0; }
// KP_STEP=<label>: an NVTX start/end range kp_step_<k> from each launch of that label to the next (at least 5 launches
// apart), so Nsight Compute can pick one step: --nvtx-include kp_step_<k>.
std::string step_label;
int step = 0, since = 1 << 30;
nvtxRangeId_t step_range = 0;
void begin(const char* kind, const char* name) {
    ++since;
    if (nvtx && !step_label.empty() && step_label == name && since >= 5) {
        if (step > 0) nvtxRangeEnd(step_range);
        step_range = nvtxRangeStartA(("kp_step_" + std::to_string(++step)).c_str());
        since = 0;
    }
    if (nvtx) nvtxRangePushA(name);
    if (out) std::fprintf(out, "K\t%.1f\t%s\t%s\n", now_us(), kind, name);
}
void end() {
    if (nvtx) nvtxRangePop();
}
}  // namespace

extern "C" {
void kokkosp_request_tool_settings(const std::uint32_t, ToolSettings* s) { s->requires_global_fencing = false; }
void kokkosp_init_library(const int, const std::uint64_t, const std::uint32_t, void*) {
    nvtx = std::getenv("KP_NVTX") && std::string(std::getenv("KP_NVTX")) == "1";
    if (const char* f = std::getenv("KP_LOG")) out = std::fopen(f, "w");
    if (const char* s = std::getenv("KP_STEP")) step_label = s;
}
void kokkosp_finalize_library() {
    if (!out) return;
    std::fprintf(out, "END\t%.1f\tpeak %llu\n", now_us(), static_cast<unsigned long long>(peak));
    std::fclose(out);
    out = nullptr;
}
void kokkosp_begin_parallel_for(const char* name, const std::uint32_t, std::uint64_t*) { begin("for", name); }
void kokkosp_end_parallel_for(const std::uint64_t) { end(); }
void kokkosp_begin_parallel_reduce(const char* name, const std::uint32_t, std::uint64_t*) { begin("reduce", name); }
void kokkosp_end_parallel_reduce(const std::uint64_t) { end(); }
void kokkosp_begin_parallel_scan(const char* name, const std::uint32_t, std::uint64_t*) { begin("scan", name); }
void kokkosp_end_parallel_scan(const std::uint64_t) { end(); }
void kokkosp_push_profile_region(const char* name) {
    if (nvtx) nvtxRangePushA(name);
    if (out) std::fprintf(out, "R\t%.1f\t%s\n", now_us(), name);
}
void kokkosp_pop_profile_region() {
    if (nvtx) nvtxRangePop();
    if (out) std::fprintf(out, "r\t%.1f\n", now_us());
}
void kokkosp_begin_fence(const char* name, const std::uint32_t, std::uint64_t*) {
    if (out) std::fprintf(out, "F\t%.1f\t%s\n", now_us(), name);
}
void kokkosp_end_fence(const std::uint64_t) {}
void kokkosp_allocate_data(const SpaceHandle h, const char* label, const void* ptr, const std::uint64_t size) {
    if (!out) return;
    std::lock_guard<std::mutex> g(lock);
    if (device(h)) {
        live += size;
        sizes[ptr] = size;
        if (live > peak) peak = live;
    }
    std::fprintf(out, "A\t%.1f\t%s\t%llu\t%p\t%s\t%llu\n", now_us(), h.name, static_cast<unsigned long long>(size), ptr, label,
                 static_cast<unsigned long long>(live));
    if (peak > logged_peak + logged_peak / 100) {
        logged_peak = peak;
        std::fprintf(out, "P\t%.1f\t%llu\n", now_us(), static_cast<unsigned long long>(peak));
    }
}
void kokkosp_deallocate_data(const SpaceHandle h, const char* label, const void* ptr, const std::uint64_t size) {
    if (!out) return;
    std::lock_guard<std::mutex> g(lock);
    if (device(h) && sizes.erase(ptr)) live -= size;
    std::fprintf(out, "D\t%.1f\t%s\t%llu\t%p\t%s\t%llu\n", now_us(), h.name, static_cast<unsigned long long>(size), ptr, label,
                 static_cast<unsigned long long>(live));
}
void kokkosp_begin_deep_copy(SpaceHandle dst, const char* dst_label, const void*, SpaceHandle src, const char* src_label,
                             const void*, std::uint64_t size) {
    if (out)
        std::fprintf(out, "C\t%.1f\t%s\t%s\t%llu\t%s\t%s\n", now_us(), dst.name, src.name, static_cast<unsigned long long>(size),
                     dst_label[0] ? dst_label : "-", src_label[0] ? src_label : "-");
}
void kokkosp_end_deep_copy() {}
}
