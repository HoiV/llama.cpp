#if defined(_WIN32)

//
// This file contains common affinity, core parking, cache enumeration code. It is included
// in zo-ggml\ggml.c and za-ggml\ggml.c.
//

#include <powrprof.h>

void xb_disable_core_parking(void)
{
#if WIN32_POWERPROF // UNDONE: powerprof.lib

    GUID* activeScheme = NULL;
    DWORD acValue = 100; // Set to 100% to disable core parking
#if 0
    DWORD dcValue = 100; // Set to 100% to disable core parking
#endif // #if 0

    //
    // Retrieve the active power scheme.
    //

    if (PowerGetActiveScheme(NULL, &activeScheme) != ERROR_SUCCESS) {
        printf("Failed to get the active power scheme.\n");
        return;
    }

    //
    // Processor Performance Core Parking Minimum Cores (AC).
    //

    GUID SUB_PROCESSOR = GUID_PROCESSOR_SETTINGS_SUBGROUP;
    GUID CORE_PARK_MIN_CORES = GUID_PROCESSOR_CORE_PARKING_MIN_CORES;

    if (PowerWriteACValueIndex(NULL, activeScheme, &SUB_PROCESSOR, &CORE_PARK_MIN_CORES, acValue) != ERROR_SUCCESS) {
        printf("Failed to set core parking minimum cores (AC).\n");
        return;
    }

#if 0
    //
    // Processor Performance Core Parking Minimum Cores (DC).
    //

    if (PowerWriteDCValueIndex(NULL, activeScheme, &SUB_PROCESSOR, &CORE_PARK_MIN_CORES, dcValue) != ERROR_SUCCESS) {
        printf("Failed to set core parking minimum cores (DC).\n");
        return;
    }
#endif // #if 0

    //
    // Apply the updated settings.
    //

    if (PowerSetActiveScheme(NULL, activeScheme) != ERROR_SUCCESS) {
        printf("Failed to apply the power scheme.\n");
        return;
    }

    //
    // Clean up.
    //

    LocalFree(activeScheme);

#endif // #if WIN32_POWERPROF

    printf("Core parking disabled successfully.\n");
}

uint64_t l1d_cache_size = 48ull * 1024ull;
uint64_t l1i_cache_size = 32ull * 1024ull;
uint64_t l2_cache_size = 1024ull * 1024ull;
uint64_t l3_cache_size = 1024ull * 1024ull;

typedef struct {
    uint64_t mask;
    uint16_t group;
    uint16_t reserved[3];
} group_affinity_t;

ULONG master_index = 0;
uint32_t maximum_logical = 0;

bool
xb_set_thread_affinity (
    uint32_t ith,
    uint64_t * affinity
    )

{

    //
    // Set the affinity of the current thread if the maximum number of logical
    // processors is less than or equal to 64, i.e., one affinity group.
    //

    if (maximum_logical <= 64) {
        uint32_t index = master_index + (2 * ith);

        if (SetThreadAffinityMask(GetCurrentThread(), 1ull << index)) {
            *affinity = SetThreadAffinityMask(GetCurrentThread(), 1ull << index);
            return true;

        } else {
            printf("failed to set thread affinity\n");
        }
    }

    return false;
}

char * ggml_cache_type[4] = {
    "null",
    "data",
    "instruction",
    "unified"
};

void
xb_set_process_affinity (
    uint32_t n_threads,
    uint64_t affinity_mask_requested
    )
{

#if 0 // Require powerprof.lib

    //
    // Disable core parking.
    //

    ggml_disable_core_parking();

#endif

    uint64_t affinity_mask = affinity_mask_requested;

    if (affinity_mask_requested != 0) {
        goto set_affinity;
    }

    //
    // Get the default rounding mode.
    //

    char * default_mode = "none";

    uint32_t mxcsr = _mm_getcsr();

    uint32_t round_mode = mxcsr & _MM_ROUND_MASK;

    switch (round_mode) {
    case _MM_ROUND_NEAREST:
        default_mode = "round nearest";
        break;

    case _MM_ROUND_DOWN:
        default_mode = "round_down";
        break;

    case _MM_ROUND_UP:
        default_mode = "round_up";
        break;

    case _MM_ROUND_TOWARD_ZERO:
        default_mode = "round_toward_zero";
        break;

    }

    printf("mxcsr 0x%08x, default rounding mode - %s\n", mxcsr, default_mode);

    //
    // Get cpuid information.
    //

    struct {
        uint32_t eax;
        uint32_t ebx;
        uint32_t ecx;
        uint32_t edx;
    } cpu_info;

    //
    // Get avx features for the current system.
    //

    printf("host system AVX capabilities:\n");
    __cpuid((int *)&cpu_info, 0x00000001);
    printf("  cpuid function 0x00000001\n");  
    if (cpu_info.ecx & (1 << 28)) {
        printf("    Avx\n");
    }

    __cpuidex((int *)&cpu_info, 0x00000007, 0);
    printf("  cpuidex function 0x00000007, subleaf 0\n");
    if (cpu_info.ebx & (1 << 5)) {
        printf("    Avx2\n");
    }

    if (cpu_info.ebx & (1 << 16)) {
        printf("    Avx512F\n");
    }

    if (cpu_info.ebx & (1 << 17)) {
        printf("    Avx512DQ\n");
    }

    if (cpu_info.ebx & (1 << 21)) {
        printf("    Avx512Ifma\n");
    }

    if (cpu_info.ebx & (1 << 27)) {
        printf("    Avx512CD\n");
    }

    if (cpu_info.ebx & (1 << 29)) {
        printf("    Avx512BW\n");
    }

    if (cpu_info.ebx & (1 << 30)) {
        printf("    Avx512VL\n");
    }

    if (cpu_info.ecx & (1 << 1)) {
        printf("    Avx512Vbmi\n");
    }

    if (cpu_info.ecx & (1 << 6)) {
        printf("    Avx512Vbmi2\n");
    }

    if (cpu_info.ecx & (1 << 11)) {
        printf("    Avx512Vnni\n");
    }

    if (cpu_info.ecx & (1 << 12)) {
        printf("    Avx512Bitalg\n");
    }

    if (cpu_info.ecx & (1 << 14)) {
        printf("    Avx512Vpopcntdq\n");
    }

    if (cpu_info.edx & (1 << 8)) {
        printf("    Avx512Vp2Intersect\n");
    }

    if (cpu_info.edx & (1 << 23)) {
        printf("    Avx512FP16\n");
    }

    __cpuidex((int *)&cpu_info, 0x00000007, 1);
    printf("  cpuidex function 0x00000007, subleaf 1\n");
    if (cpu_info.eax & (1 << 4)) {
        printf("    AvxVnni\n");
    }

    if (cpu_info.eax & (1 << 5)) {
        printf("    Avx512Bfloat16\n");
    }

    if (cpu_info.eax & (1 << 23)) {
        printf("    AvxIfma\n");
    }

    if (cpu_info.edx & (1 << 4)) {
        printf("    AvxVnniInt8\n");
    }

    if (cpu_info.edx & (1 << 5)) {
        printf("    AvxNeConvert\n");
    }

    if (cpu_info.edx & (1 << 9)) {
        printf("    AvxVnniInt16\n");
    }

    if (cpu_info.edx & (1 << 19)) {
        printf("    Avx10\n");
    }

    printf("\n");

    //
    // Get L1 instruction and data cache attributes.
    //

    __cpuid((int *)&cpu_info, 0x80000005);

//    printf("l1 d-cache line size %d\n", cpu_info.ecx & 0xff);
//    printf("l1 d-cache lines per tag %d\n", (cpu_info.ecx >> 8) & 0xff);
//    printf("l1 d-cache associativity %d\n", (cpu_info.ecx >> 16) & 0xff);

    l1d_cache_size = ((cpu_info.ecx >> 24) & 0xff) * 1024ull;

//    printf("l1 i-cache line size %d\n", cpu_info.edx & 0xff);
//    printf("l1 i-cache lines per tag %d\n", (cpu_info.edx >> 8) & 0xff);
//    printf("l1 i-cache associativity %d\n", (cpu_info.edx >> 16) & 0xff);

    l1i_cache_size = ((cpu_info.edx >> 24) & 0xff) * 1024ull;

    __cpuidex((int *)&cpu_info, 0x8000001d, 0);

    const int32_t l1d_cache_type = cpu_info.eax & 0x3;
    const int32_t l1d_cache_sharing = ((cpu_info.eax >> 14) & 0xfff) + 1;

    __cpuidex((int *)&cpu_info, 0x8000001d, 1);

    const int32_t l1i_cache_type = cpu_info.eax & 0x3;
    const int32_t l1i_cache_sharing = ((cpu_info.eax >> 14) & 0xfff) + 1;

    printf("l1 d-cache size %zdkb, type - %s, SMT sharing %d\n",
           l1d_cache_size / 1024,
           ggml_cache_type[l1d_cache_type],
           l1d_cache_sharing);

    printf("l1 i-cache size %zdkb, type - %s, SMT sharing %d\n",
           l1i_cache_size / 1024,
           ggml_cache_type[l1i_cache_type],
           l1i_cache_sharing);

    //
    // Get l2 cache information.
    //

    __cpuid((int *)&cpu_info, 0x80000006);

    l2_cache_size = ((cpu_info.ecx >> 16) & 0xffff) * 1024ull;

    __cpuidex((int *)&cpu_info, 0x8000001d, 2);

    const int32_t l2_cache_type = cpu_info.eax & 0x3;
    const int32_t l2_cache_sharing = ((cpu_info.eax >> 14) & 0xfff) + 1;

    printf("l2 cache size %zdkb, type - %s. SMT sharing %d\n",
           l2_cache_size / 1024,
           ggml_cache_type[l2_cache_type],
           l2_cache_sharing);

    //
    // Get l3 cache information.
    //

    __cpuidex((int *)&cpu_info, 0x8000001d, 3);

    uint32_t line_size = (cpu_info.ebx & 0xfff) + 1;
    uint32_t partitions = ((cpu_info.ebx >> 12) & 0x3ff) + 1;
    uint32_t associativity = ((cpu_info.ebx >> 22) & 0x3ff) + 1;
    uint32_t sets = cpu_info.ecx + 1;

//    printf("l3 line size %d\n", line_size);
//    printf("l3 partitions %d\n", partitions);
//    printf("l3 associativity %d\n", associativity);
//    printf("l3 sets %d\n", sets);

    l3_cache_size = line_size * partitions * associativity * sets;
    const int32_t l3_cache_type = cpu_info.eax & 0x3;
    const int32_t l3_cache_sharing = ((cpu_info.eax >> 14) & 0xfff) + 1;

    printf("l3 cache size %zdmb, type - %s, SMT sharing %d\n\n",
           l3_cache_size / (1024 * 1024),
           ggml_cache_type[l3_cache_type],
           l3_cache_sharing);

    //
    // Get logical processors per core and maximum logical processsors.
    //

    __cpuid((int *)&cpu_info, 0x8000001e);
    const uint32_t logical_per_core = ((cpu_info.ebx & 0x300) >> 8) + 1;

    __cpuid((int *)&cpu_info, 0x00000001);
    maximum_logical = (cpu_info.ebx & 0xff0000) >> 16;

    //
    // Compute the total l1, l2, and l3 cache.
    //

    float total_l1_cache = (float)(l1d_cache_size + l1i_cache_size);
    total_l1_cache *= (float)(maximum_logical / l1d_cache_sharing);
    total_l1_cache /= (1024.f * 1024.f);
    printf("total l1 cache %6.1fmb\n", total_l1_cache);

    float total_l2_cache = (float)(l2_cache_size);
    total_l2_cache *= (float)(maximum_logical / l2_cache_sharing);
    total_l2_cache /= (1024.f * 1024.f);
    printf("total l2 cache %6.1fmb\n", total_l2_cache);

    float total_l3_cache = (float)(l3_cache_size);
    total_l3_cache *= (float)(maximum_logical / l3_cache_sharing);
    total_l3_cache /= (1024.f * 1024.f);
    printf("total l3 cache %6.1fmb\n\n", total_l3_cache);

    printf("n_threads specified %d\n", n_threads);
    printf("logical processors per core %d\n", logical_per_core);
    printf("maximum logical processors %d\n", maximum_logical);

    //
    // Check the logical processors per core.
    //

    if (logical_per_core == 1) {
        printf("bypassing set process affinity - not SMT system\n");
        return;
    }

    //
    // Check the specified number of threads against the maximum logical processor count.
    //

    const uint32_t maximum_smt_threads = maximum_logical / 2;
    if ((n_threads & 1) || (n_threads > maximum_smt_threads)) {
        printf("bypassing set process affinity - number threads odd or gt maximum logical / 2\n");
        return;
    }

    //
    // Get the current process group count.
    //

#if 0
    uint16_t group_array[4];
    uint16_t group_count = 4;

    if (GetProcessGroupAffinity(GetCurrentProcess(), &group_count, group_array)) {
        printf("GetProcessGroupAffinity succeeded with %d groups\n", group_count);
        if (group_count != 1) {
            printf("bypassing set affinity process because group count is greater than one\n");
            return;
        }

    } else {
        printf("GetProcessGroupAffinity failed\n");
        return;
    }
#endif // #if 0

    //
    // Set process affinity.
    //

    affinity_mask = ((1ull << (n_threads * 2)) - 1) & 0x55555555ull;

    //
    // It is known that the number of threads fits within the maximum smt set. If the
    // maximum smt set is less than or equal to 32, then the threads can be pushed
    // up to higher numbered threads which will remove them from contention issues
    // with clock and device interrupts.
    //

    if (maximum_smt_threads <= 32) {

        //
        // Compute the shift up such that the thread affinity straddles CCDs.
        //

        uint32_t half_shift = maximum_logical - (n_threads * 2);

        half_shift = ((half_shift / 2) + 1) & 0x1e;

        affinity_mask <<= half_shift;
    }

    set_affinity:
    if (SetProcessAffinityMask(GetCurrentProcess(), affinity_mask)) {
        printf("process group affinity set to 0x%016llx\n", affinity_mask);

        //
        // Compute the processor index of the master thread.
        //

        BitScanForward64(&master_index, affinity_mask);
        printf("processor index of master thread %lu\n", master_index);

#if 0
        //
        // Attempt to set the affinity of the master thread.
        //

        uint64_t master_affinity;

        if (xb_set_thread_affinity(0, &master_affinity)) {
            printf("master thread affinity set to 0x%016llx\n", master_affinity);
        }
#endif // #if 0


    } else {
        printf("failed to set process affinity mask\n");
    }

    return;
}

#else

#define xb_set_process_affinity(n, a)

#endif // _WIN32
