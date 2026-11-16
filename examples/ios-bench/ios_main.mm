// iOS entry point: runs sd-bench ("bench" as the first argument) or sd-cli with the launch arguments,
// from the app's Documents directory, then prints the peak memory footprint.
#import <Foundation/Foundation.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <mach/mach.h>
#include <os/proc.h>
int sd_cli_main(int argc, const char* argv[]);
int sd_bench_main(int argc, const char* argv[]);
int main(int argc, const char* argv[]) {
    @autoreleasepool {
        NSString* docs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
        chdir(docs.UTF8String);
        setvbuf(stdout, NULL, _IOLBF, 0);
        fprintf(stdout, "[sdbench] cwd=%s avail_mem=%.0f MB\n", docs.UTF8String, os_proc_available_memory() / 1048576.0);
        NSDate* t0 = [NSDate date];
        int rc = (argc > 1 && strcmp(argv[1], "bench") == 0) ? sd_bench_main(argc, argv) : sd_cli_main(argc, argv);
        task_vm_info_data_t vm;
        mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
        task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &n);
        fprintf(stdout, "[sdbench] rc=%d total=%.2f s peak_phys_footprint=%.0f MB\n", rc, -[t0 timeIntervalSinceNow], vm.ledger_phys_footprint_peak / 1048576.0);
        return rc;
    }
}
