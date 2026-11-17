// iOS entry point: runs sd-bench ("bench" as the first argument), a CoreML precompile ("compile"), or sd-cli
// with the launch arguments,
// from the app's Documents directory, then prints the peak memory footprint and exits.
//
// The work runs on a background thread after the app has finished launching: iOS kills an app that
// has not finished launching within 20 s (watchdog 0x8BADF00D), and a first-time Neural Engine
// compile of a large CoreML model (SD 2.1's UNet) takes minutes.
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach/mach.h>
#include <os/proc.h>

#import <CoreML/CoreML.h>

int sd_cli_main(int argc, const char* argv[]);

// "compile <model.mlmodelc>": loads a CoreML model for the Neural Engine and exits, so iOS compiles it and
// caches the result. Compiling a large model (SD 2.1's UNet) takes about 100 s and over 3 GB at its peak;
// done in its own launch, that peak does not stack on top of the rest of the pipeline.
static int compile_main(int argc, const char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s compile <model.mlmodelc>\n", argv[0]);
        return 1;
    }
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    config.computeUnits          = MLComputeUnitsCPUAndNeuralEngine;
    NSError* error               = nil;
    NSDate* t0                   = [NSDate date];
    MLModel* model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]]
                                       configuration:config
                                               error:&error];
    if (model == nil) {
        fprintf(stdout, "[compile] failed: %s\n", error.localizedDescription.UTF8String);
        return 1;
    }
    fprintf(stdout, "[compile] %s loaded for the Neural Engine in %.1f s\n", argv[2], -[t0 timeIntervalSinceNow]);
    return 0;
}
int sd_bench_main(int argc, const char* argv[]);

static int g_argc;
static const char** g_argv;

// Copies everything written to stdout and stderr into Documents/sdbench.log as well as the console,
// so a run can be read back with `devicectl device copy from` when the console relay drops output.
static void tee_output_to_log(NSString* docs) {
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) {
        return;
    }
    int console = dup(STDOUT_FILENO);
    FILE* log   = fopen([docs stringByAppendingPathComponent:@"sdbench.log"].UTF8String, "w");
    dup2(pipe_fds[1], STDOUT_FILENO);
    dup2(pipe_fds[1], STDERR_FILENO);
    close(pipe_fds[1]);
    int read_fd = pipe_fds[0];
    [NSThread detachNewThreadWithBlock:^{
        char buf[4096];
        ssize_t n;
        while ((n = read(read_fd, buf, sizeof(buf))) > 0) {
            write(console, buf, n);
            if (log) {
                fwrite(buf, 1, n, log);
                fflush(log);
            }
        }
    }];
}

// With SDBENCH_MEMLOG set, logs the app's memory every second, to see how close a run gets to the limit.
static void start_memory_log() {
    if (getenv("SDBENCH_MEMLOG") == nullptr) {
        return;
    }
    [NSThread detachNewThreadWithBlock:^{
        NSDate* t0 = [NSDate date];
        for (;;) {
            task_vm_info_data_t vm;
            mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
            task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &n);
            fprintf(stdout, "[mem] t=%.0f s footprint=%.0f MB available=%.0f MB\n", -[t0 timeIntervalSinceNow],
                    vm.phys_footprint / 1048576.0, os_proc_available_memory() / 1048576.0);
            sleep(1);
        }
    }];
}

static void run_and_exit() {
    @autoreleasepool {
        NSString* docs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
        chdir(docs.UTF8String);
        tee_output_to_log(docs);
        start_memory_log();
        fprintf(stdout, "[sdbench] cwd=%s avail_mem=%.0f MB\n", docs.UTF8String, os_proc_available_memory() / 1048576.0);
        NSDate* t0 = [NSDate date];
        const char* mode = g_argc > 1 ? g_argv[1] : "";
        int rc           = strcmp(mode, "bench") == 0     ? sd_bench_main(g_argc, g_argv)
                           : strcmp(mode, "compile") == 0 ? compile_main(g_argc, g_argv)
                                                          : sd_cli_main(g_argc, g_argv);
        task_vm_info_data_t vm;
        mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
        task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &n);
        fprintf(stdout, "[sdbench] rc=%d total=%.2f s peak_phys_footprint=%.0f MB\n", rc, -[t0 timeIntervalSinceNow], vm.ledger_phys_footprint_peak / 1048576.0);
        fflush(stdout);
        fflush(stderr);
        usleep(200 * 1000);  // let the tee thread write out the last lines
        exit(rc);
    }
}

// iOS 27 requires the scene lifecycle (UIApplicationSceneManifest in Info.plist): the scene owns the window.
@interface SDBenchSceneDelegate : UIResponder <UIWindowSceneDelegate>
@property(strong, nonatomic) UIWindow* window;
@end

@implementation SDBenchSceneDelegate
- (void)scene:(UIScene*)scene willConnectToSession:(UISceneSession*)session options:(UISceneConnectionOptions*)options {
    self.window                    = [[UIWindow alloc] initWithWindowScene:(UIWindowScene*)scene];
    self.window.rootViewController = [[UIViewController alloc] init];
    [self.window makeKeyAndVisible];
}
@end

@interface SDBenchAppDelegate : UIResponder <UIApplicationDelegate>
@end

@implementation SDBenchAppDelegate
- (BOOL)application:(UIApplication*)application didFinishLaunchingWithOptions:(NSDictionary*)launchOptions {
    // Keep the screen awake so a long benchmark is not suspended halfway.
    application.idleTimerDisabled = YES;
    NSThread* worker        = [[NSThread alloc] initWithBlock:^{ run_and_exit(); }];
    worker.stackSize        = 8 << 20;
    worker.qualityOfService = NSQualityOfServiceUserInitiated;
    [worker start];
    return YES;
}
@end

int main(int argc, const char* argv[]) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    g_argc = argc;
    g_argv = argv;
    @autoreleasepool {
        return UIApplicationMain(argc, (char**)argv, nil, NSStringFromClass([SDBenchAppDelegate class]));
    }
}
