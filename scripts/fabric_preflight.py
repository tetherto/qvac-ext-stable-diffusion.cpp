import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ENGINE_TESTS = ["test-ltx-rope", "test-vae-routing", "test-fit-params", "test-safetensors-convrot", "test-graph-eval-callback"]
OPERATIONS = ["rope", "pad", "im2col3d"]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an external GGML package without changing engine pins")
    parser.add_argument("--ggml-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=["CPU", "CUDA", "Metal"], default="CPU")
    parser.add_argument("--mode", choices=["package", "diagnostic"], default="package")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cuda-architectures", default="native")
    parser.add_argument("--installed-prefix", type=Path, help="Reuse a matching package for local diagnosis")
    return parser.parse_args()


def revision(source):
    return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()


class Preflight:
    def __init__(self, args):
        self.args = args
        self.registry = "MTL" if args.backend == "Metal" else args.backend
        self.engine = Path(__file__).resolve().parents[1]
        self.source = args.ggml_source.resolve()
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.prefix = args.installed_prefix.resolve() if args.installed_prefix else self.output / "install"
        self.env = os.environ.copy()
        self.env.pop("GGML_BACKEND_PATH", None)
        self.env.pop("CPLUS_INCLUDE_PATH", None)
        self.env.pop("CPATH", None)
        self.results = {"engine": revision(self.engine), "ggml": revision(self.source),
                        "backend": args.backend, "mode": args.mode, "platform": platform.platform(),
                        "reused_package": bool(args.installed_prefix), "checks": []}

    def run(self, name, command, cwd=None, env=None):
        with (self.output / (name + ".log")).open("w") as log:
            result = subprocess.run([str(x) for x in command], cwd=cwd, env=env or self.env,
                                    stdout=log, stderr=subprocess.STDOUT)
        self.results["checks"].append({"name": name, "command": [str(x) for x in command],
                                       "returncode": result.returncode})
        self.save()
        print(name, result.returncode, flush=True)
        return result.returncode == 0

    def save(self):
        (self.output / "results.json").write_text(json.dumps(self.results, indent=2) + "\n")

    def configure(self, name, source, build, options):
        return self.run(name, ["cmake", "-S", source, "-B", build, "-G", "Ninja",
                               "-DCMAKE_BUILD_TYPE=Release", *options])

    def build(self, name, build, targets):
        return self.run(name, ["cmake", "--build", build, "--parallel", str(self.args.jobs),
                               "--target", *targets])

    def install_ggml(self):
        if self.args.installed_prefix:
            return True
        build = self.output / "ggml-build"
        options = ["-DBUILD_SHARED_LIBS=ON", "-DGGML_BACKEND_DL=ON", "-DGGML_NATIVE=OFF",
                   "-DGGML_OPENMP=OFF", "-DGGML_CPU_ALL_VARIANTS=OFF", "-DGGML_MAX_NAME=128",
                   "-DGGML_BUILD_TESTS=OFF", "-DGGML_BUILD_EXAMPLES=OFF",
                   "-DGGML_CUDA=" + ("ON" if self.args.backend == "CUDA" else "OFF"),
                   "-DGGML_METAL=" + ("ON" if self.args.backend == "Metal" else "OFF"),
                   "-DCMAKE_INSTALL_LIBDIR=lib", "-DCMAKE_INSTALL_PREFIX=" + str(self.prefix)]
        if self.args.backend == "CUDA":
            options.append("-DCMAKE_CUDA_ARCHITECTURES=" + self.args.cuda_architectures)
        return (self.configure("ggml-configure", self.source, build, options)
                and self.build("ggml-build", build, ["all"])
                and self.run("ggml-install", ["cmake", "--install", build]))

    def prepare_runtime(self):
        lib = str(self.prefix / "lib")
        variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        self.env[variable] = lib + os.pathsep + self.env.get(variable, "")

    def check_operations(self):
        build = self.output / "probes"
        if not (self.configure("probes-configure", self.engine / "tests/fabric-preflight", build,
                               ["-DCMAKE_PREFIX_PATH=" + str(self.prefix)])
                and self.build("probes-build", build, ["all"])):
            return
        self.run("discovery", [build / "test_backend_discovery", self.prefix / "lib", self.registry])
        if self.args.backend == "CUDA":
            self.run("cpu-with-gpu-hidden", [build / "test_backend_discovery", self.prefix / "lib", "CPU"],
                     env={**self.env, "CUDA_VISIBLE_DEVICES": "-1"})
        self.run_operations(build)

    def run_operations(self, build):
        for operation in OPERATIONS:
            self.run("op-" + operation, [build / "test_diffusion_ops", self.registry, operation],
                     cwd=self.prefix / "lib")

    def check_engine(self):
        build = self.output / "engine-build"
        options = ["-DSD_USE_SYSTEM_GGML=ON", "-Dggml_DIR=" + str(self.prefix / "share/ggml"),
                   "-DSD_BUILD_TESTS=ON", "-DSD_TEST_GGML_SOURCE_DIR=" + str(self.source), "-DSD_BUILD_EXAMPLES=ON", "-DSD_WEBP=OFF", "-DSD_WEBM=OFF"]
        if self.args.mode == "diagnostic":
            overlay = self.output / "private-headers"
            overlay.mkdir(exist_ok=True)
            link = overlay / "ggml"
            if not link.exists():
                link.symlink_to(self.source, target_is_directory=True)
            options.append("-DCMAKE_CXX_FLAGS=-I" + shlex.quote(str(overlay)))
            include = overlay / "include"
            include.mkdir(exist_ok=True)
            self.env["CPLUS_INCLUDE_PATH"] = str(include)
        if not self.configure("engine-configure", self.engine, build, options):
            return
        if not self.build("engine-build", build, ["sd-cli", *ENGINE_TESTS]):
            return
        self.run("engine-tests", ["ctest", "--test-dir", build, "--output-on-failure", "-R",
                                  "^(" + "|".join(ENGINE_TESTS) + ")$"],
                 env={**self.env, "GGML_BACKEND_PATH": str(self.cpu_module())})

        if self.args.backend != "CPU":
            self.run("engine-callback-gpu", [build / "bin/test-graph-eval-callback", self.registry],
                     cwd=self.prefix / "lib")

    def cpu_module(self):
        modules = list((self.prefix / "lib").glob("*ggml-cpu.so")) + list((self.prefix / "lib").glob("*ggml-cpu.dylib"))
        if len(modules) != 1:
            raise RuntimeError("Expected exactly one dynamic CPU module")
        return modules[0]

    def execute(self):
        self.save()
        if self.install_ggml():
            self.prepare_runtime()
            self.check_operations()
            self.check_engine()
        return 0 if self.results["checks"] and all(x["returncode"] == 0 for x in self.results["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(Preflight(parse_args()).execute())
