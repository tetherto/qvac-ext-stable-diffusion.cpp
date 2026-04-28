.PHONY: build clean install test help

RELEASE_DIR ?= build
BUILD_TYPE ?= Release
CMAKE_FLAGS ?=

help:
	@echo "stable-diffusion.cpp Makefile Wrapper"
	@echo "====================================="
	@echo ""
	@echo "Targets:"
	@echo "  make build         - Build the project (CPU only)"
	@echo "  make build-metal   - Build with Metal GPU support (macOS)"
	@echo "  make build-cuda    - Build with CUDA GPU support"
	@echo "  make build-vulkan  - Build with Vulkan GPU support"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  RELEASE_DIR=<dir>  - Build directory (default: build)"
	@echo "  BUILD_TYPE=<type>  - Build type: Release or Debug (default: Release)"
	@echo ""
	@echo "Examples:"
	@echo "  make build"
	@echo "  make build-metal"
	@echo "  make clean"

build: $(RELEASE_DIR)
	cd $(RELEASE_DIR) && cmake .. $(CMAKE_FLAGS) && cmake --build . --config $(BUILD_TYPE) --parallel

build-metal:
	mkdir -p $(RELEASE_DIR)
	cd $(RELEASE_DIR) && cmake .. -DSD_METAL=ON && cmake --build . --config $(BUILD_TYPE) --parallel

build-cuda:
	mkdir -p $(RELEASE_DIR)
	cd $(RELEASE_DIR) && cmake .. -DSD_CUDA=ON && cmake --build . --config $(BUILD_TYPE) --parallel

build-vulkan:
	mkdir -p $(RELEASE_DIR)
	cd $(RELEASE_DIR) && cmake .. -DSD_VULKAN=ON && cmake --build . --config $(BUILD_TYPE) --parallel

$(RELEASE_DIR):
	mkdir -p $(RELEASE_DIR)

clean:
	rm -rf $(RELEASE_DIR)

.DEFAULT_GOAL := help
