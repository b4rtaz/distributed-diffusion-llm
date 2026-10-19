CXX ?= g++
CXXFLAGS ?= -std=c++11 -Werror -Wformat -Werror=format-security
CXXFLAGS += -MMD -MP
CPPFLAGS ?= -Isrc

ifndef TERMUX_VERSION
	CXXFLAGS += -march=native -mtune=native
endif

ifdef DEBUG
	CXXFLAGS += -g -fsanitize=address
else
	CXXFLAGS += -O3
endif

ifeq ($(OS),Windows_NT)
	LIBS += -lws2_32
	DELETE_CMD = del /f
else
	LIBS += -lpthread
	DELETE_CMD = rm -fv
endif

NN_OBJECTS := \
	nn-quants.o \
	nn-core.o \
	nn-executor.o \
	nn-network.o \
	llamafile-sgemm.o \
	nn-cpu-ops.o \
	nn-cpu.o

ifdef DLLAMA_VULKAN
	CGLSLC ?= glslc
	CPPFLAGS += -DDLLAMA_VULKAN $(shell pkg-config --cflags vulkan)
	LIBS += $(shell pkg-config --libs vulkan)
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Darwin)
		LIBS += -Wl,-rpath,$(shell pkg-config --variable=libdir vulkan)
	endif
	NN_OBJECTS += nn-vulkan.o
	VULKAN_SHADER_SRCS := $(wildcard src/nn/vulkan/*.comp)
	VULKAN_SHADER_BINS := $(VULKAN_SHADER_SRCS:.comp=.spv)
endif

.PHONY: all check clean cli-check cli-integration-check converter-test tokenizer-converter-test tokenizer-check model-loader-fixture model-loader-q40-fixture model-loader-check distributed-check checkpoint-trace-check checkpoint-reference-check checkpoint-distributed-check checkpoint-cache-check attention-reference-check vulkan-check

CHECKPOINT_SOURCE ?= gemma
CHECKPOINT_MODEL ?= /tmp/diffusion-gemma-layer0-f32.ddlm
CHECKPOINT_TRACE ?= /tmp/diffusion-gemma-layer0-f32.ddrf
CHECKPOINT_LAYER ?= 0
CHECKPOINT_PYTHON ?= python3
CHECKPOINT_THREADS ?= 8

all: ddllm

clean:
	$(DELETE_CMD) *.o *.d ddllm nn-cpu-test nn-cpu-ops-test nn-diffusion-attention-probe nn-executor-test nn-vulkan-test gemma-tokenizer-test diffusion-gemma-loader-test diffusion-gemma-test diffusion-gemma-generation-test diffusion-gemma-runtime-test diffusion-gemma-distributed-test diffusion-gemma-checkpoint-probe src/nn/vulkan/*.spv *.exe

nn-quants.o: src/nn/nn-quants.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

nn-core.o: src/nn/nn-core.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

nn-executor.o: src/nn/nn-executor.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

nn-network.o: src/nn/nn-network.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

llamafile-sgemm.o: src/nn/llamafile/sgemm.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

nn-cpu-ops.o: src/nn/nn-cpu-ops.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

nn-cpu.o: src/nn/nn-cpu.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

gemma-tokenizer.o: src/gemma-tokenizer.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

diffusion-gemma-loader.o: src/diffusion-gemma-loader.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

diffusion-gemma.o: src/diffusion-gemma.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

diffusion-gemma-generation.o: src/diffusion-gemma-generation.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

diffusion-gemma-runtime.o: src/diffusion-gemma-runtime.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

diffusion-gemma-app.o: src/diffusion-gemma-app.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

nn-vulkan.o: src/nn/nn-vulkan.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

ifdef DLLAMA_VULKAN
%.spv: %.comp
	$(CGLSLC) -c $< -o $@ --target-env=vulkan1.2

nn-vulkan-test: src/nn/nn-vulkan-test.cpp nn-quants.o nn-core.o nn-executor.o nn-vulkan.o $(VULKAN_SHADER_BINS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

vulkan-check: nn-vulkan-test ddllm model-loader-fixture model-loader-q40-fixture tokenizer-converter-test
	./nn-vulkan-test
	python3 converter/vulkan-integration-test.py ./ddllm /tmp/ddllm-diffusion-gemma-loader-test.ddlm /tmp/ddllm-diffusion-gemma-loader-test-q40.ddlm /tmp/ddllm-gemma-tokenizer-test.ddlt
endif

nn-cpu-test: src/nn/nn-cpu-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

nn-cpu-ops-test: src/nn/nn-cpu-ops-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

nn-diffusion-attention-probe: src/nn/nn-diffusion-attention-probe.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

nn-executor-test: src/nn/nn-executor-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

gemma-tokenizer-test: src/gemma-tokenizer-test.cpp gemma-tokenizer.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

diffusion-gemma-loader-test: src/diffusion-gemma-loader-test.cpp diffusion-gemma-loader.o nn-quants.o nn-core.o nn-executor.o nn-network.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

diffusion-gemma-test: src/diffusion-gemma-test.cpp diffusion-gemma.o diffusion-gemma-generation.o diffusion-gemma-runtime.o diffusion-gemma-loader.o nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

diffusion-gemma-generation-test: src/diffusion-gemma-generation-test.cpp diffusion-gemma-generation.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

diffusion-gemma-runtime-test: src/diffusion-gemma-runtime-test.cpp diffusion-gemma-runtime.o diffusion-gemma-generation.o diffusion-gemma.o nn-quants.o nn-core.o nn-executor.o nn-network.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

diffusion-gemma-distributed-test: src/diffusion-gemma-distributed-test.cpp diffusion-gemma-runtime.o diffusion-gemma-generation.o diffusion-gemma.o diffusion-gemma-loader.o nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

diffusion-gemma-checkpoint-probe: src/diffusion-gemma-checkpoint-probe.cpp diffusion-gemma-runtime.o diffusion-gemma-generation.o diffusion-gemma.o diffusion-gemma-loader.o nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

ddllm: src/ddllm.cpp diffusion-gemma-app.o diffusion-gemma-runtime.o diffusion-gemma-generation.o diffusion-gemma.o diffusion-gemma-loader.o gemma-tokenizer.o $(NN_OBJECTS) $(VULKAN_SHADER_BINS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

check: cli-check cli-integration-check nn-cpu-test nn-cpu-ops-test nn-executor-test gemma-tokenizer-test diffusion-gemma-loader-test diffusion-gemma-test diffusion-gemma-generation-test diffusion-gemma-runtime-test diffusion-gemma-distributed-test model-loader-fixture model-loader-q40-fixture checkpoint-trace-check
	./nn-cpu-test
	./nn-cpu-ops-test
	./nn-executor-test
	./diffusion-gemma-test /tmp/ddllm-diffusion-gemma-loader-test.ddlm /tmp/ddllm-diffusion-gemma-loader-test-q40.ddlm
	./diffusion-gemma-generation-test
	./diffusion-gemma-runtime-test
	./diffusion-gemma-distributed-test /tmp/ddllm-diffusion-gemma-loader-test.ddlm /tmp/ddllm-diffusion-gemma-loader-test-q40.ddlm
	$(MAKE) converter-test
	$(MAKE) tokenizer-check
	$(MAKE) model-loader-check

distributed-check: diffusion-gemma-distributed-test model-loader-fixture model-loader-q40-fixture
	./diffusion-gemma-distributed-test /tmp/ddllm-diffusion-gemma-loader-test.ddlm /tmp/ddllm-diffusion-gemma-loader-test-q40.ddlm

cli-check: ddllm
	./ddllm --help

cli-integration-check: ddllm model-loader-fixture tokenizer-converter-test
	python3 converter/cli-integration-test.py ./ddllm /tmp/ddllm-diffusion-gemma-loader-test.ddlm /tmp/ddllm-gemma-tokenizer-test.ddlt

converter-test:
	python3 converter/converter-test.py

tokenizer-converter-test:
	python3 converter/gemma-tokenizer-converter-test.py /tmp/ddllm-gemma-tokenizer-test.ddlt

tokenizer-check: tokenizer-converter-test gemma-tokenizer-test
	./gemma-tokenizer-test /tmp/ddllm-gemma-tokenizer-test.ddlt

model-loader-fixture:
	python3 converter/model-loader-fixture.py /tmp/ddllm-diffusion-gemma-loader-test.ddlm

model-loader-q40-fixture:
	python3 converter/model-loader-fixture.py /tmp/ddllm-diffusion-gemma-loader-test-q40.ddlm q40

model-loader-check: model-loader-fixture diffusion-gemma-loader-test
	./diffusion-gemma-loader-test /tmp/ddllm-diffusion-gemma-loader-test.ddlm

checkpoint-trace-check: diffusion-gemma-checkpoint-probe model-loader-fixture
	./diffusion-gemma-checkpoint-probe /tmp/ddllm-diffusion-gemma-loader-test.ddlm /tmp/ddllm-diffusion-gemma-checkpoint-trace.ddrf
	python3 converter/checkpoint-trace-test.py /tmp/ddllm-diffusion-gemma-checkpoint-trace.ddrf
	./diffusion-gemma-checkpoint-probe /tmp/ddllm-diffusion-gemma-loader-test.ddlm /tmp/ddllm-diffusion-gemma-cache-chunk3.ddrf --encoder-cache 3
	./diffusion-gemma-checkpoint-probe /tmp/ddllm-diffusion-gemma-loader-test.ddlm /tmp/ddllm-diffusion-gemma-cache-chunk1.ddrf --encoder-cache 1
	python3 converter/compare-encoder-cache.py /tmp/ddllm-diffusion-gemma-cache-chunk3.ddrf /tmp/ddllm-diffusion-gemma-cache-chunk1.ddrf

checkpoint-reference-check: diffusion-gemma-checkpoint-probe
	./diffusion-gemma-checkpoint-probe $(CHECKPOINT_MODEL) $(CHECKPOINT_TRACE) $(CHECKPOINT_THREADS)
	$(CHECKPOINT_PYTHON) converter/checkpoint-trace-test.py $(CHECKPOINT_TRACE)
	$(CHECKPOINT_PYTHON) converter/compare-checkpoint-reference.py $(CHECKPOINT_SOURCE) $(CHECKPOINT_TRACE) --source-layer $(CHECKPOINT_LAYER) --threads $(CHECKPOINT_THREADS)

checkpoint-distributed-check: diffusion-gemma-distributed-test
	./diffusion-gemma-distributed-test --checkpoint $(CHECKPOINT_MODEL)

checkpoint-cache-check: diffusion-gemma-checkpoint-probe
	./diffusion-gemma-checkpoint-probe $(CHECKPOINT_MODEL) /tmp/ddllm-diffusion-gemma-cache-chunk3.ddrf --encoder-cache 3 $(CHECKPOINT_THREADS)
	./diffusion-gemma-checkpoint-probe $(CHECKPOINT_MODEL) /tmp/ddllm-diffusion-gemma-cache-chunk1.ddrf --encoder-cache 1 $(CHECKPOINT_THREADS)
	$(CHECKPOINT_PYTHON) converter/compare-encoder-cache.py /tmp/ddllm-diffusion-gemma-cache-chunk3.ddrf /tmp/ddllm-diffusion-gemma-cache-chunk1.ddrf

attention-reference-check: nn-diffusion-attention-probe
	$(CHECKPOINT_PYTHON) converter/attention-reference-test.py ./nn-diffusion-attention-probe

-include $(wildcard *.d)
