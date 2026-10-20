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

.PHONY: all check clean cli-check tokenizer-check model-loader-check distributed-check vulkan-check

MODEL ?= model/diffusion-gemma-q40.ddlm
TOKENIZER ?= model/diffusion-gemma.ddlt

all: ddllm

clean:
	$(DELETE_CMD) *.o *.d ddllm nn-cpu-test nn-cpu-ops-test nn-diffusion-attention-probe nn-executor-test nn-vulkan-test gemma-tokenizer-test diffusion-gemma-loader-test diffusion-gemma-test diffusion-gemma-generation-test diffusion-gemma-runtime-test diffusion-gemma-distributed-test src/nn/vulkan/*.spv *.exe

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

vulkan-check: nn-vulkan-test ddllm
	./nn-vulkan-test
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

ddllm: src/ddllm.cpp diffusion-gemma-app.o diffusion-gemma-runtime.o diffusion-gemma-generation.o diffusion-gemma.o diffusion-gemma-loader.o gemma-tokenizer.o $(NN_OBJECTS) $(VULKAN_SHADER_BINS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(filter %.o,$^) -o $@ $(LIBS)

check: cli-check nn-cpu-test nn-cpu-ops-test nn-executor-test diffusion-gemma-test diffusion-gemma-generation-test diffusion-gemma-runtime-test
	./nn-cpu-test
	./nn-cpu-ops-test
	./nn-executor-test
	./diffusion-gemma-test
	./diffusion-gemma-generation-test
	./diffusion-gemma-runtime-test

distributed-check: diffusion-gemma-distributed-test
	./diffusion-gemma-distributed-test --checkpoint $(MODEL)

cli-check: ddllm
	./ddllm --help

tokenizer-check: gemma-tokenizer-test
	./gemma-tokenizer-test $(TOKENIZER)

model-loader-check: diffusion-gemma-loader-test
	./diffusion-gemma-loader-test $(MODEL)

-include $(wildcard *.d)
