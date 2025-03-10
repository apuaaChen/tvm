# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

INCLUDES ?= -isystem crt/include -Icrt_config
CFLAGS ?= -Werror -Wall
CXXFLAGS ?= -Werror -Wall -std=c++11
LDFLAGS ?= -Werror -Wall

# Codegen produces spurious lines like: int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
MODEL_CFLAGS ?= -Wno-error=unused-variable

AR ?= ${PREFIX}ar
CC ?= ${PREFIX}gcc
CXX ?= ${PREFIX}g++
RANLIB ?= ${PREFIX}ranlib

QUIET ?= @

PWD = $(shell pwd)
BUILD_DIR = build
CRT_LIB_NAMES = microtvm_rpc_server microtvm_rpc_common graph_executor graph_executor_module common memory
CRT_LIBS = $(patsubst %, $(BUILD_DIR)/crt/lib%.a, $(CRT_LIB_NAMES))

CRT_INCLUDES = $(glob crt/include/**)

$(BUILD_DIR)/crt/lib%.a: $(glob crt/src/runtime/%/*.c)
	${QUIET}cd crt && $(MAKE) \
		BUILD_DIR=../$(BUILD_DIR)/crt \
		CRT_CONFIG=$(PWD)/crt_config/crt_config.h \
		EXTRA_CFLAGS="$(CFLAGS)" \
		EXTRA_CXXFLAGS="$(CXXFLAGS)" \
		EXTRA_LDFLAGS="$(EXTRA_LDFLAGS)" \
		$(patsubst $(BUILD_DIR)/crt/lib%.a,%,$@)

crt: $(CRT_LIBS)
.PHONY: crt

# Compile codegen files
$(BUILD_DIR)/model/codegen/host/%.o: model/codegen/host/%.c
	${QUIET}mkdir -p $(dir $@)
	${QUIET}$(CC) $(INCLUDES) $(CFLAGS) $(MODEL_CFLAGS) -c -o "$@" "$<"

MODEL_LIBS = \
	$(patsubst model/codegen/host/src/%.c, $(BUILD_DIR)/model/codegen/host/src/%.o, $(wildcard model/codegen/host/src/*.c)) \
	$(wildcard model/codegen/host/lib/*.o)

# Compile src/ files
build/%.o: src/%.cc
	${QUIET}mkdir -p $(dir $@)
	${QUIET}$(CXX) $(INCLUDES) $(CXXFLAGS) -c -o "$@" "$<"

SRCS = $(wildcard src/*.cc)
OBJS = $(patsubst src/%.cc,build/%.o,$(SRCS))

build/main: ${OBJS} ${MODEL_LIBS} ${CRT_LIBS}
	${QUIET}mkdir -p $(dir $@)
	${QUIET}$(CXX) $(LDFLAGS) -o "$@" $^

all: build/main
.PHONY = all

.DEFAULT_GOAL = all
