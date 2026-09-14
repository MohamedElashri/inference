# PVFinder weights pipeline: trained checkpoint -> validated Allen weight files.
#
#   make list                        models in models.tsv
#   make all      MODEL=<name>       every stage below, in order
#   make fetch    MODEL=<name>       copy the checkpoint into checkpoints/<name>.pyt
#   make convert  MODEL=<name>       write out/<name>/cnn_weights.bin and fc_weights.bin
#   make verify   MODEL=<name>       re-read both files the way Allen does, compare with the checkpoint
#   make build    MODEL=<name>       build the Allen variant the model needs (../ballen)
#   make dump     MODEL=<name>       run Allen on one slice with the FC (and UNet) dumps enabled
#   make validate MODEL=<name>       recompute FC (and UNet) from the checkpoint, compare with the dump
#   make verify-all                  fetch + convert + verify every model
#   make env      MODEL=<name>       print the PVFINDER_WEIGHTS_DIR export Allen configs need
#   make clean    MODEL=<name>       remove out/<name>
#
# MODEL defaults to unet16_lc8_iter9. Optional: DEVICE (GPU index, default 2),
# EVENTS (slice size for dump, default 500), JOBS (build jobs, default 16), PY.

SHELL         := /bin/bash
.SHELLFLAGS   := -eu -o pipefail -c
.RECIPEPREFIX := >
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:

WEIGHTS := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO    := $(abspath $(WEIGHTS)/..)
PY      ?= $(REPO)/.venv/bin/python3
CATALOG := $(WEIGHTS)/models.tsv
SCRIPTS := $(WEIGHTS)/scripts

MODEL   ?= unet16_lc8_iter9
DEVICE  ?= 2
EVENTS  ?= 500
JOBS    ?= 16

# $(call field,<model>,<column>): one column of that model's catalog row
field  = $(shell awk -F'\t' -v m='$(1)' '$$1 == m {print $$$(2)}' $(CATALOG))
MODELS := $(shell awk -F'\t' 'NR > 1 && NF && substr($$1, 1, 1) != "\043" {print $$1}' $(CATALOG))

SOURCE     := $(call field,$(MODEL),2)
UNET_FEAT  := $(call field,$(MODEL),3)
LATENT     := $(call field,$(MODEL),4)
SC_MODE    := $(call field,$(MODEL),5)
ALLEN_UNET := $(call field,$(MODEL),6)
BUILD      := $(call field,$(MODEL),7)

CKPT := $(WEIGHTS)/checkpoints/$(MODEL).pyt
OUT  := $(WEIGHTS)/out/$(MODEL)
DUMP := $(OUT)/dump
# Allen's UNet loader only supports concatenated skip connections; the other
# architectures are validated on the FC stage alone.
SEQUENCE := $(if $(filter yes,$(ALLEN_UNET)),hlt1_pp_pvfinder_unet_benchmark,hlt1_pp_pvfinder_benchmark)

.PHONY: help list check-model fetch convert verify build dump validate all verify-all env clean

help:
> @sed -n '2,17s/^# \{0,1\}//p' $(firstword $(MAKEFILE_LIST))

list:
> @awk -F'\t' '$$1 !~ /^#/ && NF {printf "%-32s %-9s %-6s %-7s %-10s %s\n", $$1, $$3, $$4, $$5, $$6, $$7}' $(CATALOG)

check-model:
> @if [[ -z "$(SOURCE)" ]]; then echo "unknown MODEL '$(MODEL)'; see 'make list'" >&2; exit 1; fi

# ---- fetch ------------------------------------------------------------------
fetch: check-model $(CKPT)

$(CKPT): $(SOURCE) | check-model
> @mkdir -p $(@D)
> cp -p "$(SOURCE)" "$@"

# ---- convert ----------------------------------------------------------------
convert: fetch $(OUT)/.converted

$(OUT)/.converted: $(CKPT) $(SCRIPTS)/convert.py
> @mkdir -p $(OUT)
> $(PY) $(SCRIPTS)/convert.py --model $(CKPT) --fc-out $(OUT)/fc_weights.bin --cnn-out $(OUT)/cnn_weights.bin
> @touch $@

# ---- verify -----------------------------------------------------------------
verify: convert $(OUT)/verify.txt

$(OUT)/verify.txt: $(OUT)/.converted $(SCRIPTS)/verify.py
> $(PY) $(SCRIPTS)/verify.py --checkpoint $(CKPT) --cnn $(OUT)/cnn_weights.bin --fc $(OUT)/fc_weights.bin | tee $@

# ---- Allen build, dump, validate ---------------------------------------------
build: check-model
> cd $(REPO) && ./ballen -a gpu --cudnn --cublas --unet-feat $(UNET_FEAT) --unet-batch-channels $(LATENT) -b $(BUILD) -j $(JOBS)

dump: verify build
> $(SCRIPTS)/allen_dump.sh --build $(REPO)/Allen/$(BUILD)gpu --weights-dir $(OUT) \
>     --sequence $(SEQUENCE) --dump-dir $(DUMP) --events $(EVENTS) --device $(DEVICE)

validate: check-model
> @test -f $(DUMP)/allen_fc_csr.bin || { echo "no dump for $(MODEL): run 'make dump MODEL=$(MODEL)'" >&2; exit 1; }
> $(PY) $(SCRIPTS)/validate_fc.py --dump-dir $(DUMP) --weights $(CKPT) --fc-bin $(OUT)/fc_weights.bin | tee $(OUT)/validate_fc.txt
> @if [[ "$(ALLEN_UNET)" == yes ]]; then \
>     $(PY) $(SCRIPTS)/validate_unet.py --dump-dir $(DUMP) --weights $(CKPT) | tee $(OUT)/validate_unet.txt; \
> else \
>     echo "UNet validation skipped: Allen's UNet loader does not support sc_mode=$(SC_MODE)" | tee $(OUT)/validate_unet.txt; \
> fi

all: check-model
> $(MAKE) --no-print-directory fetch convert verify MODEL=$(MODEL)
> $(MAKE) --no-print-directory build MODEL=$(MODEL)
> $(MAKE) --no-print-directory dump validate MODEL=$(MODEL)

verify-all:
> @for m in $(MODELS); do $(MAKE) --no-print-directory verify MODEL=$$m || exit 1; done

# ---- helpers ------------------------------------------------------------------
env: check-model
> @test -f $(OUT)/verify.txt || { echo "$(MODEL) is not converted and verified yet: make verify MODEL=$(MODEL)" >&2; exit 1; }
> @echo "export PVFINDER_WEIGHTS_DIR=$(OUT)"

clean: check-model
> rm -rf $(OUT)
