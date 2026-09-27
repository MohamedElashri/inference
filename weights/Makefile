# PVFinder weights pipeline: trained checkpoint -> validated Allen model file.
#
#   make list                        models in models.tsv
#   make all      MODEL=<name>       every stage below, in order
#   make fetch    MODEL=<name>       copy the checkpoint into checkpoints/<name>.pyt
#   make convert  MODEL=<name>       write out/<name>/pvfinder_model.json, the file Allen reads
#   make verify   MODEL=<name>       re-read it the way Allen does, compare with the checkpoint bit for bit
#   make build    MODEL=<name>       build the Allen variant the model needs (../ballen)
#   make dump     MODEL=<name>       run Allen on one slice with the FC, UNet and peak dumps enabled
#   make validate MODEL=<name>       check the dump stage by stage: track features, FC, UNet, the full
#                                    model from Allen's track features, and the peak finder; write a
#                                    run record
#   make verify-all                  fetch + convert + verify every model
#   make model-path MODEL=<name>     print the model file's path (for Allen's "model" property)
#   make clean    MODEL=<name>       remove out/<name>
#
# MODEL defaults to unet16_lc4_scnone_asym5_final. Optional: DEVICE (GPU index, default 2),
# EVENTS (slice size for dump, default 500), JOBS (build jobs, default 16), PY.
# Variants, for dump and validate alike:
#   PRECISION=bfloat16   the reduced-precision path (FC and UNet); its dump goes to
#                        out/<name>/dump_bfloat16/
#   DUMP_SET="ALG.PROP=VALUE ..."  any other Allen property overrides
#   DUMP_TAG=<tag>       keep the dump in out/<name>/dump_<tag>/
# LABEL names the run record.

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

MODEL   ?= unet16_lc4_scnone_asym5_final
DEVICE  ?= 2
EVENTS  ?= 500
JOBS    ?= 16
PRECISION ?=
DUMP_SET ?=
DUMP_TAG ?= $(PRECISION)
LABEL    ?= validate$(if $(DUMP_TAG),_$(DUMP_TAG))

# $(call field,<model>,<column>): one column of that model's catalog row
field  = $(shell awk -F'\t' -v m='$(1)' '$$1 == m {print $$$(2)}' $(CATALOG))
MODELS := $(shell awk -F'\t' 'NR > 1 && NF && substr($$1, 1, 1) != "\043" {print $$1}' $(CATALOG))

SOURCE     := $(call field,$(MODEL),2)
UNET_FEAT  := $(call field,$(MODEL),3)
LATENT     := $(call field,$(MODEL),4)
BUILD      := $(call field,$(MODEL),5)

CKPT := $(WEIGHTS)/checkpoints/$(MODEL).pyt
OUT  := $(WEIGHTS)/out/$(MODEL)
DUMP := $(OUT)/dump$(if $(DUMP_TAG),_$(DUMP_TAG))
RUNS := $(REPO)/benchmarks/runs.py
# FC, UNet, pvfinder_peak and the PV fit: every stage the validators check.
SEQUENCE := hlt1_pp_pvs_pvfinder_unet_benchmark

.PHONY: help list check-model fetch convert verify build dump validate all verify-all model-path clean

help:
> @sed -n '2,/^$$/s/^# \{0,1\}//p' $(firstword $(MAKEFILE_LIST))

list:
> @awk -F'\t' '$$1 !~ /^#/ && NF {printf "%-36s %-9s %-6s %-16s %s\n", $$1, $$3, $$4, $$5, $$7}' $(CATALOG)

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
> $(PY) $(SCRIPTS)/convert.py --model $(CKPT) --name $(MODEL) --out $(OUT)/pvfinder_model.json
> @touch $@

# ---- verify -----------------------------------------------------------------
verify: convert $(OUT)/verify.txt

$(OUT)/verify.txt: $(OUT)/.converted $(SCRIPTS)/verify.py
> $(PY) $(SCRIPTS)/verify.py --checkpoint $(CKPT) --model-file $(OUT)/pvfinder_model.json | tee $@

# ---- Allen build, dump, validate ---------------------------------------------
build: check-model
> cd $(REPO) && ./ballen -a gpu --cudnn --unet-feat $(UNET_FEAT) --unet-batch-channels $(LATENT) -b $(BUILD) -j $(JOBS)

dump: verify build
> $(SCRIPTS)/allen_dump.sh --build $(REPO)/Allen/$(BUILD)gpu --model-file $(OUT)/pvfinder_model.json \
>     --sequence $(SEQUENCE) --dump-dir $(DUMP) --events $(EVENTS) --device $(DEVICE) \
>     $(foreach s,$(DUMP_SET),--set $(s)) \
>     $(if $(PRECISION),--set pvfinder_fc_aggregation.precision=$(PRECISION) --set pvfinder_unet.precision=$(PRECISION))
> $(PY) $(RUNS) snapshot $(DUMP) --device $(DEVICE) --build-dir $(REPO)/Allen/$(BUILD)gpu --model $(MODEL)

# Every validator runs and the run record is always written; the target fails
# afterwards if any validator did.
validate: check-model
> @test -f $(DUMP)/allen_fc_csr.bin || { echo "no dump for $(MODEL): run 'make dump MODEL=$(MODEL)'" >&2; exit 1; }
> @rc=0; \
> $(PY) $(SCRIPTS)/validate_fc.py --dump-dir $(DUMP) --weights $(CKPT) \
>     --report $(DUMP)/validate_fc.json | tee $(DUMP)/validate_fc.txt || rc=1; \
> $(PY) $(SCRIPTS)/validate_unet.py --dump-dir $(DUMP) --weights $(CKPT) \
>     --report $(DUMP)/validate_unet.json | tee $(DUMP)/validate_unet.txt || rc=1; \
> $(PY) $(SCRIPTS)/validate_model.py --dump-dir $(DUMP) --weights $(CKPT) \
>     --report $(DUMP)/validate_model.json 2>&1 | grep -v '^DEBUG' | tee $(DUMP)/validate_model.txt || rc=1; \
> $(PY) $(SCRIPTS)/validate_features.py --dump-dir $(DUMP) \
>     --report $(DUMP)/validate_features.json | tee $(DUMP)/validate_features.txt || rc=1; \
> $(PY) $(SCRIPTS)/validate_peaks.py --dump-dir $(DUMP) \
>     --report $(DUMP)/validate_peaks.json | tee $(DUMP)/validate_peaks.txt || rc=1; \
> $(PY) $(RUNS) validation --model $(MODEL) --build-dir $(REPO)/Allen/$(BUILD)gpu --dump-dir $(DUMP) \
>     --device $(DEVICE) --events $(EVENTS) --sequence $(SEQUENCE) --label $(LABEL) \
>     --fc-report $(DUMP)/validate_fc.json --unet-report $(DUMP)/validate_unet.json \
>     --model-report $(DUMP)/validate_model.json --features-report $(DUMP)/validate_features.json \
>     --peaks-report $(DUMP)/validate_peaks.json; \
> exit $$rc

all: check-model
> $(MAKE) --no-print-directory fetch convert verify MODEL=$(MODEL)
> $(MAKE) --no-print-directory build MODEL=$(MODEL)
> $(MAKE) --no-print-directory dump validate MODEL=$(MODEL)

verify-all:
> @for m in $(MODELS); do $(MAKE) --no-print-directory verify MODEL=$$m || exit 1; done

# ---- helpers ------------------------------------------------------------------
model-path: check-model
> @echo $(OUT)/pvfinder_model.json

clean: check-model
> rm -rf $(OUT)
