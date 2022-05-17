# Run all experiments
ROOT := /TraceAnalysis
OUTPUT := $(ROOT)/output
# OUTPUT := $(ROOT)/tracerca-exp/data/dockeroutput
# ORIGIN_DATA_DIR := $(ROOT)/tracerca-exp/data/
ORIGIN_DATA_DIR := $(ROOT)/tracerca-exp/data/
ROOT_CAUSE_DIR := $(ORIGIN_DATA_DIR)/root_causes/
SCRIPT_DIR := $(ROOT)/tracerca-exp
CONFIG_SCRIPTS := $(ROOT)/tracerca-exp/trainticket_config.py
UPDATE_CACHE_FLAG :=


# Input data should be prepares manually
TEST_FILES = $(shell ls $(ORIGIN_DATA_DIR)/test/*.pkl)
NORMAL_TRAIN_FILES = $(shell ls $(ORIGIN_DATA_DIR)/normal/*.pkl)
ABNORMAL_TRAIN_FILES = $(shell ls ${ORIGIN_DATA_DIR}/train/*.pkl)
ALL_TRAIN_FILES = $(NORMAL_TRAIN_FILES) $(ABNORMAL_TRAIN_FILES)

# hyperparameters
DROP_SERVICE = 0
DROP_FAULT_TYPE = 0
SUPPORT = 0.1
SIGMA = 1
FISHER = 3
K = 100


INVO_TEST_FILE_RESULTS = $(addprefix $(OUTPUT)/trainticket_anomaly_detection.test/,$(addsuffix .invo.result.pkl.$(SIGMA).$(FISHER),$(basename $(notdir $(TEST_FILES)))))
TRACE_TEST_FILE_RESULTS = $(addprefix $(OUTPUT)/trainticket_anomaly_detection.test/,$(addsuffix .trace.result.pkl.$(DROP_SERVICE).$(DROP_FAULT_TYPE),$(basename $(notdir $(TEST_FILES)))))

ANOMALY_DETECTION_RESULT = $(OUTPUT)/trainticket.anomaly_detection.result.csv.$(DROP_SERVICE).$(DROP_FAULT_TYPE).$(SIGMA).$(FISHER)
FAULT_LOCALIZATION_RESULT = $(OUTPUT)/trainticket.root_cause_localization.result.csv.$(DROP_SERVICE).$(DROP_FAULT_TYPE).$(SUPPORT).$(K)

EFFECT_OF_TRACE_LOCALIZATION_RESULT = $(OUTPUT)/trainticket.root_cause_localization.effect_of_trace.result.csv.$(DROP_SERVICE).$(DROP_FAULT_TYPE)

ANOMALY_DETECTION_MODEL = $(OUTPUT)/trainticket_anomaly_detection.models.$(DROP_SERVICE).$(DROP_FAULT_TYPE)
TRACE_HISTORICAL_DATA = $(OUTPUT)/trainticket_trace_encoded/trainticket_historical_all.trace.$(DROP_SERVICE).$(DROP_FAULT_TYPE).npz
INVO_HISTORICAL_DATA = $(OUTPUT)/trainticket_invo_encoded/trainticket_historical_normal.invo.pkl

ASSOCIATION_RULE_MINING_TEST_FILE_RESULTS = $(addprefix $(OUTPUT)/trainticket_root_cause_localization/,$(addsuffix .association_rule_mining.result.pkl.$(SUPPORT).$(K),$(basename $(notdir $(TEST_FILES)))))
PAGERANK_TEST_FILE_RESULTS = $(addprefix $(OUTPUT)/trainticket_root_cause_localization/,$(addsuffix .pagerank.result.pkl,$(basename $(notdir $(TEST_FILES)))))
MEPFL_TEST_FILE_RESULTS = $(addprefix $(OUTPUT)/trainticket_root_cause_localization/,$(addsuffix .MEPFL.result.pkl.$(DROP_SERVICE).$(DROP_FAULT_TYPE),$(basename $(notdir $(TEST_FILES)))))
RCSF_TEST_FILE_RESULTS = $(addprefix $(OUTPUT)/trainticket_root_cause_localization/,$(addsuffix .RCSF.result.pkl,$(basename $(notdir $(TEST_FILES)))))
MICROSCOPE_TEST_FILE_RESULTS = $(addprefix $(OUTPUT)/trainticket_root_cause_localization/,$(addsuffix .microscope.result.pkl,$(basename $(notdir $(TEST_FILES)))))
LOCALIZATION_MODEL = $(OUTPUT)/trainticket_localization.models.$(DROP_SERVICE).$(DROP_FAULT_TYPE)

ASSOCIATION_RULE_MINING_TEST_FILE_RESULTS_EFFECT_OF_TRACE = $(addprefix $(OUTPUT)/trainticket_root_cause_localization/,$(addsuffix _effect_of_trace_type1.association_rule_mining.result.pkl.$(SUPPROT),$(basename $(notdir $(TEST_FILES)))))

# keep all temporary files
.SECONDARY:


.PHONY: target
target: $(ANOMALY_DETECTION_RESULT) $(FAULT_LOCALIZATION_RESULT) ;

.PHONY: effect-of-trace
effect-of-trace: $(EFFECT_OF_TRACE_LOCALIZATION_RESULT) ;

.PHONY: models
models: $(LOCALIZATION_MODEL) $(ANOMALY_DETECTION_MODEL) ;

.PHONY: localization
localization: $(FAULT_LOCALIZATION_RESULT) ;

.PHONY: ad
ad: $(ANOMALY_DETECTION_RESULT) ;


.PHONY: dataset-summary
dataset-summary: $(addprefix $(OUTPUT)/trainticket_trace_encoded/,$(addsuffix .trace.$(DROP_SERVICE).$(DROP_FAULT_TYPE).npz,$(basename $(notdir $(TEST_FILES))))) $(addprefix $(OUTPUT)/trainticket_invo_encoded/,$(addsuffix .invo.pkl,$(basename $(notdir $(TEST_FILES))))) $(INVO_HISTORICAL_DATA) $(TRACE_HISTORICAL_DATA) $(SCRIPT_DIR)/run_dataset_summary.py
	echo $(filter %.invo.pkl,$INVO_HISTORICAL_DATA)
	python run_dataset_summary.py \
		$(addprefix -i ",$(addsuffix ",$(filter %.trace.$(DROP_SERVICE).$(DROP_FAULT_TYPE).npz,$^))) \
		$(addprefix -i ",$(addsuffix ",$(filter %.invo.pkl,$^)))


.PHONY: prepare-all-files
prepare-all-files:
	mkdir $(ORIGIN_DATA_DIR)/all/ || echo exists
#	rm $(ORIGIN_DATA_DIR)/all/*.pkl || echo OK
	ln -sf $(ORIGIN_DATA_DIR)/test/* $(ORIGIN_DATA_DIR)/all/


.PHONY: debug
debug:
	echo TEST_FILES: $(TEST_FILES)
	echo NORMAL_TRAIN_FILES: $(NORMAL_TRAIN_FILES)
	echo ABNORMAL_TRAIN_FILES: $(ABNORMAL_TRAIN_FILES)
	echo ALL_TRAIN_FILES: $(ALL_TRAIN_FILES)
	echo TEST_FILE_RESULTS: $(TEST_FILE_RESULTS)
	echo HISTORICAL_DATA: $(HISTORICAL_DATA)
	echo ANOMALY_DETECTION_RESULT $(ANOMALY_DETECTION_RESULT)


# anomaly detection result
$(ANOMALY_DETECTION_RESULT):$(INVO_TEST_FILE_RESULTS) $(TRACE_TEST_FILE_RESULTS) $(SCRIPT_DIR)/run_anomaly_detection_collect_result.py $(CONFIG_SCRIPTS)
	python run_anomaly_detection_collect_result.py \
		$(addprefix -i ",$(addsuffix ",$(INVO_TEST_FILE_RESULTS))) \
		$(addprefix -t ",$(addsuffix ",$(TRACE_TEST_FILE_RESULTS))) \
		-o "$@"

$(ANOMALY_DETECTION_MODEL):$(TRACE_HISTORICAL_DATA) $(INVO_HISTORICAL_DATA) $(SCRIPT_DIR)/run_anomaly_detection_prepare_model.py
	python run_anomaly_detection_prepare_model.py -i $(word 2,$^) -t $(word 1,$^) -o $@


$(OUTPUT)/trainticket_anomaly_detection.test/%.invo.result.pkl.$(SIGMA).$(FISHER): $(OUTPUT)/trainticket_invo_encoded/%.invo.pkl $(INVO_HISTORICAL_DATA) $(OUTPUT)/trainticket_anomaly_detection.test/%.useful_features.$(FISHER) $(ANOMALY_DETECTION_MODEL) $(SCRIPT_DIR)/run_anomaly_detection_invo.py
	python run_anomaly_detection_invo.py -i $(word 1,$^) -o $@ -h $(word 2,$^) -u $(word 3,$^) --cache $(ANOMALY_DETECTION_MODEL) --threshold $(SIGMA)

$(OUTPUT)/trainticket_anomaly_detection.test/%.useful_features.$(FISHER): $(OUTPUT)/trainticket_invo_encoded/%.invo.pkl $(INVO_HISTORICAL_DATA) $(SCRIPT_DIR)/run_selecting_features.py
	python run_selecting_features.py -i $(word 1,$^) -o $@ -h $(word 2,$^) --fisher $(FISHER)

$(OUTPUT)/trainticket_anomaly_detection.test/%.trace.result.pkl.$(DROP_SERVICE).$(DROP_FAULT_TYPE): $(OUTPUT)/trainticket_trace_encoded/%.trace.$(DROP_SERVICE).$(DROP_FAULT_TYPE).npz $(TRACE_HISTORICAL_DATA) $(ANOMALY_DETECTION_MODEL) $(SCRIPT_DIR)/run_anomaly_detection_trace.py
	python run_anomaly_detection_trace.py -i $(word 1,$^) -o $@ -h $(word 2,$^) \
		-m $(word 3,$^) $(UPDATE_CACHE_FLAG)



$(OUTPUT)/trainticket_invo_encoded/%.invo.pkl: $(ORIGIN_DATA_DIR)/all/%.pkl $(SCRIPT_DIR)/run_invo_encoding.py
	python run_invo_encoding.py -i $(word 1,$^) -o $(word 1,$@)

$(OUTPUT)/trainticket_trace_encoded/%.trace.$(DROP_SERVICE).$(DROP_FAULT_TYPE).npz: $(ORIGIN_DATA_DIR)/all/%.pkl $(SCRIPT_DIR)/run_trace_encoding.py
	python run_trace_encoding.py -i $(word 1,$^) -o $(word 1,$@) --drop-fault-type $(DROP_FAULT_TYPE) --drop-service $(DROP_SERVICE)

$(ORIGIN_DATA_DIR)/all/trainticket_historical_normal.pkl: $(NORMAL_TRAIN_FILES) $(SCRIPT_DIR)/run_concatenate.py
	python run_concatenate.py $(addprefix -i ",$(addsuffix ",$(NORMAL_TRAIN_FILES))) -o $@

$(ORIGIN_DATA_DIR)/all/trainticket_historical_all.pkl: $(ALL_TRAIN_FILES) $(SCRIPT_DIR)/run_concatenate.py
	python run_concatenate.py $(addprefix -i ",$(addsuffix ",$(ALL_TRAIN_FILES))) -o $@ --add-root-cause

# ROOT CAUSE  LOCALIZATION
$(FAULT_LOCALIZATION_RESULT): $(ASSOCIATION_RULE_MINING_TEST_FILE_RESULTS) $(PAGERANK_TEST_FILE_RESULTS) $(MEPFL_TEST_FILE_RESULTS) $(MICROSCOPE_TEST_FILE_RESULTS) $(RCSF_TEST_FILE_RESULTS) $(SCRIPT_DIR)/run_localization_collect.py
	python run_localization_collect.py \
		$(addprefix -i ",$(addsuffix ",$(ASSOCIATION_RULE_MINING_TEST_FILE_RESULTS))) \
		$(addprefix -i ",$(addsuffix ",$(MEPFL_TEST_FILE_RESULTS))) \
		$(addprefix -i ",$(addsuffix ",$(PAGERANK_TEST_FILE_RESULTS))) \
		$(addprefix -i ",$(addsuffix ",$(RCSF_TEST_FILE_RESULTS))) \
        $(addprefix -i ",$(addsuffix ",$(MICROSCOPE_TEST_FILE_RESULTS))) \
        -r $(ROOT_CAUSE_DIR) \
		-o $@

#$(FAULT_LOCALIZATION_RESULT): $(MEPFL_TEST_FILE_RESULTS) $(SCRIPT_DIR)/run_localization_collect.py
#	python run_localization_collect.py \
#		$(addprefix -i ",$(addsuffix ",$(MEPFL_TEST_FILE_RESULTS))) \
#        -r $(ROOT_CAUSE_DIR) \
#		-o $@

$(EFFECT_OF_TRACE_LOCALIZATION_RESULT): $(SCRIPT_DIR)/run_effect_of_trace_localization_collect.py
	python run_effect_of_trace_localization_collect.py -o $(EFFECT_OF_TRACE_LOCALIZATION_RESULT)

$(OUTPUT)/trainticket_anomaly_detection.test/%_effect_of_trace_type1.invo.result.pkl: $(OUTPUT)/trainticket_anomaly_detection.test/%.invo.result.pkl $(SCRIPT_DIR)/run_effect_of_trace_inject.py
	python run_effect_of_trace_inject.py -i $(word 1,$^) -o $@ -r $(OUTPUT)/trainticket_effect_of_trace.root_cause/


$(LOCALIZATION_MODEL): $(TRACE_HISTORICAL_DATA) $(SCRIPT_DIR)/run_localization_prepare_model.py
	python run_localization_prepare_model.py -t $(word 1,$^) -o $@

$(OUTPUT)/trainticket_root_cause_localization/%.association_rule_mining.result.pkl.$(SUPPORT).$(K):$(OUTPUT)/trainticket_anomaly_detection.test/%.invo.result.pkl.$(SIGMA).$(FISHER) $(SCRIPT_DIR)/run_localization_association_rule_mining.py $(shell ls $(SCRIPT_DIR)/association_rule_mining/*.py) $(CONFIG_SCRIPTS)
	python run_localization_association_rule_mining.py -i $(word 1,$^) -o $@ \
		--min-support-rate $(SUPPORT) --quiet --k $(K)

$(OUTPUT)/trainticket_root_cause_localization/%.pagerank.result.pkl:$(OUTPUT)/trainticket_anomaly_detection.test/%.invo.result.pkl.$(SIGMA).$(FISHER) $(SCRIPT_DIR)/run_localization_pagerank.py
	python run_localization_pagerank.py -i $(word 1,$^) -o $@
    
$(OUTPUT)/trainticket_root_cause_localization/%.RCSF.result.pkl:$(OUTPUT)/trainticket_anomaly_detection.test/%.invo.result.pkl.$(SIGMA).$(FISHER) $(SCRIPT_DIR)/run_localization_RCSF.py
	python run_localization_RCSF.py -i $(word 1,$^) -o $@

$(OUTPUT)/trainticket_root_cause_localization/%.MEPFL.result.pkl.$(DROP_SERVICE).$(DROP_FAULT_TYPE):$(OUTPUT)/trainticket_trace_encoded/%.trace.$(DROP_SERVICE).$(DROP_FAULT_TYPE).npz $(LOCALIZATION_MODEL) $(SCRIPT_DIR)/run_localization_MEPFL.py
	python run_localization_MEPFL.py -i $(word 1,$^) -o $@ -c $(word 2,$^)

$(OUTPUT)/trainticket_root_cause_localization/%.microscope.result.pkl:$(OUTPUT)/trainticket_anomaly_detection.test/%.invo.result.pkl.$(SIGMA).$(FISHER) $(SCRIPT_DIR)/run_localization_microscope.py
	python run_localization_microscope.py -i $(word 1,$^) -o $@

.PHONY: clean
clean: clean-cache clean-debug
	rm $(ANOMALY_DETECTION_RESULT) || echo OK
	rm $(OUTPUT)/trainticket_anomaly_detection.test/*.result.pkl.* || echo OK
	rm $(OUTPUT)/trainticket_anomaly_detection.test/*.useful_features.* || echo OK
	rm $(OUTPUT)/trainticket_root_cause_localization/*.result.pkl.* || echo OK
	rm $(OUTPUT)/trainticket_root_cause_localization/*.result.pkl || echo OK
	rm $(OUTPUT)/trainticket_invo_encoded/*.pkl || echo OK
	rm $(OUTPUT)/trainticket_trace_encoded/*.npz || echo OK
	rm $(HISTORICAL_DATA) || echo OK
	rm $(ORIGIN_DATA_DIR)/all/*.pkl || echo OK
	rm $(ANOMALY_DETECTION_MODEL) || echo OK
	rm $(LOCALIZATION_MODEL) || echo OK

.PHONY: clean-cache
clean-cache: ;


.PHONY: clean-debug
clean-debug:
	rm $(OUTPUT)/trainticket_anomaly_detection.test/selecting_feature.debug/*.pdf || echo


.PHONY: plot
plot: $(ANOMALY_DETECTION_RESULT) $(FAULT_LOCALIZATION_RESULT) $(shell ls $(SCRIPT_DIR)/plot/*)
	python plot/run_plot_anomaly_detection.py \
		-i $(ANOMALY_DETECTION_RESULT) \
		-o $(OUTPUT)/figures/anomaly_detection_comparison.pdf
	python plot/run_plot_localization.py \
		-i $(FAULT_LOCALIZATION_RESULT) \
		-o $(OUTPUT)/figures/
	python plot/run_plot_localization_effect_of_trace.py \
		-i $(EFFECT_OF_TRACE_LOCALIZATION_RESULT) \
		-o $(OUTPUT)/figures/effect_of_trace/
	python plot/run_plot_noise_localization.py \
		-i $(FAULT_LOCALIZATION_RESULT) \
		-o $(OUTPUT)/figures/
	python plot/run_plot_noise_localization.py \
		-i $(FAULT_LOCALIZATION_RESULT) \
		-o $(OUTPUT)/figures/
	python plot/run_plot_drop.py --output ../output/figures/drop/
