DEVICE=cv181x
QUANT=BF16

# Get the list of segmented model files
SEGMENTED_ONNX := $(wildcard klm_models/segmented/*.onnx)
SEGMENTED_CVIMODEL := $(patsubst klm_models/segmented/%.onnx,klm_models/segmented/%.cvimodel,$(SEGMENTED_ONNX))
# Define segmented model files
SEGMENT_COUNT := 10
SEGMENTED_MODELS := klm_models/segmented/seg_head.cvimodel \
	$(foreach i, $(shell seq 0 $(SEGMENT_COUNT)), klm_models/segmented/seg_$(i).cvimodel) \
	klm_models/segmented/seg_tail.cvimodel

.PHONY: build clean build_segments

build: klm_models/prefill_model.cvimodel klm_models/decode_model.cvimodel

build_segments: $(SEGMENTED_CVIMODEL)

clean:
	rm -r klm_models/*

clean_segments:
	rm -r klm_models/segmented/build

klm_models/prefill_model.onnx klm_models/decode_model.onnx:
	python modeling_klm.py

klm_models/segmented/seg_head.onnx klm_models/segmented/seg_0.onnx klm_models/segmented/seg_1.onnx klm_models/segmented/seg_2.onnx klm_models/segmented/seg_3.onnx klm_models/segmented/seg_4.onnx klm_models/segmented/seg_5.onnx klm_models/segmented/seg_6.onnx klm_models/segmented/seg_7.onnx klm_models/segmented/seg_8.onnx klm_models/segmented/seg_9.onnx klm_models/segmented/seg_10.onnx klm_models/segmented/seg_tail.onnx:
	python segmenting_klm.py

klm_models/build/prefill_model.mlir: klm_models/prefill_model.onnx
	mkdir -p klm_models/build
	cd klm_models/build && model_transform.py \
	  --model_name prefill_klm \
	  --model_def ../prefill_model.onnx \
	  --input_shapes [[1,512],[1,512]] \
	  --mlir prefill_model.mlir

klm_models/build/decode_model.mlir: klm_models/decode_model.onnx
	mkdir -p klm_models/build
	cd klm_models/build && model_transform.py \
	  --model_name decode_klm \
	  --model_def ../decode_model.onnx \
	  --input_shapes [[1,1],[1,1],[1,2,511,64],[1,2,511,64]] \
	  --mlir decode_model.mlir

# Rules for segmented models

klm_models/prefill_model.cvimodel: klm_models/build/prefill_model.mlir
	cd klm_models/build && model_deploy.py \
	  --mlir prefill_model.mlir \
	  --quantize $(QUANT) \
	  --processor $(DEVICE) \
	  --model prefill_model.cvimodel
	cp klm_models/build/prefill_model.cvimodel klm_models/

klm_models/decode_model.cvimodel: klm_models/build/decode_model.mlir
	cd klm_models/build && model_deploy.py \
	  --mlir decode_model.mlir \
	  --quantize $(QUANT) \
	  --processor $(DEVICE) \
	  --model decode_model.cvimodel
	cp klm_models/build/decode_model.cvimodel klm_models/

# Head model
klm_models/segmented/build/seg_head.mlir: klm_models/segmented/seg_head.onnx
	mkdir -p klm_models/segmented/build
	cd klm_models/segmented/build && model_transform.py \
	  --model_name seg_head \
	  --model_def ../seg_head.onnx \
	  --input_shapes [[1,1],[2,1,1,64]] \
	  --mlir seg_head.mlir

# Segment models (0 to 10)
klm_models/segmented/build/seg_%.mlir: klm_models/segmented/seg_%.onnx
	mkdir -p klm_models/segmented/build
	cd klm_models/segmented/build && model_transform.py \
	  --model_name seg_$* \
	  --model_def ../seg_$*.onnx \
	  --input_shapes [[1,1,512],[2,1,1,64],[1,8,1,64],[1,2,511,64],[1,2,511,64]] \
	  --mlir seg_$*.mlir

# Tail model
klm_models/segmented/build/seg_tail.mlir: klm_models/segmented/seg_tail.onnx
	mkdir -p klm_models/segmented/build
	cd klm_models/segmented/build && model_transform.py \
	  --model_name seg_tail \
	  --model_def ../seg_tail.onnx \
	  --input_shapes [[1,1,512],[1,8,1,64],[1,2,511,64],[1,2,511,64]] \
	  --mlir seg_tail.mlir

# Rules for deploying segmented models

# Head model
klm_models/segmented/seg_head.cvimodel: klm_models/segmented/build/seg_head.mlir
	cd klm_models/segmented/build && model_deploy.py \
	  --mlir seg_head.mlir \
	  --quantize $(QUANT) \
	  --processor $(DEVICE) \
	  --model seg_head.cvimodel
	cp klm_models/segmented/build/seg_head.cvimodel klm_models/segmented/

# Segment models (0 to 10)
klm_models/segmented/seg_%.cvimodel: klm_models/segmented/build/seg_%.mlir
	cd klm_models/segmented/build && model_deploy.py \
	  --mlir seg_$*.mlir \
	  --quantize $(QUANT) \
	  --processor $(DEVICE) \
	  --model seg_$*.cvimodel
	cp klm_models/segmented/build/seg_$*.cvimodel klm_models/segmented/

# Tail model
klm_models/segmented/seg_tail.cvimodel: klm_models/segmented/build/seg_tail.mlir
	cd klm_models/segmented/build && model_deploy.py \
	  --mlir seg_tail.mlir \
	  --quantize $(QUANT) \
	  --processor $(DEVICE) \
	  --model seg_tail.cvimodel
	cp klm_models/segmented/build/seg_tail.cvimodel klm_models/segmented/
