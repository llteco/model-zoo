import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from onnxifier.logger import info
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from .. import QUANT

MODEL_ID = "Qwen/Qwen3.5-0.8B"
DATASET_ID = "neuralmagic/LLM_compression_calibration"


@QUANT.register("qwen3.5text")
class Qwen35Quantization(torch.nn.Module):
    framework: str = "modelopt"

    def __init__(
        self,
        model: str = MODEL_ID,
        dataset: str = DATASET_ID,
        max_samples: int = 256,
        max_seq_length: int = 256,
    ):
        super().__init__()
        mto.enable_huggingface_checkpointing()
        # Load model.
        info("Loading model...")
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model, dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(model)
        self.max_seq_length = max_seq_length
        info("Loading dataset...")
        ds = load_dataset(dataset, split=f"train[:{max_samples}]")
        ds = ds.shuffle(seed=42)
        data = self.processor.tokenizer(
            ds["text"][:max_samples],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )
        self.dataset = DataLoader(data["input_ids"], batch_size=16, shuffle=False)
        self.config = mtq.INT4_AWQ_CFG.copy()
        for cfg in self.config["quant_cfg"]:
            if "cfg" in cfg:
                cfg["cfg"]["trt_high_precision_dtype"] = "Half"

    @torch.inference_mode()
    def calibrate(self, model: torch.nn.Module):
        device = next(model.parameters()).device
        for data in self.dataset:
            model(data.to(device))

    def quant(self, output_dir: str):
        info("Quantizing model...")
        mtq.quantize(self.model, self.config, forward_loop=self.calibrate)
        mtq.print_quant_summary(self.model)
        # test with normal text generation
        input_text = "What is the capital of France?"
        device = next(self.model.parameters()).device
        inputs = self.processor(text=input_text, return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs)
        resp = self.processor.batch_decode(outputs, skip_special_tokens=True)
        info(f"Input: {input_text}")
        info(f"Output: {resp[0]}")
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
