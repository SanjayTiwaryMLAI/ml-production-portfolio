"""
Use Case: LLM Inference Optimisation — Model Compression & Deployment
-----------------------------------------------------------------------
Problem : ~80GB generative model too large for cost-effective deployment.
Approach: 4-bit NF4 quantisation → Auto-scaling endpoint → Benchmarking.
"""
import json
import time
import torch
from dataclasses import dataclass


@dataclass
class ModelSizeEstimate:
    name: str
    params_b: float
    dtype: str

    @property
    def size_gb(self):
        bpp = {"float32":4,"float16":2,"int8":1,"int4":0.5}
        return round(self.params_b * 1e9 * bpp[self.dtype] / 1e9, 1)


def compare_sizes(params_b=20.0):
    configs = [
        ModelSizeEstimate("FP32 (original)",  params_b, "float32"),
        ModelSizeEstimate("FP16",              params_b, "float16"),
        ModelSizeEstimate("INT8",              params_b, "int8"),
        ModelSizeEstimate("INT4 (NF4)",        params_b, "int4"),
    ]
    baseline = configs[0].size_gb
    print(f"\nModel Size Comparison ({params_b}B params):")
    print(f"  {'Config':<20} {'Size':>8} {'Reduction':>10}")
    print("  " + "-"*40)
    for c in configs:
        print(f"  {c.name:<20} {c.size_gb:>6.1f}GB {(1-c.size_gb/baseline)*100:>8.0f}%")


def get_bnb_config():
    """4-bit NF4 quantisation config for BitsAndBytes."""
    return {
        "load_in_4bit":             True,
        "bnb_4bit_quant_type":      "nf4",
        "bnb_4bit_compute_dtype":   "float16",
        "bnb_4bit_use_double_quant": True,   # Nested quantisation
    }


def endpoint_config(model_size_gb):
    gpu_mem = 24  # A10G
    replicas = max(1, int(model_size_gb / (gpu_mem * 0.8)))
    return {
        "instance_type": "ml.g5.xlarge" if model_size_gb <= 24 else "ml.g5.2xlarge",
        "min_replicas":  replicas,
        "max_replicas":  replicas * 4,
        "scale_up_pct":  70,
        "max_batch":     8,
        "env": {
            "MODEL_DTYPE": "float16",
            "MAX_SEQ_LEN": "2048",
            "FLASH_ATTN":  "true",
        }
    }


def benchmark(scenarios):
    print(f"\n{'Config':<22} {'Size':>8} {'Batch':>6} {'Latency':>10} {'Req/s':>8}")
    print("  " + "-"*56)
    for s in scenarios:
        lat = s["size_gb"] * 0.05 + 0.2 + 0.1*(s["batch"]-1)
        thr = round(s["batch"]/lat, 2)
        print(f"  {s['name']:<22} {s['size_gb']:>6.0f}GB {s['batch']:>6} {lat:>8.2f}s {thr:>8.1f}")


if __name__ == "__main__":
    compare_sizes(20.0)

    print("\n4-bit Quantisation Config:")
    print(json.dumps(get_bnb_config(), indent=2))

    print("\nEndpoint Config (18GB optimised model):")
    print(json.dumps(endpoint_config(18), indent=2))

    benchmark([
        {"name": "FP32 original",   "size_gb": 80, "batch": 1},
        {"name": "FP16",            "size_gb": 40, "batch": 1},
        {"name": "INT8",            "size_gb": 20, "batch": 1},
        {"name": "INT4 optimised",  "size_gb": 18, "batch": 1},
        {"name": "INT4 batched x4", "size_gb": 18, "batch": 4},
    ])
