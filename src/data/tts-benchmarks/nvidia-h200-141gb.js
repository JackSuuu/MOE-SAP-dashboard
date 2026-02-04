import { NVIDIA_H100_80GB_SXM_MEASURED_ROWS } from "./nvidia-h100-80gb";
const H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE = 4.8 / 3.35; // ≈ 1.4328358209


const NVIDIA_H200_141GB_SXM_MEASURED_ROWS = [

]


const NVIDIA_H200_141GB_SXM_PROJECTED_ROWS = [
    {
    model: "gpt-oss-120b-low",
    quant: "mxfp4",
    dataset: "aime25",
    engine: "vllm",
    questionsPerHour: 1.877282254 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 160,
      parallel: 1,
      samples: 1,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "gpt-oss-120b-medium",
    quant: "mxfp4",
    dataset: "aime25",
    engine: "vllm",
    questionsPerHour: 0.996808318 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 160,
      parallel: 1,
      samples: 1,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "gpt-oss-120b-high",
    quant: "mxfp4",
    dataset: "aime25",
    engine: "vllm",
    questionsPerHour: 0.466872231 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 160,
      parallel: 1,
      samples: 1,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
// GPQA Diamond
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 149.4709376 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 1,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 75.31701214 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 2,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 46.22575531 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 3,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 32.7325006 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 4,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 25.06931833 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 5,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 20.61397231 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 6,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 17.34821681 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 7,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 15.01905229 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 8,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 13.1791205 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 9,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 11.75234834 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 10,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 10.62384163 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 11,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 9.67082987 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 12,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 8.878176916 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 13,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 8.180262506 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 14,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 7.566400131 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 15,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 7.046623089 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 16,
      parallel: 1,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 106.1668076 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 1,
      parallel: 1,
      samples: null,
      maxTokens: 16384,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 46.3099079 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 2,
      parallel: 1,
      samples: null,
      maxTokens: 16384,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 26.71381971 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 3,
      parallel: 1,
      samples: null,
      maxTokens: 16384,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 17.73118339 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 4,
      parallel: 1,
      samples: null,
      maxTokens: 16384,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 13.08258783 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 5,
      parallel: 1,
      samples: null,
      maxTokens: 16384,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 10.4316992 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 6,
      parallel: 1,
      samples: null,
      maxTokens: 16384,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 8.623721359 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 7,
      parallel: 1,
      samples: null,
      maxTokens: 16384,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 7.343260673 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 8,
      parallel: 1,
      samples: null,
      maxTokens: 16384,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 6.483338878 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 1,
      parallel: 16,
      samples: null,
      maxTokens: 16384,
      tools: null,
      source: "projected",
    },
  },
  {
    model: "Qwen3-30B-A3B-Instruct-2507",
    quant: "bf16",
    dataset: "GPQA Diamond",
    engine: "SGLang",
    questionsPerHour: 9.663881716 * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE,
    accuracy: null,
    color: "#00501d", // point color (green)
    meta: {
      gpu: "H200-SXM",
      gpuCount: 1,
      sequential: 1,
      parallel: 16,
      samples: null,
      maxTokens: 8192,
      tools: null,
      source: "projected",
    },
  },
]



const NVIDIA_H200_141GB_SXM_PROJECTED_ROWS_2 = NVIDIA_H100_80GB_SXM_MEASURED_ROWS
  // (optional) only project rows that are actually H100-80GB runs
  .filter((r) => r?.meta?.gpu === "H100-SXM")
  .map((r) => {
    const qph = r.questionsPerHour * H200SXM_OVER_H100SXM_MEMORY_BANDWIDTH_SCALE;

    return {
      ...r,
      questionsPerHour: Number(qph.toFixed(2)), // keep your existing style
      accuracy: null, // "leave blank" but keep valid JS
      color: "#00501d", // darker green than #22c55e
      meta: {
        ...r.meta,
        gpu: "H200-SXM",
        source: "projected"
      },
    };
  });

export const NVIDIA_H200_141GB_BENCHMARK_ROWS = [
  ...NVIDIA_H200_141GB_SXM_MEASURED_ROWS,
  ...NVIDIA_H200_141GB_SXM_PROJECTED_ROWS,
];