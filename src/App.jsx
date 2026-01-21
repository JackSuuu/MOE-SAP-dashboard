import { useState, useMemo } from 'react';
import {
  ComposedChart, Line, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine, LabelList,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import {
  Activity, Server, Settings, Cpu, Info, Zap, Percent
} from 'lucide-react';

const Card = ({ children, className = "" }) => (
  <div className={`bg-slate-800 border border-slate-700 rounded-lg p-4 ${className}`}>
    {children}
  </div>
);

const StatCard = ({ title, value, unit, subtext, icon: Icon }) => (
  <Card>
    <div className="flex justify-between items-start mb-2">
      <span className="text-slate-400 text-sm font-medium">{title}</span>
      {Icon && <Icon className="text-blue-400 w-5 h-5" />}
    </div>
    <div className="flex items-baseline gap-1">
      <span className="text-2xl font-bold text-slate-100">{value}</span>
      <span className="text-sm text-slate-400 font-medium">{unit}</span>
    </div>
    {subtext && <div className="text-xs text-slate-500 mt-1">{subtext}</div>}
  </Card>
);

const ScenarioButton = ({ active, onClick, label, desc }) => (
  <button
    onClick={onClick}
    className={`w-full text-left p-3 rounded-md border transition-all ${
      active
        ? 'bg-blue-900/30 border-blue-500 text-blue-100'
        : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-slate-600'
    }`}
  >
    <div className="font-semibold text-sm">{label}</div>
    <div className="text-xs opacity-70 mt-1">{desc}</div>
  </button>
);

export default function App() {
  // --- Model Configurations ---
  const MODEL_CONFIGS = {
    'qwen3': {
      name: 'Qwen3-30B-A3B',
      d_model: 2048,
      n_heads: 32,
      n_kv: 4,
      d_head: 128,
      L: 48,
      d_ff: 768,
      E: 128,
      k: 8,
      V: 151936,
      P: 2
    },
    'deepseek-v2-lite': {
      name: 'Deepseek-V2-Lite',
      d_model: 2048,
      n_heads: 32,
      n_kv: 4,
      d_head: 128,
      L: 48,
      d_ff: 768,
      E: 128,
      k: 8,
      V: 151936,
      P: 2,
      expertSize: 0.229,
      restShared: 1.024,
      smbuMap: {
        1: 0.345163434301608,
        4: 0.386,
        8: 0.406,
        16: 0.429,
        32: 0.478644231191122,
        64: 0.396806645718611,
        128: 0.325042264414248
      }
    },
    'qwen1.5-moe': {
      name: 'Qwen1.5-MoE',
      d_model: 2048,
      n_heads: 32,
      n_kv: 4,
      d_head: 128,
      L: 48,
      d_ff: 768,
      E: 128,
      k: 8,
      V: 151936,
      P: 2,
      expertSize: 0.207,
      restShared: 1.871,
      smbuMap: {
        1: 0.404936278087575,
        4: 0.423,
        8: 0.442,
        16: 0.467870755068306,
        32: 0.469105695655845,
        64: 0.387568270405655,
        128: 0.320088529302412
      }
    },
    'deepseek-r1': {
      name: 'Deepseek-R1 (Fully Activated)',
      d_model: 2048,
      n_heads: 32,
      n_kv: 4,
      d_head: 128,
      L: 48,
      d_ff: 768,
      E: 256,
      k: 8,
      V: 151936,
      P: 2,
      expertSize: 0.229,
      restShared: 0.512,
      smbuMap: {
        1: 0.668,
        4: 0.742,
        8: 0.779,
        16: 0.824,
        32: 0.920,
        64: 0.762,
        128: 0.625
      }
    },
    'mixtral-8x22b': {
      name: 'Mixtral-8x22B (Fully Activated)',
      d_model: 2048,
      n_heads: 32,
      n_kv: 4,
      d_head: 128,
      L: 48,
      d_ff: 768,
      E: 8,
      k: 2,
      V: 151936,
      P: 2,
      expertSize: 2.861,
      restShared: 2.355,
      smbuMap: {
        1: 0.663,
        4: 0.737,
        8: 0.774,
        16: 0.819,
        32: 0.913,
        64: 0.758,
        128: 0.622
      }
    }
  };

  // --- State ---
  const [selectedModel, setSelectedModel] = useState('deepseek-v2-lite');
  const [scenario, setScenario] = useState('4k-256');
  
  // Inputs
  const [batchSize, setBatchSize] = useState(1);
  const [inputLen, setInputLen] = useState(4000);
  const [outputLen, setOutputLen] = useState(256);
  const [sloMs, setSloMs] = useState(50); // Target Time Per Output Token (ms)
  const [hardwareBw, setHardwareBw] = useState(768); // Default A6000
  const [smbu, setSmbu] = useState(16.33); // Fixed S-MBU
  const [numGpus, setNumGpus] = useState(1); // Supply side GPU count

  // CAP Radar Chart selections (3 configs)
  const [capConfig1, setCapConfig1] = useState('qwen3-30b-a3b-5xa5000');
  const [capConfig2, setCapConfig2] = useState('qwen1.5-moe-1xa6000');
  const [capConfig3, setCapConfig3] = useState('qwen3-235b-bf16-8xh100');
  const [capDataset, setCapDataset] = useState('gsm8k');

  // CAP benchmark data (Model-Precision-GPU configurations)
  const CAP_CONFIGS = {
    'qwen3-30b-a3b-5xa5000': {
      label: 'Qwen3-30B-A3B / BF16 / 5xRTX A5000',
      model: 'Qwen3-30B-A3B',
      precision: 'BF16',
      gpu: '5xRTX A5000',
      accuracy: 81.12,
      cost: 15342.20,
      tpot: 0.05,
      throughput: 1402.01,
      color: '#3b82f6'
    },
    'qwen1.5-moe-1xa6000': {
      label: 'Qwen1.5-MoE-A2.7B-Chat / BF16 / 1xRTX A6000',
      model: 'Qwen1.5-MoE-A2.7B-Chat',
      precision: 'BF16',
      gpu: '1xRTX A6000',
      accuracy: 45.72,
      cost: 7158.92,
      tpot: 0.03,
      throughput: 599.35,
      color: '#22c55e'
    },
    'qwen3-235b-fp8-2xh200': {
      label: 'Qwen3-235B-A22B-Thinking / FP8 / 2xH200',
      model: 'Qwen3-235B-A22B-Thinking-2507-FP8',
      precision: 'FP8',
      gpu: '2xH200',
      accuracy: 68.84,
      cost: 104052.07,
      tpot: 0.02,
      throughput: 1136.77,
      color: '#f97316'
    },
    'qwen3-235b-bf16-4xh200': {
      label: 'Qwen3-235B-A22B-Thinking / BF16 / 4xH200',
      model: 'Qwen3-235B-A22B-Thinking-2507',
      precision: 'BF16',
      gpu: '4xH200',
      accuracy: 70.28,
      cost: 195252.11,
      tpot: 0.02,
      throughput: 1206.32,
      color: '#a855f7'
    },
    'qwen3-235b-bf16-8xh100': {
      label: 'Qwen3-235B-A22B / BF16 / 8xH100',
      model: 'Qwen3-235B-A22B',
      precision: 'BF16',
      gpu: '8xH100',
      accuracy: 71.19,
      cost: 344657.14,
      tpot: 0.03,
      throughput: 1694.30,
      color: '#ef4444'
    },
    'qwen3-30b-instruct-4xa6000': {
      label: 'Qwen3-30B-A3B-Instruct / BF16 / 4xRTX A6000',
      model: 'Qwen3-30B-A3B-Instruct-2507',
      precision: 'BF16',
      gpu: '4xRTX A6000',
      accuracy: 53.30,
      cost: 21600.27,
      tpot: 0.02,
      throughput: 638.03,
      color: '#60a5fa'
    },
    'qwen3-30b-thinking-4xa6000': {
      label: 'Qwen3-30B-A3B-Thinking / BF16 / 4xRTX A6000',
      model: 'Qwen3-30B-A3B-Thinking-2507',
      precision: 'BF16',
      gpu: '4xRTX A6000',
      accuracy: 69.29,
      cost: 21600.54,
      tpot: 0.04,
      throughput: 1701.41,
      color: '#4ade80'
    },
    'qwen3-30b-4xa6000': {
      label: 'Qwen3-30B-A3B / BF16 / 4xRTX A6000',
      model: 'Qwen3-30B-A3B',
      precision: 'BF16',
      gpu: '4xRTX A6000',
      accuracy: 80.67,
      cost: 21600.69,
      tpot: 0.03,
      throughput: 1417.49,
      color: '#9333ea'
    },
    'qwen3-30b-2xa100': {
      label: 'Qwen3-30B-A3B / BF16 / 2xA100',
      model: 'Qwen3-30B-A3B',
      precision: 'BF16',
      gpu: '2xA100',
      accuracy: 80.97,
      cost: 54380.91,
      tpot: 0.01,
      throughput: 1806.09,
      color: '#f87171'
    },
  };

  const MODEL_CONFIG = MODEL_CONFIGS[selectedModel];

  // Get S-MBU based on model and batch size
  const getSmbu = (model, batchSize) => {
    const config = MODEL_CONFIGS[model];
    if (config.smbuMap && config.smbuMap[batchSize]) {
      return config.smbuMap[batchSize] * 100; // Convert to percentage
    }
    return 16.33; // Default fallback
  };

  const currentSmbu = getSmbu(selectedModel, batchSize);

  // Hardware options
  const HARDWARE_PRESETS = [
    { name: 'NVIDIA A6000', bw: 768 },
    { name: 'NVIDIA H100 (SXM)', bw: 3350 },
    { name: 'NVIDIA A100 (SXM4)', bw: 2039 },
    { name: 'NVIDIA H20', bw: 4000 },
    { name: 'NVIDIA L20', bw: 230 },
    { name: 'Consumer RTX 4090', bw: 1008 }
  ];

  // --- Handlers ---
  const handleScenarioChange = (id) => {
    setScenario(id);
    if (id === '4k-256') {
      setBatchSize(1);
      setInputLen(4000);
      setOutputLen(256);
    } else if (id === '13k-1k') {
      setBatchSize(1);
      setInputLen(13000);
      setOutputLen(1000);
    }
  };

  // --- Calculations Helper (Using table data) ---
  const calculateDemand = (customBatchSize = null) => {
    const B = customBatchSize !== null ? customBatchSize : batchSize;
    const { expertSize, restShared, E, k, P } = MODEL_CONFIG;

    // If model has pre-calculated expert size (from table), use simplified formula
    if (expertSize && restShared) {
      // From table: 
      // 1. Calculate unique experts based on batch size
      const totalExpertSelections = B * k;
      const E_unique_expected = E * (1 - Math.pow(1 - 1/E, totalExpertSelections));
      
      // 2. Activation size (GB) = rest-shared + E_unique * expert_size
      const activationWeights = restShared + (E_unique_expected * expertSize);
      
      // 3. Theoretical bandwidth (GB/s) = activation weights / tpot (in seconds)
      // Table uses tpot = 0.25s, we use sloMs converted to seconds
      const tpot_seconds = sloMs / 1000;
      const req_bw_gbs = activationWeights / tpot_seconds;
      
      return {
        reqBwGBs: req_bw_gbs,
        activeParamsGB: activationWeights,
        kvCacheGB: 0, // Table doesn't show KV cache separately
        uniqueExperts: E_unique_expected
      };
    }
    
    // Fallback to original detailed calculation for Qwen3
    const { d_model, n_heads, n_kv, d_head, L, d_ff, V } = MODEL_CONFIG;
    const avgContext = inputLen + (outputLen + 1) / 2;

    const W_embed = V * d_model * P;
    const W_Q = d_model * (n_heads * d_head) * P;
    const W_K = d_model * (n_kv * d_head) * P;
    const W_V = d_model * (n_kv * d_head) * P;
    const W_O = (n_heads * d_head) * d_model * P;
    const BW_attn = W_Q + W_K + W_V + W_O;
    const W_gate = d_model * E * P;
    const W_expert_single = 3 * d_model * d_ff * P;
    
    const totalExpertSelections = B * k;
    const E_unique_expected = E * (1 - Math.pow(1 - 1/E, totalExpertSelections));
    const BW_moe = E_unique_expected * W_expert_single;
    
    const BW_layer = BW_attn + W_gate + BW_moe;
    const BW_all_layers = L * BW_layer;
    const W_lm_head = d_model * V * P;
    const W_model_total = W_embed + BW_all_layers + W_lm_head;
    const W_kv_current = L * B * avgContext * 2 * n_kv * d_head * P;
    const BW_total = W_model_total + W_kv_current;
    const req_bw_gbs = (BW_total / (sloMs / 1000)) / 1e9;
    
    return {
      reqBwGBs: req_bw_gbs,
      activeParamsGB: W_model_total / 1e9,
      kvCacheGB: W_kv_current / 1e9,
      uniqueExperts: E_unique_expected
    };
  };

  const currentStats = useMemo(() => {
    return calculateDemand();
  }, [batchSize, inputLen, outputLen, sloMs, selectedModel]);

  // Custom label component for device names with connectors
  const CustomLabel = (props) => {
    const { x, y, value, payload } = props;
    if (!payload) return null;
    
    const color = payload.category === 'edge' ? '#f97316' : '#3b82f6';
    const pos = payload.labelPos || 'bottomRight';
    
    let dx = 0, dy = 0, textAnchor = 'start';
    let lineX1 = x, lineY1 = y, lineX2 = x, lineY2 = y;
    
    switch(pos) {
      case 'top':
        dx = 0; dy = -35; textAnchor = 'middle';
        lineX2 = x; lineY2 = y - 10;
        break;
      case 'topRight':
        dx = 28; dy = -25; textAnchor = 'start';
        lineX2 = x + 8; lineY2 = y - 8;
        break;
      case 'bottom':
        dx = 0; dy = 38; textAnchor = 'middle';
        lineX2 = x; lineY2 = y + 10;
        break;
      case 'bottomRight':
        dx = 28; dy = 28; textAnchor = 'start';
        lineX2 = x + 8; lineY2 = y + 10;
        break;
      case 'left':
        dx = -28; dy = 5; textAnchor = 'end';
        lineX2 = x - 10; lineY2 = y;
        break;
      case 'right':
      default:
        dx = 28; dy = 5; textAnchor = 'start';
        lineX2 = x + 10; lineY2 = y;
        break;
    }
    
    return (
      <g>
        {/* Connecting line */}
        <line 
          x1={lineX1} 
          y1={lineY1} 
          x2={lineX2} 
          y2={lineY2} 
          stroke={color}
          strokeWidth={1}
          strokeDasharray="2,2"
          opacity={0.6}
        />
        {/* Label text */}
        <text 
          x={x + dx} 
          y={y + dy} 
          fill={color}
          fontSize={11}
          fontWeight={600}
          textAnchor={textAnchor}
        >
          {value}
        </text>
      </g>
    );
  };

  // --- Chart Data: Power vs Bandwidth (MoE-CAP style) ---
  const chartData = useMemo(() => {
    // Calculate theoretical bandwidth for current model (Dense - based on batch size)
    const stats = calculateDemand();
    const denseBw = stats.reqBwGBs;
    
    // Calculate Fully Activated bandwidth (all experts active)
    let fullyActivatedBw = 0;
    if (MODEL_CONFIG.expertSize && MODEL_CONFIG.restShared) {
      const activationWeights = MODEL_CONFIG.restShared + (MODEL_CONFIG.E * MODEL_CONFIG.expertSize);
      const tpot_seconds = sloMs / 1000;
      fullyActivatedBw = activationWeights / tpot_seconds;
    }
    
    // Actual Bandwidth (Supply) - Horizontal Line
    const actualBw = hardwareBw * numGpus * (currentSmbu / 100);
    
    // Device points from benchmark data
    // Peak Bandwidth = HBM/GDDR memory bandwidth (blue circles)
    const peakDevices = [
      // Data Center Systems (Multi-GPU)
      { name: 'DGX-H100', bandwidth: 26800, power: 10200, category: 'datacenter-system', type: 'peak' },
      { name: 'DGX-A100', bandwidth: 16296, power: 6500, category: 'datacenter-system', type: 'peak' },
      // Data Center Cards
      { name: 'AMD MI300X', bandwidth: 5300, power: 750, category: 'datacenter-card', type: 'peak' },
      { name: 'H100-SXM', bandwidth: 3350, power: 700, category: 'datacenter-card', type: 'peak' },
      { name: 'AWS Trainium 2', bandwidth: 2900, power: 480, category: 'datacenter-card', type: 'peak' },
      { name: 'A100-80G-SXM4', bandwidth: 2037, power: 400, category: 'datacenter-card', type: 'peak' },
      { name: 'H100-PCIe', bandwidth: 2000, power: 350, category: 'datacenter-card', type: 'peak' },
      { name: 'A100-80G-PCIe', bandwidth: 1935, power: 300, category: 'datacenter-card', type: 'peak' },
      { name: 'A6000', bandwidth: 768, power: 300, category: 'datacenter-card', type: 'peak' },
      { name: 'A5000', bandwidth: 768, power: 230, category: 'datacenter-card', type: 'peak' },
      // Personal (Consumer GPUs) - 调整power避免重叠
      { name: '5090', bandwidth: 1790, power: 575, category: 'personal', type: 'peak' },
      { name: '4090', bandwidth: 1010, power: 450, category: 'personal', type: 'peak' },
      { name: '3090Ti', bandwidth: 1010, power: 400, category: 'personal', type: 'peak' },
      { name: '5080', bandwidth: 960, power: 360, category: 'personal', type: 'peak' },
      { name: '3080Ti', bandwidth: 912.4, power: 350, category: 'personal', type: 'peak' },
      { name: '4080', bandwidth: 716.8, power: 320, category: 'personal', type: 'peak' },
      // SoC (Apple Silicon) - Unified Memory - 调整power避免重叠
      { name: 'Apple M4 max', bandwidth: 546, power: 90, category: 'soc', type: 'peak' },
      { name: 'Apple M3 max', bandwidth: 400, power: 70, category: 'soc', type: 'peak' },
      { name: 'Apple M2 max', bandwidth: 400, power: 50, category: 'soc', type: 'peak' },
      { name: 'Apple M1 max', bandwidth: 400, power: 35, category: 'soc', type: 'peak' },
      // Autonomous (NVIDIA Jetson)
      { name: 'Orin AGX', bandwidth: 204.8, power: 60, category: 'autonomous', type: 'peak' },
      { name: 'Xavier AGX', bandwidth: 136.5, power: 30, category: 'autonomous', type: 'peak' },
      { name: 'Orin NX', bandwidth: 102.4, power: 25, category: 'autonomous', type: 'peak' },
      { name: 'Jetson Nano', bandwidth: 25.6, power: 10, category: 'autonomous', type: 'peak' },
    ];
    
    // Offloading Bandwidth = PCIe/ethernet bandwidth (orange squares)
    const offloadDevices = [
      // Data Center Systems (Multi-GPU with NVLink)
      { name: 'DGX-H100', bandwidth: 1280, power: 10200, category: 'datacenter-system', type: 'pcie' },
      { name: 'DGX-A100', bandwidth: 512, power: 6500, category: 'datacenter-system', type: 'pcie' },
      // Data Center Cards (PCIe) - 调整power避免与peak重叠
      { name: 'AMD MI300X', bandwidth: 128, power: 850, category: 'datacenter-card', type: 'pcie' },
      { name: 'H100-SXM', bandwidth: 128, power: 780, category: 'datacenter-card', type: 'pcie' },
      { name: 'AWS Trainium 2', bandwidth: 128, power: 520, category: 'datacenter-card', type: 'pcie' },
      { name: 'H100-PCIe', bandwidth: 128, power: 380, category: 'datacenter-card', type: 'pcie' },
      { name: 'A100-80G-SXM4', bandwidth: 64, power: 440, category: 'datacenter-card', type: 'pcie' },
      { name: 'A100-80G-PCIe', bandwidth: 64, power: 340, category: 'datacenter-card', type: 'pcie' },
      { name: 'A6000', bandwidth: 64, power: 300, category: 'datacenter-card', type: 'pcie' },
      { name: 'A5000', bandwidth: 64, power: 230, category: 'datacenter-card', type: 'pcie' },
      // Personal (Consumer GPUs - PCIe) - 调整power避免重叠
      { name: '5090', bandwidth: 128, power: 620, category: 'personal', type: 'pcie' },
      { name: '5080', bandwidth: 128, power: 400, category: 'personal', type: 'pcie' },
      { name: '4090', bandwidth: 64, power: 500, category: 'personal', type: 'pcie' },
      { name: '4080', bandwidth: 64, power: 350, category: 'personal', type: 'pcie' },
      { name: '3090Ti', bandwidth: 64, power: 420, category: 'personal', type: 'pcie' },
      { name: '3080Ti', bandwidth: 64, power: 380, category: 'personal', type: 'pcie' },
      // Autonomous (NVIDIA Jetson)
      { name: 'Orin AGX', bandwidth: 16, power: 65, category: 'autonomous', type: 'pcie' },
      { name: 'Xavier AGX', bandwidth: 16, power: 35, category: 'autonomous', type: 'pcie' },
      { name: 'Orin NX', bandwidth: 16, power: 28, category: 'autonomous', type: 'pcie' },
      { name: 'Jetson Nano', bandwidth: 4, power: 12, category: 'autonomous', type: 'pcie' },
    ];

    // Line data for horizontal requirement lines
    const lineData = [];
    for (let power = 5; power <= 20000; power *= 1.2) {
      lineData.push({
        power: Math.round(power),
        dense: denseBw,
        fullyActivated: fullyActivatedBw,
        actual: actualBw,
      });
    }

    return { lineData, peakDevices, offloadDevices, denseBw, fullyActivatedBw, actualBw };
  }, [hardwareBw, numGpus, currentSmbu, sloMs, batchSize, selectedModel]);

  // --- Chart Data: Batch Size vs Bandwidth (shows MoE curve) ---
  const batchChartData = useMemo(() => {
    const data = [];
    
    // Generate curve data for batch sizes 1 to 256
    const maxBatch = 256;
    for (let b = 1; b <= maxBatch; b += (b < 32 ? 1 : (b < 64 ? 2 : 4))) {
      const stats = calculateDemand(b);
      const batchSmbu = getSmbu(selectedModel, b);
      const actualBw = hardwareBw * numGpus * (batchSmbu / 100);
      data.push({
        batchSize: b,
        theoretical: stats.reqBwGBs,
        actual: actualBw,
        uniqueExperts: stats.uniqueExperts.toFixed(1),
      });
    }
    return data;
  }, [hardwareBw, numGpus, smbu, sloMs, selectedModel]);

  // --- CAP Radar Chart Data ---
  const capRadarData = useMemo(() => {
    const selectedConfigs = [capConfig1, capConfig2, capConfig3].filter(c => c !== '');
    
    // Find max values for normalization
    const allConfigs = Object.values(CAP_CONFIGS);
    const maxAccuracy = Math.max(...allConfigs.map(c => c.accuracy));
    const maxCost = Math.max(...allConfigs.map(c => c.cost));
    const maxTpot = Math.max(...allConfigs.map(c => c.tpot));
    const maxThroughput = Math.max(...allConfigs.map(c => c.throughput));
    
    // Normalize function (0-100 scale)
    const normalize = (val, max) => (val / max) * 100;
    const normalizeCost = (val, max) => ((max - val) / max) * 100; // Lower is better
    const normalizeTpot = (val, max) => ((max - val) / max) * 100; // Lower is better
    
    // Create radar data for each metric
    const radarData = [
      { metric: 'Accuracy (%)', fullMark: 100 },
      { metric: 'Cost ($)', fullMark: 100 },
      { metric: 'TPOT (s)', fullMark: 100 },
      { metric: 'Throughput (T/s)', fullMark: 100 },
    ];
    
    selectedConfigs.forEach(configKey => {
      const config = CAP_CONFIGS[configKey];
      if (config) {
        radarData[0][configKey] = normalize(config.accuracy, maxAccuracy);
        radarData[1][configKey] = normalizeCost(config.cost, maxCost);
        radarData[2][configKey] = normalizeTpot(config.tpot, maxTpot);
        radarData[3][configKey] = normalize(config.throughput, maxThroughput);
      }
    });
    
    return { radarData, selectedConfigs };
  }, [capConfig1, capConfig2, capConfig3]);

  // Get available options for each dropdown (excluding already selected)
  const getAvailableOptions = (currentValue, otherValues) => {
    return Object.keys(CAP_CONFIGS).filter(key => 
      key === currentValue || !otherValues.includes(key)
    );
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans p-6">
      <header className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Activity className="w-8 h-8 text-blue-500" />
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
            MoE Bandwidth Estimator
          </h1>
          <span className="px-2 py-1 bg-slate-800 border border-slate-700 rounded text-xs text-slate-400">
            {MODEL_CONFIG.name}
          </span>
        </div>
        <p className="text-slate-400">
          Theoretical vs. Actual Bandwidth Analysis (SGLang Profiling Reference)
        </p>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Sidebar - Controls */}
        <div className="space-y-6">
          
          {/* Scenarios */}
          <section>
            <h2 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
              <Server className="w-4 h-4" /> Context Size
            </h2>
            <div className="space-y-2">
              <ScenarioButton 
                active={scenario === '4k-256'}
                onClick={() => handleScenarioChange('4k-256')}
                label="IN: 4K, OUT: 256"
                desc="Standard chat scenario"
              />
              <ScenarioButton 
                active={scenario === '13k-1k'}
                onClick={() => handleScenarioChange('13k-1k')}
                label="IN: 13K, OUT: 1K"
                desc="Long context generation"
              />
            </div>
          </section>

          {/* Parameters */}
          <section className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <h2 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Settings className="w-4 h-4" /> Parameters
            </h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Model</label>
                <select 
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm focus:border-blue-500 outline-none"
                >
                  <option value="deepseek-v2-lite">Deepseek-V2-Lite</option>
                  <option value="deepseek-r1">Deepseek-R1 (Fully Activated)</option>
                  <option value="qwen1.5-moe">Qwen1.5-MoE</option>
                  <option value="mixtral-8x22b">Mixtral-8x22B (Fully Activated)</option>
                  <option value="qwen3">Qwen3-30B-A3B</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Batch Size</label>
                <select 
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm focus:border-blue-500 outline-none"
                >
                  <option value={1}>1</option>
                  <option value={4}>4</option>
                  <option value={8}>8</option>
                  <option value={16}>16</option>
                  <option value={32}>32</option>
                  <option value={64}>64</option>
                  <option value={128}>128</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Target SLO (ms/token)</label>
                <select 
                  value={sloMs}
                  onChange={(e) => setSloMs(parseInt(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm focus:border-blue-500 outline-none"
                >
                  <option value={10}>10 ms (Ultra Fast)</option>
                  <option value={20}>20 ms (Very Fast)</option>
                  <option value={50}>50 ms (Fast)</option>
                  <option value={100}>100 ms (Standard)</option>
                  <option value={200}>200 ms (Slow)</option>
                  <option value={250}>250 ms (Table Reference)</option>
                </select>
                <p className="text-[10px] text-slate-500 mt-1">Table uses 250ms (0.25s tpot)</p>
              </div>
            </div>
          </section>
        </div>

        {/* Right Side - Stats & Charts */}
        <div className="lg:col-span-2 space-y-6">

          {/* Chart 1: Power vs Bandwidth (MoE-CAP Style) */}
          <Card className="h-[700px] relative">
            <h3 className="text-lg font-semibold mb-4 pl-2 border-l-4 border-blue-500">
              MoE Deployment Benchmarking - Power vs Bandwidth
            </h3>
            
            {/* Legend Box */}
            <div className="absolute top-12 right-6 bg-slate-800/90 border border-slate-600 rounded px-3 py-2 z-10">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                <span className="text-xs text-slate-300">Peak Bandwidth</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                <span className="text-xs text-slate-300">Offloading Bandwidth</span>
              </div>
            </div>
            
            <ResponsiveContainer width="100%" height="90%">
              <ComposedChart margin={{ top: 20, right: 150, left: 80, bottom: 80 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis 
                  dataKey="power" 
                  type="number"
                  domain={[5, 20000]} 
                  scale="log"
                  tick={{ fill: '#94a3b8', fontSize: 10 }} 
                  label={{ value: 'Power (W)', position: 'insideBottom', offset: -15, fill: '#94a3b8' }}
                  allowDataOverflow={true}
                />
                <YAxis 
                  stroke="#94a3b8"
                  scale="log"
                  domain={[10, 10000]}
                  tick={{ fill: '#94a3b8', fontSize: 10 }}
                  label={{ value: 'Bandwidth (GB/s)', angle: -90, position: 'insideLeft', offset: -40, fill: '#94a3b8' }}
                  allowDataOverflow={true}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', padding: '8px 12px' }}
                  content={({ active, payload, coordinate }) => {
                    if (active && payload && payload.length > 0) {
                      // 找到最匹配的数据点（第一个有type字段的）
                      let matchedData = null;
                      let matchedType = null;
                      
                      for (let i = 0; i < payload.length; i++) {
                        const p = payload[i];
                        if (p && p.payload && p.payload.name && p.payload.type) {
                          matchedData = p.payload;
                          matchedType = p.payload.type;
                          break; // 只取第一个匹配的
                        }
                      }
                      
                      if (matchedData) {
                        const isPeak = matchedType === 'peak';
                        const color = isPeak ? '#3b82f6' : '#f97316';
                        const typeLabel = isPeak ? 'Peak BW (Memory)' : 'PCIe/NVLink BW';
                        return (
                          <div style={{ backgroundColor: '#1e293b', border: `2px solid ${color}`, padding: '10px 14px', borderRadius: '6px', boxShadow: '0 4px 12px rgba(0,0,0,0.3)' }}>
                            <div style={{ color: color, fontWeight: 600, marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                              {isPeak ? (
                                <svg width="12" height="12"><circle cx="6" cy="6" r="5" fill={color} stroke="#fff" strokeWidth="1"/></svg>
                              ) : (
                                <svg width="12" height="12"><rect x="1" y="1" width="10" height="10" fill={color} stroke="#fff" strokeWidth="1"/></svg>
                              )}
                              {matchedData.name}
                            </div>
                            <div style={{ color: color, fontSize: '13px', fontWeight: 500 }}>
                              {typeLabel}: {matchedData.bandwidth.toLocaleString()} GB/s
                            </div>
                            <div style={{ color: '#94a3b8', fontSize: '12px', marginTop: '4px' }}>
                              Power: {matchedData.power}W
                            </div>
                            <div style={{ color: '#64748b', fontSize: '11px', marginTop: '2px', textTransform: 'capitalize' }}>
                              {matchedData.category.replace(/-/g, ' ')}
                            </div>
                          </div>
                        );
                      }
                    }
                    return null;
                  }}
                />
                <Legend 
                  verticalAlign="top" 
                  height={36} 
                  wrapperStyle={{ top: -10, color: '#ffffff' }}
                  iconType="line"
                  payload={[
                    { value: `${MODEL_CONFIG.name} (BS=${batchSize})`, type: 'line', color: '#3b82f6' },
                    { value: `${MODEL_CONFIG.name} (Fully Activated)`, type: 'line', color: '#ef4444' }
                  ]}
                />
                
                {/* Horizontal lines for current model: BS=batchSize and Fully Activated */}
                <Line 
                  data={chartData.lineData}
                  type="monotone" 
                  dataKey="dense" 
                  name={`${MODEL_CONFIG.name} (BS=${batchSize})`}
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  strokeDasharray="5 5" 
                  dot={false}
                  isAnimationActive={false}
                  connectNulls
                />
                <Line 
                  data={chartData.lineData}
                  type="monotone" 
                  dataKey="fullyActivated" 
                  name={`${MODEL_CONFIG.name} (Fully Activated)`}
                  stroke="#ef4444" 
                  strokeWidth={2}
                  strokeDasharray="5 5" 
                  dot={false}
                  isAnimationActive={false}
                  connectNulls
                />

                {/* Device scatter points - blue circles for Peak Bandwidth (HBM/Memory) */}
                <Scatter 
                  data={chartData.peakDevices}
                  dataKey="bandwidth"
                  name="Peak Bandwidth"
                  fill="#3b82f6"
                  stroke="#3b82f6"
                  isAnimationActive={false}
                  legendType="none"
                  shape={(props) => {
                    const { cx, cy } = props;
                    return <circle cx={cx} cy={cy} r={6} fill="#3b82f6" stroke="#1e40af" strokeWidth={1.5} />;
                  }}
                />
                
                {/* Device scatter points - orange circles for Offloading Bandwidth (PCIe) */}
                <Scatter 
                  data={chartData.offloadDevices}
                  dataKey="bandwidth"
                  name="Offloading Bandwidth"
                  fill="#f97316"
                  stroke="#f97316"
                  isAnimationActive={false}
                  legendType="none"
                  shape={(props) => {
                    const { cx, cy } = props;
                    return <circle cx={cx} cy={cy} r={6} fill="#f97316" stroke="#c2410c" strokeWidth={1.5} />;
                  }}
                />
                
              </ComposedChart>
            </ResponsiveContainer>
          </Card>

          {/* Chart 2: CAP Radar Plot */}
          <Card className="mb-8">
            <h3 className="text-lg font-semibold mb-4 pl-2 border-l-4 border-purple-500">
              CAP Radar Plot - Cost, Accuracy, Performance
            </h3>
            
            {/* Dataset Selector */}
            <div className="mb-4">
              <label className="block text-xs font-medium text-slate-400 mb-1">Dataset</label>
              <select 
                value={capDataset}
                onChange={(e) => setCapDataset(e.target.value)}
                className="w-48 bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs focus:border-blue-500 outline-none"
              >
                <option value="gsm8k">GSM8K</option>
              </select>
            </div>
            
            {/* Config Selectors */}
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Config 1</label>
                <select 
                  value={capConfig1}
                  onChange={(e) => setCapConfig1(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs focus:border-blue-500 outline-none"
                >
                  <option value="">-- Select --</option>
                  {getAvailableOptions(capConfig1, [capConfig2, capConfig3]).map(key => (
                    <option key={key} value={key}>{CAP_CONFIGS[key].label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Config 2</label>
                <select 
                  value={capConfig2}
                  onChange={(e) => setCapConfig2(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs focus:border-blue-500 outline-none"
                >
                  <option value="">-- Select --</option>
                  {getAvailableOptions(capConfig2, [capConfig1, capConfig3]).map(key => (
                    <option key={key} value={key}>{CAP_CONFIGS[key].label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Config 3</label>
                <select 
                  value={capConfig3}
                  onChange={(e) => setCapConfig3(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs focus:border-blue-500 outline-none"
                >
                  <option value="">-- Select --</option>
                  {getAvailableOptions(capConfig3, [capConfig1, capConfig2]).map(key => (
                    <option key={key} value={key}>{CAP_CONFIGS[key].label}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Legend for selected configs */}
            <div className="flex flex-wrap gap-4 mb-4 px-2">
              {capRadarData.selectedConfigs.map(configKey => {
                const config = CAP_CONFIGS[configKey];
                return (
                  <div key={configKey} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: config.color }}></div>
                    <span className="text-xs text-slate-300">{config.label}</span>
                  </div>
                );
              })}
            </div>

            <ResponsiveContainer width="100%" height={400}>
              <RadarChart data={capRadarData.radarData} margin={{ top: 20, right: 40, bottom: 20, left: 40 }}>
                <PolarGrid stroke="#475569" />
                <PolarAngleAxis 
                  dataKey="metric" 
                  tick={{ fill: '#94a3b8', fontSize: 12 }}
                />
                <PolarRadiusAxis 
                  angle={30} 
                  domain={[0, 100]} 
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  tickCount={5}
                />
                {capRadarData.selectedConfigs.map(configKey => {
                  const config = CAP_CONFIGS[configKey];
                  return (
                    <Radar 
                      key={configKey}
                      name={config.label}
                      dataKey={configKey}
                      stroke={config.color}
                      fill={config.color}
                      fillOpacity={0.2}
                      strokeWidth={2}
                    />
                  );
                })}
                <Legend 
                  wrapperStyle={{ bottom: -10, color: '#ffffff' }}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                  formatter={(value, name) => [`${value.toFixed(1)}%`, CAP_CONFIGS[name]?.label || name]}
                />
              </RadarChart>
            </ResponsiveContainer>

            {/* Raw Values Table */}
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left py-2 px-2 text-slate-400">Config</th>
                    <th className="text-right py-2 px-2 text-slate-400">Accuracy</th>
                    <th className="text-right py-2 px-2 text-slate-400">Cost ($)</th>
                    <th className="text-right py-2 px-2 text-slate-400">TPOT (s)</th>
                    <th className="text-right py-2 px-2 text-slate-400">Throughput</th>
                  </tr>
                </thead>
                <tbody>
                  {capRadarData.selectedConfigs.map(configKey => {
                    const config = CAP_CONFIGS[configKey];
                    return (
                      <tr key={configKey} className="border-b border-slate-800">
                        <td className="py-2 px-2" style={{ color: config.color }}>{config.label}</td>
                        <td className="text-right py-2 px-2 text-slate-300">{config.accuracy}%</td>
                        <td className="text-right py-2 px-2 text-slate-300">${config.cost.toLocaleString()}</td>
                        <td className="text-right py-2 px-2 text-slate-300">{config.tpot}s</td>
                        <td className="text-right py-2 px-2 text-slate-300">{config.throughput} T/s</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Info Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <h4 className="font-semibold text-slate-300 mb-3 border-b border-slate-700 pb-2">MoE Model Config</h4>
              <ul className="space-y-2 text-sm text-slate-400">
                <li className="flex justify-between">
                  <span>Model:</span>
                  <span className="text-slate-200">{MODEL_CONFIG.name}</span>
                </li>
                <li className="flex justify-between">
                  <span>Active Experts (top-k):</span>
                  <span className="text-slate-200">{MODEL_CONFIG.k} / {MODEL_CONFIG.E}</span>
                </li>
                <li className="flex justify-between">
                  <span>Unique Experts (B={batchSize}):</span>
                  <span className="text-green-400">{currentStats.uniqueExperts?.toFixed(1)} expected</span>
                </li>
                <li className="flex justify-between">
                  <span>MBU (Efficiency):</span>
                  <span className="text-slate-200">{currentSmbu.toFixed(2)}%</span>
                </li>
              </ul>
            </Card>

            <Card>
              <h4 className="font-semibold text-slate-300 mb-3 border-b border-slate-700 pb-2">MoE-CAP Formula</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                <strong className="text-green-400">Unique Experts (Probabilistic):</strong><br/>
                <code className="text-green-300 bg-slate-900 px-1 rounded text-[10px]">E_unique = E × (1 - (1-1/E)^(B×k))</code>
              </p>
              <p className="text-xs text-slate-400 leading-relaxed mt-2">
                <strong>Theoretical Bandwidth:</strong><br/>
                <code className="text-blue-300 bg-slate-900 px-1 rounded text-[10px]">(Weights + KV_Cache) / SLO</code>
              </p>
              <p className="text-xs text-slate-500 leading-relaxed mt-2">
                The probabilistic model accounts for expert overlap as batch size increases, creating a saturating <span className="text-green-400">curve</span> instead of a linear slope.
              </p>
            </Card>
          </div>

        </div>
      </main>
    </div>
  );
}
