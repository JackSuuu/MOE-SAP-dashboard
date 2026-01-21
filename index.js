import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Settings, Server, Cpu, Activity, Info, Zap, Percent } from 'lucide-react';

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

export default function MoECalculator() {
  // --- Constants for Qwen3-30B-A3B (from PDF) ---
  const MODEL_CONFIG = {
    d_model: 2048,
    n_heads: 32,
    n_kv: 4,
    d_head: 128,
    L: 48,
    d_ff: 768,      // Intermediate size per expert
    E: 128,         // Total experts
    k: 8,           // Active experts (top-k)
    V: 151936,      // Vocab size
    P: 2            // BF16 = 2 bytes
  };

  // --- State ---
  const [scenario, setScenario] = useState('4k-256');
  
  // Inputs
  const [batchSize, setBatchSize] = useState(1);
  const [inputLen, setInputLen] = useState(4000);
  const [outputLen, setOutputLen] = useState(256);
  const [sloMs, setSloMs] = useState(50); // Target Time Per Output Token (ms)
  const [hardwareBw, setHardwareBw] = useState(2039); // Default H100
  const [smbu, setSmbu] = useState(16.33); // Fixed S-MBU
  const [numGpus, setNumGpus] = useState(8); // Supply side GPU count

  // Hardware options
  const HARDWARE_PRESETS = [
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

  // --- Calculations Helper (Strictly following PDF Steps) ---
  const calculateDemand = (ctxLen) => {
     const { d_model, n_heads, n_kv, d_head, L, d_ff, E, k, V, P } = MODEL_CONFIG;

    // Step 1: Input Embedding
    const W_embed = V * d_model * P;

    // Step 2: Self-Attention (per layer)
    const W_Q = d_model * (n_heads * d_head) * P;
    const W_K = d_model * (n_kv * d_head) * P;
    const W_V = d_model * (n_kv * d_head) * P;
    const W_O = (n_heads * d_head) * d_model * P;
    const BW_attn = W_Q + W_K + W_V + W_O;

    // Step 3: Router/Gate
    const W_gate = d_model * E * P;

    // Step 4: Expert FFN (per expert)
    // W_expert = 3 * d_model * d_ff * P
    const W_expert_single = 3 * d_model * d_ff * P;

    // Step 5: MoE Block Bandwidth (per layer)
    // Using probabilistic model for expected unique experts (Birthday Paradox / Coupon Collector)
    // E_unique = E * (1 - (1 - 1/E)^(B * k))
    // This creates a CURVE that saturates, not a linear relationship
    const totalExpertSelections = batchSize * k;
    const E_unique_expected = E * (1 - Math.pow(1 - 1/E, totalExpertSelections));
    const BW_moe = E_unique_expected * W_expert_single;

    // Step 6: Single Layer Total
    // BW_layer = BW_attn + W_gate + BW_moe
    const BW_layer = BW_attn + W_gate + BW_moe;

    // Step 9: All Layers Bandwidth (Weights)
    // BW_all_layers = L * BW_layer
    const BW_all_layers = L * BW_layer;

    // Step 10: Output LM Head
    const W_lm_head = d_model * V * P;

    // Total Active Model Weights (Constant per step)
    const W_model_total = W_embed + BW_all_layers + W_lm_head;

    // Step 7 & 11: KV Cache Size (Per Decode Step)
    // Formula: L * B * (Context) * 2 * n_kv * d_head * P
    // Here `ctxLen` represents the average context length during generation (N + (M+1)/2)
    // or the specific context length at a step.
    const W_kv_current = L * batchSize * ctxLen * 2 * n_kv * d_head * P;

    // Step 11: Total Forward Pass Bandwidth
    const BW_total = W_model_total + W_kv_current;

    // Step 12: Required Bandwidth Rate
    const req_bw_gbs = (BW_total / (sloMs / 1000)) / 1e9;
    
    return {
      reqBwGBs: req_bw_gbs,
      activeParamsGB: W_model_total / 1e9,
      kvCacheGB: W_kv_current / 1e9,
      uniqueExperts: E_unique_expected
    };
  };

  const currentStats = useMemo(() => {
    // For single point stat display, use average context
    const avgContext = inputLen + (outputLen + 1) / 2;
    return calculateDemand(avgContext);
  }, [batchSize, inputLen, outputLen, sloMs]);

  // --- Chart Data: Context vs Bandwidth (fixed batch size) ---
  const chartData = useMemo(() => {
    const data = [];
    const maxContext = Math.max(inputLen + outputLen, 16000); 
    const stepSize = Math.floor(maxContext / 10);
    
    // Actual Bandwidth (Supply)
    // Formula: HardwareBW * GPUs * MBU%
    // This value is CONSTANT relative to Context/Cost -> Horizontal Line
    const actualBw = hardwareBw * numGpus * (smbu / 100);

    for (let ctx = 1024; ctx <= maxContext + stepSize; ctx += stepSize) {
      const stats = calculateDemand(ctx);
      data.push({
        context: ctx, // X-Axis value
        theoretical: stats.reqBwGBs, // Y-Axis 1 (Linear Slope)
        actual: actualBw,            // Y-Axis 2 (Horizontal)
      });
    }
    return data;
  }, [inputLen, outputLen, hardwareBw, numGpus, smbu, sloMs, batchSize]);

  // --- Chart Data: Batch Size vs Bandwidth (shows curve) ---
  const batchChartData = useMemo(() => {
    const data = [];
    const { d_model, n_heads, n_kv, d_head, L, d_ff, E, k, V, P } = MODEL_CONFIG;
    const avgContext = inputLen + (outputLen + 1) / 2;
    
    // Actual Bandwidth (Supply) - Horizontal line
    const actualBw = hardwareBw * numGpus * (smbu / 100);
    
    // Pre-calculate constant parts
    const W_embed = V * d_model * P;
    const W_Q = d_model * (n_heads * d_head) * P;
    const W_K = d_model * (n_kv * d_head) * P;
    const W_V = d_model * (n_kv * d_head) * P;
    const W_O = (n_heads * d_head) * d_model * P;
    const BW_attn = W_Q + W_K + W_V + W_O;
    const W_gate = d_model * E * P;
    const W_expert_single = 3 * d_model * d_ff * P;
    const W_lm_head = d_model * V * P;
    
    // Generate curve data for batch sizes 1 to 256
    const maxBatch = 256;
    for (let b = 1; b <= maxBatch; b += (b < 32 ? 1 : (b < 64 ? 2 : 4))) {
      // Probabilistic unique experts: E * (1 - (1 - 1/E)^(B * k))
      const totalSelections = b * k;
      const E_unique = E * (1 - Math.pow(1 - 1/E, totalSelections));
      const BW_moe = E_unique * W_expert_single;
      const BW_layer = BW_attn + W_gate + BW_moe;
      const BW_all_layers = L * BW_layer;
      const W_model_total = W_embed + BW_all_layers + W_lm_head;
      
      // KV Cache scales with batch size
      const W_kv = L * b * avgContext * 2 * n_kv * d_head * P;
      const BW_total = W_model_total + W_kv;
      const req_bw_gbs = (BW_total / (sloMs / 1000)) / 1e9;
      
      data.push({
        batchSize: b,
        theoretical: req_bw_gbs,
        actual: actualBw,
        uniqueExperts: E_unique.toFixed(1),
      });
    }
    return data;
  }, [inputLen, outputLen, hardwareBw, numGpus, smbu, sloMs]);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans p-6">
      <header className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Activity className="w-8 h-8 text-blue-500" />
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
            MoE Bandwidth Estimator
          </h1>
          <span className="px-2 py-1 bg-slate-800 border border-slate-700 rounded text-xs text-slate-400">
            Qwen3-30B-A3B
          </span>
        </div>
        <p className="text-slate-400">
          Theoretical vs. Actual Bandwidth Analysis (SGLang Profiling Reference)
        </p>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Sidebar: Controls */}
        <div className="space-y-6">
          
          {/* Scenarios */}
          <section>
            <h2 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
              <Server className="w-4 h-4" /> Scenarios
            </h2>
            <div className="space-y-2">
              <ScenarioButton 
                active={scenario === '4k-256'}
                onClick={() => handleScenarioChange('4k-256')}
                label="Standard Chat (4k → 256)"
                desc="Batch Size 1, standard RAG-like context"
              />
              <ScenarioButton 
                active={scenario === '13k-1k'}
                onClick={() => handleScenarioChange('13k-1k')}
                label="Long Context (13k → 1k)"
                desc="Batch Size 1, document summarization"
              />
              <ScenarioButton 
                active={scenario === 'custom'}
                onClick={() => handleScenarioChange('custom')}
                label="Custom"
                desc="Define your own parameters"
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
                <label className="block text-xs font-medium text-slate-400 mb-1">Batch Size</label>
                <input 
                  type="number" 
                  min="1"
                  value={batchSize}
                  onChange={(e) => {setBatchSize(parseInt(e.target.value) || 1); setScenario('custom');}}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm focus:border-blue-500 outline-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Input Tokens</label>
                  <input 
                    type="number" 
                    value={inputLen}
                    onChange={(e) => {setInputLen(parseInt(e.target.value) || 0); setScenario('custom');}}
                    className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm focus:border-blue-500 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Output Tokens</label>
                  <input 
                    type="number" 
                    value={outputLen}
                    onChange={(e) => {setOutputLen(parseInt(e.target.value) || 0); setScenario('custom');}}
                    className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm focus:border-blue-500 outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Target SLO (ms/token)</label>
                <div className="flex items-center gap-2">
                  <input 
                    type="range" 
                    min="10" 
                    max="200" 
                    step="5"
                    value={sloMs}
                    onChange={(e) => setSloMs(parseInt(e.target.value))}
                    className="flex-1 accent-blue-500"
                  />
                  <span className="text-sm font-mono w-12 text-right">{sloMs}ms</span>
                </div>
              </div>
            </div>
          </section>

          {/* Hardware */}
          <section className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <h2 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Cpu className="w-4 h-4" /> Hardware
            </h2>
            <div className="mb-3">
              <select 
                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm outline-none"
                onChange={(e) => setHardwareBw(parseInt(e.target.value))}
                value={hardwareBw}
              >
                {HARDWARE_PRESETS.map((p) => (
                  <option key={p.name} value={p.bw}>{p.name} ({p.bw} GB/s)</option>
                ))}
                <option value={hardwareBw}>Custom</option>
              </select>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Bandwidth per Card (GB/s)</label>
                <input 
                  type="number"
                  value={hardwareBw}
                  onChange={(e) => setHardwareBw(parseInt(e.target.value) || 0)}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm focus:border-blue-500 outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Number of GPUs (Supply)</label>
                <div className="flex items-center gap-2">
                    <input 
                    type="range" 
                    min="1" 
                    max="16" 
                    step="1"
                    value={numGpus}
                    onChange={(e) => setNumGpus(parseInt(e.target.value))}
                    className="flex-1 accent-blue-500"
                    />
                    <span className="text-sm font-mono w-8 text-right">{numGpus}</span>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="block text-xs font-medium text-slate-400 flex items-center gap-1">
                    S-MBU (%) 
                    <Percent className="w-3 h-3" />
                  </label>
                  <span className="text-xs text-blue-400 font-mono">{smbu}%</span>
                </div>
                <input 
                  type="range" 
                  min="1" 
                  max="100" 
                  step="0.01"
                  value={smbu}
                  onChange={(e) => setSmbu(parseFloat(e.target.value))}
                  className="w-full accent-blue-500 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                />
                <p className="text-[10px] text-slate-500 mt-1">Sustainable Memory Bandwidth Utilization</p>
              </div>
            </div>
          </section>
        </div>

        {/* Right Content: Stats & Charts */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* KPI Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard 
              title="Weights/Step" 
              value={currentStats.activeParamsGB.toFixed(2)} 
              unit="GB"
              subtext="Active Model Params"
              icon={Server}
            />
            <StatCard 
              title="KV Cache" 
              value={currentStats.kvCacheGB.toFixed(2)} 
              unit="GB"
              subtext={`@ ${inputLen + outputLen/2} ctx`}
              icon={Info}
            />
            <StatCard 
              title="Theoretical BW" 
              value={(currentStats.reqBwGBs/1024).toFixed(2)} 
              unit="TB/s"
              subtext={`Demand @ ${sloMs}ms SLO`}
              icon={Zap}
            />
             <StatCard 
              title="Actual BW" 
              value={(hardwareBw * numGpus * smbu / 100 / 1024).toFixed(2)} 
              unit="TB/s"
              subtext={`Supply (${numGpus} GPUs)`}
              icon={Activity}
            />
          </div>

          {/* Chart Section */}
          <Card className="h-[450px] relative">
            <h3 className="text-lg font-semibold mb-6 pl-2 border-l-4 border-blue-500">
               Bandwidth Analysis: Theoretical vs Actual
            </h3>
            <ResponsiveContainer width="100%" height="85%">
              <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis 
                  dataKey="context" 
                  type="number" // Forces numeric scale to prevent curve artifacts
                  domain={['dataMin', 'dataMax']} 
                  tick={false} 
                  label={{ value: 'Cost', position: 'insideBottom', offset: -15, fill: '#94a3b8' }} 
                />
                <YAxis 
                  stroke="#94a3b8" 
                  label={{ value: 'Bandwidth (GB/s)', angle: -90, position: 'insideLeft', offset: 0, fill: '#94a3b8' }} 
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                  itemStyle={{ color: '#cbd5e1' }}
                  formatter={(value) => [`${Math.round(value)} GB/s`, '']}
                  labelFormatter={() => `Cost (Abstract)`}
                />
                <Legend verticalAlign="top" height={36} wrapperStyle={{ top: -10 }} />
                
                {/* Theoretical Bandwidth - Sloped Line (Demand) */}
                <Line 
                  type="linear" 
                  dataKey="theoretical" 
                  name="Theoretical Bandwidth (Required)" 
                  stroke="#ef4444" 
                  strokeWidth={3} 
                  dot={false}
                  isAnimationActive={false} // Prevents animation artifacts
                />

                {/* Actual Bandwidth - Horizontal Dotted Line (Supply) */}
                <Line 
                  type="linear" 
                  dataKey="actual" 
                  name="Actual Bandwidth (Available)" 
                  stroke="#3b82f6" 
                  strokeWidth={3} 
                  strokeDasharray="5 5" 
                  dot={false}
                  isAnimationActive={false}
                />
                
                {/* Reference for Current Input */}
                <ReferenceLine x={inputLen} stroke="#10b981" strokeDasharray="3 3" label={{ value: 'Current', fill: '#10b981', fontSize: 12 }} />

              </LineChart>
            </ResponsiveContainer>
            
            <div className="absolute top-16 right-6 bg-slate-900/80 p-3 rounded border border-slate-700 text-xs max-w-[200px]">
              <p className="text-slate-300 font-medium mb-1">Chart Logic:</p>
              <div className="flex items-center gap-2 text-red-400 mb-1">
                <div className="w-3 h-1 bg-red-500"></div>
                <span>Theoretical: Increases with Cost (Slope)</span>
              </div>
              <div className="flex items-center gap-2 text-blue-400">
                <div className="w-3 h-0.5 border-t border-dashed border-blue-500"></div>
                <span>Actual: Fixed Supply (Horizontal)</span>
              </div>
            </div>
          </Card>

          {/* Batch Size vs Bandwidth Chart - Shows the Curve */}
          <Card className="h-[450px] relative">
            <h3 className="text-lg font-semibold mb-6 pl-2 border-l-4 border-green-500">
              Batch Size vs Required Bandwidth (MoE Curve)
            </h3>
            <ResponsiveContainer width="100%" height="85%">
              <LineChart data={batchChartData} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis 
                  dataKey="batchSize" 
                  type="number"
                  domain={[1, 'dataMax']} 
                  tick={{ fill: '#94a3b8', fontSize: 10 }} 
                  label={{ value: 'Batch Size', position: 'insideBottom', offset: -15, fill: '#94a3b8' }} 
                />
                <YAxis 
                  stroke="#94a3b8" 
                  tickFormatter={(v) => `${(v/1024).toFixed(0)}K`}
                  label={{ value: 'Bandwidth (GB/s)', angle: -90, position: 'insideLeft', offset: 0, fill: '#94a3b8' }} 
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                  itemStyle={{ color: '#cbd5e1' }}
                  formatter={(value, name, props) => {
                    if (name === 'Theoretical Bandwidth') {
                      return [`${(value/1024).toFixed(2)} TB/s (${props.payload.uniqueExperts} experts)`, name];
                    }
                    return [`${(value/1024).toFixed(2)} TB/s`, name];
                  }}
                  labelFormatter={(label) => `Batch Size: ${label}`}
                />
                <Legend verticalAlign="top" height={36} wrapperStyle={{ top: -10 }} />
                
                {/* Theoretical Bandwidth - Curved Line (due to probabilistic experts) */}
                <Line 
                  type="monotone" 
                  dataKey="theoretical" 
                  name="Theoretical Bandwidth" 
                  stroke="#22c55e" 
                  strokeWidth={3} 
                  dot={false}
                  isAnimationActive={false}
                />

                {/* Actual Bandwidth - Horizontal Dotted Line (Supply) */}
                <Line 
                  type="linear" 
                  dataKey="actual" 
                  name="Actual Bandwidth (Available)" 
                  stroke="#3b82f6" 
                  strokeWidth={3} 
                  strokeDasharray="5 5" 
                  dot={false}
                  isAnimationActive={false}
                />
                
                {/* Reference for Current Batch Size */}
                <ReferenceLine x={batchSize} stroke="#f97316" strokeDasharray="3 3" strokeWidth={2} label={{ value: `B=${batchSize}`, fill: '#f97316', fontSize: 12 }} />

              </LineChart>
            </ResponsiveContainer>
            
            <div className="absolute top-16 right-6 bg-slate-900/80 p-3 rounded border border-slate-700 text-xs max-w-[220px]">
              <p className="text-slate-300 font-medium mb-1">MoE Expert Saturation:</p>
              <p className="text-green-400 text-[10px] mb-1">
                E_unique = E × (1 - (1-1/E)^(B×k))
              </p>
              <p className="text-slate-500 text-[10px]">
                As batch size ↑, probability of expert overlap ↑, creating a saturating curve instead of linear growth.
              </p>
            </div>
          </Card>

          {/* Detailed Breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <h4 className="font-semibold text-slate-300 mb-3 border-b border-slate-700 pb-2">MoE Model Config</h4>
              <ul className="space-y-2 text-sm text-slate-400">
                <li className="flex justify-between">
                  <span>Model:</span>
                  <span className="text-slate-200">Qwen3-30B-A3B</span>
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
                  <span className="text-slate-200">{smbu}%</span>
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