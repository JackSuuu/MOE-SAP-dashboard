import { useState, useMemo } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ZAxis
} from 'recharts';
import { Activity, Download } from 'lucide-react';
import { AGENTIC_BENCHMARK_DATA, getUniqueDatasets, getUniqueModels, getUniqueToolModes } from './data/agentic-benchmarks/index.js';

// Card component
const Card = ({ children, className = "" }) => (
  <div className={`bg-slate-800 border border-slate-700 rounded-lg p-3 sm:p-4 ${className}`}>
    {children}
  </div>
);

// CSV utilities
const escapeCSVField = (field) => {
  if (field === null || field === undefined) return '';
  const str = String(field);
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
};

const downloadCSV = (data, filename) => {
  if (!data || data.length === 0) return;
  const headers = Object.keys(data[0]);
  const csvRows = [
    headers.map(escapeCSVField).join(','),
    ...data.map(row => headers.map(h => escapeCSVField(row[h])).join(','))
  ];
  const csvContent = csvRows.join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

// Color mapping for datasets
const DATASET_COLORS = {
  'imo_answerbench_full_nt': '#22c55e',     // green
  'imo_answerbench_full_combi': '#3b82f6',  // blue
};

// Custom tooltip
const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload || !payload.length) return null;
  const data = payload[0].payload;
  return (
    <div className="bg-slate-900 border border-slate-600 rounded-lg p-3 text-sm">
      <div className="font-bold text-white mb-2">{data.modelShort}</div>
      <div className="text-slate-300">Dataset: {data.dataset}</div>
      <div className="text-slate-300">Tool Mode: {data.toolMode}</div>
      <div className="text-slate-300">Run: {data.date}/{data.run}</div>
      <div className="text-slate-300 mt-1">Accuracy: {data.accuracy.toFixed(2)}%</div>
      <div className="text-slate-300">Mean Latency: {data.meanTime.toFixed(2)}s</div>
      <div className="text-slate-300">Mean Prefill Tokens / Request: {data.meanTotalPrefill.toFixed(0)}</div>
      <div className="text-slate-300">Mean Decode Tokens / Request: {data.meanTotalDecode.toFixed(0)}</div>
      <div className="text-slate-300">Questions: {data.correctQuestions}/{data.totalQuestions}</div>
    </div>
  );
};

export function AgenticWorkflowSection() {
  // Filter states
  const [selectedDataset, setSelectedDataset] = useState('all');
  const [selectedModel, setSelectedModel] = useState('all');
  const [selectedToolMode, setSelectedToolMode] = useState('all');
  const [yAxisMetric, setYAxisMetric] = useState('latency'); // 'latency', 'input', or 'output'

  const datasets = useMemo(() => getUniqueDatasets(), []);
  const models = useMemo(() => getUniqueModels(), []);
  const toolModes = useMemo(() => getUniqueToolModes(), []);

  // Filter data
  const filteredData = useMemo(() => {
    return AGENTIC_BENCHMARK_DATA.filter(d => {
      if (d.totalQuestions !== 100) return false; // Only show complete runs
      if (selectedDataset !== 'all' && d.dataset !== selectedDataset) return false;
      if (selectedModel !== 'all' && d.modelShort !== selectedModel) return false;
      if (selectedToolMode !== 'all' && d.toolMode !== selectedToolMode) return false;
      return true;
    }).map(d => {
      let yValue;
      if (yAxisMetric === 'latency') {
        yValue = d.meanTime;
      } else if (yAxisMetric === 'input') {
        yValue = d.meanTotalPrefill;
      } else {
        yValue = d.meanTotalDecode;
      }
      return {
        ...d,
        x: d.accuracy,
        y: yValue,
        color: DATASET_COLORS[d.dataset] || '#888',
      };
    });
  }, [selectedDataset, selectedModel, selectedToolMode, yAxisMetric]);

  // Group by dataset for legend
  const dataByDataset = useMemo(() => {
    const grouped = {};
    filteredData.forEach(d => {
      if (!grouped[d.dataset]) grouped[d.dataset] = [];
      grouped[d.dataset].push(d);
    });
    return grouped;
  }, [filteredData]);

  // Export function
  const exportAgenticData = () => {
    const exportData = AGENTIC_BENCHMARK_DATA.map(d => ({
      date: d.date,
      run: d.run,
      dataset: d.dataset,
      model: d.model,
      tool_mode: d.toolMode,
      gpu: d.gpu,
      total_questions: d.totalQuestions,
      correct_questions: d.correctQuestions,
      accuracy_percent: d.accuracy,
      mean_latency_s: d.meanTime,
      mean_total_prefill: d.meanTotalPrefill,
      mean_total_decode: d.meanTotalDecode,
    }));
    const dateStr = new Date().toISOString().split('T')[0];
    downloadCSV(exportData, `agentic_workflow_benchmark_${dateStr}.csv`);
  };

  const yAxisLabel = yAxisMetric === 'latency' 
    ? 'Mean Latency per Question (s)' 
    : yAxisMetric === 'input'
    ? 'Mean Prefill Tokens / Request'
    : 'Mean Decode Tokens / Request';

  return (
    <div className="space-y-4">
      {/* Hidden button for external trigger */}
      <button 
        id="agentic-download-btn" 
        onClick={exportAgenticData} 
        style={{ display: 'none' }} 
      />
      
      {/* Controls */}
      <Card>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {/* Dataset */}
          <div>
            <label className="block text-xs text-slate-400 mb-1">Dataset</label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-white"
            >
              <option value="all">All Datasets</option>
              {datasets.map(d => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          {/* Model */}
          <div>
            <label className="block text-xs text-slate-400 mb-1">Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-white"
            >
              <option value="all">All Models</option>
              {models.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          {/* Tool Mode */}
          <div>
            <label className="block text-xs text-slate-400 mb-1">Tool Mode</label>
            <select
              value={selectedToolMode}
              onChange={(e) => setSelectedToolMode(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-white"
            >
              <option value="all">All Modes</option>
              {toolModes.map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>

          {/* Y-Axis Metric */}
          <div>
            <label className="block text-xs text-slate-400 mb-1">Y-Axis Metric</label>
            <select
              value={yAxisMetric}
              onChange={(e) => setYAxisMetric(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-white"
            >
              <option value="latency">Mean Latency (s)</option>
              <option value="input">Mean Prefill Tokens / Request</option>
              <option value="output">Mean Decode Tokens / Request</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Chart */}
      <Card>
        <div className="h-[400px] sm:h-[500px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 30, bottom: 60, left: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis 
                type="number" 
                dataKey="x" 
                name="Accuracy"
                domain={[60, 100]}
                tickFormatter={(v) => `${v}%`}
                stroke="#94a3b8"
                label={{ 
                  value: 'Accuracy (%)', 
                  position: 'bottom', 
                  offset: 40,
                  fill: '#94a3b8' 
                }}
              />
              <YAxis 
                type="number" 
                dataKey="y" 
                name={yAxisLabel}
                stroke="#94a3b8"
                label={{ 
                  value: yAxisLabel, 
                  angle: -90, 
                  position: 'insideLeft',
                  offset: -45,
                  fill: '#94a3b8',
                  style: { textAnchor: 'middle' }
                }}
              />
              <ZAxis range={[100, 100]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend 
                verticalAlign="top"
                wrapperStyle={{ paddingBottom: '10px' }}
              />
              
              {Object.entries(dataByDataset).map(([dataset, data]) => (
                <Scatter
                  key={dataset}
                  name={dataset}
                  data={data}
                  fill={DATASET_COLORS[dataset] || '#888'}
                  shape="circle"
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Summary Stats */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Summary Statistics</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
          {Object.entries(dataByDataset).map(([dataset, data]) => {
            const avgAccuracy = data.reduce((s, d) => s + d.accuracy, 0) / data.length;
            const avgLatency = data.reduce((s, d) => s + d.meanTime, 0) / data.length;
            const models = [...new Set(data.map(d => d.modelShort))].join(', ');
            const toolModes = [...new Set(data.map(d => d.toolMode))].join(', ');
            return (
              <div key={dataset} className="bg-slate-700/50 rounded p-3">
                <div className="flex items-center gap-2 mb-2">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: DATASET_COLORS[dataset] }}
                  />
                  <span className="font-medium text-white text-xs">{dataset}</span>
                </div>
                <div className="text-slate-400 text-xs">
                  <div>Model: {models}</div>
                  <div>Tool Mode: {toolModes || 'N/A'}</div>
                  <div>Runs: {data.length}</div>
                  <div>Avg Accuracy: {avgAccuracy.toFixed(1)}%</div>
                  <div>Avg Latency: {avgLatency.toFixed(1)}s</div>
                </div>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

export default AgenticWorkflowSection;
