"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";
import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [ram, setRam] = useState(8);
  const [cpu, setCpu] = useState(4);
  const [gpu, setGpu] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleOptimize = async () => {
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/optimize", {
        data_path: "dataset",
        ram: ram,
        cpu: cpu,
        gpu: gpu,
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black text-white p-10">
      
      <h1 className="text-4xl font-bold mb-8 text-center">
        🚀 AutoML CV Hyperparameter Tuner
      </h1>

      {/* INPUTS */}
      <div className="grid md:grid-cols-3 gap-4 mb-6">
        <input
          type="number"
          placeholder="RAM (GB)"
          className="p-3 rounded bg-gray-800"
          value={ram}
          onChange={(e) => setRam(Number(e.target.value))}
        />
        <input
          type="number"
          placeholder="CPU Cores"
          className="p-3 rounded bg-gray-800"
          value={cpu}
          onChange={(e) => setCpu(Number(e.target.value))}
        />
        <label className="flex items-center gap-2">
          GPU:
          <input
            type="checkbox"
            checked={gpu}
            onChange={() => setGpu(!gpu)}
          />
        </label>
      </div>

      {/* BUTTON */}
      <div className="text-center mb-6">
        <button
          onClick={handleOptimize}
          className="bg-blue-500 hover:bg-blue-600 px-6 py-3 rounded text-lg"
        >
          {loading ? "Running..." : "Find Best Hyperparameters"}
        </button>
      </div>

      {/* RESULTS */}
      {result && (
        <>
          <div className="grid md:grid-cols-2 gap-6 mt-8">

            {/* SYSTEM CARD */}
            <div className="bg-gray-800 p-6 rounded-xl shadow-lg">
              <h2 className="text-xl font-semibold mb-4">
                🧠 System Recommendation
              </h2>
              <p>Batch Size: {result.system_recommendation.recommended_batch}</p>
            </div>

            {/* HYPERPARAMETERS CARD */}
            <div className="bg-gray-800 p-6 rounded-xl shadow-lg">
              <h2 className="text-xl font-semibold mb-4">
                ⚙️ Best Hyperparameters
              </h2>
              <p>Learning Rate: {result.best_hyperparameters.lr}</p>
              <p>Batch Size: {result.best_hyperparameters.batch_size}</p>
              <p>Image Size: {result.best_hyperparameters.image_size}</p>
              <p>Optimizer: {result.best_hyperparameters.optimizer}</p>

              {result.accuracy && (
                <p className="mt-4 text-green-400">
                  Accuracy: {(result.accuracy * 100).toFixed(2)}%
                </p>
              )}
            </div>

          </div>

          {/* 📊 BASIC HYPERPARAMETER CHART */}
          <div className="mt-10 bg-gray-800 p-6 rounded-xl">
            <h2 className="text-xl mb-4">📊 Hyperparameter Visualization</h2>

            <BarChart
              width={500}
              height={300}
              data={[
                { name: "Batch", value: result.best_hyperparameters.batch_size },
                { name: "Image", value: result.best_hyperparameters.image_size },
              ]}
            >
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" />
            </BarChart>
          </div>

          {/* 🔥 NEW: OPTUNA TRIALS CHART */}
          {result.trials && (
            <div className="mt-10 bg-gray-800 p-6 rounded-xl">
              <h2 className="text-xl mb-4">📈 Optuna Trials Performance</h2>

              <BarChart width={600} height={300} data={result.trials}>
                <XAxis dataKey="trial" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="loss" />
              </BarChart>
            </div>
          )}
        </>
      )}
    </div>
  );
}