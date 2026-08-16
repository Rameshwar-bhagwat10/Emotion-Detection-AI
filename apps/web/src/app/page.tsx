import React from "react";
import { CheckCircle2, Server, Cpu, Database, Layers } from "lucide-react";

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-6 bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      <div className="w-full max-w-2xl bg-slate-900/80 border border-slate-800 rounded-xl p-8 shadow-2xl backdrop-blur">
        <div className="flex items-center gap-3 border-b border-slate-800 pb-6 mb-6">
          <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg text-blue-400">
            <Layers className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">
              AI-Based Facial Expression Emotion Detection
            </h1>
            <p className="text-sm text-slate-400 mt-1">
              Phase 01 — Project Foundation & Environment Smoke Test
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-slate-950/60 border border-slate-800/80 rounded-lg">
            <div className="flex items-center gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <div>
                <h2 className="font-semibold text-sm text-slate-200">System Status</h2>
                <p className="text-xs text-slate-400">Frontend Environment</p>
              </div>
            </div>
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Operational
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
            <div className="p-3.5 bg-slate-950/40 border border-slate-800/60 rounded-lg">
              <div className="flex items-center gap-2 text-slate-400 text-xs font-medium mb-1">
                <Server className="w-4 h-4 text-blue-400" />
                <span>Backend API</span>
              </div>
              <p className="text-xs text-slate-300 font-mono">FastAPI / Uvicorn</p>
            </div>

            <div className="p-3.5 bg-slate-950/40 border border-slate-800/60 rounded-lg">
              <div className="flex items-center gap-2 text-slate-400 text-xs font-medium mb-1">
                <Database className="w-4 h-4 text-purple-400" />
                <span>Infrastructure</span>
              </div>
              <p className="text-xs text-slate-300 font-mono">PostgreSQL + Redis</p>
            </div>

            <div className="p-3.5 bg-slate-950/40 border border-slate-800/60 rounded-lg">
              <div className="flex items-center gap-2 text-slate-400 text-xs font-medium mb-1">
                <Cpu className="w-4 h-4 text-amber-400" />
                <span>AI Engine</span>
              </div>
              <p className="text-xs text-slate-300 font-mono">PyTorch / OpenCV</p>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-4 border-t border-slate-800/60 flex items-center justify-between text-xs text-slate-500">
          <span>Environment: {process.env.NODE_ENV || "development"}</span>
          <span>Phase 01 Complete</span>
        </div>
      </div>
    </main>
  );
}
