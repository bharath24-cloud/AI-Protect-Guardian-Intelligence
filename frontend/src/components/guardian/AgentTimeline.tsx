"use client";

import React, { useState, useEffect } from "react";
import {
  Activity,
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  Terminal,
  ChevronDown,
  ChevronUp,
  Cpu,
  Layers,
  Database,
  BarChart3,
} from "lucide-react";
import { getGuardianJobStatus, GuardianJobStatusResponse } from "@/lib/api";

interface AgentTimelineProps {
  executionId: string | null;
  onExecutionCompleted?: (data: GuardianJobStatusResponse) => void;
}

export default function AgentTimeline({
  executionId,
  onExecutionCompleted,
}: AgentTimelineProps) {
  const [data, setData] = useState<GuardianJobStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedLog, setExpandedLog] = useState<Record<number, boolean>>({});

  useEffect(() => {
    if (!executionId) {
      setData(null);
      return;
    }

    let isMounted = true;
    let pollTimer: NodeJS.Timeout;

    const fetchStatus = async () => {
      try {
        const res = await getGuardianJobStatus(executionId);
        if (!isMounted) return;
        setData(res);
        setError(null);

        if (res.status === "completed" || res.status === "failed" || res.status === "cancelled") {
          if (res.status === "completed" && onExecutionCompleted) {
            onExecutionCompleted(res);
          }
        } else {
          pollTimer = setTimeout(fetchStatus, 1500);
        }
      } catch (err: any) {
        if (!isMounted) return;
        setError(err.response?.data?.detail || err.message || "Failed to fetch execution timeline.");
      }
    };

    fetchStatus();

    return () => {
      isMounted = false;
      if (pollTimer) clearTimeout(pollTimer);
    };
  }, [executionId]);

  if (!executionId) {
    return (
      <div className="p-8 text-center bg-slate-900/60 border border-slate-800 rounded-2xl text-slate-400">
        <Activity className="w-8 h-8 text-slate-600 mx-auto mb-2 animate-pulse" />
        <p className="text-sm font-medium text-slate-300">No Active Execution</p>
        <p className="text-xs text-slate-500 mt-1">
          Approve and execute an experiment plan to observe live tool invocations and measurements.
        </p>
      </div>
    );
  }

  const toggleLog = (idx: number) => {
    setExpandedLog((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden backdrop-blur-md shadow-xl text-slate-100">
      {/* Execution Header */}
      <div className="p-6 border-b border-slate-800 bg-slate-900/40">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
            <span className="text-xs text-slate-400 font-mono">Job ID: {executionId.slice(0, 8)}...</span>
          </div>

          <span
            className={`text-xs uppercase tracking-wider font-semibold px-2.5 py-1 rounded-full border ${
              data?.status === "completed"
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                : data?.status === "failed"
                ? "bg-rose-500/10 text-rose-400 border-rose-500/30"
                : "bg-sky-500/10 text-sky-400 border-sky-500/30 animate-pulse"
            }`}
          >
            {data?.status || "Connecting"}
          </span>
        </div>

        <h3 className="text-lg font-bold text-white tracking-tight">
          {data?.plan_title || "Experiment Execution Timeline"}
        </h3>

        {/* Progress Bar */}
        <div className="mt-4">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5 font-mono">
            <span>
              Step {data?.current_step || 0} of {data?.total_steps || 0}
            </span>
            <span>{data?.progress || 0}%</span>
          </div>
          <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-sky-500 to-cyan-400 transition-all duration-500"
              style={{ width: `${data?.progress || 0}%` }}
            />
          </div>
        </div>
      </div>

      {/* Error Message */}
      {data?.error_message && (
        <div className="m-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300 flex items-start gap-2">
          <XCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-rose-200">Execution Error:</span>
            <p className="mt-0.5">{data.error_message}</p>
          </div>
        </div>
      )}

      {/* Timeline Steps Stream */}
      <div className="p-6 space-y-4">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-cyan-400" />
          <span>Real-Time Operation Log</span>
        </h4>

        {(!data?.logs || data.logs.length === 0) && (
          <div className="flex items-center justify-center p-8 text-xs text-slate-500 gap-2">
            <Loader2 className="w-4 h-4 animate-spin text-sky-400" />
            <span>Initializing agent execution pipeline...</span>
          </div>
        )}

        <div className="relative pl-6 space-y-4 before:absolute before:left-2.5 before:top-2 before:bottom-2 before:w-0.5 before:bg-slate-800">
          {data?.logs?.map((log, idx) => {
            const isCompleted = log.status === "completed";
            const isFailed = log.status === "failed";
            const isRunning = log.status === "running";
            const isExpanded = !!expandedLog[idx];

            return (
              <div key={idx} className="relative group">
                {/* Timeline node icon */}
                <div
                  className={`absolute -left-6 top-1 w-5 h-5 rounded-full flex items-center justify-center text-[10px] border ${
                    isCompleted
                      ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"
                      : isFailed
                      ? "bg-rose-500/20 text-rose-400 border-rose-500/40"
                      : "bg-sky-500/20 text-sky-400 border-sky-500/40 animate-pulse"
                  }`}
                >
                  {isCompleted ? (
                    <CheckCircle2 className="w-3 h-3" />
                  ) : isFailed ? (
                    <XCircle className="w-3 h-3" />
                  ) : (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  )}
                </div>

                {/* Step Card */}
                <div className="p-3.5 rounded-xl border border-slate-800 bg-slate-900/60 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sky-300 font-medium">
                      {log.tool_name}
                    </span>
                    <span className="text-[10px] font-mono text-slate-500">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                  </div>

                  <p className="text-slate-300 mt-1 text-[11px] leading-relaxed">{log.message}</p>

                  {/* Measurements preview */}
                  {log.measurements && Object.keys(log.measurements).length > 0 && (
                    <div className="mt-2 pt-2 border-t border-slate-800/80">
                      <button
                        onClick={() => toggleLog(idx)}
                        className="flex items-center gap-1.5 text-[10px] font-mono text-cyan-400 hover:text-cyan-300 transition"
                      >
                        <BarChart3 className="w-3 h-3" />
                        <span>{isExpanded ? "Hide Outputs" : "View Measurements"}</span>
                        {isExpanded ? (
                          <ChevronUp className="w-3 h-3" />
                        ) : (
                          <ChevronDown className="w-3 h-3" />
                        )}
                      </button>

                      {isExpanded && (
                        <pre className="mt-2 p-2 rounded bg-slate-950 border border-slate-800 font-mono text-[10px] text-slate-300 overflow-x-auto">
                          {JSON.stringify(log.measurements, null, 2)}
                        </pre>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
