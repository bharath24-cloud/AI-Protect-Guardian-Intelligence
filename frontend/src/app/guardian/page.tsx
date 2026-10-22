"use client";

import React, { useState, useEffect } from "react";
import GuardianChat from "@/components/guardian/GuardianChat";
import ExperimentPlan from "@/components/guardian/ExperimentPlan";
import ApprovalDialog from "@/components/guardian/ApprovalDialog";
import AgentTimeline from "@/components/guardian/AgentTimeline";
import EvidencePanel from "@/components/guardian/EvidencePanel";
import {
  GuardianPlanResponse,
  GuardianEvidenceSummary,
  GuardianJobStatusResponse,
  approveGuardianPlan,
  executeGuardianPlan,
  getGuardianModels,
  listGuardianExperiments,
  compareGuardianExperiments,
} from "@/lib/api";
import {
  Sparkles,
  Bot,
  Layers,
  Activity,
  FileSearch,
  History,
  Cpu,
  CheckCircle2,
  AlertTriangle,
  Scale,
  RefreshCw,
  Loader2,
  Shield,
  Zap,
} from "lucide-react";

export default function GuardianPage() {
  const [activeTab, setActiveTab] = useState<"plan" | "timeline" | "evidence" | "history">("plan");
  const [activePlan, setActivePlan] = useState<GuardianPlanResponse | null>(null);
  const [currentExecutionId, setCurrentExecutionId] = useState<string | null>(null);
  const [evidence, setEvidence] = useState<GuardianEvidenceSummary | null>(null);
  const [isApprovalOpen, setIsApprovalOpen] = useState(false);
  const [isSubmittingApproval, setIsSubmittingApproval] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);

  // Model pool status
  const [modelStatus, setModelStatus] = useState<Record<string, any>>({});
  const [experiments, setExperiments] = useState<any[]>([]);
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([]);
  const [comparisonResult, setComparisonResult] = useState<any | null>(null);
  const [isComparing, setIsComparing] = useState(false);

  useEffect(() => {
    getGuardianModels()
      .then((data) => {
        if (data && data.models) setModelStatus(data.models);
      })
      .catch(() => {});

    listGuardianExperiments()
      .then((data) => {
        if (Array.isArray(data)) setExperiments(data);
      })
      .catch(() => {});
  }, []);

  const handlePlanProposed = (plan: GuardianPlanResponse) => {
    setActivePlan(plan);
    setActiveTab("plan");
  };

  const handleAuthorizeConfirm = async (notes?: string) => {
    if (!activePlan) return;
    setIsSubmittingApproval(true);
    try {
      const res = await approveGuardianPlan({
        plan_id: activePlan.plan_id,
        config_hash: activePlan.config_hash,
        decision: "approved",
        notes: notes || "Authorized via Guardian Intelligence Dashboard",
      });

      if (res.success) {
        setActivePlan((prev) => (prev ? { ...prev, status: "approved", approval_status: "approved" } : null));
        setIsApprovalOpen(false);
      }
    } catch (err: any) {
      alert(`Approval error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setIsSubmittingApproval(false);
    }
  };

  const handleExecute = async () => {
    if (!activePlan) return;
    setIsExecuting(true);
    try {
      const res = await executeGuardianPlan({
        plan_id: activePlan.plan_id,
        config_hash: activePlan.config_hash,
      });

      if (res.success && res.execution_id) {
        setCurrentExecutionId(res.execution_id);
        setActivePlan((prev) => (prev ? { ...prev, status: "running" } : null));
        setActiveTab("timeline");
      }
    } catch (err: any) {
      alert(`Execution error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setIsExecuting(false);
    }
  };

  const handleExecutionCompleted = (jobData: GuardianJobStatusResponse) => {
    if (jobData.evidence) {
      setEvidence(jobData.evidence);
    }
    setActivePlan((prev) => (prev ? { ...prev, status: "completed" } : null));
    // Refresh history
    listGuardianExperiments().then((data) => {
      if (Array.isArray(data)) setExperiments(data);
    });
  };

  const handleSelectRecommendation = (rec: string) => {
    setActiveTab("plan");
  };

  const handleToggleCompareSelection = (expId: string) => {
    setSelectedForCompare((prev) => {
      if (prev.includes(expId)) {
        return prev.filter((id) => id !== expId);
      }
      if (prev.length >= 2) {
        return [prev[1], expId];
      }
      return [...prev, expId];
    });
  };

  const handleRunComparison = async () => {
    if (selectedForCompare.length < 2) return;
    setIsComparing(true);
    try {
      const res = await compareGuardianExperiments(selectedForCompare);
      setComparisonResult(res);
    } catch (err: any) {
      alert(`Comparison error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setIsComparing(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-16">
      {/* Header Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-6 bg-[#181C24] border border-[#262C38] rounded-2xl backdrop-blur-md">
        <div className="flex items-center gap-3.5">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-[#5CE1D2] to-[#A9A5FF] p-0.5 shadow-lg shadow-[#5CE1D2]/15">
            <div className="w-full h-full bg-[#101217] rounded-[10px] flex items-center justify-center">
              <Bot className="w-6 h-6 text-[#5CE1D2]" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-[#F4F1E9] tracking-tight">Guardian Intelligence</h1>
              <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded bg-[#5CE1D2]/10 text-[#5CE1D2] border border-[#5CE1D2]/20">
                Orchestrator
              </span>
            </div>
            <p className="text-xs text-[#9BA5B4] mt-0.5">
              Adaptive experiment planning, policy gates, registered ML tools, and grounded evidence analysis.
            </p>
          </div>
        </div>

        {/* Model Status Pills */}
        <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono">
          {Object.entries(modelStatus).map(([mName, mInfo]: [string, any]) => (
            <span
              key={mName}
              className="px-2.5 py-1 rounded-lg bg-[#101217] border border-[#262C38] flex items-center gap-1.5"
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  mInfo.status === "ready" ? "bg-[#5CE1D2]" : "bg-[#FF8C79]"
                }`}
              />
              <span className="text-[#F4F1E9] font-semibold">{mName}:</span>
              <span className="text-[#9BA5B4] uppercase">{mInfo.status}</span>
            </span>
          ))}
        </div>
      </div>

      {/* Main Two-Column Cockpit */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Guardian Chat Dialogue (5 cols) */}
        <div className="lg:col-span-5">
          <GuardianChat
            onPlanProposed={handlePlanProposed}
            activePlan={activePlan}
            onOpenApproval={() => setIsApprovalOpen(true)}
          />
        </div>

        {/* Right Column: Dynamic Workspace Tabs (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {/* Tab Navigation */}
          <div className="flex items-center gap-1 p-1 bg-[#181C24] border border-[#262C38] rounded-xl">
            <button
              onClick={() => setActiveTab("plan")}
              className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 text-xs font-semibold rounded-lg transition ${
                activeTab === "plan"
                  ? "bg-[#202631] text-[#5CE1D2] font-semibold shadow-sm"
                  : "text-[#9BA5B4] hover:text-[#F4F1E9] hover:bg-[#202631]/50"
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Experiment Plan</span>
            </button>

            <button
              onClick={() => setActiveTab("timeline")}
              className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 text-xs font-semibold rounded-lg transition ${
                activeTab === "timeline"
                  ? "bg-[#202631] text-[#5CE1D2] font-semibold shadow-sm"
                  : "text-[#9BA5B4] hover:text-[#F4F1E9] hover:bg-[#202631]/50"
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>Live Timeline</span>
              {currentExecutionId && (
                <span className="w-1.5 h-1.5 rounded-full bg-[#5CE1D2] animate-ping" />
              )}
            </button>

            <button
              onClick={() => setActiveTab("evidence")}
              className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 text-xs font-semibold rounded-lg transition ${
                activeTab === "evidence"
                  ? "bg-[#202631] text-[#A9A5FF] font-semibold shadow-sm"
                  : "text-[#9BA5B4] hover:text-[#F4F1E9] hover:bg-[#202631]/50"
              }`}
            >
              <FileSearch className="w-3.5 h-3.5" />
              <span>Evidence & Insights</span>
              {evidence && (
                <span className="w-1.5 h-1.5 rounded-full bg-[#A9A5FF]" />
              )}
            </button>

            <button
              onClick={() => setActiveTab("history")}
              className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 text-xs font-semibold rounded-lg transition ${
                activeTab === "history"
                  ? "bg-[#202631] text-[#5CE1D2] font-semibold shadow-sm"
                  : "text-[#9BA5B4] hover:text-[#F4F1E9] hover:bg-[#202631]/50"
              }`}
            >
              <History className="w-3.5 h-3.5" />
              <span>History & Compare</span>
            </button>
          </div>

          {/* Tab 1: Experiment Plan View */}
          {activeTab === "plan" && (
            <div>
              {activePlan ? (
                <ExperimentPlan
                  plan={activePlan}
                  onAuthorize={() => setIsApprovalOpen(true)}
                  onExecute={handleExecute}
                  isExecuting={isExecuting}
                />
              ) : (
                <div className="p-12 text-center bg-[#181C24] border border-[#262C38] rounded-2xl text-[#9BA5B4]">
                  <Layers className="w-10 h-10 text-[#9BA5B4]/40 mx-auto mb-3" />
                  <h3 className="text-sm font-semibold text-[#F4F1E9]">No Experiment Plan Active</h3>
                  <p className="text-xs text-[#9BA5B4] mt-1 max-w-sm mx-auto">
                    State an experimental goal in Guardian Dialogue to formulate an adaptive, reproducible plan.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Agent Timeline View */}
          {activeTab === "timeline" && (
            <div>
              <AgentTimeline
                executionId={currentExecutionId}
                onExecutionCompleted={handleExecutionCompleted}
              />
            </div>
          )}

          {/* Tab 3: Evidence Panel View */}
          {activeTab === "evidence" && (
            <div>
              <EvidencePanel
                evidence={evidence}
                onSelectRecommendation={handleSelectRecommendation}
              />
            </div>
          )}

          {/* Tab 4: History & Comparison View */}
          {activeTab === "history" && (
            <div className="bg-[#181C24] border border-[#262C38] rounded-2xl p-6 space-y-4 backdrop-blur-md text-[#F4F1E9]">
              <div className="flex items-center justify-between border-b border-[#262C38] pb-3">
                <div>
                  <h3 className="text-sm font-bold text-[#F4F1E9]">Historical Experiment Records</h3>
                  <p className="text-xs text-[#9BA5B4]">
                    Select two experiments to calculate differential metric comparison.
                  </p>
                </div>
                {selectedForCompare.length >= 2 && (
                  <button
                    onClick={handleRunComparison}
                    disabled={isComparing}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#5CE1D2] hover:bg-[#5CE1D2] text-[#101217] font-semibold text-xs transition"
                  >
                    {isComparing ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Scale className="w-3.5 h-3.5" />
                    )}
                    <span>Compare Selected ({selectedForCompare.length})</span>
                  </button>
                )}
              </div>

              {/* Comparison Results Card */}
              {comparisonResult && (
                <div className="p-4 rounded-xl bg-[#101217] border border-[#5CE1D2]/30 space-y-3 text-xs">
                  <span className="font-semibold text-[#5CE1D2] uppercase tracking-wider text-[10px]">
                    Side-By-Side Differential Analysis
                  </span>
                  <div className="grid grid-cols-2 gap-3">
                    {comparisonResult.experiments?.map((exp: any, i: number) => (
                      <div key={i} className="p-3 rounded-lg bg-[#181C24] border border-[#262C38]">
                        <span className="font-mono text-[10px] text-[#9BA5B4]">Exp #{i + 1}: {exp.experiment_id.slice(0, 8)}</span>
                        <div className="mt-1 space-y-1 font-mono text-[11px]">
                          <div>Overall: {exp.scores?.overall?.toFixed(1) || "N/A"}%</div>
                          <div>Disruption: {exp.scores?.disruption?.toFixed(1) || "N/A"}%</div>
                          <div>Transfer: {exp.scores?.transferability?.toFixed(1) || "N/A"}%</div>
                          <div>Robustness: {exp.scores?.robustness?.toFixed(1) || "N/A"}%</div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {comparisonResult.deltas && (
                    <div className="pt-2 border-t border-[#262C38] font-mono text-[11px]">
                      <span className="text-[#9BA5B4] text-[10px] block mb-1">Metric Deltas (Exp 2 - Exp 1):</span>
                      {Object.entries(comparisonResult.deltas).map(([k, d]: [string, any]) => (
                        <div key={k} className="flex justify-between text-[#F4F1E9] py-0.5">
                          <span className="capitalize">{k}:</span>
                          <span className={d.delta > 0 ? "text-[#5CE1D2]" : d.delta < 0 ? "text-[#FF8C79]" : "text-[#9BA5B4]"}>
                            {d.delta > 0 ? `+${d.delta}` : d.delta} ({d.direction})
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Experiments Table */}
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {experiments.map((exp: any) => {
                  const isSelected = selectedForCompare.includes(exp.experiment_id);
                  return (
                    <div
                      key={exp.experiment_id}
                      onClick={() => handleToggleCompareSelection(exp.experiment_id)}
                      className={`p-3 rounded-xl border text-xs cursor-pointer transition flex items-center justify-between ${
                        isSelected
                          ? "bg-[#5CE1D2]/10 border-[#5CE1D2]/40"
                          : "bg-[#101217] border-[#262C38] hover:border-[#5CE1D2]/30"
                      }`}
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-medium text-[#F4F1E9]">
                            {exp.experiment_id.slice(0, 8)}...
                          </span>
                          <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-[#202631] text-[#9BA5B4]">
                            {exp.type}
                          </span>
                          <span className="text-[10px] text-[#5CE1D2]">
                            Status: {exp.status}
                          </span>
                        </div>
                        <span className="text-[10px] text-[#9BA5B4] font-mono">
                          {exp.created_at ? new Date(exp.created_at).toLocaleString() : ""}
                        </span>
                      </div>

                      <div className="flex items-center gap-3 font-mono text-[11px] text-[#9BA5B4]">
                        {exp.scores && (
                          <div>
                            Score: <strong className="text-[#F4F1E9]">{exp.scores.overall?.toFixed(1) || "N/A"}%</strong>
                          </div>
                        )}
                        <input
                          type="checkbox"
                          checked={isSelected}
                          readOnly
                          className="rounded border-[#262C38] text-[#5CE1D2]"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Approval Dialog Modal */}
      {activePlan && (
        <ApprovalDialog
          isOpen={isApprovalOpen}
          onClose={() => setIsApprovalOpen(false)}
          onConfirm={handleAuthorizeConfirm}
          plan={activePlan}
          isSubmitting={isSubmittingApproval}
        />
      )}
    </div>
  );
}
