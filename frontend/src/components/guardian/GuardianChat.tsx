"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Sparkles,
  Bot,
  User,
  ShieldCheck,
  Cpu,
  Layers,
  ArrowRight,
  Loader2,
  RefreshCw,
  Terminal,
} from "lucide-react";
import {
  chatWithGuardian,
  GuardianChatResponse,
  GuardianPlanResponse,
} from "@/lib/api";

interface MessageItem {
  sender: "user" | "guardian";
  text: string;
  plan?: GuardianPlanResponse | null;
  suggestedActions?: string[];
  timestamp: string;
}

interface GuardianChatProps {
  onPlanProposed: (plan: GuardianPlanResponse) => void;
  activePlan: GuardianPlanResponse | null;
  onOpenApproval: () => void;
}

const DEFAULT_SUGGESTIONS = [
  "Evaluate cloaking configuration on artwork",
  "Test cross-model transferability on ViT and ConvNeXt",
  "Benchmark JPEG compression and resizing robustness",
  "Investigate vision model checkpoints and readiness",
  "Compare last two experiments",
];

export default function GuardianChat({
  onPlanProposed,
  activePlan,
  onOpenApproval,
}: GuardianChatProps) {
  const [messages, setMessages] = useState<MessageItem[]>([
    {
      sender: "guardian",
      text: (
        "Welcome to Guardian Intelligence. I am your autonomous experiment-planning and evidence-analysis orchestrator.\n\n" +
        "State your experimental objective or select a suggested prompt to begin. I will formulate a reproducible plan, " +
        "enforce resource safety bounds, request your authorization, and synthesize grounded evidence."
      ),
      suggestedActions: DEFAULT_SUGGESTIONS,
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async (textToSend?: string) => {
    const text = textToSend || inputValue;
    if (!text.trim() || isLoading) return;

    const userMsg: MessageItem = {
      sender: "user",
      text: text.trim(),
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!textToSend) setInputValue("");
    setIsLoading(true);

    try {
      const res: GuardianChatResponse = await chatWithGuardian({
        message: text.trim(),
        plan_id: activePlan?.plan_id,
      });

      const guardianMsg: MessageItem = {
        sender: "guardian",
        text: res.reply,
        plan: res.plan,
        suggestedActions: res.suggested_actions,
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages((prev) => [...prev, guardianMsg]);

      if (res.plan) {
        onPlanProposed(res.plan);
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "guardian",
          text: `Error processing request: ${err.response?.data?.detail || err.message || "Failed to reach Guardian API."}`,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[650px] bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden backdrop-blur-md shadow-xl text-slate-100">
      {/* Chat Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/60">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-sky-400 p-0.5">
            <div className="w-full h-full bg-slate-950 rounded-[6px] flex items-center justify-center">
              <Bot className="w-4 h-4 text-cyan-400" />
            </div>
          </div>
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
              <span>Guardian Dialogue</span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            </h3>
            <span className="text-[10px] text-slate-400">Autonomous Experiment Agent</span>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <Cpu className="w-3.5 h-3.5 text-sky-400" />
          <span>Policy: Safe Bounds</span>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 text-xs">
        {messages.map((msg, idx) => {
          const isUser = msg.sender === "user";
          return (
            <div
              key={idx}
              className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}
            >
              {!isUser && (
                <div className="w-7 h-7 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 flex items-center justify-center shrink-0 mt-0.5">
                  <Bot className="w-3.5 h-3.5" />
                </div>
              )}

              <div
                className={`max-w-[85%] rounded-2xl p-4 leading-relaxed ${
                  isUser
                    ? "bg-sky-600 text-white rounded-tr-none shadow-md"
                    : "bg-slate-950/80 border border-slate-800/80 text-slate-200 rounded-tl-none shadow-sm"
                }`}
              >
                <div className="whitespace-pre-wrap">{msg.text}</div>

                {/* Plan Proposal Card in Chat */}
                {msg.plan && (
                  <div className="mt-3 p-3 rounded-xl bg-slate-900 border border-sky-500/30 text-xs text-slate-200">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-semibold text-sky-400">{msg.plan.title}</span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-sky-500/10 text-sky-300 font-mono">
                        {msg.plan.steps.length} Steps
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 line-clamp-2">{msg.plan.goal}</p>

                    <div className="mt-2.5 flex items-center justify-between pt-2 border-t border-slate-800">
                      <span className="text-[10px] font-mono text-slate-400">
                        VRAM: {msg.plan.resource_estimate.gpu_vram_gb.toFixed(1)} GB | ~{msg.plan.resource_estimate.estimated_runtime_seconds}s
                      </span>
                      {msg.plan.requires_approval && (
                        <button
                          onClick={onOpenApproval}
                          className="px-2.5 py-1 rounded bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold text-[10px] transition"
                        >
                          Review & Authorize
                        </button>
                      )}
                    </div>
                  </div>
                )}

                {/* Suggested Action Pills */}
                {msg.suggestedActions && msg.suggestedActions.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-slate-800/80 flex flex-wrap gap-1.5">
                    {msg.suggestedActions.map((act, i) => (
                      <button
                        key={i}
                        onClick={() => handleSendMessage(act)}
                        disabled={isLoading}
                        className="px-2.5 py-1 rounded-lg bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/60 text-slate-300 text-[10px] transition flex items-center gap-1 hover:text-white"
                      >
                        <Sparkles className="w-2.5 h-2.5 text-cyan-400" />
                        <span>{act}</span>
                      </button>
                    ))}
                  </div>
                )}

                <div className="text-[9px] text-slate-500 text-right mt-1.5 select-none font-mono">
                  {msg.timestamp}
                </div>
              </div>

              {isUser && (
                <div className="w-7 h-7 rounded-lg bg-sky-600/20 border border-sky-500/30 text-sky-400 flex items-center justify-center shrink-0 mt-0.5">
                  <User className="w-3.5 h-3.5" />
                </div>
              )}
            </div>
          );
        })}

        {isLoading && (
          <div className="flex gap-3 justify-start">
            <div className="w-7 h-7 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 flex items-center justify-center shrink-0 mt-0.5">
              <Bot className="w-3.5 h-3.5" />
            </div>
            <div className="p-4 rounded-2xl rounded-tl-none bg-slate-950/80 border border-slate-800 text-slate-400 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
              <span>Analyzing objective and inspecting ML pipeline...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <div className="p-4 border-t border-slate-800 bg-slate-900/60">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="flex items-center gap-2"
        >
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type your experimental goal (e.g., 'Test cross-model transferability')..."
            disabled={isLoading}
            className="flex-1 px-4 py-2.5 text-xs bg-slate-950 border border-slate-800 rounded-xl text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition"
          />
          <button
            type="submit"
            disabled={isLoading || !inputValue.trim()}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-sky-500 hover:from-cyan-400 hover:to-sky-400 disabled:opacity-40 text-slate-950 font-semibold transition flex items-center gap-1.5"
          >
            <Send className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Send</span>
          </button>
        </form>
      </div>
    </div>
  );
}
