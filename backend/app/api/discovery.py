import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.datasets.materials_data import MaterialsScienceDataset
from src.loop.discovery_loop import AutonomousDiscoveryLoop
from src.memory.scientific_memory import ScientificMemory

router = APIRouter(prefix="/discovery", tags=["discovery"])

# Global singleton discovery loop
_memory = ScientificMemory()
_loop: Optional[AutonomousDiscoveryLoop] = None

def get_loop():
    global _loop
    if _loop is None:
        _loop = AutonomousDiscoveryLoop(memory=_memory)
    return _loop

class StartDiscoveryRequest(BaseModel):
    num_cycles: int = 5
    budget_limit: float = 30.0
    task_family: Optional[str] = "all"

@router.get("/status")
async def get_discovery_status():
    loop = get_loop()
    mem_stats = _memory.get_summary_statistics()
    return {
        "status": "running" if loop.is_running else "idle",
        "current_cycle": loop.current_cycle,
        "remaining_budget": round(loop.env.remaining_budget, 2),
        "total_hypotheses_stored": mem_stats["total_hypotheses"],
        "total_experiments_executed": mem_stats["total_experiments_executed"],
        "knowledge_graph_nodes": mem_stats["knowledge_graph_nodes"],
        "knowledge_graph_edges": mem_stats["knowledge_graph_edges"]
    }

@router.post("/start")
async def start_discovery(req: StartDiscoveryRequest, background_tasks: BackgroundTasks):
    loop = get_loop()
    if loop.is_running:
        raise HTTPException(status_code=400, detail="Discovery loop is already actively executing.")
        
    def run_bg():
        loop.run_cycles(num_cycles=req.num_cycles)
        
    background_tasks.add_task(run_bg)
    return {"message": f"Initiated {req.num_cycles} autonomous discovery cycles in background.", "status": "started"}

@router.get("/hypotheses")
async def get_hypotheses(status: Optional[str] = None):
    hyps = _memory.hypothesis_store.get_all(status=status)
    return {"total": len(hyps), "hypotheses": hyps}

@router.get("/experiments")
async def get_experiments():
    exps = _memory.experiment_store.get_all()
    return {"total": len(exps), "experiments": exps}

@router.get("/knowledge-graph")
async def get_knowledge_graph():
    return _memory.knowledge_graph.to_dict()

@router.get("/journal")
async def get_research_journal():
    p = Path("results/research_journal.jsonl")
    if not p.exists():
        return {"records": []}
    records = []
    with open(p, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return {"total": len(records), "records": records}

@router.get("/baselines")
async def get_baseline_results():
    p = Path("results/metrics/baseline_comparison.csv")
    if not p.exists():
        return {"data": []}
    import pandas as pd
    df = pd.read_csv(p)
    return {"data": df.to_dict(orient="records")}

@router.get("/ablations")
async def get_ablation_results():
    p = Path("results/metrics/ablation_study.csv")
    if not p.exists():
        return {"data": []}
    import pandas as pd
    df = pd.read_csv(p)
    return {"data": df.to_dict(orient="records")}
