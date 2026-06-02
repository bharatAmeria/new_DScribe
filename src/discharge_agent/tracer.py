"""Observability: structured step-by-step trace logging."""
from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


class TraceStep:
    def __init__(
        self,
        step_num: int,
        node: str,
        reasoning: str,
        action: str,
        inputs: dict[str, Any],
        result: Any,
        next_decision: str,
        duration_ms: float,
    ):
        self.step_num = step_num
        self.node = node
        self.reasoning = reasoning
        self.action = action
        self.inputs = inputs
        self.result = result
        self.next_decision = next_decision
        self.duration_ms = duration_ms
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "step": self.step_num,
            "timestamp": self.timestamp,
            "node": self.node,
            "reasoning": self.reasoning,
            "action": self.action,
            "inputs": self.inputs,
            "result": str(self.result)[:500] if self.result else None,
            "next_decision": self.next_decision,
            "duration_ms": round(self.duration_ms, 2),
        }


class AgentTracer:
    def __init__(self, patient_id: str, output_dir: Path):
        self.patient_id = patient_id
        self.output_dir = output_dir
        self.steps: list[TraceStep] = []
        self._step_counter = 0
        self._start_times: dict[str, float] = {}

    def start_step(self, node: str) -> None:
        self._start_times[node] = time.time()

    def record(
        self,
        node: str,
        reasoning: str,
        action: str,
        inputs: dict[str, Any],
        result: Any,
        next_decision: str,
    ) -> None:
        self._step_counter += 1
        duration_ms = (time.time() - self._start_times.get(node, time.time())) * 1000
        step = TraceStep(
            step_num=self._step_counter,
            node=node,
            reasoning=reasoning,
            action=action,
            inputs=inputs,
            result=result,
            next_decision=next_decision,
            duration_ms=duration_ms,
        )
        self.steps.append(step)
        self._print_step(step)

    def _print_step(self, step: TraceStep) -> None:
        color = "cyan"
        title = f"[{color}]Step {step.step_num} · {step.node}[/{color}]"
        body = (
            f"[bold]Reasoning:[/bold] {step.reasoning}\n"
            f"[bold]Action:[/bold] {step.action}\n"
            f"[bold]Result:[/bold] {str(step.result)[:200]}\n"
            f"[bold]Next:[/bold] {step.next_decision}\n"
            f"[dim]{step.duration_ms:.0f}ms[/dim]"
        )
        console.print(Panel(body, title=title, border_style=color))

    def save(self, filename: Optional[str] = None) -> Path:
        filename = filename or f"trace_{self.patient_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "patient_id": self.patient_id,
                    "total_steps": self._step_counter,
                    "steps": [s.to_dict() for s in self.steps],
                },
                f,
                indent=2,
            )
        console.print(f"[green]Trace saved → {path}[/green]")
        return path
