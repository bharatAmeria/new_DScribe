"""Observability — structured step-by-step trace logging."""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
        self.step_num     = step_num
        self.node         = node
        self.reasoning    = reasoning
        self.action       = action
        self.inputs       = inputs
        self.result       = result
        self.next_decision= next_decision
        self.duration_ms  = duration_ms
        self.timestamp    = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "step":          self.step_num,
            "timestamp":     self.timestamp,
            "node":          self.node,
            "reasoning":     self.reasoning,
            "action":        self.action,
            "inputs":        self.inputs,
            "result":        str(self.result)[:500] if self.result else None,
            "next_decision": self.next_decision,
            "duration_ms":   round(self.duration_ms, 2),
        }


class AgentTracer:
    def __init__(self, patient_id: str, output_dir: Path):
        self.patient_id = patient_id
        self.output_dir = Path(output_dir)
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
        self._log_step(step)

    def _log_step(self, step: TraceStep) -> None:
        """Emit step to logger (appears in both log file and console)."""
        separator = "─" * 60
        logger.info(
            "\n%s\n"
            "Step %-2d │ Node: %s\n"
            "Reasoning : %s\n"
            "Action    : %s\n"
            "Result    : %s\n"
            "Next      : %s\n"
            "Duration  : %.0fms\n"
            "%s",
            separator,
            step.step_num, step.node,
            step.reasoning,
            step.action,
            str(step.result)[:200],
            step.next_decision,
            step.duration_ms,
            separator,
        )

    def save(self, filename: Optional[str] = None) -> Path:
        filename = filename or f"trace_{self.patient_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "patient_id":  self.patient_id,
                    "total_steps": self._step_counter,
                    "steps":       [s.to_dict() for s in self.steps],
                },
                f,
                indent=2,
            )
        logger.info("Trace saved → %s", path)
        return path
