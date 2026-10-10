"""Optional routing observations; no action payloads or physical-motion claims.

The sink runs on the delivery thread and must consume events without blocking.
A sink exception disables recording for this run, leaving an incomplete trace.
It must not prevent robot shutdown or change which action is delivered.
"""

from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class Origin:
    kind: str
    request: int
    observation: int
    slot: int

    def record(self):
        return {
            "kind": self.kind,
            "request": self.request,
            "observation": self.observation,
            "slot": self.slot,
        }


class DeliveryTrace:
    def __init__(self, sink=None, run_id=None):
        self.sink = sink
        self.run_id = run_id or (uuid4().hex if sink is not None else None)
        self.sequence = 0
        self.error = None

    def emit(self, kind, step=None, **fields):
        if self.sink is None:
            return
        event = {
            "schema": 1,
            "run": self.run_id,
            "seq": self.sequence,
            "kind": kind,
            "step": step,
        }
        event.update(fields)
        self.sequence += 1
        try:
            self.sink(event)
        except Exception as error:  # noqa: BLE001 - a failed observer cannot stop delivery.
            self.error = error
            self.sink = None

    def origin(self, kind, step, origin, **fields):
        if self.sink is not None:
            self.emit(kind, step, origin=origin.record(), **fields)

    def send(self, robot, action, step, origin):
        self.origin("send_begin", step, origin)
        try:
            robot.send_action(action)
        except BaseException:
            self.origin("send_end", step, origin, status="error")
            raise
        self.origin("send_end", step, origin, status="ok")

    def stop(self, robot):
        try:
            try:
                self.emit("stop_begin")
            finally:
                robot.stop()
        except BaseException:
            self.emit("stop_end", status="error")
            raise
        self.emit("stop_end", status="ok")
