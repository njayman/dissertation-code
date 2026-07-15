from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from devmind.models import EdgeContextReport, OperationalState, ResourceStress


class MiscalibrationClassifier:
    def __init__(self) -> None:
        self._model: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = LogisticRegression()
        self._model.fit(X, y)

    def predict(self, resource_stress: ResourceStress, error_rate: float) -> OperationalState:
        if self._model is None:
            return self._fallback_heuristic(resource_stress, error_rate)
        x = np.concatenate([resource_stress.as_array(), [error_rate]]).reshape(1, -1)
        proba = self._model.predict_proba(x)[0]
        labels = list(self._model.classes_)
        return OperationalState(labels[int(proba.argmax())])

    @staticmethod
    def _fallback_heuristic(resource_stress: ResourceStress, error_rate: float) -> OperationalState:
        cpu_thermal = max(resource_stress.cpu, resource_stress.thermal)
        mem_disk = max(resource_stress.memory, resource_stress.disk_io)
        if cpu_thermal > 0.85 or error_rate > 0.10:
            return OperationalState.DEGRADING
        if cpu_thermal > 0.65 or mem_disk > 0.80:
            return OperationalState.STRESSED
        return OperationalState.NOMINAL

    @property
    def fitted(self) -> bool:
        return self._model is not None


def compute_calibration_delta(confidence_raw: float, cpu: float, thermal: float) -> float:
    temp = 1.0 + 0.5 * cpu + 0.3 * thermal
    calibrated = 1.0 / (1.0 + np.exp(-(np.log(confidence_raw / (1.0 - confidence_raw + 1e-8)) / temp)))
    return float(abs(confidence_raw - calibrated))


class EdgeDevice:
    def __init__(self, classifier: MiscalibrationClassifier | None = None):
        self.classifier = classifier or MiscalibrationClassifier()
        self._resource_stress = ResourceStress()
        self._error_buffer: list[bool] = []
        self._last_report: EdgeContextReport | None = None

    @property
    def last_report(self) -> EdgeContextReport | None:
        return self._last_report

    def emit_report(self, confidence_raw: float, is_correct: bool | None = None) -> EdgeContextReport:
        cpu = self._resource_stress.cpu
        thermal = self._resource_stress.thermal
        calibrated = confidence_raw - compute_calibration_delta(confidence_raw, cpu, thermal)
        delta = abs(confidence_raw - calibrated)

        if is_correct is not None:
            self._error_buffer.append(not is_correct)
            if len(self._error_buffer) > 60:
                self._error_buffer.pop(0)
        error_rate = sum(self._error_buffer) / max(len(self._error_buffer), 1)

        state = self.classifier.predict(self._resource_stress, error_rate)

        self._last_report = EdgeContextReport(
            resource_stress=ResourceStress(**{
                k: getattr(self._resource_stress, k) for k in ["cpu", "gpu", "memory", "disk_io", "thermal"]
            }),
            operational_state=state,
            confidence_raw=confidence_raw,
            confidence_calibrated=calibrated,
            calibration_delta=delta,
            error_rate_1m=error_rate,
        )
        return self._last_report

    def apply_stress(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if hasattr(self._resource_stress, k):
                setattr(self._resource_stress, k, v)

    def update_from_outcome(self, latency_ms: float, sla_met: bool, accuracy: float) -> None:
        # Backward loop: true label isn't known at emit_report() time in the real
        # gateway path (only after dispatch), so feed the outcome's accuracy proxy
        # into the same error buffer emit_report() reads next request.
        self._error_buffer.append(accuracy < 0.5)
        if len(self._error_buffer) > 60:
            self._error_buffer.pop(0)
