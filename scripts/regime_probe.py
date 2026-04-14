from __future__ import annotations

import json
from datetime import datetime, timezone

from core.regime_detection import get_brain_prediction
from core.runtime_calibration import load_runtime_calibration


def main():
    runtime = load_runtime_calibration()
    macro_strategy, macro_confidence, _ = get_brain_prediction()
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "macro_strategy": macro_strategy,
        "macro_confidence": round(float(macro_confidence), 4),
        "runtime_regime": runtime.current_regime,
        "runtime_market_state": runtime.current_market_state,
        "runtime_profile": runtime.selected_profile,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
