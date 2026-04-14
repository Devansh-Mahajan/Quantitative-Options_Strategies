from __future__ import annotations

import json

from core.system_telemetry import build_system_resource_snapshot


def main() -> None:
    print(json.dumps(build_system_resource_snapshot(), indent=2))


if __name__ == "__main__":
    main()
