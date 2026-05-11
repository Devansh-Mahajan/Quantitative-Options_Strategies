import { useEffect, useState } from "react";
import { ApiState, fetchJson } from "../lib/api";

export function useApiResource<T>(path: string, refreshMs = 0): ApiState<T> {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: true,
    error: null,
    source: "unavailable"
  });

  useEffect(() => {
    let mounted = true;
    const controller = new AbortController();

    const load = async () => {
      try {
        const data = await fetchJson<T>(path, controller.signal);
        if (mounted) setState({ data, loading: false, error: null, source: "live" });
      } catch (error) {
        if (mounted) {
          setState({
            data: null,
            loading: false,
            error: error instanceof Error ? error.message : "request failed",
            source: "unavailable"
          });
        }
      }
    };

    void load();
    const timer = refreshMs > 0 ? window.setInterval(load, refreshMs) : null;
    return () => {
      mounted = false;
      controller.abort();
      if (timer) window.clearInterval(timer);
    };
  }, [path, refreshMs]);

  return state;
}
