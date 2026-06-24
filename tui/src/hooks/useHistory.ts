import { useState, useCallback } from "react";
import { loadHistory, appendHistory, updateHistory } from "../history";
import type { RunRecord } from "../types";

export function useHistory() {
  const [history, setHistory] = useState<RunRecord[]>(() => loadHistory());

  const add = useCallback((record: RunRecord) => {
    appendHistory(record);
    setHistory((prev) => [...prev, record]);
  }, []);

  const update = useCallback((record: RunRecord) => {
    updateHistory(record);
    setHistory((prev) => prev.map((r) => (r.id === record.id ? record : r)));
  }, []);

  return { history, add, update };
}
