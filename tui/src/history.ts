import { appendFileSync, readFileSync, existsSync, writeFileSync } from "fs";
import { join } from "path";
import { getConfigDir } from "./utils/paths";
import type { RunRecord } from "./types";

const HISTORY_PATH = join(getConfigDir(), "history.jsonl");

export function loadHistory(): RunRecord[] {
  if (!existsSync(HISTORY_PATH)) return [];
  return readFileSync(HISTORY_PATH, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line) as RunRecord;
      } catch {
        return null;
      }
    })
    .filter((record): record is RunRecord => record !== null);
}

export function appendHistory(record: RunRecord): void {
  appendFileSync(HISTORY_PATH, JSON.stringify(record) + "\n");
}

export function updateHistory(updated: RunRecord): void {
  const history = loadHistory().map((r) => (r.id === updated.id ? updated : r));
  writeFileSync(HISTORY_PATH, history.map((r) => JSON.stringify(r)).join("\n") + "\n");
}
