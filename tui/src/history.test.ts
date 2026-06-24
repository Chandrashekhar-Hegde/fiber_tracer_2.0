import { describe, expect, it, beforeEach } from "bun:test";
import { loadHistory, appendHistory, updateHistory } from "./history";
import { join } from "path";
import { getConfigDir } from "./utils/paths";
import type { RunRecord } from "./types";

const HISTORY_PATH = join(getConfigDir(), "history.jsonl");

function makeRecord(id: string, status: RunRecord["status"] = "success"): RunRecord {
  return {
    id,
    name: id,
    config: {
      dataPath: "data/test.tif",
      outputDir: "results/test",
      voxelSpacing: [1, 1, 1],
      fiberDiameter: 10,
      regime: "auto",
      method: "otsu",
      model: "",
      batchSize: 1,
      computeMorphometry: true,
      computeOrientationTensor: true,
      computeTda: false,
    },
    status,
    startedAt: new Date().toISOString(),
    outputDir: "results/test",
  };
}

describe("history", () => {
  beforeEach(() => {
    try { require("fs").unlinkSync(HISTORY_PATH); } catch {}
  });

  it("returns empty history when file missing", () => {
    expect(loadHistory()).toEqual([]);
  });

  it("appends and updates records", () => {
    appendHistory(makeRecord("a"));
    appendHistory(makeRecord("b", "failed"));
    expect(loadHistory()).toHaveLength(2);
    updateHistory({ ...makeRecord("b", "failed"), status: "success" });
    expect(loadHistory()[1].status).toBe("success");
  });

  it("skips corrupted history lines", () => {
    const fs = require("fs");
    fs.writeFileSync(HISTORY_PATH, JSON.stringify(makeRecord("good")) + "\nnot valid json\n");
    const records = loadHistory();
    expect(records).toHaveLength(1);
    expect(records[0].id).toBe("good");
  });
});
