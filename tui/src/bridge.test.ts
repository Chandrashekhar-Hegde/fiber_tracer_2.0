import { describe, expect, it } from "bun:test";
import { runAnalysis } from "./bridge";

describe("bridge", () => {
  it("reports error when fiber-tracer CLI is missing", async () => {
    const result = await runAnalysis({
      dataPath: "/nonexistent",
      outputDir: "/tmp/out",
      voxelSpacing: [1, 1, 1],
      fiberDiameter: 10,
      regime: "auto",
      method: "otsu",
      model: "",
      batchSize: 1,
      computeMorphometry: true,
      computeOrientationTensor: true,
      computeTda: false,
    });
    expect(result.success).toBe(false);
  });
});
