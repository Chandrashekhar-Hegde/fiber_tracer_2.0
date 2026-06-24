import { describe, expect, it, beforeEach, afterEach } from "bun:test";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import React from "react";
import { render } from "ink-testing-library";
import { App } from "./app";

describe("App", () => {
  let tmpDir: string;
  let originalConfigDir: string | undefined;

  beforeEach(() => {
    originalConfigDir = process.env.FIBER_TRACER_CONFIG_DIR;
    tmpDir = mkdtempSync(join(tmpdir(), "fiber-tracer-tui-"));
    process.env.FIBER_TRACER_CONFIG_DIR = tmpDir;
  });

  afterEach(() => {
    if (originalConfigDir === undefined) {
      delete process.env.FIBER_TRACER_CONFIG_DIR;
    } else {
      process.env.FIBER_TRACER_CONFIG_DIR = originalConfigDir;
    }
    try {
      rmSync(tmpDir, { recursive: true, force: true });
    } catch {}
  });

  it("renders the sidebar with key sections", () => {
    const { lastFrame } = render(<App />);
    const frame = lastFrame();
    expect(frame).toContain("Fiber Tracer");
    expect(frame).toContain("New Analysis");
    expect(frame).toContain("Dashboard");
    expect(frame).toContain("History");
    expect(frame).toContain("Settings");
  });
});
