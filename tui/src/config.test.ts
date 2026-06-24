import { describe, expect, it, beforeEach } from "bun:test";
import { loadConfig, saveConfig, DEFAULT_CONFIG } from "./config";
import { join } from "path";
import { getConfigDir } from "./utils/paths";

const CONFIG_PATH = join(getConfigDir(), "tui-config.json");

describe("config", () => {
  beforeEach(() => {
    try { require("fs").unlinkSync(CONFIG_PATH); } catch {}
  });

  it("returns default config when file missing", () => {
    expect(loadConfig().theme).toBe("dracula");
  });

  it("persists config changes", () => {
    saveConfig({ ...DEFAULT_CONFIG, theme: "nord" });
    expect(loadConfig().theme).toBe("nord");
  });
});
