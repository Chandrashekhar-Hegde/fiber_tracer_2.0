import { describe, expect, it, beforeAll, afterAll } from "bun:test";
import { mkdirSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { BUILTIN_THEMES, DEFAULT_THEME, isTheme, loadTheme, type Theme } from "./theme";

const testConfigDir = join(tmpdir(), `fiber-tracer-theme-test-${Date.now()}`);
process.env.FIBER_TRACER_CONFIG_DIR = testConfigDir;

beforeAll(() => {
  mkdirSync(join(testConfigDir, "themes"), { recursive: true });
});

afterAll(() => {
  rmSync(testConfigDir, { recursive: true, force: true });
});

describe("theme", () => {
  it("loads all built-in themes", () => {
    expect(Object.keys(BUILTIN_THEMES).sort()).toEqual([
      "catppuccin-latte",
      "catppuccin-mocha",
      "dracula",
      "nord",
      "one-dark",
    ]);

    for (const theme of Object.values(BUILTIN_THEMES)) {
      expect(isTheme(theme)).toBe(true);
    }

    expect(BUILTIN_THEMES.dracula.name).toBe("Dracula");
    expect(BUILTIN_THEMES["catppuccin-mocha"].name).toBe("Catppuccin Mocha");
    expect(BUILTIN_THEMES["catppuccin-latte"].name).toBe("Catppuccin Latte");
    expect(BUILTIN_THEMES["one-dark"].name).toBe("One Dark");
    expect(BUILTIN_THEMES.nord.name).toBe("Nord");
  });

  it("falls back to dracula for unknown theme", () => {
    const theme = loadTheme("nonexistent");
    expect(theme.name).toBe("Dracula");
    expect(theme).toEqual(DEFAULT_THEME);
  });

  it("loads a custom theme from the config directory", () => {
    const custom: Theme = {
      name: "Custom",
      background: "#000000",
      foreground: "#ffffff",
      accent: "#ff00ff",
      success: "#00ff00",
      warning: "#ffff00",
      error: "#ff0000",
      muted: "#666666",
      border: "#333333",
      highlight: "#00ffff",
    };

    writeFileSync(join(testConfigDir, "themes", "custom.json"), JSON.stringify(custom));

    const loaded = loadTheme("custom");
    expect(loaded.name).toBe("Custom");
    expect(loaded).toEqual(custom);
  });

  it("falls back to dracula for a malformed custom theme file", () => {
    writeFileSync(join(testConfigDir, "themes", "bad.json"), "{not valid json");
    expect(loadTheme("bad")).toEqual(DEFAULT_THEME);

    writeFileSync(
      join(testConfigDir, "themes", "incomplete.json"),
      JSON.stringify({ name: "Incomplete", background: "#000000" }),
    );
    expect(loadTheme("incomplete")).toEqual(DEFAULT_THEME);
  });
});
