import { describe, expect, it } from "bun:test";
import { BUILTIN_THEMES, loadTheme } from "./theme";

describe("theme", () => {
  it("loads all built-in themes", () => {
    expect(Object.keys(BUILTIN_THEMES)).toContain("dracula");
    expect(BUILTIN_THEMES.dracula.name).toBe("Dracula");
  });

  it("falls back to dracula for unknown theme", () => {
    const theme = loadTheme("nonexistent");
    expect(theme.name).toBe("Dracula");
  });
});
