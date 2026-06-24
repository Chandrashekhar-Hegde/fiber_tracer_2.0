import { readFileSync, existsSync } from "fs";
import { join } from "path";
import { getConfigDir } from "./utils/paths";

export interface Theme {
  name: string;
  background: string;
  foreground: string;
  accent: string;
  success: string;
  warning: string;
  error: string;
  muted: string;
  border: string;
  highlight: string;
}

export const BUILTIN_THEMES: Record<string, Theme> = {
  dracula: loadBuiltinTheme("dracula"),
  "catppuccin-mocha": loadBuiltinTheme("catppuccin-mocha"),
  "catppuccin-latte": loadBuiltinTheme("catppuccin-latte"),
  "one-dark": loadBuiltinTheme("one-dark"),
  nord: loadBuiltinTheme("nord"),
};

function loadBuiltinTheme(name: string): Theme {
  const path = join(import.meta.dirname, "../themes", `${name}.json`);
  return JSON.parse(readFileSync(path, "utf8")) as Theme;
}

export function loadTheme(name: string): Theme {
  if (BUILTIN_THEMES[name]) return BUILTIN_THEMES[name];
  const custom = join(getConfigDir(), "themes", `${name}.json`);
  if (existsSync(custom)) {
    return JSON.parse(readFileSync(custom, "utf8")) as Theme;
  }
  return BUILTIN_THEMES.dracula;
}
