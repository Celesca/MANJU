export type ThemeMode = "light" | "dark" | "auto";
export declare function useThemeMode(): {
    mode: ThemeMode;
    computedMode: ThemeMode;
    setMode: (mode: ThemeMode) => void;
    toggleMode: () => void;
    clearMode: () => void;
};
