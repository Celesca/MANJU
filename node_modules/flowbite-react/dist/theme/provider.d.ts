import { type PropsWithChildren } from "react";
import type { FlowbiteProps, FlowbiteTheme, ThemingProps } from "../types";
export interface ThemeProviderValue extends ThemingProps<FlowbiteTheme> {
    props?: Partial<FlowbiteProps>;
}
export interface ThemeProviderProps extends ThemeProviderValue, PropsWithChildren {
    /**
     * Prevent merging with parent context value.
     *
     * @type {boolean}
     * @default false
     */
    root?: boolean;
}
export declare function ThemeProvider({ children, root, props, theme, clearTheme, applyTheme }: ThemeProviderProps): import("react/jsx-runtime").JSX.Element;
export declare namespace ThemeProvider {
    var displayName: string;
}
export declare function useThemeProvider(): ThemeProviderValue;
