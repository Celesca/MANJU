import { type ComponentProps, type FC } from "react";
import type { ThemingProps } from "../../types";
export interface DarkThemeToggleTheme {
    root: DarkThemeToggleRootTheme;
}
export interface DarkThemeToggleRootTheme {
    base: string;
    icon: {
        base: string;
        light: string;
        dark: string;
    };
}
export interface DarkThemeToggleProps extends ComponentProps<"button">, ThemingProps<DarkThemeToggleTheme> {
    iconDark?: FC<ComponentProps<"svg">>;
    iconLight?: FC<ComponentProps<"svg">>;
}
export declare const DarkThemeToggle: import("react").ForwardRefExoticComponent<Omit<DarkThemeToggleProps, "ref"> & import("react").RefAttributes<HTMLButtonElement>>;
