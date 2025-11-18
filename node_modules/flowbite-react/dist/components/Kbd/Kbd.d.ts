import { type ComponentProps, type FC } from "react";
import type { ThemingProps } from "../../types";
export interface KbdTheme {
    root: KbdRootTheme;
}
export interface KbdRootTheme {
    base: string;
    icon: string;
}
export interface KbdProps extends ComponentProps<"span">, ThemingProps<KbdTheme> {
    icon?: FC<ComponentProps<"svg">>;
}
export declare const Kbd: import("react").ForwardRefExoticComponent<Omit<KbdProps, "ref"> & import("react").RefAttributes<HTMLSpanElement>>;
