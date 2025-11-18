import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface FooterDividerTheme {
    base: string;
}
export interface FooterDividerProps extends ComponentProps<"hr">, ThemingProps<FooterDividerTheme> {
}
export declare const FooterDivider: import("react").ForwardRefExoticComponent<Omit<FooterDividerProps, "ref"> & import("react").RefAttributes<HTMLHRElement>>;
