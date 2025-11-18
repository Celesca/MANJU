import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface MegaMenuDropdownToggleTheme {
    base: string;
}
export interface MegaMenuDropdownToggleProps extends ComponentProps<"button">, ThemingProps<MegaMenuDropdownToggleTheme> {
}
export declare const MegaMenuDropdownToggle: import("react").ForwardRefExoticComponent<Omit<MegaMenuDropdownToggleProps, "ref"> & import("react").RefAttributes<HTMLButtonElement>>;
