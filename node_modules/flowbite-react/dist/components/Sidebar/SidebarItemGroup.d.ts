import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface SidebarItemGroupTheme {
    base: string;
}
export interface SidebarItemGroupProps extends ComponentProps<"ul">, ThemingProps<SidebarItemGroupTheme> {
}
export declare const SidebarItemGroup: import("react").ForwardRefExoticComponent<Omit<SidebarItemGroupProps, "ref"> & import("react").RefAttributes<HTMLUListElement>>;
