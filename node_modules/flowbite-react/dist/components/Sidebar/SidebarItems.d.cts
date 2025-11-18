import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface SidebarItemsTheme {
    base: string;
}
export interface SidebarItemsProps extends ComponentProps<"div">, ThemingProps<SidebarItemsTheme> {
}
export declare const SidebarItems: import("react").ForwardRefExoticComponent<Omit<SidebarItemsProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
