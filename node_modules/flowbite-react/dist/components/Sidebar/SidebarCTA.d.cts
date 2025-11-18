import { type ComponentProps } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteColors, ThemingProps } from "../../types";
export interface SidebarCTATheme {
    base: string;
    color: SidebarCTAColors;
}
export interface SidebarCTAProps extends Omit<ComponentProps<"div">, "color">, ThemingProps<SidebarCTATheme> {
    color?: DynamicStringEnumKeysOf<SidebarCTAColors>;
}
export interface SidebarCTAColors extends Pick<FlowbiteColors, "blue" | "dark" | "failure" | "gray" | "green" | "light" | "purple" | "red" | "success" | "warning" | "yellow"> {
    [key: string]: string;
}
export declare const SidebarCTA: import("react").ForwardRefExoticComponent<Omit<SidebarCTAProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
