import type { ComponentProps } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
export interface SidebarLogoTheme {
    base: string;
    collapsed: FlowbiteBoolean;
    img: string;
}
export interface SidebarLogoProps extends ComponentProps<"a">, ThemingProps<SidebarLogoTheme> {
    href: string;
    img: string;
    imgAlt?: string;
}
export declare const SidebarLogo: import("react").ForwardRefExoticComponent<Omit<SidebarLogoProps, "ref"> & import("react").RefAttributes<HTMLAnchorElement>>;
