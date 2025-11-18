import { type ComponentProps, type ElementType } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
import type { SidebarCollapseTheme } from "./SidebarCollapse";
import type { SidebarCTATheme } from "./SidebarCTA";
import type { SidebarItemTheme } from "./SidebarItem";
import type { SidebarItemGroupTheme } from "./SidebarItemGroup";
import type { SidebarItemsTheme } from "./SidebarItems";
import type { SidebarLogoTheme } from "./SidebarLogo";
export interface SidebarTheme {
    root: {
        base: string;
        collapsed: FlowbiteBoolean;
        inner: string;
    };
    collapse: SidebarCollapseTheme;
    cta: SidebarCTATheme;
    item: SidebarItemTheme;
    items: SidebarItemsTheme;
    itemGroup: SidebarItemGroupTheme;
    logo: SidebarLogoTheme;
}
export interface SidebarProps extends ComponentProps<"div">, ThemingProps<SidebarTheme> {
    as?: ElementType;
    collapseBehavior?: "collapse" | "hide";
    collapsed?: boolean;
}
export declare const Sidebar: import("react").ForwardRefExoticComponent<Omit<SidebarProps, "ref"> & import("react").RefAttributes<HTMLElement>>;
