import type { ComponentProps } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
import type { NavbarBrandTheme } from "./NavbarBrand";
import type { NavbarCollapseTheme } from "./NavbarCollapse";
import type { NavbarLinkTheme } from "./NavbarLink";
import type { NavbarToggleThem } from "./NavbarToggle";
export interface NavbarTheme {
    root: NavbarRootTheme;
    brand: NavbarBrandTheme;
    collapse: NavbarCollapseTheme;
    link: NavbarLinkTheme;
    toggle: NavbarToggleThem;
}
export interface NavbarRootTheme {
    base: string;
    rounded: FlowbiteBoolean;
    bordered: FlowbiteBoolean;
    inner: {
        base: string;
        fluid: FlowbiteBoolean;
    };
}
export interface NavbarProps extends ComponentProps<"nav">, ThemingProps<NavbarTheme> {
    menuOpen?: boolean;
    fluid?: boolean;
    rounded?: boolean;
    border?: boolean;
}
export declare const Navbar: import("react").ForwardRefExoticComponent<Omit<NavbarProps, "ref"> & import("react").RefAttributes<HTMLElement>>;
