import { type ComponentProps, type ElementType } from "react";
import type { ThemingProps } from "../../types";
export interface NavbarBrandTheme {
    base: string;
}
export interface NavbarBrandProps extends ComponentProps<"a">, ThemingProps<NavbarBrandTheme> {
    as?: ElementType;
    href?: string;
}
export declare const NavbarBrand: import("react").ForwardRefExoticComponent<Omit<NavbarBrandProps, "ref"> & import("react").RefAttributes<HTMLAnchorElement>>;
