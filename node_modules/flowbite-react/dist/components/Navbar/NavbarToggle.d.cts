import { type ComponentProps, type FC } from "react";
import type { ThemingProps } from "../../types";
export interface NavbarToggleThem {
    base: string;
    icon: string;
    title: string;
}
export interface NavbarToggleProps extends ComponentProps<"button">, ThemingProps<NavbarToggleThem> {
    barIcon?: FC<ComponentProps<"svg">>;
}
export declare const NavbarToggle: import("react").ForwardRefExoticComponent<Omit<NavbarToggleProps, "ref"> & import("react").RefAttributes<HTMLButtonElement>>;
