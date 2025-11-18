import { type ComponentProps, type ElementType } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
export interface NavbarLinkTheme {
    base: string;
    active: FlowbiteBoolean;
    disabled: FlowbiteBoolean;
}
export interface NavbarLinkProps extends ComponentProps<"a">, ThemingProps<NavbarLinkTheme> {
    active?: boolean;
    as?: ElementType;
    disabled?: boolean;
    href?: string;
}
export declare const NavbarLink: import("react").ForwardRefExoticComponent<Omit<NavbarLinkProps, "ref"> & import("react").RefAttributes<HTMLLIElement>>;
