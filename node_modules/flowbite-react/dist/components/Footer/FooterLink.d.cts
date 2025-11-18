import { type ComponentProps, type ElementType } from "react";
import type { ThemingProps } from "../../types";
export interface FooterLinkTheme {
    base: string;
    href: string;
}
export interface FooterLinkProps extends ComponentProps<"a">, ThemingProps<FooterLinkTheme> {
    as?: ElementType;
    href: string;
}
export declare const FooterLink: import("react").ForwardRefExoticComponent<Omit<FooterLinkProps, "ref"> & import("react").RefAttributes<HTMLLIElement>>;
