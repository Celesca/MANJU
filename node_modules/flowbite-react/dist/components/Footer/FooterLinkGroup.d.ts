import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
import type { FooterLinkTheme } from "./FooterLink";
export interface FooterLinkGroupTheme {
    base: string;
    link: FooterLinkTheme;
    col: string;
}
export interface FooterLinkGroupProps extends ComponentProps<"ul">, ThemingProps<FooterLinkGroupTheme> {
    col?: boolean;
}
export declare const FooterLinkGroup: import("react").ForwardRefExoticComponent<Omit<FooterLinkGroupProps, "ref"> & import("react").RefAttributes<HTMLUListElement>>;
