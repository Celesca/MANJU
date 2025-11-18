import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
import type { FooterBrandTheme } from "./FooterBrand";
import type { FooterCopyrightTheme } from "./FooterCopyright";
import type { FooterDividerTheme } from "./FooterDivider";
import type { FooterIconTheme } from "./FooterIcon";
import type { FooterLinkGroupTheme } from "./FooterLinkGroup";
import type { FooterTitleTheme } from "./FooterTitle";
export interface FooterTheme {
    brand: FooterBrandTheme;
    copyright: FooterCopyrightTheme;
    divider: FooterDividerTheme;
    groupLink: FooterLinkGroupTheme;
    icon: FooterIconTheme;
    root: FooterRootTheme;
    title: FooterTitleTheme;
}
export interface FooterRootTheme {
    base: string;
    bgDark: string;
    container: string;
}
export interface FooterProps extends ComponentProps<"footer">, ThemingProps<FooterTheme> {
    bgDark?: boolean;
    container?: boolean;
}
export declare const Footer: import("react").ForwardRefExoticComponent<Omit<FooterProps, "ref"> & import("react").RefAttributes<HTMLElement>>;
