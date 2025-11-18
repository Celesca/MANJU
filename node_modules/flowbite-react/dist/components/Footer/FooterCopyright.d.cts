import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface FooterCopyrightTheme {
    base: string;
    href: string;
    span: string;
}
export interface FooterCopyrightProps extends ComponentProps<"div">, ThemingProps<FooterCopyrightTheme> {
    by: string;
    href?: string;
    year?: number;
}
export declare const FooterCopyright: import("react").ForwardRefExoticComponent<Omit<FooterCopyrightProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
