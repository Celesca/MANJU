import { type ComponentProps, type ElementType } from "react";
import type { ThemingProps } from "../../types";
export interface FooterTitleTheme {
    base: string;
}
export interface FooterTitleProps extends ComponentProps<"h2">, ThemingProps<FooterTitleTheme> {
    as?: ElementType;
    title: string;
}
export declare const FooterTitle: import("react").ForwardRefExoticComponent<Omit<FooterTitleProps, "ref"> & import("react").RefAttributes<HTMLHeadElement>>;
