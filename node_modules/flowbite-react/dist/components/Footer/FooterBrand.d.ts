import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface FooterBrandTheme {
    base: string;
    img: string;
    span: string;
}
type GenericLinkImageProps = ComponentProps<"a"> & ComponentProps<"img">;
export interface FooterBrandProps extends GenericLinkImageProps, ThemingProps<FooterBrandTheme> {
    name?: string;
    src: string;
}
export declare const FooterBrand: import("react").ForwardRefExoticComponent<Omit<FooterBrandProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
export {};
