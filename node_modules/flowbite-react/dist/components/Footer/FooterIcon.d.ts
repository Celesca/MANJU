import { type ComponentProps, type FC } from "react";
import type { ThemingProps } from "../../types";
export interface FooterIconTheme {
    base: string;
    size: string;
}
type GenericLinkSvgProps = ComponentProps<"a"> & ComponentProps<"svg">;
export interface FooterIconProps extends GenericLinkSvgProps, ThemingProps<FooterIconTheme> {
    ariaLabel?: string;
    icon: FC<ComponentProps<"svg">>;
}
export declare const FooterIcon: import("react").ForwardRefExoticComponent<Omit<FooterIconProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
export {};
