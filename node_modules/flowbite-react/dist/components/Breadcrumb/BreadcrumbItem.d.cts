import type { ComponentProps, FC } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
export interface BreadcrumbItemTheme {
    base: string;
    chevron: string;
    href: FlowbiteBoolean;
    icon: string;
}
export interface BreadcrumbItemProps extends Omit<ComponentProps<"li">, "ref">, ThemingProps<BreadcrumbItemTheme> {
    href?: string;
    icon?: FC<ComponentProps<"svg">>;
}
export declare const BreadcrumbItem: import("react").ForwardRefExoticComponent<BreadcrumbItemProps & import("react").RefAttributes<HTMLAnchorElement | HTMLSpanElement>>;
