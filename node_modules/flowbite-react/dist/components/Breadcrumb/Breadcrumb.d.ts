import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
import type { BreadcrumbItemTheme } from "./BreadcrumbItem";
export interface BreadcrumbTheme {
    root: BreadcrumbRootTheme;
    item: BreadcrumbItemTheme;
}
export interface BreadcrumbRootTheme {
    base: string;
    list: string;
}
export interface BreadcrumbProps extends ComponentProps<"nav">, ThemingProps<BreadcrumbRootTheme> {
}
export declare const Breadcrumb: import("react").ForwardRefExoticComponent<Omit<BreadcrumbProps, "ref"> & import("react").RefAttributes<HTMLElement>>;
