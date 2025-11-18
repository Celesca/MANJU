import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface DrawerItemsTheme {
    base: string;
}
export interface DrawerItemsProps extends ComponentProps<"div">, ThemingProps<DrawerItemsTheme> {
}
export declare const DrawerItems: import("react").ForwardRefExoticComponent<Omit<DrawerItemsProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
