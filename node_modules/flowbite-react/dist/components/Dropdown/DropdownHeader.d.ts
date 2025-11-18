import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface DropdownHeaderTheme {
    header: string;
}
export interface DropdownHeaderProps extends ComponentProps<"div">, ThemingProps<DropdownHeaderTheme> {
}
export declare const DropdownHeader: import("react").ForwardRefExoticComponent<Omit<DropdownHeaderProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
