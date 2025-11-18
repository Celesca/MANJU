import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface DropdownDividerTheme {
    divider: string;
}
export interface DropdownDividerProps extends ComponentProps<"div">, ThemingProps<DropdownDividerTheme> {
}
export declare const DropdownDivider: import("react").ForwardRefExoticComponent<Omit<DropdownDividerProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
