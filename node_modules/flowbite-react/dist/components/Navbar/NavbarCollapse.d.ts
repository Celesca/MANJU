import { type ComponentProps } from "react";
import type { FlowbiteBoolean, ThemingProps } from "../../types";
export interface NavbarCollapseTheme {
    base: string;
    list: string;
    hidden: FlowbiteBoolean;
}
export interface NavbarCollapseProps extends ComponentProps<"div">, ThemingProps<NavbarCollapseTheme> {
}
export declare const NavbarCollapse: import("react").ForwardRefExoticComponent<Omit<NavbarCollapseProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
