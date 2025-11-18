import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
import { type DropdownTheme } from "../Dropdown";
export interface MegaMenuDropdownTheme {
    base: string;
    toggle: DropdownTheme;
}
export interface MegaMenuDropdownProps extends ComponentProps<"div">, ThemingProps<MegaMenuDropdownTheme> {
    toggle?: JSX.Element;
}
export declare function MegaMenuDropdown(props: MegaMenuDropdownProps): import("react/jsx-runtime").JSX.Element;
export declare namespace MegaMenuDropdown {
    var displayName: string;
}
