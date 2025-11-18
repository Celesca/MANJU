import type { NavbarProps, NavbarTheme } from "../Navbar";
import type { MegaMenuDropdownTheme } from "./MegaMenuDropdown";
import type { MegaMenuDropdownToggleTheme } from "./MegaMenuDropdownToggle";
export interface MegaMenuTheme extends NavbarTheme {
    dropdown: MegaMenuDropdownTheme;
    dropdownToggle: MegaMenuDropdownToggleTheme;
}
export type MegaMenuProps = NavbarProps;
export declare const MegaMenu: import("react").ForwardRefExoticComponent<Omit<NavbarProps, "ref"> & import("react").RefAttributes<HTMLElement>>;
