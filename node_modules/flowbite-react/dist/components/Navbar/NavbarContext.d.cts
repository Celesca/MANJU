import type { ThemingProps } from "../../types";
import type { NavbarTheme } from "./Navbar";
export interface NavbarContextValue extends ThemingProps<NavbarTheme> {
    isOpen?: boolean;
    setIsOpen: (isOpen: boolean) => void;
}
export declare const NavbarContext: import("react").Context<NavbarContextValue | undefined>;
export declare function useNavbarContext(): NavbarContextValue;
