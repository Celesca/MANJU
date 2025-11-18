import type { useInteractions } from "@floating-ui/react";
import type { ThemingProps } from "../../types";
import type { DropdownTheme } from "./Dropdown";
export interface DropdownContextValue extends ThemingProps<DropdownTheme> {
    activeIndex: number | null;
    dismissOnClick?: boolean;
    getItemProps: ReturnType<typeof useInteractions>["getItemProps"];
    handleSelect: (index: number | null) => void;
}
export declare const DropdownContext: import("react").Context<DropdownContextValue | undefined>;
export declare function useDropdownContext(): DropdownContextValue;
