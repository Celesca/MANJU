import type { ExtendedRefs } from "@floating-ui/react";
import type { Dispatch, HTMLProps, ReactElement, ReactNode, SetStateAction } from "react";
import type { ThemingProps } from "../../types";
import { type ButtonProps } from "../Button";
import type { FloatingProps, FloatingTheme } from "../Floating";
import type { DropdownDividerTheme } from "./DropdownDivider";
import type { DropdownHeaderTheme } from "./DropdownHeader";
import type { DropdownItemTheme } from "./DropdownItem";
export interface DropdownFloatingTheme extends FloatingTheme, DropdownDividerTheme, DropdownHeaderTheme {
    item: DropdownItemTheme;
}
export interface DropdownTheme {
    floating: DropdownFloatingTheme;
    content: string;
    inlineWrapper: string;
    arrowIcon: string;
}
export interface DropdownProps extends Pick<FloatingProps, "placement" | "trigger">, Omit<ButtonProps, keyof ThemingProps<DropdownTheme>>, ThemingProps<DropdownTheme> {
    arrowIcon?: boolean;
    dismissOnClick?: boolean;
    floatingArrow?: boolean;
    inline?: boolean;
    label?: ReactNode;
    enableTypeAhead?: boolean;
    renderTrigger?: (theme: DropdownTheme) => ReactElement;
    "data-testid"?: string;
}
export interface TriggerProps extends Omit<ButtonProps, keyof ThemingProps<DropdownTheme>> {
    refs: ExtendedRefs<HTMLElement>;
    inline?: boolean;
    theme: DropdownTheme;
    setButtonWidth?: Dispatch<SetStateAction<number | undefined>>;
    getReferenceProps: (userProps?: HTMLProps<Element> | undefined) => Record<string, unknown>;
    renderTrigger?: (theme: DropdownTheme) => ReactElement;
}
export declare function Dropdown(props: DropdownProps): import("react/jsx-runtime").JSX.Element;
export declare namespace Dropdown {
    var displayName: string;
}
