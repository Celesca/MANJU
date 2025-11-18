import type { Placement } from "@floating-ui/core";
import type { ComponentProps, ReactNode } from "react";
import type { ThemingProps } from "../../types";
import { type FloatingTheme } from "../Floating";
export type TooltipTheme = FloatingTheme;
export interface TooltipProps extends Omit<ComponentProps<"div">, "content" | "style">, ThemingProps<TooltipTheme> {
    animation?: false | `duration-${number}`;
    arrow?: boolean;
    content: ReactNode;
    placement?: "auto" | Placement;
    style?: "dark" | "light" | "auto";
    trigger?: "hover" | "click";
}
/**
 * @see https://floating-ui.com/docs/react-dom-interactions
 */
export declare function Tooltip(props: TooltipProps): import("react/jsx-runtime").JSX.Element;
export declare namespace Tooltip {
    var displayName: string;
}
