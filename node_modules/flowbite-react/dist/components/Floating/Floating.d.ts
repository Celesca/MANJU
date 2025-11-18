import type { Placement } from "@floating-ui/core";
import type { ComponentProps, ReactNode } from "react";
export interface FloatingTheme {
    arrow: FloatingArrowTheme;
    animation: string;
    base: string;
    content: string;
    hidden: string;
    style: {
        auto: string;
        dark: string;
        light: string;
    };
    target: string;
}
export interface FloatingArrowTheme {
    base: string;
    placement: string;
    style: {
        dark: string;
        light: string;
        auto: string;
    };
}
export type FloatingStyle = "dark" | "light" | "auto";
export interface FloatingProps extends Omit<ComponentProps<"div">, "content" | "style"> {
    animation?: false | `duration-${number}`;
    arrow?: boolean;
    content: ReactNode;
    placement?: "auto" | Placement;
    style?: FloatingStyle;
    theme: FloatingTheme;
    trigger?: "hover" | "click";
    minWidth?: number;
}
/**
 * @see https://floating-ui.com/docs/react-dom-interactions
 */
export declare function Floating({ animation, arrow, children, className, content, placement, style, theme, trigger, minWidth, ...props }: FloatingProps): import("react/jsx-runtime").JSX.Element;
export declare namespace Floating {
    var displayName: string;
}
