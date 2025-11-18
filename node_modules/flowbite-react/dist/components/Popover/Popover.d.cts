import type { Placement } from "@floating-ui/react";
import type { ComponentProps, Dispatch, ReactNode, SetStateAction } from "react";
import type { ThemingProps } from "../../types";
import type { FloatingArrowTheme } from "../Floating";
export interface PopoverTheme {
    base: string;
    inner: string;
    content: string;
    arrow: Omit<FloatingArrowTheme, "style">;
}
export interface PopoverProps extends Omit<ComponentProps<"div">, "content" | "style">, ThemingProps<PopoverTheme> {
    arrow?: boolean;
    content: ReactNode;
    placement?: "auto" | Placement;
    trigger?: "hover" | "click";
    initialOpen?: boolean;
    open?: boolean;
    onOpenChange?: Dispatch<SetStateAction<boolean>>;
}
export declare function Popover(props: PopoverProps): import("react/jsx-runtime").JSX.Element;
export declare namespace Popover {
    var displayName: string;
}
