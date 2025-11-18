import type { ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface ToastTheme {
    root: {
        base: string;
        closed: string;
    };
    toggle: {
        base: string;
        icon: string;
    };
}
export type ToastDuration = 75 | 100 | 150 | 200 | 300 | 500 | 700 | 1000;
export interface ToastProps extends ComponentProps<"div">, ThemingProps<ToastTheme> {
    duration?: ToastDuration;
}
export declare const Toast: import("react").ForwardRefExoticComponent<Omit<ToastProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
