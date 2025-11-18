import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface TimelineBodyTheme {
    base: string;
}
export interface TimelineBodyProps extends ComponentProps<"p">, ThemingProps<TimelineBodyTheme> {
}
export declare const TimelineBody: import("react").ForwardRefExoticComponent<Omit<TimelineBodyProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
