import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
export interface TimelineTimeTheme {
    base: string;
}
export interface TimelineTimeProps extends ComponentProps<"time">, ThemingProps<TimelineTimeTheme> {
}
export declare const TimelineTime: import("react").ForwardRefExoticComponent<Omit<TimelineTimeProps, "ref"> & import("react").RefAttributes<HTMLTimeElement>>;
