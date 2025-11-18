import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
import type { TimelineItemTheme } from "./TimelineItem";
export interface TimelineTheme {
    root: {
        direction: {
            horizontal: string;
            vertical: string;
        };
    };
    item: TimelineItemTheme;
}
export interface TimelineProps extends ComponentProps<"ol">, ThemingProps<TimelineTheme> {
    horizontal?: boolean;
}
export declare const Timeline: import("react").ForwardRefExoticComponent<Omit<TimelineProps, "ref"> & import("react").RefAttributes<HTMLOListElement>>;
