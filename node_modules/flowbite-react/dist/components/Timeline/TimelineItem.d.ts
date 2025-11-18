import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
import type { TimelineContentTheme } from "./TimelineContent";
import type { TimelinePointTheme } from "./TimelinePoint";
export interface TimelineItemTheme {
    root: {
        horizontal: string;
        vertical: string;
    };
    content: TimelineContentTheme;
    point: TimelinePointTheme;
}
export interface TimelineItemProps extends ComponentProps<"li">, ThemingProps<TimelineItemTheme> {
}
export declare const TimelineItem: import("react").ForwardRefExoticComponent<Omit<TimelineItemProps, "ref"> & import("react").RefAttributes<HTMLLIElement>>;
