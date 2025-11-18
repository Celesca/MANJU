import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
import type { TimelineBodyTheme } from "./TimelineBody";
import type { TimelineTimeTheme } from "./TimelineTime";
import type { TimelineTitleTheme } from "./TimelineTitle";
export interface TimelineContentTheme {
    root: {
        base: string;
        horizontal: string;
        vertical: string;
    };
    time: TimelineTitleTheme;
    title: TimelineTimeTheme;
    body: TimelineBodyTheme;
}
export interface TimelineContentProps extends ComponentProps<"div">, ThemingProps<TimelineContentTheme> {
}
export declare const TimelineContent: import("react").ForwardRefExoticComponent<Omit<TimelineContentProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
