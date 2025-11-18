import { type ComponentProps } from "react";
import type { FlowbiteHeadingLevel, ThemingProps } from "../../types";
export interface TimelineTitleTheme {
    base: string;
}
export interface TimelineTitleProps extends ComponentProps<"h1">, ThemingProps<TimelineTitleTheme> {
    as?: FlowbiteHeadingLevel;
}
export declare const TimelineTitle: import("react").ForwardRefExoticComponent<Omit<TimelineTitleProps, "ref"> & import("react").RefAttributes<HTMLHeadingElement>>;
